"""
ekf.py — Composite-Manifold Continuous-Discrete EKF.

Implements the M-CD-EKF for state X = ⟨T, v, ω, g^B, d^B, p_1^B,...⟩
as derived in §sec:ekf, §sec:system_model, §sec:measurement.

Two measurement channels:
  - Pixel reprojection of EKF feature points (§sec:measurement)
  - Gravity-magnitude pseudo-measurement (§eq:h_gravity)

Variable state dimension: features added/removed via augment / marginalise.

All physical parameters and noise spectral densities are read from
calibration.yaml. No hardcoded values.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from jax.scipy.linalg import block_diag
import jax.numpy as jnp
import jax
import jaxlie


from .points import Point, PointSet
from .utils import skew

# =============================================================================
# Shared functions
# =============================================================================
def project_point(p_B: jnp.ndarray, calib: dict, camera: str) -> jnp.ndarray:
    """Project a body-frame point into camera  ∈ {"L", "R"}.

        û = π(R_cB · p_B + t_c)

    Args:
        p_B    : body-frame point. Shape (1,3) for a single point or
                 (N, 3) for a batch.
        calib  : parsed calibration.yaml; reads camera_intrinsics and
                 camera_extrinsics for the requested camera.
        camera : "L" or "R".

    Returns:
        Pixel coordinate(s) in the requested camera. Shape (1,2) for a
        single input, (N, 2) for a batch.

    Raises:
        AssertionError if camera is not "L" or "R"
    """
    assert camera in ["L", "R"], "Camera must be one of 'L' or 'R'"
    
    if camera == "L":
        T_cb = calib["camera_extrinsics"]["left"]["cog_cam"]
        dist_coeff = calib["camera_intrinsics"]["left"]["dist_coeff"]
        k_matrix = calib["camera_intrinsics"]["left"]["k_matrix"]

    else:
        T_cb = calib["camera_extrinsics"]["right"]["cog_cam"]
        dist_coeff = calib["camera_intrinsics"]["right"]["dist_coeff"]
        k_matrix = calib["camera_intrinsics"]["right"]["k_matrix"]
   
    # Distortion guard
    dist_coeff = jnp.asarray(dist_coeff)                # (4,)
    if jnp.any(dist_coeff != 0.0):
        raise NotImplementedError("Non-zero distortion coefficients are not supported.")
   
    T_cb = jaxlie.SE3.from_matrix(jnp.asarray(T_cb))
    k_matrix = jnp.asarray(k_matrix)                    # 3x3

    # Normalise to batch shape (N, 3)
    single = (p_B.ndim == 1)
    if single:
        p_B  = p_B[None, :]                              # (1, 3)

    # 1. Rigid body → camera frame  (SE3 acts on points directly)
    p_c = jax.vmap(T_cb.apply)(p_B)                    # (N, 3)

   
    # 2. Homogenize → K·[I|0]·p̃_c → dehomogenize
    ones = jnp.ones((p_c.shape[0], 1))
    p_h  = jnp.concatenate([p_c, ones], axis=-1)       # (N, 4)
    P    = k_matrix @ jnp.eye(3, 4)                    # (3, 4)
    uv_h = (P @ p_h.T).T                               # (N, 3)
    uv   = uv_h[:, :2] / uv_h[:, 2:3]                 # (N, 2)
    return uv  


def relative_pose(T_a: jaxlie.SE3, T_b: jaxlie.SE3,
                  covar_a: jnp.ndarray, covar_b: jnp.ndarray,
                  state_matrix: jnp.ndarray,
                  update_matrix: jnp.ndarray = None,
                  a_ids: list = None, b_ids: list = None,
                  ) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """Relative pose Δ = T_b T_a⁻¹ and 6x6 covariance.
 
    Args
    ----
    T_a, T_b        : SE(3) endpoints (T̂_{k-1}^+ and T̂_k^- or T̂_k^+).
    covar_a         : full covariance at k-1, shape (n_a, n_a).
    covar_b         : full covariance at k,   shape (n_b, n_b).
    state_matrix    : Φ_{k-1} = I + dt·F, shape (n_a, n_a).
    update_matrix   : (I - K_k H_k), shape (n_b, n_b). Pass None for
                      pre-update.
    a_ids, b_ids    : feature-id ordering at k-1 and at k. Required for
                      post-update; used to map surviving rows of Φ P^{k-1,+}
                      to the (possibly smaller) post-marginalisation dim.
 
    Returns
    -------
    (ΔT, Σ_Δξ).
    """
    delta_T = T_b @ T_a.inverse()
 
    PhiPa = state_matrix @ covar_a              # (n_a, n_a)
 
    if update_matrix is None:
        cross = PhiPa[:6, :6]                   # eq 183
    else:
        assert a_ids is not None and b_ids is not None, \
            "post-update path requires a_ids and b_ids"
        rows = list(range(18))
        for fid in b_ids:
            i_a = a_ids.index(fid)
            rows.extend([18 + 3*i_a + j for j in range(3)])
        rows = jnp.asarray(rows)
        PhiPa_sliced = PhiPa[rows, :]           # (n_b, n_a)
        cross = (update_matrix @ PhiPa_sliced)[:6, :6]    # eq 187
 
    Caa = covar_a[:6, :6]
    Cbb = covar_b[:6, :6]
    P_joint = jnp.block([[Caa,    cross.T],
                         [cross,  Cbb    ]])
 
    Ad = T_a.adjoint()
    J  = jnp.concatenate([-Ad, Ad], axis=1)     # (6, 12)
    Sigma_DeltaXi = J @ P_joint @ J.T
    Sigma_DeltaXi = 0.5 * (Sigma_DeltaXi + Sigma_DeltaXi.T)
    return delta_T, Sigma_DeltaXi
 

# =============================================================================
# State container
# =============================================================================

@dataclass
class CoreEkfState:
    """Mean of X = ⟨T, v, ω, g^B, d^B⟩."""
    T:       jaxlie.SE3                                  # T_{B_k, B_0}
    v:       jnp.ndarray                                 # (3,)
    omega:   jnp.ndarray                                 # (3,)
    g_B:     jnp.ndarray                                 # (3,)
    d_B:     jnp.ndarray                                 # (3,)
    

@dataclass
class EkfState(CoreEkfState):
    """Mean of X = ⟨T, v, ω, g^B, d^B, p_1^B, ..., p_{N_f}^B⟩."""
    p_F:     jnp.ndarray = None                          # (N_f, 3); rows aligned with feature_ids
    feature_ids: list[int] = field(default_factory=list) # ids matching p_F rows

    def get_core_state(self) -> CoreEkfState:
        return CoreEkfState(
            T=self.T,
            v=self.v,
            omega=self.omega,
            g_B=self.g_B,
            d_B=self.d_B,
        )
    

# =============================================================================
# EKF
# =============================================================================

class Ekf:
    """Composite-manifold continuous-discrete EKF.

    Per-frame lifecycle:
        1. propagate(u, dt)            — §eq:T_euler, §eq:f
        2. add_pixel_measurements(F)   — accumulate by point id
        3. add_gravity_measurement()   — accumulate ‖g‖² pseudo-meas
        4. update()                    — §eq:49–52, single combined update
        5. augment / marginalise       — §sec:augment
    """

    # ---- construction ----------------------------------------------------

    def __init__(self, calib: dict) -> None:
        """Bootstrap. calib = parsed calibration.yaml."""
        self.calib = calib
        
        #Initial state
        init_x = calib["ekf_sys"]["init_state"]
        self.X = EkfState(
            T=jaxlie.SE3.exp(jnp.asarray(init_x["pose"])),
            v=jnp.asarray(init_x["lin_vel"]),
            omega=jnp.asarray(init_x["ang_vel"]),
            g_B=jnp.asarray(init_x["grav"]),
            d_B=jnp.asarray(init_x["dist"])
        )

        #Initial state covariance
        init_p = calib["ekf_sys"]["init_covar"]
        self.P = block_diag(
            jnp.diag(jnp.asarray(init_p["var_pose"])),
            jnp.diag(jnp.asarray(init_p["var_lin_vel"])), 
            jnp.diag(jnp.asarray(init_p["var_ang_vel"])), 
            jnp.diag(jnp.asarray(init_p["var_grav"])), 
            jnp.diag(jnp.asarray(init_p["var_dist"]))
        )
        
        #Gravity parameter
        self.g =  calib["ekf_meas"]["g_norm"]
        
        #Continuous time Process noise spectral density
        sys_noise = calib["ekf_sys"]["covar"]
        self.Q_c = block_diag(
            jnp.diag(jnp.asarray(sys_noise["var_act"])), 
            jnp.diag(jnp.asarray(sys_noise["var_ang_vel"])), 
            jnp.diag(jnp.asarray(sys_noise["var_grav"])), 
            jnp.diag(jnp.asarray(sys_noise["var_dist"]))
        )
        
        # Measurement noise
        meas_noise = calib["ekf_meas"]["covar"]
        self.var_px = meas_noise["var_px"] 
        self.var_g = meas_noise["var_g"]
       
        #Camera extrinsics
        self.T_LB = jaxlie.SE3.from_matrix(jnp.asarray(
            calib["camera_extrinsics"]["left"]["cog_cam"])
        )
        self.T_RB = jaxlie.SE3.from_matrix(jnp.asarray(
            calib["camera_extrinsics"]["right"]["cog_cam"])
        )
        
        # ---- drone params ---------------------------------------------
        plat = self.calib["drone_parameters"]
        self._drone = {}
        self._drone["m"]    = plat["mass"]
        self._drone["Cd"]   = plat["drag_coeff"]
        self._drone["l"]    = plat["arm_length"]
        self._drone["J"]    = jnp.diag(jnp.asarray(plat["inertia_tensor"]))         
        self._drone["k_f"]  = jnp.mean(jnp.asarray([
            plat["rotor_1"]["ang_thr_coeff"], 
            plat["rotor_2"]["ang_thr_coeff"], 
            plat["rotor_3"]["ang_thr_coeff"], 
            plat["rotor_4"]["ang_thr_coeff"]
        ]))
        self._drone["k_m"]  = jnp.mean(jnp.asarray([
            plat["rotor_1"]["ang_tor_coeff"], 
            plat["rotor_2"]["ang_tor_coeff"], 
            plat["rotor_3"]["ang_tor_coeff"], 
            plat["rotor_4"]["ang_tor_coeff"]   
        ]))
        self._drone["Jinv"] = jnp.linalg.inv(self._drone["J"])

        # Augment param
        self.aug_cross_seed = float(self.calib["ekf_sys"]["aug_cross_seed"])


    # ---- system model: f and its derivatives -----------------------------

    def f(self, u: jnp.ndarray,
        w: jnp.ndarray | None = None) -> jnp.ndarray:
        """System model dX/dt (§eq:f) returned in tangent-space ordering
        matching the perturbation vector (eq:delta_x):
            (ξ_v, v, ω, g^B, d^B, p_1^B, ..., p_{N_f}^B)

        where ξ_v = (v, ω) is the velocity twist driving the pose
        (eq:Tdot: Ṫ = -ξ_v^∧ T). The pose block of dX/dt is the body-frame
        twist itself; the propagation step composes it via Exp() on SE(3).

        w=None ⇒ noise-free dynamics (used for state propagation).
        """
        # ---- unpack noise -------------------------------------------------
        if w is None:
            w_u     = jnp.zeros(4)
            w_omega = jnp.zeros(3)
            w_g     = jnp.zeros(3)
            w_d     = jnp.zeros(3)
        else:
            w_u, w_omega, w_g, w_d = w[:4], w[4:7], w[7:10], w[10:13]

        # ---- Unpack variables -------------------------------------------
        X = self.X
        m = self._drone["m"]
        Cd = self._drone["Cd"]
        l = self._drone["l"] 
        J = self._drone["J"]          
        k_f = self._drone["k_f"]
        k_m = self._drone["k_m"] 
        Jinv = self._drone["Jinv"] 
        
        # ---- noisy input -------------------------------------------------
        u_n  = u + w_u                                                  # (4,)
        T_sum = jnp.sum(u_n)

        # ---- pose: ξ_v = (v, ω). Ṫ = -ξ_v^∧ T handled by propagation. ----
        xi_v = jnp.concatenate([X.v, X.omega])                          # (6,)

        # ---- v̇  (eq:vdot) ------------------------------------------------
        F_thrust = jnp.array([0.0, 0.0, -T_sum])                        # (3,) FRD
        v_dot = (F_thrust / m
                - jnp.cross(X.omega, X.v)
                - (Cd / m) * X.v
                + X.g_B
                + X.d_B)

        # ---- ω̇  (eq:omegadot) -------------------------------------------
        l_s = l / jnp.sqrt(2.0)
        km_kf = k_m / k_f
        tau = jnp.array([
            l_s   * (u_n[3] + u_n[2] - u_n[0] - u_n[1]),   # roll
            l_s   * (u_n[0] + u_n[3] - u_n[1] - u_n[2]),   # pitch
            km_kf * (u_n[1] + u_n[3] - u_n[0] - u_n[2]),   # yaw
        ])
        omega_dot = Jinv @ (tau - jnp.cross(X.omega, J @ X.omega)) + w_omega

        # ---- ġ^B  (eq:gdot) ----------------------------------------------
        g_dot = -jnp.cross(X.omega, X.g_B) + w_g

        # ---- ḋ^B  (eq:ddot) ----------------------------------------------
        d_dot = -jnp.cross(X.omega, X.d_B) + w_d

        # ---- ṗ_i^B  (eq:pdot) for each feature ---------------------------
        if len(X.feature_ids) > 0:
            # ṗ_i = -v - ω × p_i, vectorised over features
            p_dot = -X.v[None, :] - jnp.cross(X.omega[None, :], X.p_F)   # (N_f, 3)
            p_dot_flat = p_dot.reshape(-1)
        else:
            p_dot_flat = jnp.zeros(0)

        # ---- assemble in tangent ordering --------------------------------
        return jnp.concatenate([xi_v, v_dot, omega_dot, g_dot, d_dot, p_dot_flat])

    def get_system_jacobian(self) -> jnp.ndarray:
        """F = Df/DX about the current mean (§eq:F).

        Row/column order (matches the perturbation vector eq:delta_x):
            ξ, v, ω, g^B, d^B, p_1^B, ..., p_{N_f}^B
        i.e. block sizes 6, 3, 3, 3, 3, then 3 per feature.

        Pose row (δξ-row, §eq:F_se3):
            F[δξ, (δv, δω)] = -Ad_{T̂⁻¹},   all other entries in row = 0.
        Pose column (δξ-column):
            Zero everywhere (no ℝ³ dynamics depend on pose).
        """
        X = self.X
        v     = X.v
        omega = X.omega
        g_B   = X.g_B
        d_B   = X.d_B
        J     = self._drone["J"]
        Jinv  = self._drone["Jinv"]
        Cd    = self._drone["Cd"]
        m     = self._drone["m"]

        N_f = len(X.feature_ids)
        n   = 18 + 3 * N_f
        F   = jnp.zeros((n, n))

        # ---- ℝ³ block-level building blocks ------------------------------
        Ox    = skew(omega)                       # [ω]_×
        Vx    = skew(v)                           # [v]_×
        Gx    = skew(g_B)                         # [g^B]_×
        Dx    = skew(d_B)                         # [d^B]_×
        JOx   = skew(J @ omega)                   # [J ω]_×
        I3    = jnp.eye(3)

        # row offsets
        iξ, iv, iω, ig, id = 0, 6, 9, 12, 15

        # ---- δξ-row (pose error dynamics) -------------------------------
        # F[ξ, (v, ω)] = -Ad_{T̂⁻¹}  (6×6)
        Ad_inv = X.T.inverse().adjoint()           # (6, 6)
        F = F.at[iξ:iξ+6, iv:iv+6].set(-Ad_inv)

        # ---- δv-row (translational, eq:vdot derivatives) ----------------
        # F_vv = -[ω]_× - (Cd/m) I_3
        # F_vω =  [v]_×
        # F_vg = I_3
        # F_vd = I_3
        F = F.at[iv:iv+3, iv:iv+3].set(-Ox - (Cd / m) * I3)
        F = F.at[iv:iv+3, iω:iω+3].set(Vx)
        F = F.at[iv:iv+3, ig:ig+3].set(I3)
        F = F.at[iv:iv+3, id:id+3].set(I3)

        # ---- δω-row (rotational, eq:omegadot derivatives) ---------------
        # F_ωω = J⁻¹ ([J ω]_× - [ω]_× J)
        F_ww = Jinv @ (JOx - Ox @ J)
        F = F.at[iω:iω+3, iω:iω+3].set(F_ww)

        # ---- δg-row (gravity transport) ---------------------------------
        # F_gω = [g^B]_×
        # F_gg = -[ω]_×
        F = F.at[ig:ig+3, iω:iω+3].set(Gx)
        F = F.at[ig:ig+3, ig:ig+3].set(-Ox)

        # ---- δd-row (disturbance transport) -----------------------------
        # F_dω = [d^B]_×
        # F_dd = -[ω]_×
        F = F.at[id:id+3, iω:iω+3].set(Dx)
        F = F.at[id:id+3, id:id+3].set(-Ox)

        # ---- feature rows -----------------------------------------------
        # F_p_iv = -I_3
        # F_p_iω = [p_i^B]_×
        # F_p_ip_i = -[ω]_×
        # cross-point entries = 0
        for i in range(N_f):
            ip = 18 + 3 * i
            F = F.at[ip:ip+3, iv:iv+3].set(-I3)
            F = F.at[ip:ip+3, iω:iω+3].set(skew(X.p_F[i]))
            F = F.at[ip:ip+3, ip:ip+3].set(-Ox)

        return F


    def get_noise_jacobian(self) -> jnp.ndarray:
        """G = ∂f/∂w about the current mean (§eq:G).

        Noise vector w ∈ ℝ¹³ ordered as (w_u, w_ω, w_g, w_d) with sizes 4,3,3,3.
        Row/column order: rows match the perturbation vector eq:delta_x,
        columns match w.

        Non-zero blocks:
            G[v, w_u]  = B_v        (3 × 4)   actuator → linear force
            G[ω, w_u]  = B_ω        (3 × 4)   actuator → torque (mixed)
            G[ω, w_ω]  = I_3        (3 × 3)   unmodelled torque
            G[g, w_g]  = I_3        (3 × 3)   gravity drift
            G[d, w_d]  = I_3        (3 × 3)   disturbance drift
        Pose and feature rows are zero (no direct noise).
        """
        X     = self.X
        m     = self._drone["m"]
        l     = self._drone["l"]
        k_f   = self._drone["k_f"]
        k_m   = self._drone["k_m"]
        Jinv  = self._drone["Jinv"]

        N_f = len(X.feature_ids)
        n   = 18 + 3 * N_f
        G   = jnp.zeros((n, 13))

        # ---- B_v: actuator → linear force/mass (eq:B_v) ----------------
        B_v = jnp.array([
            [ 0.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]) / m                                                        # (3, 4)

        # ---- B_ω: actuator → torque, premultiplied by J⁻¹ (eq:B_ω) -----
        l_s   = l / jnp.sqrt(2.0)
        km_kf = k_m / k_f
        dtau_du = jnp.array([
            [-l_s,  -l_s,   l_s,   l_s],   # roll
            [ l_s,  -l_s,  -l_s,   l_s],   # pitch
            [-km_kf, km_kf, -km_kf, km_kf],# yaw
        ])                                                            # (3, 4)
        B_omega = Jinv @ dtau_du                                      # (3, 4)

        # column offsets in w: w_u [0:4], w_ω [4:7], w_g [7:10], w_d [10:13]
        iv, iω, ig, id = 6, 9, 12, 15
        I3 = jnp.eye(3)

        G = G.at[iv:iv+3,  0:4 ].set(B_v)
        G = G.at[iω:iω+3,  0:4 ].set(B_omega)
        G = G.at[iω:iω+3,  4:7 ].set(I3)
        G = G.at[ig:ig+3,  7:10].set(I3)
        G = G.at[id:id+3, 10:13].set(I3)

        # pose rows and feature rows stay zero
        return G

    # ---- measurement model: h and its derivatives ------------------------
    def h(self, F_set: PointSet,
                v_meas: jnp.ndarray | None = None) -> jnp.ndarray:
        ...
        

    def h_pixels(self, F_set: PointSet,
                v_meas: jnp.ndarray | None = None) -> jnp.ndarray:
        """Stacked pixel-projection measurement (§eq:h_full).

        Iterates F_set in id-order, projecting each Point's state-tracked
        position p_i^B through the relevant camera(s). Visibility (mono /
        stereo) read from each Point's get_px_type. v_meas=None ⇒ noise-free.
        """
        ...

    def h_gravity(self, v_meas: float | None = None) -> float:
        """Gravity-magnitude pseudo-measurement ‖g^B‖² (§eq:h_gravity).
        Independent of features."""
        ...

    def get_measurement_jacobian(self, F_set: PointSet) -> jnp.ndarray:
        """H_k = Dh/DX about the current mean (§eq:H_full).

        Iterates F_set in the same id-order as h_pixels. 
        """
        ...

    def get_measurement_noise_jacobian(self, F_set: PointSet) -> jnp.ndarray:
        """H_v = ∂h/∂v (§eq:Hv).

        """
        ...

    # ---- noise covariances (read from calibration.yaml) ------------------


    def get_measurement_noise(self, F_set: PointSet) -> jnp.ndarray:
        """Measurement noise R for the staged measurements (§eq:R + R_g if
        gravity staged). Block-diagonal, σ_px² weights from calibration.yaml,
        sized by visibility per point in F_set.
        """
        ...

    # ---- continuous propagation -----------------------------------------

    def propagate(self, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Integrate mean and covariance to t + dt (§Propagation).

        Discrete Euler step on the composite manifold (§eq:T_euler):
            x̂_{k+1}^- = x̂_k^+ ⊕ dt · f(x̂_k^+, u_k, 0)
            Φ_k       = I + dt · F_k
            P_{k+1}^- = Φ_k P_k^+ Φ_k^T + dt · G_k Q_c G_k^T
        """
        # ---- linearisations about the *current* mean -
        F = self.get_system_jacobian()
        G = self.get_noise_jacobian()
        
        # ---- evaluate the derivative at the current mean (noise-free) ----
        f_val = self.f(u, w=None)        
        
        # ---- mean update via ⊕ ------------------------------------------
        self._apply_oplus(dt * f_val)            # mutates self.X in place
        
        # ---- covariance update ------------------------------------------
        n   = int(self.P.shape[0])
        Phi = jnp.eye(n) + dt * F
        self.P = Phi @ self.P @ Phi.T + dt * G @ self.Q_c @ G.T
        self.P = 0.5 * (self.P + self.P.T)       # symmetrise
        
        return Phi
   
    def _apply_oplus(self, delta: jnp.ndarray) -> None:
        """Apply δ to the current state in tangent space.

        Block layout matches the perturbation vector (eq:delta_x):
            delta[:6]   → SE(3) pose: T̂ ← Exp(-δξ) · T̂   (left-acting, eq:T_euler)
            delta[6:9]  → v̂ ← v̂ + δv
            delta[9:12] → ω̂ ← ω̂ + δω
            delta[12:15]→ ĝ ← ĝ + δg
            delta[15:18]→ d̂ ← d̂ + δd
            delta[18:]  → p̂_F ← p̂_F + δp  (per feature, reshape (N_f, 3))

        Note the negation on the pose: f returns +ξ_v (the body-frame twist),
        while Ṫ = -ξ_v^∧ T means the actual SE(3) increment is Exp(-ξ_v·dt).
        """
        X = self.X

        # Pose: left-acting Exp with negation (eq:T_euler)
        new_T = jaxlie.SE3.exp(-delta[:6]) @ X.T

        # Euclidean blocks
        new_v     = X.v     + delta[6:9]
        new_omega = X.omega + delta[9:12]
        new_g     = X.g_B   + delta[12:15]
        new_d     = X.d_B   + delta[15:18]

        # Feature points (Euclidean in body frame)
        N_f = len(X.feature_ids)
        if N_f > 0:
            d_p = delta[18:].reshape(N_f, 3)
            new_p_F = X.p_F + d_p
        else:
            new_p_F = X.p_F

        # Reassign EkfState
        X.T = new_T
        X.v = new_v
        X.omega = new_omega
        X.g_B = new_g
        X.d_B = new_d
        X.p_F = new_p_F

    # ---- discrete update: stage then apply -------------------------------

    def add_pixel_measurements(self, F_set: PointSet) -> None:
        """Stage pixel reprojection terms for the upcoming update.

        Matches state features to F_set entries by Point.id. Points in
        F_set without a corresponding state entry are ignored (must be
        augmented first); state entries without an F_set match are
        skipped this frame (e.g. occluded). Per-point block dim depends
        on visibility (νᵢ = 2 mono, 4 stereo).
        """
        ...

    def add_gravity_measurement(self) -> None:
        """Stage the ‖g^B‖² = g² pseudo-measurement (§eq:h_gravity)."""
        ...

    def get_gain(self) -> jnp.ndarray:
        """Kalman gain for the currently staged measurements (§eq:49).
        """
        ...

    def update(self) -> jnp.ndarray:
        """Apply all staged measurements in a single Kalman update.

        Returns I-KH for the staged measurements. Needed for the post-update
        cross-covariance (I - K H) Φ P^{k-1,+} in the joint pose-pose block
        (§eq:187).
        """
        ...

    def augment(self, p: Point) -> None:
        """Add p to the EKF state (§sec:augment).

        Grows state by 3 entries and covariance by 3 rows/cols. The new
        block uses Σ_new = p.Sigma_curr; cross-correlations are seeded as
        P J_aug^T with J_aug = δ 1 per the design choice in §sec:augment.
        """
        assert p.p_curr is not None and p.Sigma_curr is not None, \
            f"Augment requires p_curr and Sigma_curr populated on Point {p.id}"
        assert p.id not in self.X.feature_ids, \
            f"Feature {p.id} already in EKF state"

        p_new     = jnp.asarray(p.p_curr)             # (3,)
        Sigma_new = jnp.asarray(p.Sigma_curr[:3, :3]) # (3, 3)
        delta     =  self.aug_cross_seed

        # ---- grow mean -----------------------------------------------------
        if self.X.p_F is None: 
            new_p_F = p_new[None, :]                 # (1, 3)
        else:
            if self.X.p_F.shape[0] == 0:
                new_p_F = p_new[None, :]
            else:
                new_p_F = jnp.concatenate([self.X.p_F, p_new[None, :]], axis=0)

        self.X.p_F = new_p_F
        self.X.feature_ids = self.X.feature_ids + [p.id]

        # ---- grow covariance ----------------------------------------------
        n = int(self.P.shape[0])
        
        J_aug = delta * jnp.ones((3, n))             # (3, n),
        cross = J_aug @ self.P                       # (3, n) 

        P_new = jnp.zeros((n + 3, n + 3))
        P_new = P_new.at[:n,  :n ].set(self.P)
        P_new = P_new.at[:n,  n:n+3].set(cross.T)    # P J_aug^T
        P_new = P_new.at[n:n+3, :n].set(cross)       # J_aug P
        P_new = P_new.at[n:n+3, n:n+3].set(Sigma_new)

        # symmetrise (cheap insurance against float asymmetry)
        self.P = 0.5 * (P_new + P_new.T)


    def marginalise(self, id: int) -> None:
        """Remove feature `id` from the state (§sec:augment).

        Deletes 3 rows/cols from the covariance and the corresponding p_F
        row.
        """
        if id not in self.X.feature_ids:
            raise KeyError(f"feature {id} not in EKF state")

        # ---- which rows/cols correspond to this feature ------------------
        feat_idx  = self.X.feature_ids.index(id)
        start     = 18 + 3 * feat_idx
        keep_rows = jnp.concatenate([
            jnp.arange(start),
            jnp.arange(start + 3, int(self.P.shape[0])),
        ])

        # ---- drop rows/cols from P ---------------------------------------
        self.P = self.P[keep_rows, :][:, keep_rows]

        # ---- drop row from p_F and id from feature_ids -------------------
        keep_p = jnp.concatenate([
            self.X.p_F[:feat_idx],
            self.X.p_F[feat_idx + 1:],
        ], axis=0)
        new_ids = self.X.feature_ids[:feat_idx] + self.X.feature_ids[feat_idx + 1:]

        self.X.p_F = keep_p
        self.X.feature_ids = new_ids 
        

    # ---- outputs ---------------------------------------------------------

    @property
    def state(self) -> EkfState:
        """Return a snapshot of the current state.
        """
        return EkfState(
            T=self.X.T,
            v=self.X.v,
            omega=self.X.omega,
            g_B=self.X.g_B,
            d_B=self.X.d_B,
            p_F=self.X.p_F,
            feature_ids=list(self.X.feature_ids),
        )

    @property
    def covariance(self) -> jnp.ndarray:
        """Return a snapshot of the current covariance.
        """
        return self.P     

    def get_fp_body(self, id: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Body-frame position and 3x3 covariance for feature `id`.

        Reads p̂_i^B from the state and the 3x3 marginal of P at the
        feature's slot.

        Raises:
            KeyError: if `id` is not in the EKF state.
        """
        if id not in self.X.feature_ids:
            raise KeyError(f"feature {id} not in EKF state")

        i      = self.X.feature_ids.index(id)
        p_B    = self.X.p_F[i]                              # (3,)
        start  = 18 + 3 * i
        Sigma  = self.P[start:start + 3, start:start + 3]   # (3, 3)
        return p_B, Sigma


    def get_fp_px(self, id: int, camera: str) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Pixel-space prediction and 2x2 innovation covariance for feature `id`
        in the requested camera.

        Returns:
            u_hat : (2,) projected pixel.
            S     : (2, 2) innovation covariance H_i P⁻ H_iᵀ + R_i, restricted
                    to the requested camera's mono block (eq:Hv, eq:R).
        """
        assert camera in ("L", "R"), "camera must be 'L' or 'R'"

        # ---- pixel prediction --------------------------------------------
        p_B, Sigma_p = self.get_fp_body(id)
        u_hat = project_point(p_B, self.calib, camera)          # (2,)
        ...


    def feature_output(self) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[int]]:
        """Returns (p_F, Σ_per_point, ids):
            p_F          : (N_f, 3) body-frame positions
            Σ_per_point  : (N_f, 3, 3) per-feature covariances
            ids          : list of N_f point ids matching the row order
        """
        N_f = len(self.X.feature_ids)
        if N_f == 0:
            return (None,
                    None,
                    [])

        # Per-feature 3×3 blocks
        Sigmas = [
            self.P[18 + 3*i:18 + 3*i + 3, 18 + 3*i:18 + 3*i + 3]
            for i in range(N_f)
        ]                                              # (N_f, 3, 3)
        Ps = [self.X.p_F[i,:] for i in range(N_f)]
        return Ps, Sigmas, list(self.X.feature_ids)