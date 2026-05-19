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
from cv2 import KeyPoint
import jax.numpy as jnp
import jax
import jaxlie


from .points import Point, PointSet
from .utils import skew, make_psd

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
                  a_ids: list = None,
                  c_ids: list = None,
                  b_ids: list = None,
                  ) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """Relative pose Δ = T_b T_a⁻¹ and 6x6 covariance.

    Args
    ----
    T_a, T_b      : SE(3) endpoints.
    covar_a       : (n_a, n_a) at k-1 post-update, ordered by a_ids.
    covar_b       : (n_b, n_b) at k post-update (or pre-update), ordered by b_ids.
    state_matrix  : Φ_{k-1}, (n_c, n_c), ordered by c_ids — the EKF state
                    dim at propagation time, which may differ from a_ids if
                    features were marginalised between t_a and propagation.
    update_matrix : (I - K_k H_k), (n_b, n_b), ordered by b_ids. None pre-update.
    a_ids, c_ids, b_ids : feature-id orderings at the three time points.

    Returns (ΔT, Σ_Δξ).
    """
    delta_T = T_b @ T_a.inverse()
 
    # Build S_ac: select a → c. (n_c, n_a). Core 18 dims always identity.
    n_a = covar_a.shape[0]
    n_c = state_matrix.shape[0]
    S_ac = _select_matrix(a_ids, c_ids, n_a, n_c)
    PhiPa_in_c = state_matrix @ S_ac @ covar_a    # (n_c, n_a)

    if update_matrix is None:
        cross = PhiPa_in_c[:6, :6]                # eq 183
    else:
        assert b_ids is not None, "post-update path requires b_ids"
        # b_ids ⊆ c_ids (b is c post-update-and-marginalise/augment).
        S_cb = _select_matrix(c_ids, b_ids, n_c, update_matrix.shape[0])  # (n_b, n_c)
        cross = (update_matrix @ S_cb @ PhiPa_in_c)[:6, :6]                # eq 187

    Caa = covar_a[:6, :6]
    Cbb = covar_b[:6, :6]
    P_joint = jnp.block([[Caa,    cross.T],
                         [cross,  Cbb    ]])

    Ad = T_a.adjoint()
    J  = jnp.concatenate([-Ad, Ad], axis=1)
    Sigma_DeltaXi = J @ P_joint @ J.T
    Sigma_DeltaXi = make_psd(Sigma_DeltaXi) 
    return delta_T, Sigma_DeltaXi


def _select_matrix(src_ids: list[int], dst_ids: list[int],
                   n_src: int, n_dst: int) -> jnp.ndarray:
    """(n_dst, n_src) selector. Core 18 always identity-to-identity.
    Each fid in dst that exists in src maps that 3-block; missing fids
    contribute zero rows (won't happen if dst ⊆ src)."""
    S = jnp.zeros((n_dst, n_src))
    S.at[:18, :18].set(jnp.eye(18))
    for i_dst, fid in enumerate(dst_ids):
        if fid not in src_ids:
            continue
        i_src = src_ids.index(fid)
        S.at[18 + 3*i_dst:18 + 3*i_dst + 3, 18 + 3*i_src:18 + 3*i_src + 3].set(jnp.eye(3))
    return S
 

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
        
        #Camera intrinsics
        self.K_L = jnp.asarray(self.calib["camera_intrinsics"]["left"]["k_matrix"])
        self.K_R = jnp.asarray(self.calib["camera_intrinsics"]["right"]["k_matrix"])
        
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
        
        #For measurement and update
        self._staged_pixels = None                          
        self._staged_gravity = False

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

    def h_pixels(self, F_set: PointSet,
                v_meas: jnp.ndarray | None = None) -> jnp.ndarray:
        """Stacked pixel-projection predictions (§eq:h_full).

        For each point in F_set (id-order), project the state-tracked
        position p_i^B through the relevant camera(s). Visibility from
        get_px_type. v_meas=None ⇒ noise-free.
        """
        blocks = []
        for p in self._ordered_features(F_set):
            i = self.X.feature_ids.index(p.id)
            p_B = self.X.p_F[i]
            
            # Visibility determines L/R/both. Read pixel slots to decide.
            if p.uL_curr is not None:
                blocks.append(project_point(p_B, self.calib, "L")[0])
            if p.uR_curr is not None:
                blocks.append(project_point(p_B, self.calib, "R")[0])
        
        y = jnp.concatenate(blocks) if blocks else jnp.zeros(0)
        if v_meas is not None:
            y = y + v_meas
        return y


    def h_gravity(self, v_meas: float | None = None) -> float:
        """‖g^B‖² + v_g."""
        val = float(jnp.dot(self.X.g_B, self.X.g_B))
        if v_meas is not None:
            val = val + v_meas
        return val

    def _proj_jacobian(self, p_B: jnp.ndarray, camera: str) -> jnp.ndarray:
        """∂π(R_cB·p_B + t_c)/∂p_B for one body-frame point (§eq:H_pi).

        Returns 2x3 matrix. No distortion assumed (rectified images).
        """
        assert camera in ["L", "R"], "Camera must be one of 'L' or 'R'"
        if camera == "L":
            K = self.K_L
            T_cb = self.T_LB
        else:
            K = self.K_R
            T_cb = self.T_RB
         
        R_cB = T_cb.rotation().as_matrix()
        t_c  = T_cb.translation()

        p_c = R_cB @ p_B + t_c                            # (3,)
        z   = p_c[2]
        fx, fy = K[0, 0], K[1, 1]

        # ∂π/∂p_c = [[fx/z, 0, -fx·x/z²], [0, fy/z, -fy·y/z²]]
        dpi_dpc = jnp.array([
            [fx / z, 0.0,    -fx * p_c[0] / (z * z)],
            [0.0,    fy / z, -fy * p_c[1] / (z * z)],
        ])                                                # (2, 3)

        return dpi_dpc @ R_cB                             # (2, 3)

    def get_measurement_jacobian(self, F_set: PointSet) -> jnp.ndarray:
        """H_k for the pixel measurements (§eq:H_full).

        Block-sparse: each feature contributes 2-row (mono) or 4-row (stereo)
        nonzero only in its p_i^B columns. All other state columns zero.
        """
        n = self.P.shape[0]
        rows = []

        for p in self._ordered_features(F_set):
            i = self.X.feature_ids.index(p.id)
            col = 18 + 3 * i
            p_B = self.X.p_F[i]
            
            if p.uL_curr is not None:
                row = jnp.zeros((2, n))
                J = self._proj_jacobian(p_B, "L")          # (2, 3)
                row = row.at[:, col:col+3].set(J)
                rows.append(row)
            if p.uR_curr is not None:
                row = jnp.zeros((2, n))
                J = self._proj_jacobian(p_B, "R")          # (2, 3)
                row = row.at[:, col:col+3].set(J)
                rows.append(row)

        return jnp.concatenate(rows, axis=0) if rows else jnp.zeros((0, n))

    def _H_gravity(self) -> jnp.ndarray:
        """H_g = [0 | 2 g^Bᵀ | 0] : 1xn, nonzero only in g^B columns."""
        n = self.P.shape[0]
        H = jnp.zeros((1, n))
        return H.at[0, 12:15].set(2.0 * self.X.g_B)


    # ---- noise covariances (read from calibration.yaml) ------------------


    def get_measurement_noise(self, F_set: PointSet) -> jnp.ndarray:
        """R for the staged pixel measurements (§eq:R).

        Block-diagonal σ_px² I, sized by per-point visibility. Each mono
        point contributes a 2x2 block; each stereo a 4x4.
        """
        rows = []
        for p in self._ordered_features(F_set):
            nu = 0
            if p.uL_curr is not None: nu += 2
            if p.uR_curr is not None: nu += 2
            rows.append(self.var_px * jnp.eye(nu))
        return block_diag(*rows) if rows else jnp.zeros((0, 0))


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
    def _ordered_features(self, F_set: PointSet) -> list[Point]:
        """Features that are both in F_set and in EKF state, ordered by EKF
        feature_ids. Ensures consistency between h, H, R, and the staged
        observation vector."""
        return [F_set.get(fid) for fid in self.X.feature_ids
                if fid in F_set]
   
    def _filter_to_state(self, F_set: PointSet) -> PointSet:
        """Drop points from F_set that aren't in the EKF state (not yet
        augmented)."""
        state_ids = set(self.X.feature_ids)
        return F_set.filter(lambda p: p.id in state_ids)

    def _stack_pixels(self, ordered_pts: list[Point]) -> jnp.ndarray:
        """Stack observed (u, v) pairs from each point's uL_curr/uR_curr,
        in the same order as h_pixels would predict."""
        px = []
        for p in ordered_pts:
            if p.uL_curr is not None:
                px.append(jnp.array([p.uL_curr.pt[0], p.uL_curr.pt[1]]))
            if p.uR_curr is not None:
                px.append(jnp.array([p.uR_curr.pt[0], p.uR_curr.pt[1]]))
        return jnp.concatenate(px) if px else jnp.zeros(0)

    def add_pixel_measurements(self, F_set: PointSet) -> None:
        """Stage pixel reprojection terms for the upcoming update.

        Builds the observation vector y_pix from each Point's uL_curr/uR_curr,
        in the same id-order h_pixels and get_measurement_jacobian use.
        Points without a state entry (not yet augmented) are filtered out.
        """
        in_state = self._filter_to_state(F_set)
        y_pix = self._stack_pixels(self._ordered_features(in_state))
        self._staged_pixels = (in_state, y_pix)
        

    def add_gravity_measurement(self) -> None:
        """Stage ‖g^B‖² = g² (§eq:h_gravity)."""
        self._staged_gravity = True


    def update(self) -> jnp.ndarray:
        """Apply all staged measurements in a single Kalman update.

        Builds:
            y_pred = h(X, 0), stacked over all staged measurement groups
            y_meas = stacked observations
            H      = stacked measurement Jacobian
            H_v    = stacked noise Jacobian
            R      = stacked measurement noise covariance

        Then applies:
            innov = y_meas - y_pred
            K     = P H^T (H P H^T + H_v R H_v^T)^{-1}
            X    ← X ⊕ K · innov                       (composite manifold)
            P    ← (I - K H) P
            
        Returns (I - K H) for use in the post-update relative-pose cross
        block (§eq:187).
        """
        assert self._staged_pixels is not None or self._staged_gravity, \
            "update called with no measurements staged"

        # ---- assemble stacked measurement system -------------------------
        y_pred_blocks = []
        y_meas_blocks = []
        H_blocks      = []
        R_blocks      = []

        if self._staged_pixels is not None:
            F_set, y_pix = self._staged_pixels
            y_pred_blocks.append(self.h_pixels(F_set))
            y_meas_blocks.append(y_pix)
            H_blocks.append(self.get_measurement_jacobian(F_set))
            R_blocks.append(self.get_measurement_noise(F_set))

        if self._staged_gravity:
            y_pred_blocks.append(jnp.array([self.h_gravity()]))
            y_meas_blocks.append(jnp.array([self.g ** 2]))
            H_g = self._H_gravity()                       # (1, n)
            H_blocks.append(H_g)
            R_blocks.append(jnp.array([[self.var_g]]))    # (1, 1)

        y_pred = jnp.concatenate(y_pred_blocks)
        y_meas = jnp.concatenate(y_meas_blocks)
        H      = jnp.concatenate(H_blocks, axis=0)        # (m, n)
        R      = block_diag(*R_blocks)                    # (m, m)
        # H_v = I since noise is additive in both pixel and gravity channels (§eq:Hv)
        # Equivalent to: H_v R H_v^T = R
        n = self.P.shape[0]

        # ---- innovation, gain, update ------------------------------------
        innov   = y_meas - y_pred                         # (m,)
        S       = H @ self.P @ H.T + R                    # (m, m)
        K       = self.P @ H.T @ jnp.linalg.inv(S)        # (n, m)

        correction = K @ innov                            # (n,) tangent vector
        self._apply_correction(correction)

        I_minus_KH = jnp.eye(n) - K @ H
        self.P = I_minus_KH @ self.P
        self.P = 0.5 * (self.P + self.P.T)                # symmetrise

        # ---- clear staging ------------------------------------------------
        self._staged_pixels = None
        self._staged_gravity = False

        return I_minus_KH


    def _apply_correction(self, delta: jnp.ndarray) -> None:
        """Apply the Kalman correction K·innov to the state (composite ⊕).

        Distinct from _apply_oplus (used by propagate):
            - Pose: T ← Exp(+δξ) · T̂  (right-perturbation correction, positive sign)
            - ℝ³ blocks: standard addition

        The positive sign matches the right-perturbation convention
        (Section sec:composite): T_true = T̂ · Exp(δξ).
        """
        X = self.X
        X.T       = X.T @ jaxlie.SE3.exp(delta[:6])
        X.v       = X.v     + delta[6:9]
        X.omega   = X.omega + delta[9:12]
        X.g_B     = X.g_B   + delta[12:15]
        X.d_B     = X.d_B   + delta[15:18]
        if len(X.feature_ids) > 0:
            X.p_F = X.p_F + delta[18:].reshape(-1, 3)


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

        Builds a single-feature mono-`camera` view of the EKF state and runs
        the same h_pixels / H / R machinery as the update step:
            S = H P H^T + R                (H_v = I, additive noise §eq:Hv)

        Returns:
            u_hat : (2,) projected pixel.
            S     : (2, 2) innovation covariance.
        """
        assert camera in ("L", "R"), "camera must be 'L' or 'R'"
        if id not in self.X.feature_ids:
            raise KeyError(f"feature {id} not in EKF state")

        # ---- single-feature mono view of this camera ---------------------
        # The point's pixel slot only signals visibility for h/H/R dispatch;
        # the actual value doesn't matter for prediction or covariance.
        dummy = KeyPoint(0.0, 0.0, 1.0)
        p = Point(id=id)
        if camera == "L":
            p.uL_curr = dummy
        else:
            p.uR_curr = dummy
        F_set = PointSet("fp_px_view")
        F_set.add(p)

        # ---- prediction and innovation covariance ------------------------
        u_hat = self.h_pixels(F_set)                              # (2,)
        H     = self.get_measurement_jacobian(F_set)              # (2, n)
        R     = self.get_measurement_noise(F_set)                 # (2, 2)
        S     = H @ self.P @ H.T + R                              # (2, 2)
        S     = 0.5 * (S + S.T)                                   # symmetrise
        return u_hat, S


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