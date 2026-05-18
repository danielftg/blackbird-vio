"""
solver.py — Joint two-frame stereo point-velocity solver with pose prior.

Two-view bundle adjustment over the joint state
    x_joint = ⟨ ΔT, s_1, ..., s_N ⟩
where ΔT = T_{B_k, B_{k-1}} ∈ SE(3) is the pose change, and s_i is
per-point position+velocity (R^6 for SS/SM/MS, R^4 for MM).

The EKF supplies a pose prior (ΔT̂_EKF, Σ_prior) that anchors the
6-DOF gauge freedom (§sec:joint_solver). Solver runs Gauss-Newton on
the whitened normal equations until convergence, then transports
per-point states from B_{k-1} to B_k.

"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp
import jaxlie

from .points import Point, PointSet, PixelType
from .vision import CandidateSample
from .utils  import np_skew

# =============================================================================
# Joint state container
# =============================================================================

@dataclass
class JointState:
    """Mean of x_joint = ⟨ ΔT, s_1, ..., s_N ⟩.

    Per-point states stored as a flat (Σ dim(s_i),) array with row
    offsets in `offsets`. `corr_types[i]` and `point_ids[i]` align
    row-wise with `offsets[i]`.
    """
    delta_T:    jaxlie.SE3
    s:          jnp.ndarray
    point_ids:  list[int]       = field(default_factory=list)
    corr_types: list[PixelType] = field(default_factory=list)
    offsets:    list[int]       = field(default_factory=list)

    def dim(self) -> int:
        """6 (pose) + Σ dim(s_i)."""
        return 6 + int(self.s.shape[0])

    def s_block(self, i: int) -> jnp.ndarray:
        """Slice s_i from the flat array."""
        d = _s_dim(self.corr_types[i])
        return self.s[self.offsets[i]:self.offsets[i] + d]


def _s_dim(corr: PixelType) -> int:
    """Per-point state dim: 6 for SS/SM/MS, 4 for MM."""
    return 4 if corr == PixelType.M_M else 6



# =============================================================================
# Solver
# =============================================================================

class Solver:
    """Gauss-Newton joint solver with EKF pose prior (§sec:joint_solver)."""

    # ---- construction ----------------------------------------------------

    def __init__(self, calib: dict, alg: dict) -> None:
        self.calib = calib
        self.alg   = alg

        # Camera intrinsics + extrinsics
        K_L = np.asarray(calib["camera_intrinsics"]["left"]["k_matrix"],  dtype=np.float64)
        K_R = np.asarray(calib["camera_intrinsics"]["right"]["k_matrix"], dtype=np.float64)
        T_LB = np.asarray(calib["camera_extrinsics"]["left"]["cog_cam"],  dtype=np.float64)
        T_RB = np.asarray(calib["camera_extrinsics"]["right"]["cog_cam"], dtype=np.float64)
        self.K    = {"L": K_L, "R": K_R}
        self.R_cB = {"L": T_LB[:3, :3], "R": T_RB[:3, :3]}
        self.t_c  = {"L": T_LB[:3,  3], "R": T_RB[:3,  3]}
        self.R_Bc = {"L": T_LB[:3, :3].T, "R": T_RB[:3, :3].T}

        # Whitening: Σ_z = σ²·I_ν is isotropic ⇒ Σ^{-1/2} = (1/σ)·I.
        # Equivalent to a generic Cholesky-based whitener applied per-block.
        self.sigma_px  = float(calib["ekf_meas"]["covar"]["var_px"]) ** 0.5
        self.inv_sigma = 1.0 / self.sigma_px

        # GN tolerances
        self.eps_tau   = float(alg["solver"]["eps_tau"])
        self.max_iters = int  (alg["solver"]["max_iters"])

        # Per-call state (cleared each initialise)
        self._x:           JointState | None = None
        self._z:           dict[int, jnp.ndarray] = {}
        self._mono_cam:    dict[int, str]    = {}
        self._prior_T:     jaxlie.SE3 | None = None
        self._Sigma_prior: np.ndarray | None = None
        self._L_prior:     np.ndarray | None = None  # Σ_prior^{-1/2}
        self._dt: float = 0.0

    # ---- initialisation --------------------------------------------------

    def initialise(self,
                   dt: float,
                   points: PointSet,
                   delta_T_prior: jaxlie.SE3,
                   Sigma_prior: jnp.ndarray,
                   search_inits: dict[int, CandidateSample]
                   ) -> None:
        """Build JointState and prepare measurements from F_pre ∪ I.

        Initial values:
          ΔT^0          = delta_T_prior
          p_i^{B_{k-1}} = CandidateSample.p_Bkm1
          v_i^p         = CandidateSample.v_p   (3-vector for SS/SM/MS,
                                                  projection v_p · d_⊥
                                                  for MM, sign-preserving)

        Σ_prior taken as Σ_{Δξ} directly: J_r^{-1} ≈ I near identity
        for one frame at typical rates (§eq:DeltaXi_prior).
        """
        # Reset per-call state
        self._dt          = float(dt)
        self._prior_T     = delta_T_prior
        self._Sigma_prior = np.asarray(Sigma_prior, dtype=np.float64)
        self._L_prior     = self._whitening(self._Sigma_prior)
        self._z           = {}
        self._mono_cam    = {}

        s_blocks:   list[np.ndarray] = []
        point_ids:  list[int]        = []
        corr_types: list[PixelType]  = []
        offsets:    list[int]        = []
        offset = 0

        T_k_to_km1 = delta_T_prior.inverse()
        
        for p in points:
            if p.id not in search_inits:
                continue
            
            cand = search_inits[p.id]
            corr  = p.get_px_type()

            z_i   = self._stack_measurement(p, corr)
            if z_i is None:
                continue
            
            cam_m = self._mm_camera(p, corr)
            self._mono_cam[p.id] = cam_m
            
            if cand.p_Bk is not None and cand.p_Bkm1 is None:
                cand.p_Bkm1 = T_k_to_km1.apply(jnp.asarray(cand.p_Bk))
            p_init = np.asarray(cand.p_Bkm1, dtype=np.float64).reshape(3)
            
            if corr == PixelType.M_M:
                # v_⊥ = sign-preserving projection of v_p onto d_⊥
                d_perp = np.asarray(self.d_perp(p_init, delta_T_prior, cam_m))
                v_perp = float(np.asarray(cand.v_p, dtype=np.float64).reshape(-1) @ d_perp)
                s_i = np.concatenate([p_init, [v_perp]])
            else:
                v_init = np.asarray(cand.v_p, dtype=np.float64).reshape(3)
                s_i = np.concatenate([p_init, v_init])

            s_blocks.append(s_i)
            point_ids.append(p.id)
            corr_types.append(corr)
            offsets.append(offset)
            offset += s_i.shape[0]
            self._z[p.id] = jnp.asarray(z_i)

        s_flat = jnp.asarray(np.concatenate(s_blocks)) if s_blocks else jnp.zeros(0)
        self._x = JointState(
            delta_T    = delta_T_prior,
            s          = s_flat,
            point_ids  = point_ids,
            corr_types = corr_types,
            offsets    = offsets,
        )

    def _whitening(self, Sigma: np.ndarray) -> np.ndarray:
        """Σ^{-1/2} via Cholesky: Σ = L Lᵀ ⇒ Σ^{-1/2} ≡ L^{-1}
        satisfies (Σ^{-1/2})ᵀ Σ^{-1/2} = Σ^{-1}."""
        L = np.linalg.cholesky(Sigma)
        return np.linalg.inv(L)

    def _stack_measurement(self, p: Point, corr: PixelType) -> np.ndarray | None:
        """Stack pixel measurements per correspondence type. Order matches h_point."""
        def _kp(kp): return np.array(kp.pt, dtype=np.float64) if kp is not None else None

        l_p, r_p = _kp(getattr(p, "uL_prev", None)), _kp(getattr(p, "uR_prev", None))
        l_c, r_c = _kp(getattr(p, "uL_curr", None)), _kp(getattr(p, "uR_curr", None))

        if corr == PixelType.S_S:
            if any(v is None for v in (l_p, r_p, l_c, r_c)): return None
            return np.concatenate([l_p, r_p, l_c, r_c])
        if corr == PixelType.S_M:
            mono = l_c if l_c is not None else r_c
            if any(v is None for v in (l_p, r_p, mono)): return None
            return np.concatenate([l_p, r_p, mono])
        if corr == PixelType.M_S:
            mono = l_p if l_p is not None else r_p
            if any(v is None for v in (mono, l_c, r_c)): return None
            return np.concatenate([mono, l_c, r_c])
        if corr == PixelType.M_M:
            mono_p = l_p if l_p is not None else r_p
            mono_c = l_c if l_c is not None else r_c
            if any(v is None for v in (mono_p, mono_c)): return None
            return np.concatenate([mono_p, mono_c])
        return None

    def _mm_camera(self, p: Point, corr: PixelType) -> str:
        """Pick mono cameras"""
        if corr == PixelType.S_M:
            if p.uL_curr is not None:
                return "L"
            if p.uR_curr is not None:
                return "R"
        if corr == PixelType.M_S or corr == PixelType.M_M: #TODO: Fix M_M handling for L-R and R-L in the future.
            if p.uL_prev is not None:
                return "L"
            if p.uR_prev is not None:
                return "R"
        return None

    # ---- model: projection, M, transport, h, J, d_perp -------------------

    def _project(self, p_B: np.ndarray, camera: str) -> np.ndarray:
        """Project body-frame point. Returns (2,)."""
        p_c = self.R_cB[camera] @ p_B + self.t_c[camera]
        K = self.K[camera]
        return np.array([K[0, 0] * p_c[0] / p_c[2] + K[0, 2],
                         K[1, 1] * p_c[1] / p_c[2] + K[1, 2]])

    def _M(self, p_B: np.ndarray, camera: str) -> np.ndarray:
        """M_c = (∂π/∂q) · R_cB ∈ R^{2x3} at q = R_cB·p_B + t_c."""
        p_c = self.R_cB[camera] @ p_B + self.t_c[camera]
        K = self.K[camera]
        fx, fy = K[0, 0], K[1, 1]
        X, Y, Z = p_c
        inv_Z, inv_Z2 = 1.0 / Z, 1.0 / (Z * Z)
        dpi = np.array([[fx * inv_Z, 0.0,        -fx * X * inv_Z2],
                        [0.0,        fy * inv_Z, -fy * Y * inv_Z2]])
        return dpi @ self.R_cB[camera]

    def transport(self, s_i: np.ndarray, corr: PixelType,
                  delta_T: jaxlie.SE3, dt: float,
                  d_perp: np.ndarray | None = None) -> np.ndarray:
        """Transport p_i: B_{k-1} → B_k (§eq:transport_joint / _mm)."""
        p = np.asarray(s_i[:3], dtype=np.float64)
        if corr == PixelType.M_M:
            p_pre = p + float(s_i[3]) * d_perp * dt
        else:
            p_pre = p + np.asarray(s_i[3:6], dtype=np.float64) * dt
        T = np.asarray(delta_T.as_matrix(), dtype=np.float64)
        return T[:3, :3] @ p_pre + T[:3, 3]

    def h_point(self, s_i: np.ndarray, corr: PixelType, camera_mono: str,
                delta_T: jaxlie.SE3, dt: float,
                d_perp: np.ndarray | None = None) -> np.ndarray:
        """Stacked predicted pixels, order matches _stack_measurement."""
        p_kml = np.asarray(s_i[:3], dtype=np.float64)
        p_k   = self.transport(s_i, corr, delta_T, dt, d_perp)
        if corr == PixelType.S_S:
            return np.concatenate([self._project(p_kml, "L"), self._project(p_kml, "R"),
                                   self._project(p_k,   "L"), self._project(p_k,   "R")])
        if corr == PixelType.S_M:
            return np.concatenate([self._project(p_kml, "L"), self._project(p_kml, "R"),
                                   self._project(p_k,   camera_mono)])
        if corr == PixelType.M_S:
            return np.concatenate([self._project(p_kml, camera_mono),
                                   self._project(p_k,   "L"), self._project(p_k,   "R")])
        return np.concatenate([self._project(p_kml, camera_mono),
                               self._project(p_k,   camera_mono)])

    def J_point(self, s_i: np.ndarray, corr: PixelType, camera_mono: str,
                delta_T: jaxlie.SE3, dt: float,
                d_perp: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (J_pose (v,6), J_si (v, dim(s_i))) per §eq:J_SS and MM analogue."""
        T  = np.asarray(delta_T.as_matrix(), dtype=np.float64)
        dR = T[:3, :3]
        p_kml = np.asarray(s_i[:3], dtype=np.float64)

        if corr == PixelType.M_M:
            v_perp = float(s_i[3])
            p_pre  = p_kml + v_perp * d_perp * dt
        else:
            v = np.asarray(s_i[3:6], dtype=np.float64)
            p_pre = p_kml + v * dt
        p_k = dR @ p_pre + T[:3, 3]

        M_L_km = self._M(p_kml, "L"); M_R_km = self._M(p_kml, "R")
        M_L_k  = self._M(p_k,   "L"); M_R_k  = self._M(p_k,   "R")
        M_mono_km = M_L_km if camera_mono == "L" else M_R_km
        M_mono_k  = M_L_k  if camera_mono == "L" else M_R_k

        # Action Jacobian (§eq:J_action): [ΔR, -ΔR·[p_pre]_×]
        J_action = np.hstack([dR, -dR @ np_skew(p_pre)])                       # (3, 6)

        def Jpose_at_k(M_c):       return M_c @ J_action                      # (2, 6)
        def Js_kml(M_c, dim):      return np.hstack([M_c, np.zeros((2, dim - 3))])
        def Js_k_full(M_c):        return np.hstack([M_c @ dR, M_c @ dR * dt])
        def Js_k_mm(M_c):
            return np.hstack([M_c @ dR, (M_c @ dR @ d_perp * dt).reshape(2, 1)])

        if corr == PixelType.S_S:
            J_pose = np.vstack([np.zeros((2, 6)), np.zeros((2, 6)),
                                Jpose_at_k(M_L_k), Jpose_at_k(M_R_k)])
            J_si   = np.vstack([Js_kml(M_L_km, 6), Js_kml(M_R_km, 6),
                                Js_k_full(M_L_k), Js_k_full(M_R_k)])
            return J_pose, J_si
        if corr == PixelType.S_M:
            J_pose = np.vstack([np.zeros((2, 6)), np.zeros((2, 6)),
                                Jpose_at_k(M_mono_k)])
            J_si   = np.vstack([Js_kml(M_L_km, 6), Js_kml(M_R_km, 6),
                                Js_k_full(M_mono_k)])
            return J_pose, J_si
        if corr == PixelType.M_S:
            J_pose = np.vstack([np.zeros((2, 6)),
                                Jpose_at_k(M_L_k), Jpose_at_k(M_R_k)])
            J_si   = np.vstack([Js_kml(M_mono_km, 6),
                                Js_k_full(M_L_k), Js_k_full(M_R_k)])
            return J_pose, J_si
        # MM
        J_pose = np.vstack([np.zeros((2, 6)), Jpose_at_k(M_mono_k)])
        J_si   = np.vstack([Js_kml(M_mono_km, 4), Js_k_mm(M_mono_k)])
        return J_pose, J_si

    def d_perp(self, p_Bkm1: np.ndarray, delta_T: jaxlie.SE3,
               camera: str) -> np.ndarray:
        """d_⊥ in B_{k-1} (§eq:d_perp). Recomputed each iteration."""
        T = np.asarray(delta_T.as_matrix(), dtype=np.float64)
        dR, dt_vec = T[:3, :3], T[:3, 3]
        R_cB, R_Bc, t_c = self.R_cB[camera], self.R_Bc[camera], self.t_c[camera]

        # §eq:t_base
        t_base = t_c - R_cB @ dR.T @ (R_Bc @ t_c + dt_vec)

        # Ray in camera frame
        p_c = R_cB @ p_Bkm1 + t_c
        ray = np.array([p_c[0] / p_c[2], p_c[1] / p_c[2], 1.0])

        n_c = np.cross(t_base, ray)
        d_B = R_Bc @ n_c
        norm = np.linalg.norm(d_B)
        if norm < 1e-12:
            # Degenerate (pure rotation between frames): d_⊥ undefined.
            # Fall back to body-z; downstream gates should catch this.
            return np.array([0.0, 0.0, 1.0])
        return d_B / norm

    # ---- normal equations ------------------------------------------------

    def build_normal_equations(self, x: JointState
                               ) -> tuple[np.ndarray, np.ndarray]:
        """Stack whitened prior + per-point blocks (§eq:stacked)."""
        D = x.dim()
        A_rows, b_rows = [], []

        A0, b0 = self.prior_block(x.delta_T, self._prior_T, self._L_prior, D)
        A_rows.append(A0); b_rows.append(b0)

        for i, pid in enumerate(x.point_ids):
            corr   = x.corr_types[i]
            s_i    = np.asarray(x.s_block(i), dtype=np.float64)
            cam_m  = self._mono_cam[pid]
            dperp  = self.d_perp(s_i[:3], x.delta_T, cam_m) if corr == PixelType.M_M else None

            h_i    = self.h_point(s_i, corr, cam_m, x.delta_T, self._dt, dperp)
            z_i    = np.asarray(self._z[pid], dtype=np.float64)
            J_pose, J_si = self.J_point(s_i, corr, cam_m, x.delta_T, self._dt, dperp)

            nu  = h_i.shape[0]
            d_i = _s_dim(corr)
            A_i = np.zeros((nu, D))
            A_i[:, :6] = J_pose
            A_i[:, 6 + x.offsets[i]:6 + x.offsets[i] + d_i] = J_si
            b_i = z_i - h_i

            # Scalar whitening
            A_rows.append(self.inv_sigma * A_i)
            b_rows.append(self.inv_sigma * b_i)

        return np.vstack(A_rows), np.concatenate(b_rows)

    def prior_block(self, delta_T: jaxlie.SE3,
                    delta_T_prior: jaxlie.SE3,
                    L_prior: np.ndarray,
                    D: int) -> tuple[np.ndarray, np.ndarray]:
        """Whitened pose-prior factor (§eq:pose_prior).

        b_0 = L · Log((ΔT^t)^{-1} · ΔT_prior)   (prior ⊖ estimate)
        A_0 = L · [I_6, 0]
        """
        delta = (delta_T.inverse() @ delta_T_prior).log()
        b0 = L_prior @ np.asarray(delta, dtype=np.float64)
        A0 = np.zeros((6, D))
        A0[:, :6] = L_prior
        return A0, b0

    # ---- iteration -------------------------------------------------------

    def step(self, x: JointState) -> np.ndarray:
        """One GN iteration: build (A,b), solve (AᵀA)τ = Aᵀb, return τ."""
        A, b = self.build_normal_equations(x)
        return np.linalg.solve(A.T @ A, A.T @ b)

    def apply_increment(self, x: JointState, tau: np.ndarray) -> JointState:
        """ΔT^{t+1} = ΔT^t · Exp(τ_ξ);  s_i^{t+1} = s_i^t + τ_si."""
        new_T = x.delta_T @ jaxlie.SE3.exp(jnp.asarray(tau[:6]))
        new_s = x.s + jnp.asarray(tau[6:])
        return JointState(delta_T=new_T, s=new_s,
                          point_ids=x.point_ids, corr_types=x.corr_types,
                          offsets=x.offsets)

    def run(self) -> tuple[JointState, jnp.ndarray]:
        """Iterate GN to convergence. Returns (x_post, Σ_post_full in B_{k-1})."""
        assert self._x is not None, "Solver not initialised."
        x = self._x
        for _ in range(self.max_iters):
            tau = self.step(x)
            x = self.apply_increment(x, tau)
            if float(np.linalg.norm(np.asarray(tau))) < self.eps_tau:
                break

        A, _ = self.build_normal_equations(x)
        Sigma_full = np.linalg.inv(A.T @ A)
        self._x = x
        return x, jnp.asarray(Sigma_full)

    # ---- output transport: B_{k-1} → B_k ---------------------------------

    def transport_to_Bk(self,
                        x_post: JointState,
                        Sigma_full: jnp.ndarray
                        ) -> tuple[JointState, dict]:
        """Transport per-point states and *diagonal* covariance to B_k.

        Returns (x_Bk, {"pose": (6,6), "points": {id: (d,d)}, "mm_d_perp_Bk": {id: (3,)}}).
        """
        Sigma = np.asarray(Sigma_full, dtype=np.float64)
        T = np.asarray(x_post.delta_T.as_matrix(), dtype=np.float64)
        dR, dt_vec = T[:3, :3], T[:3, 3]
        dt = self._dt

        Sigma_pose = Sigma[:6, :6]
        new_s_blocks:  list[np.ndarray]      = []
        point_sigmas:  dict[int, np.ndarray] = {}
        mm_d_perp_Bk:  dict[int, np.ndarray] = {}

        for i, pid in enumerate(x_post.point_ids):
            corr = x_post.corr_types[i]
            d_i  = _s_dim(corr)
            off  = 6 + x_post.offsets[i]
            s_i  = np.asarray(x_post.s_block(i), dtype=np.float64)
            Sig_i = Sigma[off:off + d_i, off:off + d_i]

            if corr == PixelType.M_M:
                d_perp_km = self.d_perp(s_i[:3], x_post.delta_T, self._mono_cam[pid])
                v_perp    = float(s_i[3])
                p_pre     = s_i[:3] + v_perp * d_perp_km * dt
                p_Bk      = dR @ p_pre + dt_vec
                s_Bk      = np.concatenate([p_Bk, [v_perp]])
                T_di      = np.block([[dR,                   (dR @ d_perp_km * dt).reshape(3, 1)],
                                       [np.zeros((1, 3)),     np.array([[1.0]])]])
                mm_d_perp_Bk[pid] = dR @ d_perp_km
            else:
                v    = s_i[3:6]
                p_Bk = dR @ (s_i[:3] + v * dt) + dt_vec
                v_Bk = dR @ v
                s_Bk = np.concatenate([p_Bk, v_Bk])
                T_di = np.block([[dR,                dR * dt],
                                  [np.zeros((3, 3)), dR     ]])

            new_s_blocks.append(s_Bk)
            point_sigmas[pid] = T_di @ Sig_i @ T_di.T

        s_flat_Bk = jnp.asarray(np.concatenate(new_s_blocks)) if new_s_blocks else jnp.zeros(0)
        x_Bk = JointState(delta_T=x_post.delta_T, s=s_flat_Bk,
                          point_ids=x_post.point_ids, corr_types=x_post.corr_types,
                          offsets=x_post.offsets)
        return x_Bk, {
            "pose":         jnp.asarray(Sigma_pose),
            "points":       point_sigmas,
            "mm_d_perp_Bk": mm_d_perp_Bk,
        }

    # ---- output assembly -------------------------------------------------

    def write_back(self, x_Bk: JointState, Sigma_Bk: dict,
                   points: PointSet) -> None:
        """Write (p_curr, v_curr, Σ_curr) onto each Point in B_k frame.

        For MM, v_curr = v_⊥ · d_⊥ in B_k (3-vector form).
        """
        point_sigmas   = Sigma_Bk["points"]
        mm_d_perp_Bk   = Sigma_Bk["mm_d_perp_Bk"]
        point_map      = {p.id: p for p in points}

        for i, pid in enumerate(x_Bk.point_ids):
            if pid not in point_map:
                continue
            corr = x_Bk.corr_types[i]
            s_i  = np.asarray(x_Bk.s_block(i), dtype=np.float64)
            p    = point_map[pid]

            p.p_curr     = s_i[:3]
            p.Sigma_curr = np.asarray(point_sigmas[pid], dtype=np.float64)

            if corr == PixelType.M_M:
                p.v_curr = float(s_i[3]) * mm_d_perp_Bk[pid]
            else:
                p.v_curr = s_i[3:6]