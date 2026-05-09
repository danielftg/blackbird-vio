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
import jax.numpy as jnp
import jax
import jaxlie

from .points import Point, PointSet


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
    p_F:     jnp.ndarray                                 # (N_f, 3); rows aligned with feature_ids
    feature_ids: list[int] = field(default_factory=list) # ids matching p_F rows

    def dim(self) -> int:
        """Tangent dimension: 6 + 3·4 + 3·N_f = 18 + 3 N_f."""
        ...
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
        ...

    # ---- system model: f and its derivatives -----------------------------

    def f(self, X: EkfState, u: jnp.ndarray,
          w: jnp.ndarray | None = None) -> jnp.ndarray:
        """System model dX/dt (§eq:f). w=None ⇒ noise-free."""
        ...

    def get_system_jacobian(self) -> jnp.ndarray:
        """F = Df/DX about the current mean (§eq:F).
        """
        ...

    def get_noise_jacobian(self) -> jnp.ndarray:
        """G = ∂f/∂w about the current mean (§eq:G).
        """
        ...

    # ---- measurement model: h and its derivatives ------------------------

    def h_pixels(self, X: EkfState, F_set: PointSet,
                v_meas: jnp.ndarray | None = None) -> jnp.ndarray:
        """Stacked pixel-projection measurement (§eq:h_full).

        Iterates F_set in id-order, projecting each Point's state-tracked
        position p_i^B through the relevant camera(s). Visibility (mono /
        stereo) read from each Point's get_px_type. v_meas=None ⇒ noise-free.
        """
        ...

    def h_gravity(self, X: EkfState,
                v_meas: float | None = None) -> float:
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

    def get_process_noise_density(self) -> jnp.ndarray:
        """Continuous-time spectral density Q ∈ R^{13×13} (§eq:Q).

        Block-diagonal: Q_u (4×4), Q_ω (3×3), Q_g (3×3), Q_d (3×3).
        Pulled from calib['ekf_sys']['covar'].
        """
        ...

    def get_measurement_noise(self, F_set: PointSet) -> jnp.ndarray:
        """Measurement noise R for the staged measurements (§eq:R + R_g if
        gravity staged). Block-diagonal, σ_px² weights from calibration.yaml,
        sized by visibility per point in F_set.
        """
        ...

    # ---- continuous propagation -----------------------------------------

    def propagate(self, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Integrate mean and covariance to t + dt (§Propagation).

        Returns the discrete state-transition matrix Φ_{k-1} = I + dt·F,
        needed for the joint pose-pose covariance across the propagation step
        (§eq:183, the (Φ P^{k-1,+})_ξξ block).     
        """
        ...

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

    # ---- state augmentation / marginalisation (§sec:augment) -------------

    def augment(self, p: Point) -> None:
        """Add p to the EKF state.

        Initial position and covariance read from p.p_curr, p.Sigma_curr;
        cross-correlations seeded as J_aug = δ·I per the design choice
        in §sec:augment.
        """
        ...

    def marginalise(self, id: int) -> None:
        """Remove point with the id from the EKF state.

        Deletes the corresponding 3 rows/columns from X̂ and P. No
        information loss for remaining states.
        """
        ...

    # ---- outputs ---------------------------------------------------------

    @property
    def state(self) -> EkfState: ...

    @property
    def covariance(self) -> jnp.ndarray: ...

    def get_fp_body(self, id) -> tuple[jnp.ndarray, jnp.ndarray]:
        """In body frame: Retrieve and return the feature point 'id' from the current state.
        Also return its 3x3 covariance.
        """
        ...
    def get_fp_px(self, id, camera:str) -> tuple[jnp.ndarray, jnp.ndarray]:
        """In camere frame: Retrieve and return the feature point 'id' from the current state.
        Project point and covariance (H_i P⁻ H_iᵀ + R_i) into the given camera. Return the pixel and 2x2 covariance.
        
        """

    def feature_output(self) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[int]]:
        """Returns (p_F, Σ_per_point, ids) where:
            p_F : (N_f, 3) positions
            Σ_per_point : (N_f, 3, 3) per-feature marginal covariances
            ids : list of N_f point ids
        """
        ...