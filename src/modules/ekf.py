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
import jaxlie

from .points import Point, PointSet


# =============================================================================
# State container
# =============================================================================

@dataclass
class EkfState:
    """Mean of X = ⟨T, v, ω, g^B, d^B, p_1^B, ..., p_{N_f}^B⟩."""
    T:       jaxlie.SE3                                  # T_{B_k, B_0}
    v:       jnp.ndarray                                 # (3,)
    omega:   jnp.ndarray                                 # (3,)
    g_B:     jnp.ndarray                                 # (3,)
    d_B:     jnp.ndarray                                 # (3,)
    p_F:     jnp.ndarray                                 # (N_f, 3); rows aligned with feature_ids
    feature_ids: list[int] = field(default_factory=list) # ids matching p_F rows

    def dim(self) -> int:
        """Tangent dimension: 6 + 3·4 + 3·N_f = 18 + 3 N_f."""
        ...


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
        sized by visibility per point in F_set."""
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


    def update(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply all staged measurements in a single Kalman update.

        Returns (K, H) for the staged measurements. Needed for the post-update
        cross-covariance (I - K H) Φ P^{k-1,+} in the joint pose-pose block
        (§eq:187).
        """
        ...

    def get_gain(self) -> jnp.ndarray:
        """Kalman gain for the currently staged measurements (§eq:49).
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

    def marginalise(self, p: Point) -> None:
        """Remove p from the EKF state.

        Deletes the corresponding 3 rows/columns from X̂ and P. No
        information loss for remaining states.
        """
        ...

    # ---- outputs ---------------------------------------------------------

    @property
    def state(self) -> EkfState: ...

    @property
    def covariance(self) -> jnp.ndarray: ...

    def pose_prior(self) -> tuple[jaxlie.SE3, jaxlie.SE3, jnp.ndarray]:
        """(T̂_{k-1}^+, T̂_k^-, P_pose,joint^{12×12}) for the solver's pose
        prior factor (§eq:187).
        """
        ...

    def feature_output(self) -> tuple[jnp.ndarray, jnp.ndarray, list[int]]:
        """(p_F, P_FF, ids) — EKF feature points emitted directly to output,
        bypassing the joint solver to avoid double-counting (the EKF's
        feature info enters the solver via the pose prior).
        """
        ...