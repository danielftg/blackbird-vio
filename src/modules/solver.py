"""
solver.py — Joint Two-frame stereo point-velocity solver with pose prior.

Two-view stereo bundle adjustment over the joint state
    x_joint = ⟨ ΔT, s_1, ..., s_N ⟩
where ΔT = T_{B_k, B_{k-1}} ∈ SE(3) is the pose change,
and s_i ∈ R^6 (SS/SM/MS) or R^4 (MM) is per-point position+velocity.

The EKF supplies a pose prior (ΔT̂_EKF, Σ_prior) that anchors the
6-DOF gauge freedom (§sec:joint_solver). The solver runs Gauss-Newton
on the whitened normal equations until convergence, then transports
per-point states from B_{k-1} to B_k.

All physical parameters and noise levels read from calibration.yaml and algorithm.yaml
No hardcoded values.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import jax.numpy as jnp
import jaxlie

from .points import Point, PointSet, PixelType


# =============================================================================
# Joint state container
# =============================================================================

@dataclass
class JointState:
    """Mean of x_joint = ⟨ ΔT, s_1, ..., s_N ⟩.

    Per-point states are stored as a flat (Σ dim(s_i),) array with row
    offsets recorded in `offsets` so each point's block can be sliced
    in O(1). `corr_types[i]` and `point_ids[i]` align row-wise with
    `offsets[i]`.
    """
    delta_T:    jaxlie.SE3                       # ΔT = T_{B_k, B_{k-1}}
    s:          jnp.ndarray                      # (Σ dim(s_i),) flat
    point_ids:  list[int] = field(default_factory=list)
    corr_types: list[PixelType] = field(default_factory=list)
    offsets:    list[int] = field(default_factory=list)
                                                 # row index of s_i in s

    def dim(self) -> int:
        """Total tangent dim: 6 + Σ dim(s_i). 6 for ΔT, then 6 (SS/SM/MS)
        or 4 (MM) per point."""
        ...

    def s_block(self, i: int) -> jnp.ndarray:
        """Return s_i slice from the flat s array."""
        ...


# =============================================================================
# Solver
# =============================================================================

class Solver:
    """Gauss-Newton joint solver with EKF pose prior (§sec:joint_solver).

    Per-frame lifecycle:
        1. initialise(F_pre ∪ I, ΔT̂_EKF, Σ_prior, search_inits)
        2. run() — iterates GN to convergence, returns posterior
        3. transport_to_Bk() — applies T_Δ to per-point states & cov
    """

    # ---- construction ----------------------------------------------------

    def __init__(self, calib: dict, alg: dict) -> None:
        """Bootstrap. calib = parsed calibration.yaml. Reads camera
        intrinsics/extrinsics, σ_px, GN tolerances and iteration cap from
        algorithm.yaml.
        """
        ...

    # ---- initialisation --------------------------------------------------

    def initialise(self,
                   points: PointSet,
                   delta_T_prior: jaxlie.SE3,
                   Sigma_prior: jnp.ndarray,
                   search_inits: dict[int, tuple[jnp.ndarray, jnp.ndarray]]
                   ) -> None:
        """Build the JointState from F_pre U I and the EKF pose prior.

        points : the candidate-feature and interest-point sets that enter
                 the joint solve (EKF features F do NOT enter — their
                 information enters via the pose prior).
        delta_T_prior, Sigma_prior : §eq:DeltaXi_prior. Used as both the
                 prior factor and the initial guess for ΔT.
        search_inits : per-id (p_init, v_init) from the search step
                 (§sec:search_region). v_init is a 3-vector for full
                 velocity types or a scalar for MM.

        Filters points by PixelType — points that don't
        produce a valid solver correspondence are skipped.
        """
        ...

    # ---- model: measurement function and Jacobians -----------------------

    def transport(self, s_i: jnp.ndarray, corr: PixelType,
                  delta_T: jaxlie.SE3, dt: float,
                  d_perp: jnp.ndarray | None = None) -> jnp.ndarray:
        """Transport p_i from B_{k-1} to B_k via ΔT (§eq:transport_joint
        for SS/SM/MS, §eq:transport_mm for MM). dt = Δt_{k-1}.
        d_perp required only for MM."""
        ...

    def h_point(self, s_i: jnp.ndarray, corr: CorrType,
                delta_T: jaxlie.SE3, dt: float,
                d_perp: jnp.ndarray | None = None) -> jnp.ndarray:
        """Stacked pixel projection for one point at (k-1, k) per its
        correspondence type (§Assembled measurement models).
        Returns a (ν_i,) array."""
        ...

    def J_point(self, s_i: jnp.ndarray, corr: CorrType,
                delta_T: jaxlie.SE3, dt: float,
                d_perp: jnp.ndarray | None = None
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per-point measurement Jacobian split into (J_pose, J_si) where:
            J_pose : (ν_i, 6) Jacobian w.r.t. right perturbation of ΔT
                     (§eq:Jh_xi). Zero block for k-1 rows (no ΔT dependence
                     at the previous frame).
            J_si   : (ν_i, dim(s_i)) per-point block (§eq:Jh_v_stereo or
                     MM analogue).
        Stacked into the global Jacobian by build_normal_equations.
        """
        ...

    def d_perp(self, s_i: jnp.ndarray, delta_T: jaxlie.SE3,
               camera: str) -> jnp.ndarray:
        """Perpendicular direction for MM points (§sec:d_perp).

        Recomputed each GN iteration since it depends on ΔT through
        t_base^c (§eq:t_base). camera ∈ {'L', 'R'}.
        """
        ...

    # ---- normal equations ------------------------------------------------

    def build_normal_equations(self,
                               x: JointState,
                               z: dict[int, jnp.ndarray]
                               ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Whiten and stack the prior + per-point measurement blocks
        (§eq:stacked).

        z : observed pixel coordinates per point id. Each value is a
            (ν_i,) array stacked in the order h_point would produce.

        Returns (A, b) such that the GN normal equations are
            (Aᵀ A) τ = Aᵀ b.
        """
        ...

    def prior_block(self, delta_T: jaxlie.SE3,
                    delta_T_prior: jaxlie.SE3,
                    Sigma_prior: jnp.ndarray
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Whitened pose-prior factor (§eq:pose_prior).

        Returns (A_0, b_0) sized to the full joint tangent dim, with the
        ΔT block populated by Σ_prior^{-1/2} and the residual computed
        as Σ_prior^{-1/2} · (ΔT̂_EKF ⊖ ΔT^t).
        """
        ...

    # ---- iteration -------------------------------------------------------

    def step(self, x: JointState, z: dict[int, jnp.ndarray]) -> jnp.ndarray:
        """One GN iteration: build normal equations, solve, return τ.
        Caller applies the increment via apply_increment.
        """
        ...

    def apply_increment(self, x: JointState,
                        tau: jnp.ndarray) -> JointState:
        """Apply a GN increment (§eq:increment): right-compose Exp(τ_ξ)
        onto ΔT, vector-add τ_si onto each s_i."""
        ...

    def run(self, z: dict[int, jnp.ndarray]
            ) -> tuple[JointState, jnp.ndarray]:
        """Iterate GN to convergence. Returns (x_post, Σ_x_post) where
        Σ_x_post = (AᵀA)^{-1} at the final iterate (§Covariance at
        convergence).

        Termination on ‖τ‖ < ε or max_iters reached (algorithm.yaml).
        """
        ...

    # ---- output transport: B_{k-1} → B_k ---------------------------------

    def transport_to_Bk(self,
                        x_post: JointState,
                        Sigma_post: jnp.ndarray
                        ) -> tuple[JointState, jnp.ndarray]:
        """Apply T_Δ to per-point states and joint covariance
        (§eq:T_block, §eq:Sigma_Bk).

        Per-point T_{Δ,i} from §eq:T_Delta_SS (SS/SM/MS) or §eq:T_Delta_MM
        (MM). Pose tangent block is left unchanged. Pose-point
        cross-covariances are transformed on the point side only.

        Returns (x_post in B_k, Σ in B_k).
        """
        ...

    # ---- output assembly -------------------------------------------------

    def write_back(self, x_Bk: JointState, Sigma_Bk: jnp.ndarray,
                   F_pre: PointSet, I_set: PointSet) -> None:
        """Write per-point (p_curr, v_curr, Σ_curr) and pose-point
        cross-covariances back onto the corresponding Point objects in
        F_pre and I_set, in B_k coordinates.
        """
        ...