"""
stats.py — Statistical tests for the EKF / solver pipeline.

Three tests, all χ² under their null hypotheses:

  - feature_nis_gate         : per-feature stationarity test (§sec:feature_movement)
  - joint_consistency        : group-level inlier consistency (§sec:feature_movement)
  - admission_velocity_gate  : pre-EKF velocity test for new features (§sec:mono_admission_stats)

All confidence levels (alpha_NIS, alpha_adm) read from algorithm.yaml.
No hardcoded thresholds.

PERF NOTE: feature_nis_gate currently computes innovations and innovation
covariances. Then the EKF computes equivalent quantities inside update() to build the gain.
A future optimisation pass should, eliminating the duplicate work.
"""

from __future__ import annotations
import jax.numpy as jnp
from scipy.stats import chi2

from .points import Point, PointSet, CorrType
from .ekf import Ekf


# =============================================================================
# Building block: chi-squared cutoff
# =============================================================================

def chi2_threshold(alpha: float, dof: int) -> float:
    """χ²_{alpha, dof} cutoff. Used by every test below.

    alpha is the confidence level (e.g. 0.99); the test rejects when the
    statistic exceeds this cutoff.
    """
    ...


# =============================================================================
# Feature movement detection (§sec:feature_movement)
# =============================================================================

def feature_nis_gate(ekf: Ekf,
                     F_set: PointSet,
                     z: dict[int, jnp.ndarray],
                     alg: dict
                     ) -> tuple[set[int], dict[int, float]]:
    """Per-feature normalised innovation squared (NIS) gate.

    For each feature i in F_set, computes:
        ŷ_i = h_i(x̂⁻)                          (predicted measurement)
        y_i = z_i - ŷ_i                          (innovation)
        S_i = H_i P⁻ H_iᵀ + R_i                  (innovation covariance)
        γ_i = y_iᵀ S_i⁻¹ y_i                     ~ χ²_{ν_i}
    with ν_i = 4 (stereo) or 2 (mono) per the Point's pixel type, and
    rejects stationarity if γ_i > χ²_{α_NIS, ν_i}.

    Args:
        ekf   : filter state, exposes h_pixels, get_measurement_jacobian,
                covariance, get_measurement_noise.
        F_set : EKF feature set
        z     : per-id observed pixels (stacked per Point in id-order
                consistent with ekf.h_pixels).

    Returns:
        (rejected_ids, gammas) — set of point ids that failed, plus γ_i
        per id (used by joint_consistency without recomputation).
    """
    ...


def joint_consistency(inlier_ids: set[int],
                      gammas: dict[int, float],
                      F_set: PointSet,
                      alg: dict
                      ) -> bool:
    """Joint χ² test on the inlier set after per-feature gating.

    γ_joint = Σ_{i ∈ S} γ_i  ~  χ²_{Σ ν_i}.

    Returns True if the inlier set is jointly consistent (γ_joint 
    cutoff), False otherwise. False ⇒ pipeline coasts and triggers
    emergency replenishment per the algorithm pseudocode.
    """
    ...


# =============================================================================
# Feature admission (§sec:mono_admission_stats)
# =============================================================================

def admission_velocity_gate(F_pre: PointSet,
                            alg: dict
                            ) -> tuple[set[int], set[int]]:
    """Pre-EKF velocity test for points after their first stage-2 solve.

    Per-point statistic depends on correspondence type (read from each
    Point's get_px_type):
        SS, SM, MS:  γ_v = v̂ᵀ Σ_vv⁻¹ v̂                    ~ χ²_3
        MM:          γ_v = v̂_⊥² / σ²_{v_⊥}                ~ χ²_1

    Reject (do not admit) if γ_v > χ²_{α_adm, dof}.

    Args:
        F_pre    : candidate-feature set (pre-admission)
     
    Returns:
        (admit_ids, reject_ids) — partition of F_pre by test outcome.
        Caller moves admit set → F (and augments into EKF), reject set → I.

    """
    ...