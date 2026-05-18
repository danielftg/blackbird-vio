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
A future optimisation pass should eliminate the duplicate work.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import logging 
from scipy.stats import chi2

from .points import PointSet
from .ekf import Ekf

log = logging.getLogger(__name__)

# =============================================================================
# Building block: chi-squared cutoff
# =============================================================================

def chi2_threshold(alpha: float, dof: int) -> float:
    """χ²_{alpha, dof} cutoff. Used by every test below.

    alpha is the confidence level (e.g. 0.99); the test rejects when the
    statistic exceeds this cutoff.
    """
    return float(chi2.ppf(alpha, dof))


# =============================================================================
# Feature movement detection (§sec:feature_movement)
# =============================================================================

def feature_nis_gate(ekf: Ekf,
                     F_set: PointSet,
                     alg: dict
                     ) -> tuple[list[int], dict[int, tuple[float, int]]]:
    """Per-feature normalised innovation squared (NIS) gate.

    For each feature i in F_set, computes:
        ŷ_i = h_i(x̂⁻)                          (predicted measurement)
        y_i = z_i - ŷ_i                          (innovation)
        S_i = H_i P⁻ H_iᵀ + R_i                  (innovation covariance)
        γ_i = y_iᵀ S_i⁻¹ y_i                     ~ χ²_{ν_i}
    with ν_i = 4 (stereo) or 2 (mono) per the Point's pixel type, and
    rejects stationarity if γ_i > χ²_{α_NIS, ν_i}.

    Args:
        ekf   : filter state, exposes get_fp_px for ŷ_i and S_i 
        F_set : EKF feature set, has z_i
       
    Returns:
        (rejected_ids, gammas) — list of point ids that failed, plus γ_i
        per id (used by joint_consistency without recomputation).
    """
    alpha = float(alg["signif"]["alpha_NIS"])
    rejected: list[int] = []
    # gammas: dict[int, float] = {}
    gammas: dict[int, tuple[float, int]] = {}

    for p in F_set:
        if p.uL_curr is None and p.uR_curr is None:
            continue

        single = PointSet("single")
        single.add(p)
        y_hat = ekf.h_pixels(single)
        y_meas = ekf._stack_pixels([p])
        H = ekf.get_measurement_jacobian(single)
        R = ekf.get_measurement_noise(single)
        P = ekf.covariance
        S = H @ P @ H.T + R

        if S.size == 0:
            continue

        y = y_meas - y_hat

        res = jnp.linalg.solve(S, y)
        if jnp.any(jnp.isnan(res)) or jnp.any(jnp.isinf(res)):
            gamma = float(y.T @ (jnp.linalg.pinv(S) @ y))
        else:
            gamma = float(y.T @ res)
            
        dof = 4 if (
            p.uL_curr is not None and
            p.uR_curr is not None
        ) else 2

        gammas[p.id] = (gamma, dof)

        if gamma > chi2_threshold(alpha, dof):
            rejected.append(p.id)

    return rejected, gammas



def joint_consistency(inlier_ids: set[int],
                      gammas: dict[int, tuple[float, int]],
                      alg: dict
                      ) -> bool:
    """Forms the approximate statistics

    γ_joint = Σ_{i ∈ S} γ_i 
     
    and compares it against χ²_{Σ ν_i}.
    Exact χ² behaviour assumes independent innovations.

    Returns True if the inlier set is jointly consistent (γ_joint 
    cutoff), False otherwise.
    """
    alpha = float(alg["signif"]["alpha_NIS"])

    if len(inlier_ids) == 0:
        return True

    gamma_joint = 0.0
    dof_joint = 0

    for pid in inlier_ids:

        if pid not in gammas:
            continue

        gamma_i, dof_i = gammas[pid]

        gamma_joint += gamma_i
        dof_joint += dof_i

    return gamma_joint <= chi2_threshold(alpha, dof_joint)


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
       
    """
    """Reads each Point's v_curr (3-vec, magnitude is scalar for MM)
    and Sigma_curr (6x6 or 4x4 for MM) directly from the PointSet."""
    admit_ids: set[int] = set()
    reject_ids: set[int] = set()
    alpha = float(alg["signif"]["alpha_adm"])

    for p in F_pre:
        if p.v_curr is None or p.Sigma_curr is None:
            reject_ids.add(p.id)
            continue

        px_type = p.get_px_type().value
        if px_type == "M-M":
            v_hat = np.asarray(p.v_curr).reshape(-1)
            if v_hat.size == 0:
                reject_ids.add(p.id)
                continue
            v_perp = np.linalg.norm(v_hat) 
            Sigma = np.asarray(p.Sigma_curr, dtype=np.float64)
            if Sigma.ndim != 2 or Sigma.shape[0] < 1 or Sigma.shape[1] < 1:
                reject_ids.add(p.id)
                continue
            sigma_v = float(Sigma[-1, -1])
            if sigma_v <= 0.0:
                reject_ids.add(p.id)
                continue
            gamma = (v_perp * v_perp) / sigma_v
            dof = 1
        else:
            v_hat = np.asarray(p.v_curr, dtype=np.float64).reshape(3,)
            
            Sigma = np.asarray(p.Sigma_curr, dtype=np.float64)
            if Sigma.ndim == 2 and Sigma.shape == (6, 6):
                Sigma_v = Sigma[3:6, 3:6]
            else:
                reject_ids.add(p.id)
                continue
        
            res = np.linalg.solve(Sigma_v, v_hat)
            
            if jnp.any(jnp.isnan(res)) or jnp.any(jnp.isinf(res)):
                log.error(f"Point: {p} has issues sigma: {Sigma_v}, v_hat: {v_hat}")
                gamma = float(v_hat.T @ (np.linalg.pinv(Sigma_v) @ v_hat))
            else:
                gamma = float(v_hat.T @ res)

            dof = 3

        if gamma <= chi2_threshold(alpha, dof):
            admit_ids.add(p.id)
        else:
            reject_ids.add(p.id)

    return admit_ids, reject_ids