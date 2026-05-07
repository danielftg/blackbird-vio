"""
algo.py — Top-level orchestration of the body-frame state estimator.

Implements the pseudocode in §sec:algorithm_overview as a single class.

"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jaxlie
import cv2 as cv 

import utils as util
from .points  import Point, PointSet, PixelType, IdSource
from .vision  import (
    detect_keypoints,
    grid_select_features, focus_select_interest,
    stereo_match, reconstruct_depth,
    temporal_match_one, stereo_promote,
)
from .ekf     import Ekf, EkfState
from .solver  import Solver, JointState, CorrType
from .stats   import feature_nis_gate, joint_consistency, admission_velocity_gate
from .utils   import relative_pose
from cv2.typing import MatLike


# =============================================================================
# Accumulator container
# =============================================================================

@dataclass
class Accumulator:
    """Algorithm's internal state (§sec:algorithm_overview)."""
    F:                  PointSet=PointSet("Feature points")
    I:                  PointSet=PointSet("Interest points")
    F_pre:              PointSet=PointSet("Feature points. Pre admission.")
    L_prev:             MatLike=None
    R_prev:             MatLike=None
    X_prev:             EkfState=None
    EKF:                Ekf=None
    P_prev:             jnp.ndarray=None
    focus_prev:         np.ndarray=None
    focus_sigma_prev:   float=None 
    t_prev:             float=None 


# =============================================================================
# Output container
# =============================================================================

@dataclass
class IterOutput:
    """One frame's worth of estimator output (§sec:alg_io)."""
    X_core:         EkfState                     # core state, no features
    P_core:         jnp.ndarray                  # (18, 18)
    delta_T_solver: jaxlie.SE3                   # solver pose change
    Sigma_DeltaXi:  jnp.ndarray                  # (6, 6)
    point_cloud:    list[tuple]                  # see assemble_point_cloud


# =============================================================================
# Algorithm
# =============================================================================

class Algo:
    """Body-frame state estimator. One instance per flight."""

    # =====================================================================
    # Construction — wire up internal state
    # =====================================================================

    def __init__(self) -> None:
        """Build internal state: empty PointSets, IdSource, EKF, Solver.
        No image processing here — call init() with the first frame.
        """
        self.accum = Accumulator()
        self.id_gen = IdSource()
        self.calib = util.load_yaml("modules/constants/calibration.yaml")
        self.alg = util.load_yaml("modules/constants/algorithm.yaml")

    # =====================================================================
    # First frame — Initialise (k = 0) per pseudocode
    # =====================================================================

    def init(self,
             L_0: MatLike, R_0: MatLike,
             t_0: float,
             F_init: np.ndarray, sigma_F_init: float
             ) -> IterOutput:
        """First-frame bootstrap (§Pseudocode "Initialise").

        - FAST + Shi-Tomasi on L_0, R_0
        - NCC stereo match per detected keypoint
        - Grid select into F_pre and I (focus-weighted for I)
        - Stage-1 stereo: triangulate (p̂, Σ_p) for stereo-matched points
        - Bootstrap EKF with no features (X_0, P_0 from calibration)
        - Roll k → k-1 in the accumulator

        Returns the k=0 IterOutput. F is empty (no admitted features yet);
        delta_T_solver is identity with prior covariance.
        """
        accum = self.accum
        calib = self.calib
        alg = self.alg 

        accum.t_prev = t_0
        accum.focus_prev = F_init
        accum.focus_sigma_prev = sigma_F_init
        accum.L_prev = L_0
        accum.R_prev = R_0


        key_L = detect_keypoints(L_0, calib, alg)
        key_R = detect_keypoints(R_0, calib, alg)

        I_L = focus_select_interest(key_L, F_init, sigma_F_init, accum.I, calib, alg, "L")
        I_R = focus_select_interest(key_R, F_init, sigma_F_init, accum.I, calib, alg, "R")
        
        I_m_L_R = stereo_match(
                image_src=L_0, image_dst=R_0,
                keypoints_src=I_L,
                calib=calib, alg=alg,
                direction="L→R"
        ) 
        I_m_R_L = stereo_match(
                image_src=R_0, image_dst=L_0,
                keypoints_src=I_R,
                calib=calib, alg=alg,
                direction="R→L"
        )

        #Build non-duplicate matches
        I_kpts = [(I_L[idx],m) for idx, m in I_m_L_R]
        I_kpts.extend([(m,I_R[idx]) for idx, m in I_m_R_L])
        I_kpts = self._dedupe(I_kpts)

        #Build point set.
        I_kpts.sort(key=lambda p: p[0] is None or p[1] is None)
        for j, kp_pair  in enumerate(I_kpts):
            if j >= alg["cv"]["N_I_max"]:
                break
            id = self.id_gen.next()
            accum.I.add(Point(id, uL_curr=kp_pair[0], uR_curr=kp_pair[1]))
        

        F_L = grid_select_features(key_L, accum.I, calib, alg, "L") 
        F_R = grid_select_features(key_R, accum.I, calib, alg, "R")
        
        F_m_L_R = stereo_match(
                image_src=L_0, image_dst=R_0,
                keypoints_src=F_L,
                calib=calib, alg=alg,
                direction="L→R"
        ) 
        F_m_R_L = stereo_match(
                image_src=R_0, image_dst=L_0,
                keypoints_src=F_R,
                calib=calib, alg=alg,
                direction="R→L"
        ) 
        
        #Build lists of matches.
        F_kpts = [(F_L[idx],m) for idx, m in F_m_L_R]
        F_kpts.extend([(m,F_R[idx]) for idx, m in F_m_R_L])
        F_kpts = self._dedupe(F_kpts)

        #Build point set.
        F_kpts.sort(key=lambda p: p[0] is None or p[1] is None)
        for j, kp_pair  in enumerate(F_kpts):
            if j >= alg["cv"]["N_F_max"]:
                break
            id = self.id_gen.next()
            accum.F_pre.add(Point(id, uL_curr=kp_pair[0], uR_curr=kp_pair[1]))

        accum.I = reconstruct_depth(accum.I, calib)
        accum.F_pre = reconstruct_depth(accum.F_pre, calib)
    
        accum.EKF = Ekf(calib)
        accum.X_prev = accum.EKF.state
        accum.P_prev = accum.EKF.covariance 

        delta_T = jaxlie.SE3.identity()
        Sigma_DeltaXi = jnp.zeros((6,6))
        return self._emit_and_roll(delta_T, Sigma_DeltaXi)


    # =====================================================================
    # Main loop — Iterate (k = 1, 2, ...)
    # =====================================================================

    def iter(self,
             L_k: MatLike, R_k: MatLike,
             t_k: float,
             u_km1: jnp.ndarray,
             F_km1: np.ndarray, sigma_F_km1: float
             ) -> IterOutput:
        """One iteration (§Pseudocode "Iterate"). Sub-steps below; this
        method is the orchestration shell."""

        # ---- Receive and propagate ---------------------------------------
        # dt = t_k - t_{k-1}; EKF predict with u_{k-1}; detect on (L_k, R_k).
        ...

        # ---- Pre-update relative pose ------------------------------------
        # Compute (ΔT⁻, Σ_Δξ⁻) from T̂_{k-1}^+ and T̂_k^- using P_pose_joint^-.
        # Used as prior for search regions if EKF coasts.
        ...

        # ---- Search and track --------------------------------------------
        # For each point with u^{k-1} populated: candidate set → SSD coarse
        # → LK refine → forward-backward check. Failures: drop from I,
        # marginalise from F. Solver-init metadata cached per surviving
        # point (the winning candidate's (ΔT, p, v) sample).
        ...

        # ---- Stereo promotion --------------------------------------------
        # Mono points at k that are flanked by stereo neighbours: attempt
        # NCC in the other camera; fill the empty slot in u^k.
        ...

        # ---- EKF update --------------------------------------------------
        # Assemble feature measurements for F. If |F| < N_F^min: coast.
        # Else: NIS gate → joint consistency → update on inliers + gravity
        # pseudo-measurement. Failures move F → I.
        ...

        # ---- Post-update relative pose -----------------------------------
        # Compute (ΔT⁺, Σ_Δξ⁺) from T̂_{k-1}^+ and T̂_k^+. If coasting,
        # reuse the pre-update version. Pass into the joint solver.
        ...

        # ---- Joint solver ------------------------------------------------
        # Stage-2 points in F_pre ∪ I: classify CorrType, solve with the
        # post-update pose prior, transport per-point states/covariances
        # to B_k, write back to Point objects.
        ...

        # ---- Feature admission -------------------------------------------
        # F_pre points with first stage-2 solve: velocity χ² gate.
        # Pass: augment into EKF (move to F). Fail: move to I.
        ...

        # ---- Focus update ------------------------------------------------
        # If focus changed: drop interest points outside new focus,
        # select replacements from the keypoint pool, NCC stereo match.
        ...

        # ---- Housekeeping ------------------------------------------------
        # Retire points past n_max; pre-emptive replenish for points that
        # would retire next frame; replenish F_pre / I from the keypoint
        # pool; stage-1 stereo triangulation for new stereo points.
        ...

        # ---- Output ------------------------------------------------------
        # Strip feature rows from EKF state/covariance for X_core / P_core.
        # Assemble C_k from the three sources (F via EKF, F_pre/I via
        # solver, stage-1 stereo via disparity).
        # Roll accumulator k → k-1.
        ...

    # =====================================================================
    # Helpers (each one block of the pseudocode)
    # =====================================================================
    # The methods below are private; iter() calls them in order.

    def _propagate(self, u_km1: jnp.ndarray, dt: float) -> jnp.ndarray:
        """EKF propagate. Returns Φ for joint pose covariance assembly."""
        ...

    def _detect(self, L_k: MatLike, R_k: MatLike
                ) -> tuple[list, list]:
        """FAST + Shi-Tomasi on both images. Returns (pool_L, pool_R)."""
        ...

    def _search_and_track(self, L_k: MatLike, R_k: MatLike,
                          delta_T_minus: jaxlie.SE3,
                          Sigma_xi_minus: jnp.ndarray, dt: float
                          ) -> dict[int, "CandidateSample"]:
        """Temporal match all points with u^{k-1}; return solver-init
        samples per surviving id. Failures dropped/marginalised here."""
        ...

    def _stereo_promote(self, L_k: MatLike, R_k: MatLike) -> None:
        """Attempt mono → stereo for newly-tracked mono points."""
        ...

    def _ekf_update(self) -> tuple[bool, jnp.ndarray, jnp.ndarray]:
        """Run NIS gate, joint consistency, EKF update + gravity.
        Returns (coasting, K, H). coasting=True if |F| < N_F^min or joint
        consistency failed; in that case K, H are None."""
        ...

    def _solve_joint(self, delta_T_prior: jaxlie.SE3,
                     Sigma_prior: jnp.ndarray,
                     init_samples: dict[int, "CandidateSample"]
                     ) -> tuple[JointState, jnp.ndarray]:
        """Run the joint Gauss-Newton solver, transport to B_k, write
        back to Points. Returns (x_post in B_k, Σ_post in B_k)."""
        ...

    def _admit_features(self, x_post: JointState,
                        Sigma_post: jnp.ndarray) -> None:
        """Velocity gate on F_pre points with first solve. Move passes
        to F (augment EKF), fails to I."""
        ...

    def _update_focus(self, F_k: np.ndarray, sigma_F_k: float,
                      L_k: MatLike, R_k: MatLike,
                      pool_L: list, pool_R: list) -> None:
        """If focus changed: drop outside-focus interests, select new
        ones from the pool with NCC stereo match."""
        ...

    def _housekeeping(self, L_k: MatLike, R_k: MatLike,
                      pool_L: list, pool_R: list) -> None:
        """Retire/replenish points; stage-1 stereo for new stereo pairs."""
        ...

    def _emit_and_roll(self, delta_T_solver: jaxlie.SE3,
                         Sigma_DeltaXi: jnp.ndarray) -> IterOutput:
        """Strip feature rows for X_core/P_core; build point cloud
        list with per-point (id, role, stage, p, v, Σ). Roll point sets k -> k-1!"""
        ...

    def _roll_accumulator(self) -> None:
        """End-of-iteration: roll all PointSets (k → k-1), shuffle
        images and timestamps, sync EKF feature reps to Points."""
        ...
    
    def _kpt_key(self, kp: cv.KeyPoint) -> tuple[int, int]:
        """Round to integer pixels for hashable identity."""
        return (round(kp.pt[0]), round(kp.pt[1]))

    def _pair_key(self, pair: tuple[cv.KeyPoint | None, cv.KeyPoint | None]
                ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """Hashable identity for a stereo pair: (left_key, right_key)."""
        left, right = pair
        return (self._kpt_key(left)  if left  is not None else None,
                self._kpt_key(right) if right is not None else None)


    def _dedupe(self, pairs: list[tuple[cv.KeyPoint | None, cv.KeyPoint | None]]
                ) -> list[tuple[cv.KeyPoint | None, cv.KeyPoint | None]]:
        """Drop duplicates within a pair list. Stereo pairs (both populated)
        use the joint (left, right) identity; mono pairs use whichever side
        is populated."""
        seen: set = set()
        out = []
        for p in pairs:
            k = self._pair_key(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        return out


