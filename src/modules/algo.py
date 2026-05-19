"""
algo.py — Top-level orchestration of the body-frame state estimator.

Implements the pseudocode in §sec:algorithm_overview as a single class.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import chain
import logging 
import numpy as np
import jax.numpy as jnp
import jaxlie
import cv2 as cv 

from .points  import Point, PointSet, PixelType, IdSource
from .vision  import (
    detect_keypoints, CandidateSample,
    grid_select_features, focus_select_interest,
    stereo_match, reconstruct_depth,
    temporal_match_one, temporal_match_one_feature,
    )
from .ekf     import Ekf, CoreEkfState, EkfState, relative_pose
from .solver  import Solver, JointState
from .stats   import feature_nis_gate, joint_consistency, admission_velocity_gate
from cv2.typing import MatLike

log = logging.getLogger(__name__)

# =============================================================================
# Accumulator container
# =============================================================================
    
@dataclass
class Accumulator:
    """Algorithm's internal state (§sec:algorithm_overview)."""
    F:                  PointSet=field(default_factory=lambda: PointSet("Feature points"))
    I:                  PointSet=field(default_factory=lambda: PointSet("Interest points"))
    F_pre:              PointSet=field(default_factory=lambda: PointSet("Feature points. Pre admission."))
    L_prev:             MatLike=None
    R_prev:             MatLike=None
    EKF:                Ekf=None
    X_prev:             EkfState=None
    P_prev:             jnp.ndarray=None 
    focus_prev:         np.ndarray=None
    focus_sigma_prev:   float=None 
    t_prev:             float=None 
    reject_ids:         list[int]=field(default_factory=lambda: [])


# =============================================================================
# Output container
# =============================================================================

@dataclass
class IterOutput:
    """One frame's worth of estimator output (§sec:alg_io)."""
    X_core:         CoreEkfState                 # core state, no features
    P_core:         jnp.ndarray                  # (18, 18)
    delta_T_solver: jaxlie.SE3                   # solver pose change
    Sigma_DeltaXi:  jnp.ndarray                  # (6, 6)
    point_cloud:    dict[int, CloudPoint]        # see assemble_point_cloud


@dataclass
class CloudPoint:
    role: str             # 'F', 'F_pre', 'I'
    stage: int            # '-1', '-2'. -1 is stage 1 and -2 is stage2.
    p: np.ndarray         # (3,)
    v: np.ndarray | None  # None  for stage1, (3,)  for stage2
    Sigma: np.ndarray     # (3,3) for stage1, (6,6) or (4,4) 
                          # for stage2 depending on correspondance type

# =============================================================================
# Algorithm
# =============================================================================

class Algo:
    """Body-frame state estimator. One instance per flight."""

    # =====================================================================
    # Construction — wire up internal state
    # =====================================================================

    def __init__(self, calib, alg) -> None:
        """Build internal state: empty PointSets, IdSource, EKF, Solver.
        No image processing here — call init() with the first frame.
        """
        self.accum = Accumulator()
        self.id_gen = IdSource()
        self.calib = calib
        self.alg = alg
        self.solvr = Solver(self.calib, self.alg)
       
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
        I_kpts = [(I_L[idx],m) for idx, m in I_m_L_R.items()]
        I_kpts.extend([(m,I_R[idx]) for idx, m in I_m_R_L.items()])
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
        F_kpts = [(F_L[idx],m) for idx, m in F_m_L_R.items()]
        F_kpts.extend([(m,F_R[idx]) for idx, m in F_m_R_L.items()])
        F_kpts = self._dedupe(F_kpts)

        #Build point set.
        F_kpts.sort(key=lambda p: p[0] is None or p[1] is None)
        for j, kp_pair  in enumerate(F_kpts):
            if j >= alg["cv"]["N_F_max"]:
                break
            id = self.id_gen.next()
            accum.F_pre.add(Point(id, uL_curr=kp_pair[0], uR_curr=kp_pair[1]))

        log.debug(f"Number of interest points: {len(self.accum.I)}. Number of non-admitted feature points: {len(self.accum.F_pre)}.")

        accum.I = reconstruct_depth(accum.I, calib, alg)
        accum.F_pre = reconstruct_depth(accum.F_pre, calib, alg)
    
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
             u_km1: np.ndarray,
             F_km1: np.ndarray, sigma_F_km1: float
             ) -> IterOutput:
        """One iteration (§Pseudocode "Iterate"). Sub-steps below; this
        method is the orchestration shell."""
        accum = self.accum
        calib = self.calib
        alg = self.alg 
        
        # ---- Receive and propagate ---------------------------------------
        delta_t = t_k - accum.t_prev
        prop_state_matrix = accum.EKF.propagate(u_km1, delta_t)
        key_L = detect_keypoints(L_k, calib, alg)
        key_R = detect_keypoints(R_k, calib, alg)
        
        # ---- Pre-update relative pose ------------------------------------
        # Compute (ΔT⁻, Σ_Δξ⁻) from T̂_{k-1}^+ and T̂_k^- using P_pose_joint^-.
        T_a = accum.X_prev.T 
        T_b = accum.EKF.state.T
        covar_a = accum.P_prev
        covar_b = accum.EKF.covariance
        c_ids = accum.EKF.state.feature_ids.copy()

        delta_T, covar_delta_T = relative_pose(
            T_a, T_b
            ,covar_a, covar_b,
            prop_state_matrix,
            a_ids=accum.X_prev.feature_ids, c_ids=c_ids
        )

        # ---- Search and track --------------------------------------------
        # For each point with u^{k-1} populated: 
        # candidate set -> Predict → SSD coarse → LK refine → forward-backward check. 
        # Failures: drop from I, marginalise from F. 
        # Solver-init metadata cached per surviving point (the winning candidate's (ΔT, p, v) sample).
        
        w_cands = self._search_and_track(L_k, R_k,
                          delta_T,
                          covar_delta_T, delta_t
        )
        
        # ---- Stereo promotion --------------------------------------------
        # Attempt stereo promotion for mono points. 
        # NCC in the other camera; fill the empty slot in u^k.
        for p_set in [accum.I, accum.F_pre, accum.F]:
            mono_L = p_set.filter(lambda p: p.uR_curr is None)
            mono_R = p_set.filter(lambda p: p.uL_curr is None)

            if len(mono_L) != 0:
                self._stereo_promote(L_k, R_k,
                mono_L,
                calib, alg,
                "L→R"
                )

            if len(mono_R) != 0:
                self._stereo_promote(R_k, L_k,
                mono_R,
                calib, alg,
                "R→L"
                )
        
        # ---- EKF update --------------------------------------------------
        upd_state_matrix = self._ekf_update()
        

        # ---- Post-update relative pose -----------------------------------
        # Compute (ΔT⁺, Σ_Δξ⁺) from T̂_{k-1}^+ and T̂_k^+. 
        
        T_b = accum.EKF.state.T
        covar_b = accum.EKF.covariance
        
        delta_T, covar_delta_T = relative_pose(
            T_a, T_b
            ,covar_a, covar_b,
            prop_state_matrix, upd_state_matrix,
            accum.X_prev.feature_ids, c_ids, accum.EKF.state.feature_ids
        )


        # ---- Joint solver ------------------------------------------------
        # Stage-2 points in F_pre ∪ I: classify PixelType, solve with the
        # post-update pose prior, transport per-point states/covariances
        # to B_k, write back to Point objects.
        
        sol_dT, covar_dT = self._solve_joint(
            delta_t, delta_T, covar_delta_T, w_cands
        )
        
        

        # ---- Feature admission -------------------------------------------
        # F_pre points with first stage-2 solve: velocity χ² gate.
        # Pass: augment into EKF (move to F). Fail: move to I.
        self._admit_features()
        
        
        # ---- Replenish ------------------------------------------------
        I_hit_limit, F_hit_limit = self._replenish(L_k, R_k, key_L, key_R, F_km1, sigma_F_km1)     
        

        # ---- Output ------------------------------------------------------
        # Strip feature rows from EKF state/covariance for X_core / P_core.
        # Assemble C_k from the three sources (F via EKF, F_pre/I via
        # solver, stage-1 stereo via disparity).
        # Roll accumulator k → k-1.
        out = self._emit_and_roll(sol_dT, covar_dT)
        
        
        # ---- Housekeeping  ------------------------------------------------
        # If focus changed: drop interest points outside new focus. #NOTE: Implement when focus changes.
        accum.t_prev = t_k
        accum.focus_prev = F_km1
        accum.focus_sigma_prev = sigma_F_km1
        accum.L_prev = L_k
        accum.R_prev = R_k
        accum.X_prev = accum.EKF.state
        accum.P_prev = accum.EKF.covariance

        to_drop = list(I_hit_limit.ids())
        for id in to_drop: self.accum.I.discard(id)
        for id in self.accum.reject_ids: self.accum.I.discard(id)
        to_drop = list(F_hit_limit.ids())
        for id in to_drop: 
            self.accum.F.discard(id)
            self.accum.EKF.marginalise(id)
        self.accum.reject_ids = []
        return out

    # =====================================================================
    # Helpers (each one block of the pseudocode)
    # =====================================================================
    # The methods below are private; iter() calls them in order.


    def _search_and_track(self, L_k: MatLike, R_k: MatLike,
                          delta_T: jaxlie.SE3,
                          Sigma_xi: jnp.ndarray, dt: float
                          ) -> dict[int, CandidateSample]:
        """Temporal match all points with u^{k-1}; return solver-init
        samples per surviving id. Failures dropped/marginalised here."""
        cands = {}
        for point in chain(self.accum.I, self.accum.F_pre):
            cand_L, cand_R = self._st_point(point,"I", L_k, R_k, delta_T, Sigma_xi, dt, self.calib, self.alg)
            point.uL_curr = cand_L.keypoint if cand_L is not None else None
            point.uR_curr = cand_R.keypoint if cand_R is not None else None            
            cands[point.id] = cand_L if cand_R is None else cand_R

        no_match = [id for id,c in cands.items() if c is None]
        
        for point in self.accum.F:
            cand_L, cand_R = self._st_point(point,"F", L_k, R_k, delta_T, Sigma_xi, dt, self.calib, self.alg)
            point.uL_curr = cand_L.keypoint if cand_L is not None else None
            point.uR_curr = cand_R.keypoint if cand_R is not None else None  
            cands[point.id] = cand_L if cand_R is None else cand_R

            if (point.uL_curr is None) and (point.uR_curr is None):
                no_match.append(point.id) 
        
        for id in no_match: 
            if id in self.accum.I:
                log.debug(f"Failed to temporal match interest point: {id}")
                self.accum.I.discard(id)
                continue
            if id in self.accum.F_pre:
                log.debug(f"Failed to temporal match non-admitted feature point: {id}")
                self.accum.F_pre.discard(id)
                continue
            if id in self.accum.F:
                log.debug(f"Failed to temporal match feature point: {id}")
                self.accum.F.discard(id)
                self.accum.EKF.marginalise(id)
        return cands 

    def _stereo_promote(self, 
                        image_src: MatLike, image_dst: MatLike,
                        points_src: PointSet,
                        calib: dict, alg: dict,
                        direction: str = "L→R") -> None:
        """Attempt mono → stereo for newly-tracked mono points."""
        assert direction in ["L→R", "R→L"], "Direction is given by 'L→R' or 'R→L'"
        
        points_list = list(points_src)       
        if direction == "L→R":
            keypoints_src = [p.uL_curr for p in points_list]
        else:
            keypoints_src = [p.uR_curr for p in points_list]
        
        matches = stereo_match(image_src, image_dst, 
                               keypoints_src, calib, alg, direction
        )
        
        for idx, p in enumerate(points_list):
            if direction == "L→R":
                p.uR_curr = matches[idx]
            else:
                p.uL_curr = matches[idx]

    def _ekf_update(self) -> jnp.ndarray:
        """Run NIS gate, joint consistency, EKF update.
        Returns I-KH."""
        ...
        
        accum = self.accum
        alg = self.alg
        any_points = lambda F: len(F) != 0
        
        accum.EKF.add_gravity_measurement()
        if any_points(accum.F):
            moving, gammas = feature_nis_gate(accum.EKF, accum.F, alg)

            for id in moving:
                accum.EKF.marginalise(id)
                p = accum.F.remove(id)
                self.accum.reject_ids.append(id)
                accum.I.add(p)
            
            if any_points(accum.F):
                stationary = joint_consistency(
                    accum.F.ids(), gammas, alg
                )
                if not stationary:
                    to_drop = list(accum.F.ids())
                    for id in to_drop:
                        accum.EKF.marginalise(id)
                        p = accum.F.remove(id)
                        self.accum.reject_ids.append(id)
                        accum.I.add(p)
                else:
                    accum.EKF.add_pixel_measurements(accum.F)
        

        upd_state_matrix = accum.EKF.update()
        return upd_state_matrix


    def _solve_joint(self, dt:float,
                     delta_T_prior: jaxlie.SE3,
                     Sigma_prior: jnp.ndarray,
                     init_samples: dict[int, CandidateSample]
                     ) -> tuple[jaxlie.SE3, jnp.ndarray]:
        """Run the joint Gauss-Newton solver, transport to B_k, write
        back to Points. Returns (x_post.delta_T in B_k, Σ_post in B_k of delta_T)."""
        all_pts = self.accum.I.union(self.accum.F_pre)
        solvr = self.solvr

        solvr.initialise(dt, all_pts, delta_T_prior, Sigma_prior, init_samples)
        sol, sol_covar = solvr.run()     
        sol, sol_covar = solvr.transport_to_Bk(sol, sol_covar)
        solvr.write_back(sol, sol_covar, all_pts)
        return sol.delta_T, sol_covar["pose"]

    def _admit_features(self) -> None:
        """Velocity gate on F_pre points with first solve. Move passes
        to F (augment EKF), fails to I."""

        admit_ids, reject_ids = admission_velocity_gate(self.accum.F_pre, self.alg)
        
        self.accum.reject_ids.extend(reject_ids)

        self.accum.F_pre.move_to(self.accum.F, admit_ids)
        self.accum.F_pre.move_to(self.accum.I, reject_ids)
        for id in admit_ids: self.accum.EKF.augment(self.accum.F.get(id))

    def _replenish(self, L_k: MatLike, R_k: MatLike,
                      pool_L: list, pool_R: list
                      ,F_km1: np.ndarray, sigma_F_km1: float
                      ) -> tuple[PointSet, PointSet]:
        """Replenish points; stage-1 stereo for new stereo pairs.
           Returns points which hit the lifetime limit this frame per pointset.
           Left is I, right is F.
        """
        
        # Replenish  F / I from the keypoint pools. Bring them up to max
        # Pre-emptive replenish for points that would retire next frame. (Draw above max) 
        # NCC stereo match. 
        # stage-1 stereo triangulation for new stereo points.
        
        focus_changed = (not np.allclose(self.accum.focus_prev, F_km1)) or (not np.isclose(self.accum.focus_sigma_prev, sigma_F_km1))
        log.debug(f"Focus changed: {focus_changed}")

        N_I = len(self.accum.I)
        N_F = len(self.accum.F)
        
        N_I_max = self.alg["cv"]["N_I_max"]
        N_F_max = self.alg["cv"]["N_F_max"]
        lifespan_max = self.alg["cv"]["n_max"]

        hits_limit = lambda p: p.n_max + 1 == lifespan_max
        hit_limit = lambda p: p.n_max == lifespan_max

        I_hits_limit = self.accum.I.filter(hits_limit) 
        F_hits_limit = self.accum.F.filter(hits_limit)
        
        I_hit_limit = self.accum.I.filter(hit_limit)
        F_hit_limit = self.accum.F.filter(hit_limit)
        
        I_pre_empt = len(I_hits_limit)        
        F_pre_empt = len(F_hits_limit)

        I_ign = len(I_hit_limit)
        F_ign = len(F_hit_limit)
        
        N_I_repl = N_I_max - (N_I - I_ign) + I_pre_empt + len(self.accum.reject_ids)
        N_F_repl = N_F_max - (N_F - F_ign) + F_pre_empt

        log.debug(f"Max Interest points: {N_I_max}. Number of interest points: {N_I}. Interest points ending this frame: {I_ign}. Interest points ending next frame: {I_pre_empt}")
        _log_pixel_type_counts("Interest", self.accum.I)
        log.debug(f"Max Feature points: {N_F_max}. Number of feature points: {N_F}. Feature points ending this frame: {F_ign}. Feature points ending next frame: {F_pre_empt}")
        _log_pixel_type_counts("Features", self.accum.F) 

        log.debug(f"Interest points to replenish: {N_I_repl}, Feature points to replenish: {N_F_repl}")
        
        if focus_changed:
            raise NotImplementedError("focus change handling not yet implemented")


        if (N_I_repl > 0) and (not focus_changed):
            all_pts = self.accum.I.union(self.accum.F)
            I_L = focus_select_interest(pool_L, F_km1, sigma_F_km1, all_pts, self.calib, self.alg, "L")
            I_R = focus_select_interest(pool_R, F_km1, sigma_F_km1, all_pts, self.calib, self.alg, "R")

            I_m_L_R = stereo_match(
                image_src=L_k, image_dst=R_k,
                keypoints_src=I_L,
                calib=self.calib, alg=self.alg,
                direction="L→R"
            ) 
            I_m_R_L = stereo_match(
                    image_src=R_k, image_dst=L_k,
                    keypoints_src=I_R,
                    calib=self.calib, alg=self.alg,
                    direction="R→L"
            )

            #Build non-duplicate matches
            I_kpts = [(I_L[idx],m) for idx, m in I_m_L_R.items()]
            I_kpts.extend([(m,I_R[idx]) for idx, m in I_m_R_L.items()])
            I_kpts = self._dedupe(I_kpts)

            I_kpts.sort(key=lambda p: p[0] is None or p[1] is None)
            for j, kp_pair  in enumerate(I_kpts):
                if j >= N_I_repl:
                    break
                id = self.id_gen.next()
                self.accum.I.add(Point(id, uL_curr=kp_pair[0], uR_curr=kp_pair[1]))
            
            self.accum.I = reconstruct_depth(self.accum.I, self.calib, self.alg)
        
        if N_F_repl > 0:
            all_pts = self.accum.I.union(self.accum.F)
            F_L = grid_select_features(pool_L, all_pts, self.calib, self.alg, "L") 
            F_R = grid_select_features(pool_R, all_pts, self.calib, self.alg, "R")
            
            F_m_L_R = stereo_match(
                image_src=L_k, image_dst=R_k,
                keypoints_src=F_L,
                calib=self.calib, alg=self.alg,
                direction="L→R"
            ) 
            F_m_R_L = stereo_match(
                    image_src=R_k, image_dst=L_k,
                    keypoints_src=F_R,
                    calib=self.calib, alg=self.alg,
                    direction="R→L"
            ) 
            F_kpts = [(F_L[idx],m) for idx, m in F_m_L_R.items()]
            F_kpts.extend([(m,F_R[idx]) for idx, m in F_m_R_L.items()])
            F_kpts = self._dedupe(F_kpts)

            F_kpts.sort(key=lambda p: p[0] is None or p[1] is None)
            for j, kp_pair  in enumerate(F_kpts):
                if j >= N_F_repl:
                    break
                id = self.id_gen.next()
                self.accum.F_pre.add(Point(id, uL_curr=kp_pair[0], uR_curr=kp_pair[1]))
        
            self.accum.F_pre = reconstruct_depth(self.accum.F_pre, self.calib, self.alg)

        return (I_hit_limit, F_hit_limit)

    def _assemble_point_cloud(self) -> dict[int, CloudPoint]:
        """Build per-frame point cloud C_k from all three sources (§sec:alg_io).

        Source priority — each point appears exactly once:
            F           : (p, Σ_p) from EKF state. Position only.
            F_pre, I    : (p, v, Σ) from joint solver, transported to B_k.
                        Value depends on correspondance type.
            F_pre, I    : (p, Σ_p) from reconstruct_depth.

        Returns:
            Dict of Cloudpoint id:(role, stage, p, v, Σ). 
        """
        cloud = {}

        # F — EKF feature points
        p_F, P_FF, ids_F = self.accum.EKF.feature_output()
        for i, pid in enumerate(ids_F):
            cloud[pid] = CloudPoint(
                role="F",
                stage=-1,
                p=p_F[i],
                v=None,
                Sigma=P_FF[i],
            )
            
        # F_pre and I — solver output (stage-2) or stereo triangulation (stage-1)
        for role, pset in (('F_pre', self.accum.F_pre), ('I', self.accum.I)):
            for p in pset:
                log.debug(f"Role: {role}. Point {p}.")
                if p.p_curr is None:
                    continue
                stage = -2 if p.v_curr is not None else -1
                cloud[p.id]= CloudPoint(
                    role=role,
                    stage=stage,
                    p=p.p_curr,
                    v=p.v_curr,
                    Sigma=p.Sigma_curr,
                )
        return cloud


    def _emit_and_roll(self, delta_T_solver: jaxlie.SE3,
                         Sigma_DeltaXi: jnp.ndarray) -> IterOutput:
        """Strip feature rows for X_core/P_core; build point cloud
        list with per-point (id, role, stage, p, v, Σ). THEN roll point sets k -> k-1!"""
        
        out = IterOutput(
            X_core = self.accum.EKF.state.get_core_state(),
            P_core=self.accum.EKF.covariance[:18,:18],
            delta_T_solver=delta_T_solver,
            Sigma_DeltaXi=Sigma_DeltaXi,
            point_cloud=self._assemble_point_cloud(),
        )
        self.accum.F.roll()
        self.accum.I.roll()
        self.accum.F_pre.roll()
        return out 

    
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
    
    def _st_point(self, point:Point, pset, L_k, R_k, delta_T, covar_delta_T, delta_t, calib, alg):
        had_L = point.uL_prev is not None
        had_R = point.uR_prev is not None
        cand_L, cand_R = None, None 
        if had_L:
            cand_L = self._st_point_cam(point, pset, L_k, delta_T, covar_delta_T, delta_t, calib, alg, "L")
        if had_R:
            cand_R = self._st_point_cam(point, pset, R_k, delta_T, covar_delta_T, delta_t, calib, alg, "R")

        return (cand_L, cand_R)

    def _st_point_cam(self, point:Point, pset:str, image_dst, delta_T, covar_delta_T, delta_t, calib, alg, camera:str):
        assert pset in ["I", "F"], "The point set is given by 'I' or 'F'"
        assert camera in ["L", "R"], "The camera is given by 'L' or 'R'"

        if camera == "L":
            image_src = self.accum.L_prev    
        if camera == "R":
            image_src = self.accum.R_prev

        if pset=="I":
            return temporal_match_one(
            point, image_src, image_dst, delta_T, 
            covar_delta_T, delta_t, calib, alg,
            camera
            )
        if pset =="F":
            p_Bk, sigma = self.accum.EKF.get_fp_body(point.id)
            return  temporal_match_one_feature(
            point, image_src, image_dst, 
            p_Bk, sigma, calib, alg,
            camera 
            )

def _log_pixel_type_counts(label: str, pset: PointSet) -> None:
    """Log per-pixel-type counts in a PointSet."""
    counts = {t: len(pset.filter(lambda p, t=t: p.get_px_type() == t))
            for t in PixelType}
    log.debug(f"{label}: " + ", ".join(f"N_{t.name}={counts[t]}" for t in PixelType))
