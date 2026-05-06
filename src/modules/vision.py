"""
vision.py — Computer vision primitives.

Detection, stereo matching, temporal matching, depth reconstruction,
and grid/focus sampling. All physical and algorithm parameters read
from calibration.yaml and algorithm.yaml. No hardcoded values.

Stateless module: each function takes images and parameters explicitly.
The keypoint pool is a per-frame pre-computation passed in where needed,
not cached internally.

Distinguishes two roles for "keypoints":
  - cv.KeyPoint  : OpenCV detector output (with response, size, etc.)
  - Point        : tracked entity from points.py (carries 3D state, ids)
"""

from __future__ import annotations
from dataclasses import dataclass
import cv2 as cv
import numpy as np
import jax.numpy as jnp
import jaxlie

from .points import Point, PointSet, PixelType


# =============================================================================
# Detection — FAST + Shi-Tomasi (§sec:cv)
# =============================================================================

def detect_keypoints(image: np.ndarray, calib: dict, alg: dict
                     ) -> list[cv.KeyPoint]:
    """Run FAST detection then Shi-Tomasi cornerness scoring on `image`.

    Returns the unfiltered keypoint pool — caller decides how to allocate
    via grid_select / focus_select. Sorted by Shi-Tomasi response,
    descending. Threshold τ_min from algorithm.yaml.

    Stored once per frame in the pipeline; passed to allocators below.
    """
    ...


# =============================================================================
# Sampling / replenishment (§sec:cv "Grid Selection and Focus Weighting")
# =============================================================================

def grid_select_features(pool: list[cv.KeyPoint],
                         existing: PointSet,
                         calib: dict, alg: dict
                         ) -> list[cv.KeyPoint]:
    """Allocate feature replenishments uniformly across an N×M grid.

    Picks N_feat,gl total across the image, with at least N_feat,lo per
    cell (cells with no candidates left empty). Uses Shi-Tomasi response
    as the per-cell ranker. Suppresses cells already saturated by points
    in `existing` (avoid clustering on existing trackers).
    """
    ...


def focus_select_interest(pool: list[cv.KeyPoint],
                          focus_B: np.ndarray,
                          sigma_F: float,
                          existing: PointSet,
                          calib: dict, alg: dict
                          ) -> dict[str, list[cv.KeyPoint]]:
    """Allocate interest points around the focus projection per camera.

    focus_B : (3,) focus point F_{k-1} in body frame
    sigma_F : Gaussian spread

    Projects focus into each camera, builds a truncated 2D Gaussian over
    the grid centred on the projection, allocates floor(N_I/2 · p_g) per
    cell with residual to the peak. Within each cell the top-by-response
    keypoints are picked.

    Returns {"L": [...], "R": [...]} — one list per camera. Stereo
    matching of these into mono/stereo Points is the caller's job
    (via stereo_match below).
    """
    ...


# =============================================================================
# Stereo matching — NCC (§sec:cv)
# =============================================================================

def stereo_match(image_src: np.ndarray, image_dst: np.ndarray,
                 keypoints_src: list[cv.KeyPoint],
                 calib: dict, alg: dict,
                 direction: str = "L→R"
                 ) -> dict[int, cv.KeyPoint | None]:
    """NCC stereo match each src keypoint along its epipolar line in dst.

    direction ∈ {"L→R", "R→L"}: which image is source, which is target.
    For rectified stereo (ZJU bag) the epipolar line is a horizontal row,
    so search reduces to 1D along v = v_src.

    Returns a dict mapping the index in `keypoints_src` to the matched
    cv.KeyPoint in dst, or None if NCC peak below threshold. Sub-pixel
    refinement by parabolic peak fitting on the NCC response.

    NCC threshold and search-window parameters from algorithm.yaml.
    """
    ...


# =============================================================================
# Stereo depth and covariance (§sec:depth_estimation)
# =============================================================================

def reconstruct_depth(uL: np.ndarray, uR: np.ndarray,
                      calib: dict
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate (u_L, u_R) pixel pairs into a body-frame 3D point.

    For rectified stereo:  Z = b·fx / (uL.x − uR.x)
                            X = (uL.x − cx) · Z / fx
                            Y = (uL.y − cy) · Z / fy
    Then transform from left-camera frame into body frame via T_BL.

    Returns (p_B, Σ_p) where Σ_p is the 3×3 position covariance from
    propagating σ_px² (calibration.yaml) through the triangulation
    Jacobian. 

    Vectorised: uL, uR shape (N, 2); returns p_B (N, 3) and Σ_p (N, 3, 3).
    """
    ...


# =============================================================================
# Temporal matching (§sec:search_region)
# =============================================================================
#
# Pipeline per point:
#   1. predict_pixel        — h(ΔT, p, v) for a single sample
#   2. candidate_set        — sample uncertainty sets, project, hull
#   3. ssd_coarse_match     — pick best candidate by SSD
#   4. lk_refine            — pyramidal LK from coarse winner
#   5. fb_check             — forward-backward consistency
#
# Solver init metadata (which sample produced the winner) is returned
# alongside the matched pixel so the joint solver can warm-start.
# =============================================================================

def predict_pixel(delta_T: jaxlie.SE3,
                  p_Bkm1: np.ndarray, v_p: np.ndarray,
                  dt: float, calib: dict, camera: str
                  ) -> np.ndarray:
    """Prediction function (§eq:u_pred): pose change × point + velocity
    transport, projected through camera `camera` ∈ {"L", "R"}..
    """
    ...


@dataclass
class CandidateSample:
    """One sample from the uncertainty product space.

    Carries the input that produced the candidate so the winning sample
    can warm-start the joint solver (§search_region step 6).
    """
    pixel: np.ndarray            # (2,) projected pixel
    delta_T: jaxlie.SE3          # ΔT sample
    p_Bkm1: np.ndarray           # (3,) p sample
    v_p: np.ndarray              # (3,) velocity sample (or v_⊥ scalar for MM)


def candidate_set(p: Point,
                  delta_T_hat: jaxlie.SE3, Sigma_xi: np.ndarray,
                  dt: float, calib: dict, alg: dict, camera: str
                  ) -> list[CandidateSample]:
    """Generate candidate pixels for `p` by sampling the uncertainty
    product space appropriate to its available state (§search_region).

    Three branches per the table:
      - p̂ and v̂   : E_ξ × (E_pv ⊕ S_a)
      - p̂ only     : E_ξ × E_p × S_v
      - neither    : E_ξ × S_Z × S_v

    Returns the list of (pixel, sample) candidates whose projections
    fall in the image. Empty list ⇒ nothing to match (caller skips).
    """
    ...


def ssd_coarse_match(image_src: np.ndarray, image_dst: np.ndarray,
                     u_src: np.ndarray,
                     candidates: list[CandidateSample],
                     calib: dict, alg: dict
                     ) -> CandidateSample | None:
    """Evaluate SSD between the reference patch at u_src in image_src
    and each candidate location in image_dst. Return the lowest-SSD
    candidate, or None if none beat the SSD threshold (algorithm.yaml).

    Patch size and SSD threshold from algorithm.yaml. SSD chosen over
    NCC because src and dst are the same camera one frame apart —
    brightness/contrast invariance not needed.
    """
    ...


def lk_refine(image_src: np.ndarray, image_dst: np.ndarray,
              u_src: np.ndarray, u_init: np.ndarray,
              alg: dict
              ) -> np.ndarray:
    """Pyramidal Lucas-Kanade refinement from coarse seed `u_init`.

    Wraps cv.calcOpticalFlowPyrLK. Window size, pyramid levels,
    termination criteria from algorithm.yaml.
    Returns the refined sub-pixel position.
    """
    ...


def fb_check(image_src: np.ndarray, image_dst: np.ndarray,
             u_src: np.ndarray, u_dst: np.ndarray,
             alg: dict
             ) -> bool:
    """Forward-backward consistency. Re-track u_dst back into image_src;
    accept iff round-trip pixel error < ε_fb (algorithm.yaml).
    Primary failure detector for LK tracking.
    """
    ...


# =============================================================================
# Convenience wrappers
# =============================================================================

def temporal_match_one(p: Point,
                       image_src: np.ndarray, image_dst: np.ndarray,
                       delta_T_hat: jaxlie.SE3, Sigma_xi: np.ndarray,
                       dt: float, calib: dict, alg: dict,
                       camera: str
                       ) -> tuple[np.ndarray, CandidateSample] | None:
    """Full temporal-matching pipeline for one point in one camera.

    Composes candidate_set → ssd_coarse_match → lk_refine → fb_check.
    Returns (u_matched, init_sample) on success, None on any failure.
    `init_sample` is the candidate input that won — caller passes it to
    the joint solver as the warm-start (§search_region step 6).
    """
    ...


def stereo_promote(p: Point, image_other: np.ndarray,
                   calib: dict, alg: dict
                   ) -> cv.KeyPoint | None:
    """Mono-to-stereo: `p` has only one camera populated at frame k.
    Attempt NCC match in the other camera along the epipolar line.

    Returns the matched cv.KeyPoint or None. Caller fills the empty slot
    (uL_curr or uR_curr) on `p` if successful — vision.py does not mutate
    Point objects.
    """
    ...