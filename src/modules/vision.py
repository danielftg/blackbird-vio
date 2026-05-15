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
from cv2.typing import MatLike
import cv2 as cv
import numpy as np
import jax.numpy as jnp
import jaxlie

from .points import Point, PointSet, PixelType
from .ekf import project_point


def _camera_key(camera: str) -> str:
    assert camera in ["L", "R"], "Camera must be 'L' or 'R'"
    return "left" if camera == "L" else "right"


def _pixel_in_bounds(pixel: np.ndarray, size: tuple[int, int]) -> bool:
    x, y = float(pixel[0]), float(pixel[1])
    width, height = size
    return 0.0 <= x < width and 0.0 <= y < height


def _get_patch(image: MatLike, center: tuple[float, float], patch_size: tuple[int, int]) -> np.ndarray | None:
    height, width = image.shape[:2]
    w, h = patch_size
    x, y = float(center[0]), float(center[1])
    half_w = (w - 1) / 2.0
    half_h = (h - 1) / 2.0
    if (x - half_w) < 0 or (x + half_w) >= width or (y - half_h) < 0 or (y + half_h) >= height:
        return None

    if image.ndim == 3 and image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    patch = cv.getRectSubPix(image, (w, h), (x, y))
    if patch is None or patch.shape[:2] != (h, w):
        return None
    return patch.astype(np.float32)

# =============================================================================
# Detection — FAST + Shi-Tomasi (§sec:cv)
# =============================================================================

def detect_keypoints(image: MatLike, calib: dict, alg: dict
                     ) -> list[cv.KeyPoint]:
    """Run FAST detection then Shi-Tomasi cornerness scoring on `image`.

    Returns the unfiltered keypoint pool — caller decides how to allocate
    via grid_select / focus_select. Sorted by Shi-Tomasi response,
    descending. Threshold τ_min from algorithm.yaml.

    Stored once per frame in the pipeline; passed to allocators below.
    """
    ...
    alg["cv"]["tau_min"] #absolute threshold = tau_min · max eigenvalue in image)


# =============================================================================
# Sampling / replenishment (§sec:cv "Grid Selection and Focus Weighting")
# =============================================================================

def grid_select_features(pool: list[cv.KeyPoint],
                         existing: PointSet,
                         calib: dict, alg: dict, camera: "str"
                         ) -> list[cv.KeyPoint]:
    """Allocate feature replenishments uniformly across an N×M grid.

    Picks N_feat,gl total across the image, with at least N_feat,lo per
    cell (cells with no candidates left empty, cell size from alg). Uses Shi-Tomasi response
    as the per-cell ranker. Suppresses cells already saturated by points
    in `existing` (avoid clustering on existing trackers, dont select existing points.).
    """
    alg["cv"]["grid_cells"]
    calib["camera_intrinsics"]["left"]["size"]
    alg["cv"]["N_feat_gl"]
    alg["cv_N_feat_lo"]
    alg["cv"]["N_F_max"]
    ...


def focus_select_interest(pool: list[cv.KeyPoint],
                          focus_B: np.ndarray,
                          sigma_F: float,
                          existing: PointSet,
                          calib: dict, alg: dict, camera: "str"
                          ) -> list[cv.KeyPoint]:
    """Allocate interest points around the focus projection per camera.

    focus_B : (3,) focus point F_{k-1} in body frame
    sigma_F : Gaussian spread

    Projects focus into the camera ("L" or "R"), builds a truncated 2D Gaussian over
    the grid centred on the projection, allocates floor(N_I/2 · p_g) per
    cell with residual to the peak. Within each cell the top-by-response
    keypoints are picked (See paper for details).
    Don't select existing points.
    """
    alg["cv"]["grid_cells"]
    calib["camera_intrinsics"]["size"]
    project_point
    alg["cv"]["N_I_max"]

# =============================================================================
# Stereo matching — NCC (§sec:cv)
# =============================================================================

def stereo_match(image_src: MatLike, image_dst: MatLike,
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
    alg["cv"]["disp_min"]
    alg["cv"]["disp_max"]
    alg["cv"]["stereo_subpix"]
    alg["cv"]["ncc_min"]
    alg["cv"]["ncc_patch_size"]
    ...


# =============================================================================
# Stereo depth and covariance (§sec:depth_estimation)
# =============================================================================

def reconstruct_depth(points: PointSet,
                      calib:  dict
                      ) -> PointSet:
    """Triangulate current (u_L, u_R) pixel pairs into a body-frame 3D point.
        Only do so for points without 3D point data.
    For rectified stereo:  Z = b·fx / (uL.x − uR.x)
                            X = (uL.x − cx) · Z / fx
                            Y = (uL.y − cy) · Z / fy
    Then transform from left-camera frame into body frame via T_BL.

    Add (p_B, Σ_p) to the stereo points in the pointset.
    Where Σ_p is the 3×3 position covariance from
    propagating σ_px² (calibration.yaml) through the triangulation
    Jacobian. 
    
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
    """Prediction function (§eq:u_pred): pose change x point + velocity
    transport, projected through camera `camera` ∈ {"L", "R"}..
    """
    #SE3 apply
    p_Bk: np.ndarray = np.array(delta_T.apply(p_Bkm1), dtype=np.float64)

    #velocity transport
    v_p_Bk: np.ndarray = np.array( 
        delta_T.rotation().apply(v_p), dtype=np.float64
    )
    p_Bk_transported: np.ndarray = p_Bk + v_p_Bk * dt

    # project throug pinhole
    u_pred: np.ndarray = project_point(p_Bk_transported, calib, camera)
 
    return u_pred

@dataclass
class CandidateSample:
    """One sample from the uncertainty product space.

    Carries the input that produced the candidate so the winning sample
    can warm-start the joint solver (§search_region step 6).
    """
    def predict(self, delta_t: float, calib: dict, camera: str) -> None:
        self.pixel = predict_pixel(self.delta_T, self.p_Bkm1, self.v_p,
                                   delta_t, calib, camera)
        
    keypoint: cv.KeyPoint = None  # Populated by SSD match
    pixel: np.ndarray     = None  # (2,) Predicted pixel 
    delta_T: jaxlie.SE3   = None  # ΔT sample
    p_Bkm1: np.ndarray    = None  # (3,) p sample 
    v_p: np.ndarray       = None  # (3,) velocity sample (or v_⊥ scalar for MM)
    p_Bk: np.ndarray      = None  # (3,) p sample (Only relevant for feature points)
   

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
    assert camera in ["L", "R"], "Camera must be 'L' or 'R'"
    if p.p_prev is None:
        return []

    p_Bkm1 = np.asarray(p.p_prev, dtype=np.float64).reshape(3,)
    if p.v_prev is None:
        v_p = np.zeros(3, dtype=np.float64)
    else:
        v_p = np.asarray(p.v_prev, dtype=np.float64).reshape(3,)

    samples: list[CandidateSample] = []
    cand = CandidateSample(delta_T=delta_T_hat, p_Bkm1=p_Bkm1, v_p=v_p)
    cand.predict(dt, calib, camera)
    size = tuple(calib["camera_intrinsics"][_camera_key(camera)]["size"])
    if _pixel_in_bounds(cand.pixel, size):
        samples.append(cand)
    return samples

def candidate_feature_set(p: Point,
                  p_Bk:jnp.ndarray, sigma:jnp.ndarray, 
                  calib: dict, alg: dict, camera: str
                  ) -> list[CandidateSample]:
    """Generate candidate pixels for `p` by sampling the uncertainty
    ellipsoid given by p_Bk and sigma (§search_region).

    p_Bk is the center of the ellipsoid. sigma is a 3x3 covariance.

    Returns the list of candidates within the ellipsoid for some significance level.
    Points must project to pixels in the image.
    Empty list ⇒ nothing to match (caller skips).
    
    """
    assert camera in ["L", "R"], "Camera must be 'L' or 'R'"
    alpha_sr = float(alg["signif"]["alpha_sr"])
    size = tuple(calib["camera_intrinsics"][_camera_key(camera)]["size"])

    p_Bk = np.asarray(p_Bk, dtype=np.float64).reshape(3,)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(3, 3)

    candidates: list[CandidateSample] = []
    try:
        eigvals, eigvecs = np.linalg.eigh(sigma)
    except np.linalg.LinAlgError:
        eigvals = np.clip(np.diag(sigma), 0.0, None)
        eigvecs = np.eye(3)

    eigvals = np.clip(eigvals, 0.0, None)
    # Use the significance level as a conservative multiplier for the
    # covariance-based search radius.
    scale = np.sqrt(alpha_sr)

    ellipsoids = [p_Bk]
    for i in range(3):
        std = np.sqrt(eigvals[i])
        if std == 0.0:
            continue
        axis = eigvecs[:, i]
        radius = axis * (std * scale)
        ellipsoids.append(p_Bk + radius)
        ellipsoids.append(p_Bk - radius)

    seen = set()
    for p_sample in ellipsoids:
        pixel = np.asarray(project_point(jnp.asarray(p_sample), calib, camera)).reshape(2,)
        pixel_key = (round(float(pixel[0]), 2), round(float(pixel[1]), 2))
        if pixel_key in seen:
            continue
        seen.add(pixel_key)
        if not _pixel_in_bounds(pixel, size):
            continue
        candidates.append(CandidateSample(pixel=pixel,
                                          p_Bk=p_sample,
                                          v_p=np.zeros(3, dtype=np.float64)))
    return candidates


def ssd_coarse_match(image_src: MatLike, image_dst: MatLike,
                     u_src: cv.KeyPoint,
                     candidates: list[CandidateSample],
                     calib: dict, alg: dict
                     ) -> CandidateSample | None:
    """Evaluate SSD between the reference patch at u_src in image_src
    and each candidate location in image_dst. Return the lowest-SSD
    candidate.

    Patch size from algorithm.yaml. SSD chosen over
    NCC because src and dst are the same camera one frame apart —
    brightness/contrast invariance not needed.
    """
    patch_size = tuple(alg["cv"]["ssd_patch_size"])
    if u_src is None or len(candidates) == 0:
        return None

    ref_patch = _get_patch(image_src, u_src.pt, patch_size)
    if ref_patch is None:
        return None

    best = None
    best_ssd = float("inf")
    for cand in candidates:
        if cand.pixel is None:
            continue
        patch = _get_patch(image_dst, tuple(cand.pixel.tolist()), patch_size)
        if patch is None:
            continue
        diff = ref_patch - patch
        ssd = float(np.sum(diff * diff))
        if ssd < best_ssd:
            best_ssd = ssd
            best = cand

    if best is None:
        return None

    x, y = float(best.pixel[0]), float(best.pixel[1])
    best.keypoint = cv.KeyPoint(x, y, 1.0)
    return best


def lk_refine(image_src: MatLike, image_dst: MatLike,
              u_src: cv.KeyPoint, u_init: cv.KeyPoint,
              alg: dict
              ) -> cv.keyPoint:
    """Pyramidal Lucas-Kanade refinement from coarse seed `u_init`.

    Wraps cv.calcOpticalFlowPyrLK. Window size, pyramid levels,
    termination criteria from algorithm.yaml.
    Returns the refined keypoint.
    """
    if u_src is None or u_init is None:
        raise ValueError("Both source and initial keypoints are required for LK refinement")

    win_size = int(alg["cv"]["klt_window"])
    max_level = int(alg["cv"]["klt_pyramid"])
    if image_src.ndim == 3 and image_src.shape[2] == 3:
        image_src = cv.cvtColor(image_src, cv.COLOR_BGR2GRAY)
    if image_dst.ndim == 3 and image_dst.shape[2] == 3:
        image_dst = cv.cvtColor(image_dst, cv.COLOR_BGR2GRAY)

    prev_pts = np.array([[u_src.pt]], dtype=np.float32)
    next_pts = np.array([[u_init.pt]], dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    next_pts_refined, status, _ = cv.calcOpticalFlowPyrLK(
        image_src, image_dst,
        prev_pts, next_pts,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=criteria,
        flags=cv.OPTFLOW_USE_INITIAL_FLOW,
    )

    if next_pts_refined is None or status is None or status.shape[0] == 0:
        return u_init
    if int(status[0, 0]) != 1:
        return u_init

    x, y = float(next_pts_refined[0, 0, 0]), float(next_pts_refined[0, 0, 1])
    return cv.KeyPoint(x, y, u_init.size, u_init.angle,
                       u_init.response, u_init.octave, u_init.class_id)


def fb_check(image_src: MatLike, image_dst: MatLike,
             u_src: cv.KeyPoint, u_dst: cv.KeyPoint,
             alg: dict
             ) -> bool:
    """Forward-backward consistency. Re-track u_dst back into image_src;
    accept iff round-trip pixel error < ε_fb (algorithm.yaml).
    Primary failure detector for LK tracking.
    True if passed check. 
    """
    if u_src is None or u_dst is None:
        return False

    if image_src.ndim == 3 and image_src.shape[2] == 3:
        image_src = cv.cvtColor(image_src, cv.COLOR_BGR2GRAY)
    if image_dst.ndim == 3 and image_dst.shape[2] == 3:
        image_dst = cv.cvtColor(image_dst, cv.COLOR_BGR2GRAY)

    prev_pts = np.array([[u_dst.pt]], dtype=np.float32)
    next_pts_back, status, _ = cv.calcOpticalFlowPyrLK(
        image_dst, image_src,
        prev_pts, None,
        winSize=(int(alg["cv"]["klt_window"]), int(alg["cv"]["klt_window"])),
        maxLevel=int(alg["cv"]["klt_pyramid"]),
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_pts_back is None or status is None or int(status[0, 0]) != 1:
        return False

    x_back, y_back = float(next_pts_back[0, 0, 0]), float(next_pts_back[0, 0, 1])
    x_src, y_src = float(u_src.pt[0]), float(u_src.pt[1])
    error = np.hypot(x_back - x_src, y_back - y_src)
    return error <= float(alg["cv"]["fb_check"])


# =============================================================================
# Convenience wrappers
# =============================================================================


def temporal_match_one_feature(p: Point,
                       image_src: MatLike, image_dst: MatLike,
                       P_Bk: jnp.ndarray, sigma: jnp.ndarray, 
                       calib: dict, alg: dict,
                       camera: str
                       ) -> CandidateSample | None:
    """Full temporal-matching pipeline for one feature point in one camera.

    Composes feature_candidate_set → ssd_coarse_match → lk_refine → fb_check.
    Returns CandidateSample on success, None on any failure.
    """
    assert camera in ["L", "R"], "The camera is given by 'L' or 'R'"
    if camera == "L":
        kpt_src = p.uL_prev
    if camera == "R":
        kpt_src = p.uR_prev
    
    cands = candidate_feature_set(p, P_Bk, sigma, calib, alg, camera)
    if len(cands) != 0:
        cand = ssd_coarse_match(image_src, image_dst, kpt_src, cands, calib, alg)
        cand.keypoint = lk_refine(image_src, image_dst, kpt_src, cand.keypoint, alg)
        passed = fb_check(image_src, image_dst, kpt_src, cand.keypoint, alg)
        if passed:
            return cand
        
    return None


def temporal_match_one(p: Point,
                       image_src: MatLike, image_dst: MatLike,
                       delta_T: jaxlie.SE3, Sigma_xi: np.ndarray,
                       dt: float, calib: dict, alg: dict,
                       camera: str
                       ) -> CandidateSample | None:
    """Full temporal-matching pipeline for one point in one camera.

    Composes candidate_set → ssd_coarse_match → lk_refine → fb_check.
    Returns CandidateSample on success, None on any failure.
    """
    assert camera in ["L", "R"], "The camera is given by 'L' or 'R'"
    if camera == "L":
        kpt_src = p.uL_prev
    if camera == "R":
        kpt_src = p.uR_prev

    cands = candidate_set(p, delta_T, Sigma_xi, dt, calib, alg, camera)
    if len(cands) != 0:
        cand = ssd_coarse_match(image_src, image_dst, kpt_src, cands, calib, alg)
        cand.keypoint = lk_refine(image_src, image_dst, kpt_src, cand.keypoint, alg)
        passed = fb_check(image_src, image_dst, kpt_src, cand.keypoint, alg)
        if passed:
            return cand
        
    return None
