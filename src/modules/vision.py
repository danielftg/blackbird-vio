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
    # FAST detector
    fast = cv.FastFeatureDetector_create(threshold=alg["cv"]["fast_threshold"] if "fast_threshold" in alg["cv"] else 10,
                                         nonmaxSuppression=True)
    keypoints = fast.detect(image, None)
    
    if not keypoints:
        return []
    
    # Compute Shi-Tomasi scores (min eigenvalue)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    scores = cv.cornerMinEigenVal(gray, blockSize=alg["cv"]["shi_tomasi_block_size"] if "shi_tomasi_block_size" in alg["cv"] else 3)
    
    # Get max eigenvalue in image for normalization
    max_eigenval = np.max(scores)
    tau_min = alg["cv"]["tau_min"] * max_eigenval
    
    # Filter and score keypoints
    filtered_keypoints = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < scores.shape[1] and 0 <= y < scores.shape[0]:
            score = scores[y, x]
            if score > tau_min:
                kp.response = score
                filtered_keypoints.append(kp)
    
    # Sort by response descending
    filtered_keypoints.sort(key=lambda kp: kp.response, reverse=True)
    
    return filtered_keypoints


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
    cam_key = "left" if camera == "L" else "right"
    width, height = calib["camera_intrinsics"][cam_key]["size"]
    grid_rows, grid_cols = alg["cv"]["grid_cells"]
    cell_width = width / grid_cols
    cell_height = height / grid_rows
    N_feat_gl = alg["cv"]["N_feat_gl"]
    N_feat_lo = alg["cv"]["N_feat_lo"]
    N_F_max = alg["cv"]["N_F_max"]
    
    # Get existing pixels for this camera
    existing_pixels = []
    for point in existing.points:
        if camera == "L" and point.uL is not None:
            existing_pixels.append(point.uL.pt)
        elif camera == "R" and point.uR is not None:
            existing_pixels.append(point.uR.pt)
    existing_pixels = np.array(existing_pixels)
    
    selected = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Cell bounds
            x_min = col * cell_width
            x_max = (col + 1) * cell_width
            y_min = row * cell_height
            y_max = (row + 1) * cell_height
            
            # Count existing in cell
            if len(existing_pixels) > 0:
                in_cell = ((existing_pixels[:, 0] >= x_min) & (existing_pixels[:, 0] < x_max) &
                           (existing_pixels[:, 1] >= y_min) & (existing_pixels[:, 1] < y_max))
                count_existing = np.sum(in_cell)
            else:
                count_existing = 0
            
            if count_existing >= N_F_max:
                continue  # Saturated
            
            # Collect keypoints in cell, not close to existing
            cell_kp = []
            for kp in pool:
                x, y = kp.pt
                if x_min <= x < x_max and y_min <= y < y_max:
                    # Check if close to existing
                    close = False
                    for ex_pt in existing_pixels:
                        if np.linalg.norm(np.array(kp.pt) - ex_pt) < 1.0:  # 1 pixel threshold
                            close = True
                            break
                    if not close:
                        cell_kp.append(kp)
            
            # Sort by response descending
            cell_kp.sort(key=lambda kp: kp.response, reverse=True)
            
            # Take top N_feat_lo
            num_to_take = min(N_feat_lo, len(cell_kp))
            selected.extend(cell_kp[:num_to_take])
    
    # If total selected > N_feat_gl, take top N_feat_gl by response
    if len(selected) > N_feat_gl:
        selected.sort(key=lambda kp: kp.response, reverse=True)
        selected = selected[:N_feat_gl]
    
    return selected


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
    cam_key = "left" if camera == "L" else "right"
    width, height = calib["camera_intrinsics"][cam_key]["size"]
    grid_rows, grid_cols = alg["cv"]["grid_cells"]
    cell_width = width / grid_cols
    cell_height = height / grid_rows
    N_I = alg["cv"]["N_I"] if "N_I" in alg["cv"] else 50  # assume default
    N_I_max = alg["cv"]["N_I_max"]
    
    # Project focus into camera
    focus_pixel = project_point(focus_B, calib, camera)
    if focus_pixel is None:
        return []
    focus_pixel = np.asarray(focus_pixel).reshape(-1)[:2]
    
    # Get existing pixels
    existing_pixels = []
    for point in existing.points:
        if camera == "left" and point.uL is not None:
            existing_pixels.append(point.uL.pt)
        elif camera == "right" and point.uR is not None:
            existing_pixels.append(point.uR.pt)
    existing_pixels = np.array(existing_pixels)
    
    # Compute p_g for each cell
    p_g = np.zeros((grid_rows, grid_cols))
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_center_x = (col + 0.5) * cell_width
            cell_center_y = (row + 0.5) * cell_height
            dist = np.linalg.norm(np.array([cell_center_x, cell_center_y]) - focus_pixel)
            p_g[row, col] = np.exp(-dist**2 / (2 * sigma_F**2))
    
    # Normalize p_g
    total_p = np.sum(p_g)
    if total_p > 0:
        p_g /= total_p
    
    selected = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Cell bounds
            x_min = col * cell_width
            x_max = (col + 1) * cell_width
            y_min = row * cell_height
            y_max = (row + 1) * cell_height
            
            # Collect kp in cell, not close to existing
            cell_kp = []
            for kp in pool:
                x, y = kp.pt
                if x_min <= x < x_max and y_min <= y < y_max:
                    close = False
                    for ex_pt in existing_pixels:
                        if np.linalg.norm(np.array(kp.pt) - ex_pt) < 1.0:
                            close = True
                            break
                    if not close:
                        cell_kp.append(kp)
            
            # Sort by response
            cell_kp.sort(key=lambda kp: kp.response, reverse=True)
            
            # Allocate num = floor(N_I * p_g[row, col])
            num = int(np.floor(N_I * p_g[row, col]))
            num = min(num, N_I_max, len(cell_kp))
            selected.extend(cell_kp[:num])
    
    return selected

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
    disp_min = alg["cv"]["disp_min"]
    disp_max = alg["cv"]["disp_max"]
    stereo_subpix = alg["cv"]["stereo_subpix"]
    ncc_min = alg["cv"]["ncc_min"]
    patch_size = alg["cv"]["ncc_patch_size"]
    half_patch = patch_size // 2
    
    height_src, width_src = image_src.shape[:2]
    height_dst, width_dst = image_dst.shape[:2]
    
    # Convert to gray if needed
    if len(image_src.shape) == 3:
        gray_src = cv.cvtColor(image_src, cv.COLOR_BGR2GRAY)
    else:
        gray_src = image_src
    if len(image_dst.shape) == 3:
        gray_dst = cv.cvtColor(image_dst, cv.COLOR_BGR2GRAY)
    else:
        gray_dst = image_dst
    
    matches = {}
    for idx, kp_src in enumerate(keypoints_src):
        u_src, v_src = kp_src.pt
        v = int(v_src)
        
        # Define search range
        if direction == "L→R":
            u_min = u_src - disp_max
            u_max = u_src - disp_min
        elif direction == "R→L":
            u_min = u_src + disp_min
            u_max = u_src + disp_max
        else:
            matches[idx] = None
            continue
        
        # Collect NCC values
        ncc_values = []
        u_positions = []
        for u in np.arange(u_min, u_max + 1, 1):
            u_int = int(u)
            if u_int - half_patch < 0 or u_int + half_patch >= width_dst or v - half_patch < 0 or v + half_patch >= height_dst:
                continue
            if int(u_src) - half_patch < 0 or int(u_src) + half_patch >= width_src or v - half_patch < 0 or v + half_patch >= height_src:
                continue
            
            patch_src = gray_src[v - half_patch:v + half_patch + 1, int(u_src) - half_patch:int(u_src) + half_patch + 1]
            patch_dst = gray_dst[v - half_patch:v + half_patch + 1, u_int - half_patch:u_int + half_patch + 1]
            
            # Compute NCC
            numerator = np.sum(patch_src.astype(np.float32) * patch_dst.astype(np.float32))
            denom_src = np.sqrt(np.sum(patch_src.astype(np.float32)**2))
            denom_dst = np.sqrt(np.sum(patch_dst.astype(np.float32)**2))
            if denom_src > 0 and denom_dst > 0:
                ncc = numerator / (denom_src * denom_dst)
                ncc_values.append(ncc)
                u_positions.append(u)
        
        if not ncc_values:
            matches[idx] = None
            continue
        
        # Find max NCC
        max_idx = np.argmax(ncc_values)
        max_ncc = ncc_values[max_idx]
        u_peak = u_positions[max_idx]
        
        if max_ncc < ncc_min:
            matches[idx] = None
            continue
        
        # Sub-pixel refinement
        if stereo_subpix and 0 < max_idx < len(ncc_values) - 1:
            # Fit parabola: y = a*(x-x0)^2 + b
            x0 = u_peak
            y0 = max_ncc
            xm1 = u_positions[max_idx - 1]
            ym1 = ncc_values[max_idx - 1]
            xp1 = u_positions[max_idx + 1]
            yp1 = ncc_values[max_idx + 1]
            
            # Solve for a, b
            # At xm1: a*(xm1-x0)^2 + b = ym1
            # At x0: b = y0
            # At xp1: a*(xp1-x0)^2 + b = yp1
            denom = (xm1 - x0)**2 - (xp1 - x0)**2
            if abs(denom) > 1e-6:
                a = ((ym1 - y0) - (yp1 - y0) * (xm1 - x0)**2 / (xp1 - x0)**2) / denom
                if a < 0:  # Maximum
                    delta = - (xp1 - x0)**2 / (2 * a * (xm1 - x0))
                    u_refined = x0 + delta
                else:
                    u_refined = u_peak
            else:
                u_refined = u_peak
        else:
            u_refined = u_peak
        
        # Create matched keypoint
        matched_kp = cv.KeyPoint(u_refined, v_src, kp_src.size, kp_src.angle, kp_src.response, kp_src.octave, kp_src.class_id)
        matches[idx] = matched_kp
    
    return matches


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
    left_intr = calib["camera_intrinsics"]["left"]
    right_intr = calib["camera_intrinsics"]["right"]
    fx = left_intr["fx"]
    fy = left_intr["fy"]
    cx = left_intr["cx"]
    cy = left_intr["cy"]
    b = calib["extrinsics"]["baseline"]  # assume b is there
    T_BL = np.array(calib["extrinsics"]["T_BL"])  # 4x4
    
    sigma_px = calib["sigma_px"] if "sigma_px" in calib else 1.0  # assume
    
    for point in points.points:
        if point.p_B is not None:
            continue
        if point.uL is None or point.uR is None:
            continue
        
        uL_x, uL_y = point.uL.pt
        uR_x, uR_y = point.uR.pt
        
        # Assume rectified, so vL == vR
        d = uL_x - uR_x
        if d <= 0:
            continue  # invalid disparity
        
        Z = b * fx / d
        X = (uL_x - cx) * Z / fx
        Y = (uL_y - cy) * Z / fy
        
        p_L = np.array([X, Y, Z, 1.0])
        p_B_homo = T_BL @ p_L
        point.p_B = p_B_homo[:3]
        
        # Covariance propagation
        # Jacobian of triangulation
        # p_L = [X, Y, Z]
        # partial Z / partial uL_x = -b*fx / d^2
        # partial Z / partial uR_x = b*fx / d^2
        # partial X / partial uL_x = Z/fx + (uL_x - cx) * (-b*fx / d^2) / fx = Z/fx - (uL_x - cx)*b / d^2
        # partial X / partial uR_x = (uL_x - cx) * (b*fx / d^2) / fx = (uL_x - cx)*b / d^2
        # partial Y / partial uL_y = Z/fy
        # partial Y / partial uR_y = 0 (assuming rectified)
        
        dZ_duLx = -b * fx / (d**2)
        dZ_duRx = b * fx / (d**2)
        dX_duLx = Z / fx + (uL_x - cx) * dZ_duLx / fx
        dX_duRx = (uL_x - cx) * dZ_duRx / fx
        dY_duLy = Z / fy
        dY_duRy = 0.0
        
        # Jacobian J: 3x4, since u = [uL_x, uL_y, uR_x, uR_y]
        J = np.array([
            [dX_duLx, dY_duLy, dX_duRx, dY_duRy],
            [0, 0, 0, 0],  # Y doesn't depend on uL_x, uR_x
            [dZ_duLx, 0, dZ_duRx, 0]
        ])
        
        # But Y depends on uL_y, so correct
        J[1, 0] = 0  # dY_duLx = 0
        J[1, 1] = dY_duLy
        J[1, 2] = 0  # dY_duRx = 0
        J[1, 3] = dY_duRy
        
        # Σ_u = sigma_px^2 * I_4
        Sigma_u = sigma_px**2 * np.eye(4)
        
        # Σ_p_L = J @ Sigma_u @ J.T
        Sigma_p_L = J @ Sigma_u @ J.T
        
        # Now, transform to body frame
        # p_B = T_BL @ p_L, but since T_BL is rigid, Σ_p_B = R @ Sigma_p_L @ R.T
        R_BL = T_BL[:3, :3]
        Sigma_p_B = R_BL @ Sigma_p_L @ R_BL.T
        
        point.Sigma_p = Sigma_p_B
    
    return points


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
    project_point
    

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
    alg["signif"]["alpha_sr"]
    alg["cv"]["Z_min"], alg["cv"]["Z_max"]
    alg["cv"]["v_max"]
    alg["cv"]["a_max"]
    CandidateSample()
    CandidateSample().predict
    calib["camera_intrinsics"]["L"]["size"]
    ...

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
    alg["signif"]["alpha_sr"]
    CandidateSample().pixel
    CandidateSample().p_Bk
    CandidateSample().v_p #Set this to (0,0,0). 
    project_point
    calib["camera_intrinsics"]["L"]["size"]
    ...


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
    alg["cv"]["ssd_patch_size"]
    #Evaluate SSD at candidates[i].pixel 
    #Populate candidates[i].keypoint for winning candidate
    ...


def lk_refine(image_src: MatLike, image_dst: MatLike,
              u_src: cv.KeyPoint, u_init: cv.KeyPoint,
              alg: dict
              ) -> cv.keyPoint:
    """Pyramidal Lucas-Kanade refinement from coarse seed `u_init`.

    Wraps cv.calcOpticalFlowPyrLK. Window size, pyramid levels,
    termination criteria from algorithm.yaml.
    Returns the refined keypoint.
    """
    ...


def fb_check(image_src: MatLike, image_dst: MatLike,
             u_src: cv.KeyPoint, u_dst: cv.KeyPoint,
             alg: dict
             ) -> bool:
    """Forward-backward consistency. Re-track u_dst back into image_src;
    accept iff round-trip pixel error < ε_fb (algorithm.yaml).
    Primary failure detector for LK tracking.
    True if passed check. 
    """
    ...


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
