"""
vision.py — Computer vision primitives.

Detection, stereo matching, temporal matching, depth reconstruction,
and grid/focus sampling. All physical and algorithm parameters read
from calibration.yaml and algorithm.yaml. No hardcoded values.

Stateless module: each function takes images and parameters explicitly.

Input image convention: caller passes grayscale images. Module does
not convert or check.

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
from scipy.stats import chi2,norm

from .points import Point, PointSet, PixelType
from .ekf import project_point


# =============================================================================
# Internal helpers
# =============================================================================

def _camera_key(camera: str) -> str:
    assert camera in ["L", "R"], "Camera must be 'L' or 'R'"
    return "left" if camera == "L" else "right"


def _pixel_in_bounds(pixel: np.ndarray, size: tuple[int, int]) -> bool:
    x, y = float(pixel[0]), float(pixel[1])
    width, height = size
    return 0.0 <= x < width and 0.0 <= y < height


def _clamp_center(center: tuple[float, float],
                  patch_size: tuple[int, int],
                  image_shape: tuple[int, ...]) -> tuple[float, float]:
    """Clamp a patch center so the patch fits in the image."""
    h, w = image_shape[:2]
    pw, ph = patch_size
    half_w = (pw - 1) / 2.0
    half_h = (ph - 1) / 2.0
    cx = max(half_w, min(w - 1 - half_w, float(center[0])))
    cy = max(half_h, min(h - 1 - half_h, float(center[1])))
    return cx, cy


def _ellipsoid_samples(mean: np.ndarray, sigma: np.ndarray,
                       alpha: float) -> np.ndarray:
    """Sample at mean and ±sqrt(χ²_α,d · λ_i) along each eigenvector.

    Returns (M, d) array with the mean first, then 2·d_eff axis extremes
    (some axes dropped if their eigenvalue is zero).
    """
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64)
    d = mean.shape[0]
    radius = np.sqrt(chi2.ppf(alpha, d))

    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)

    out = [mean.copy()]
    for i in range(d):
        std = np.sqrt(eigvals[i])
        if std == 0.0:
            continue
        axis = eigvecs[:, i] * (std * radius)
        out.append(mean + axis)
        out.append(mean - axis)
    return np.stack(out, axis=0)


def _se3_perturbation_samples(sigma_xi: np.ndarray, alpha: float
                              ) -> list[np.ndarray]:
    """Sample SE(3) tangent perturbations on E_ξ.

    Returns list of (6,) ξ vectors including zero (the mean). Caller
    applies each as ΔT = ΔT_hat ⊕ Exp(ξ).
    """
    sigma = np.asarray(sigma_xi, dtype=np.float64)
    radius = np.sqrt(chi2.ppf(alpha, 6))

    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)

    xi_samples = [np.zeros(6, dtype=np.float64)]
    for i in range(6):
        std = np.sqrt(eigvals[i])
        if std == 0.0:
            continue
        axis = eigvecs[:, i] * (std * radius)
        xi_samples.append(axis)
        xi_samples.append(-axis)
    return xi_samples


def _ball_samples(radius: float, dim: int = 3) -> np.ndarray:
    """Axis-aligned ball samples: origin plus ±r along each axis. (2d+1, d)"""
    out = [np.zeros(dim, dtype=np.float64)]
    for i in range(dim):
        e = np.zeros(dim, dtype=np.float64); e[i] = radius
        out.append(e)
        out.append(-e)
    return np.stack(out, axis=0)


def _depth_samples(Z_min: float, Z_max: float, n: int = 5) -> np.ndarray:
    """Linspace depth samples for S_Z. (n,)"""
    return np.linspace(Z_min, Z_max, n)


def _backproject_pixel(u_pix: np.ndarray, Z: float,
                       calib: dict, camera: str) -> np.ndarray:
    """Back-project pixel u at camera-frame depth Z into body frame.

    Inverse of π(R_cB·p_B + t_c) at fixed Z = p_cam[2].
    """
    side = _camera_key(camera)
    K = np.asarray(calib["camera_intrinsics"][side]["k_matrix"], dtype=np.float64)
    T_cb = np.asarray(calib["camera_extrinsics"][side]["cog_cam"], dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_norm = (float(u_pix[0]) - cx) / fx
    v_norm = (float(u_pix[1]) - cy) / fy
    p_cam = np.array([u_norm * Z, v_norm * Z, Z], dtype=np.float64)

    R_cB = T_cb[:3, :3]
    t_c  = T_cb[:3,  3]
    return R_cB.T @ (p_cam - t_c)


def _build_deltaT_arrays(delta_T_hat: jaxlie.SE3,
                         xi_samples: list[np.ndarray]
                         ) -> tuple[list[jaxlie.SE3], np.ndarray, np.ndarray]:
    """For each ξ in xi_samples, build ΔT_i = ΔT_hat @ Exp(ξ).
    Returns (list of SE3, R_stack (N, 3, 3), t_stack (N, 3)).
    The matrix stacks are for bulk numpy transform; the SE3 list is for
    storing back on CandidateSample for solver warm-start.
    """
    se3_list: list[jaxlie.SE3] = []
    R_list = []
    t_list = []
    for xi in xi_samples:
        if np.linalg.norm(xi) == 0.0:
            dT = delta_T_hat
        else:
            dT = delta_T_hat @ jaxlie.SE3.exp(jnp.asarray(xi))
        T_mat = np.asarray(dT.as_matrix(), dtype=np.float64)
        se3_list.append(dT)
        R_list.append(T_mat[:3, :3])
        t_list.append(T_mat[:3,  3])
    return se3_list, np.stack(R_list, axis=0), np.stack(t_list, axis=0)


# =============================================================================
# Detection — FAST + Shi-Tomasi (§sec:cv)
# =============================================================================

def detect_keypoints(image: MatLike, calib: dict, alg: dict) -> list[cv.KeyPoint]:
    """Run FAST detection then Shi-Tomasi cornerness scoring on `image`.

    Returns the unfiltered keypoint pool — caller decides how to allocate
    via grid_select / focus_select. Sorted by Shi-Tomasi response,
    descending. 

    Stored once per frame in the pipeline; passed to allocators below.
    """
    tau_min        = float(alg["cv"]["tau_min"])
    fast_threshold = int(alg["cv"]["fast_threshold"])
    fast_non_max   = bool(alg["cv"]["fast_non_max"])
    sht_block_size = int(alg["cv"]["sht_block_size"])
    sht_ksize      = int(alg["cv"]["sht_ksize"])

    # FAST detection
    fast = cv.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=fast_non_max)
    fast_kps = fast.detect(image, None)
    if len(fast_kps) == 0:
        return []

    # Shi-Tomasi response per FAST keypoint
    response_img = cv.cornerMinEigenVal(image, blockSize=sht_block_size, ksize=sht_ksize)
    max_response = float(response_img.max())
    threshold = tau_min * max_response

    keypoints = []
    for kp in fast_kps:
        ix, iy = int(round(kp.pt[0])), int(round(kp.pt[1]))
        ix = max(0, min(response_img.shape[1] - 1, ix))
        iy = max(0, min(response_img.shape[0] - 1, iy))
        score = float(response_img[iy, ix])
        if score < threshold:
            continue
        kp.response = score
        keypoints.append(kp)

    keypoints.sort(key=lambda k: k.response, reverse=True)
    return keypoints


# =============================================================================
# Sampling / replenishment (§sec:cv "Grid Selection and Focus Weighting")
# =============================================================================

def grid_select_features(pool: list[cv.KeyPoint],
                         existing: PointSet,
                         calib: dict, alg: dict, camera: str
                         ) -> list[cv.KeyPoint]:
    """Allocate feature replenishments across an N×M grid for spatial spread.

    Strategy:
      1. Drop pool entries too close to existing points (min_distance_px).
      2. Pick top N_feat_gl globally by response (pool is already sorted).
      3. Bin the leftovers into the grid; take top N_feat_lo from each cell.

    Returns up to N_feat_gl + grid_rows·grid_cols·N_feat_lo keypoints. The
    caller truncates to its replenishment budget.
    """
    assert camera in ("L", "R"), "camera must be 'L' or 'R'"

    cam_key = _camera_key(camera)
    width, height = calib["camera_intrinsics"][cam_key]["size"]
    grid_cols, grid_rows = alg["cv"]["grid_cells"]
    N_feat_gl = int(alg["cv"]["N_feat_gl"])
    N_feat_lo = int(alg["cv"]["N_feat_lo"])
    min_dist  = float(alg["cv"]["min_distance_px"])

    existing_pixels = _current_pixels(existing, camera)
    filtered = _filter_near_existing(pool, existing_pixels, min_dist, width, height)

    # Top N_feat_gl globally (filtered preserves response-sort from detect_keypoints)
    selected  = filtered[:N_feat_gl]
    remaining = filtered[N_feat_gl:]

    # Top N_feat_lo per cell from the leftovers
    cells = _bin_into_grid(remaining, width, height, grid_rows, grid_cols)
    for row in range(grid_rows):
        for col in range(grid_cols):
            selected.extend(cells[row][col][:N_feat_lo])

    return selected


def focus_select_interest(pool: list[cv.KeyPoint],
                          focus_B: np.ndarray,
                          sigma_F: float,
                          existing: PointSet,
                          calib: dict, alg: dict, camera: str
                          ) -> list[cv.KeyPoint]:
    """Allocate interest points around the focus projection.

    The focus point F_{k-1} projects to a pixel f_c in `camera`. A 2D
    isotropic Gaussian N(f_c, sigma_F·I) is defined over the image plane.
    Each grid cell's weight is the probability mass of that Gaussian
    over the cell rectangle (X ⊥ Y, so it factors into CDF differences
    in x and y). Re-normalized to the mass over the image rectangle.
    Allocation per cell is floor(N · w_g). When a cell
    cannot fulfil its allocation (insufficient keypoints), the residual
    spills to other cells in weight-descending order — peak first, then
    outward.
    """
    assert camera in ("L", "R"), "camera must be 'L' or 'R'"

    cam_key = _camera_key(camera)
    width, height       = calib["camera_intrinsics"][cam_key]["size"]
    grid_cols, grid_rows = alg["cv"]["grid_cells"]
    N        = int(alg["cv"]["N_I_max"]) // 2
    min_dist = float(alg["cv"]["min_distance_px"])

    # Project focus into camera image plane
    focus_pixel = np.asarray(project_point(jnp.asarray(focus_B), calib, camera))[0]
    mu_x, mu_y = float(focus_pixel[0]), float(focus_pixel[1])
    sigma = float(sigma_F)

    # Filter pool against existing trackers, bin into grid
    existing_pixels = _current_pixels(existing, camera)
    filtered = _filter_near_existing(pool, existing_pixels, min_dist, width, height)
    cells = _bin_into_grid(filtered, width, height, grid_rows, grid_cols)

    # Per-cell weight = P(cell) under N(f_c, σ²·I), renormalised over the
    # image rectangle. Since X ⊥ Y, P(cell) factors and so does the
    # normaliser: Z = P(0 ≤ X ≤ W) · P(0 ≤ Y ≤ H) = Z_x · Z_y.
    x_edges = np.linspace(0.0, width,  grid_cols + 1)
    y_edges = np.linspace(0.0, height, grid_rows + 1)
    Z_x = norm.cdf(width,  loc=mu_x, scale=sigma) - norm.cdf(0.0, loc=mu_x, scale=sigma)
    Z_y = norm.cdf(height, loc=mu_y, scale=sigma) - norm.cdf(0.0, loc=mu_y, scale=sigma)
    p_x = np.diff(norm.cdf(x_edges, loc=mu_x, scale=sigma)) / Z_x   # (grid_cols,)
    p_y = np.diff(norm.cdf(y_edges, loc=mu_y, scale=sigma)) / Z_y   # (grid_rows,)
    weights = np.outer(p_y, p_x)                                    # (grid_rows, grid_cols)

    # Allocation
    alloc = np.floor(N * weights).astype(int)

    # First pass: take up to alloc[r,c] from each cell, top-by-response.
    # (cells[r][c] is response-descending because pool came in that order.)
    selected: list[cv.KeyPoint] = []
    taken = np.zeros((grid_rows, grid_cols), dtype=int)
    for r in range(grid_rows):
        for c in range(grid_cols):
            n_take = min(int(alloc[r, c]), len(cells[r][c]))
            selected.extend(cells[r][c][:n_take])
            taken[r, c] = n_take

    # Spillover: distribute the remaining budget by weight-descending
    # order. Under an isotropic Gaussian this is peak-first then outward.
    remaining = N - len(selected)
    if remaining > 0:
        order = np.argsort(weights.flatten())[::-1]
        for idx in order:
            if remaining == 0:
                break
            r, c = np.unravel_index(idx, weights.shape)
            available = len(cells[r][c]) - taken[r, c]
            if available <= 0:
                continue
            n_extra = min(remaining, available)
            start = taken[r, c]
            selected.extend(cells[r][c][start:start + n_extra])
            taken[r, c] += n_extra
            remaining -= n_extra

    return selected


def _current_pixels(points: PointSet, camera: str) -> np.ndarray:
    """Return current-frame pixels already occupied in one camera."""
    pixels = []
    for point in points:
        keypoint = point.uL_curr if camera == "L" else point.uR_curr
        if keypoint is not None:
            pixels.append(keypoint.pt)

    return np.asarray(pixels, dtype=float).reshape(-1, 2)

def _filter_near_existing(pool: list[cv.KeyPoint],
                          existing_pixels: np.ndarray,
                          min_dist: float,
                          width: int, height: int
                          ) -> list[cv.KeyPoint]:
    """Drop out-of-image and too-close-to-existing keypoints. Preserves order."""
    out = []
    for kp in pool:
        x, y = kp.pt
        if not (0 <= x < width and 0 <= y < height):
            continue
        if existing_pixels.shape[0] > 0:
            dists = np.linalg.norm(existing_pixels - np.asarray(kp.pt), axis=1)
            if np.any(dists < min_dist):
                continue
        out.append(kp)
    return out

def _bin_into_grid(pool: list[cv.KeyPoint],
                   width: int, height: int,
                   grid_rows: int, grid_cols: int
                   ) -> list[list[list[cv.KeyPoint]]]:
    """Bucket keypoints by image grid cell. Preserves order within each cell."""
    cells = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
    cell_width  = width  / grid_cols
    cell_height = height / grid_rows
    for kp in pool:
        x, y = kp.pt
        col = min(int(x // cell_width),  grid_cols - 1)
        row = min(int(y // cell_height), grid_rows - 1)
        cells[row][col].append(kp)
    return cells

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
    For rectified stereo the epipolar line is a horizontal row, so the
    search reduces to 1D along v = v_src.

    Uses cv.matchTemplate with TM_CCOEFF_NORMED — the zero-mean
    normalised cross-correlation per §sec:cv eq:NCC. Sub-pixel
    refinement by parabolic peak fitting on the NCC response.

    Returns a dict {idx → cv.KeyPoint | None} mapping each
    keypoints_src index to its match in dst, None if peak < ncc_min or
    if bounds make matching infeasible.
    """
    if direction not in ("L→R", "R→L"):
        raise ValueError(f"direction must be 'L→R' or 'R→L', got {direction!r}")

    disp_min       = float(alg["cv"]["disp_min"])
    disp_max       = float(alg["cv"]["disp_max"])
    stereo_subpix  = bool(alg["cv"]["stereo_subpix"])
    ncc_min        = float(alg["cv"]["ncc_min"])
    patch_size     = int(alg["cv"]["ncc_patch_size"])
    half           = patch_size // 2

    h_src, w_src = image_src.shape[:2]
    h_dst, w_dst = image_dst.shape[:2]

    matches: dict[int, cv.KeyPoint | None] = {}
    for idx, kp_src in enumerate(keypoints_src):
        u_src, v_src = kp_src.pt
        u_int = int(round(u_src))
        v_int = int(round(v_src))

        # Template bounds in src
        if (u_int - half < 0 or u_int + half >= w_src
                or v_int - half < 0 or v_int + half >= h_src):
            matches[idx] = None
            continue

        # Disparity range → column search bounds in dst
        if direction == "L→R":
            u_lo = u_src - disp_max
            u_hi = u_src - disp_min
        else:
            u_lo = u_src + disp_min
            u_hi = u_src + disp_max
        u_lo_int = max(half,             int(np.floor(u_lo)))
        u_hi_int = min(w_dst - 1 - half, int(np.ceil(u_hi)))
        if u_hi_int < u_lo_int:
            matches[idx] = None
            continue

        template = image_src[v_int - half:v_int + half + 1,
                             u_int - half:u_int + half + 1]
        strip    = image_dst[v_int - half:v_int + half + 1,
                             u_lo_int - half:u_hi_int + half + 1]

        # Zero-mean normalised cross-correlation
        response = cv.matchTemplate(strip, template, cv.TM_CCOEFF_NORMED)
        ncc = response[0]                                       # (N,)

        peak_idx = int(np.argmax(ncc))
        peak_ncc = float(ncc[peak_idx])
        if np.isnan(peak_ncc) or peak_ncc < ncc_min:
            matches[idx] = None
            continue

        u_peak = u_lo_int + peak_idx                           # integer

        # Parabolic sub-pixel fit on 3 NCC values around the peak
        if stereo_subpix and 0 < peak_idx < len(ncc) - 1:
            y_m1 = float(ncc[peak_idx - 1])
            y_0  = float(ncc[peak_idx])
            y_p1 = float(ncc[peak_idx + 1])
            denom = y_m1 - 2.0 * y_0 + y_p1
            if abs(denom) > 1e-12:
                offset = float(np.clip(0.5 * (y_m1 - y_p1) / denom, -1.0, 1.0))
                u_refined = u_peak + offset
            else:
                u_refined = float(u_peak)
        else:
            u_refined = float(u_peak)

        matches[idx] = cv.KeyPoint(u_refined, float(v_src),
                                   kp_src.size, kp_src.angle,
                                   kp_src.response, kp_src.octave,
                                   kp_src.class_id)
    return matches

# =============================================================================
# Stereo depth and covariance (§sec:depth_estimation)
# =============================================================================

def reconstruct_depth(points: PointSet, calib: dict, alg: dict) -> PointSet:
    """Triangulate (u_L, u_R) pairs into body-frame 3D points (§sec:depth_estimation).
    For each point with stereo pixels but no p_curr yet.

    Distortion is not currently supported; raises NotImplementedError if
    non-zero distortion coefficients are present. Points without stereo
    pairs are left unchanged.

    Mutates `points` in place; returns the same PointSet for chaining.
    """
    dist_coeff = np.asarray(calib["camera_intrinsics"]["left"]["dist_coeff"])
    if np.any(dist_coeff != 0.0):
        raise NotImplementedError("Non-zero distortion not supported in reconstruct_depth.")

    K_L = np.asarray(calib["camera_intrinsics"]["left"]["k_matrix"])
    fu, fv = K_L[0, 0], K_L[1, 1]
    cu, cv_c = K_L[0, 2], K_L[1, 2]
    baseline = float(calib["camera_extrinsics"]["baseline"])
    fb = baseline * fu

    T_LB = np.asarray(calib["camera_extrinsics"]["left"]["cog_cam"])
    R_BL = T_LB[:3, :3].T
    t_L  = T_LB[:3,  3]

    var_sigma_px = float(alg["cv"]["var_sigma_px"])
    
    inv_fu, inv_fv = 1.0 / fu, 1.0 / fv

    for point in points:
        if point.p_curr is not None:
            continue
        if point.uL_curr is None or point.uR_curr is None:
            continue

        uLx, uLy = point.uL_curr.pt
        uRx      = point.uR_curr.pt[0]
        disparity = uLx - uRx
       
        # Camera-frame point
        x_n = (uLx - cu) * inv_fu        # no distortion → x_n = x_d
        y_n = (uLy - cv_c) * inv_fv
        Z = fb / disparity
        p_c = np.array([x_n * Z, y_n * Z, Z], dtype=float)

        # Triangulation Jacobian ∂p^c / ∂(u_L, v_L, u_R) (eq:Jpc_stereo)
        # No distortion → C = diag(1/f_u, 1/f_v).
        inv_d = 1.0 / disparity
        J = Z * np.array([
            [inv_fu - x_n * inv_d,  0.0,    x_n * inv_d],
            [-y_n * inv_d,          inv_fv, y_n * inv_d],
            [-inv_d,                0.0,    inv_d],
        ])

        Sigma_pc = var_sigma_px * (J @ J.T)
        Sigma_pB = R_BL @ Sigma_pc @ R_BL.T

        point.p_curr = R_BL @ (p_c - t_L)
        point.Sigma_curr = Sigma_pB

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

# =============================================================================
# Sample-carrying dataclass (carries solver warm-start info)
# =============================================================================

@dataclass
class CandidateSample:
    """One sample from the uncertainty product space.

    Carries the input that produced the candidate so the winning sample
    can warm-start the joint solver (§search_region step 6).
    """
    keypoint: cv.KeyPoint = None   # Populated by SSD coarse match / LK refine
    pixel:    np.ndarray  = None   # (2,) predicted pixel
    delta_T:  jaxlie.SE3  = None   # ΔT sample that produced this candidate
    p_Bkm1:   np.ndarray  = None   # (3,) p sample at k-1 (or back-projected)
    v_p:      np.ndarray  = None   # (3,) velocity sample
    p_Bk:     np.ndarray  = None   # (3,) p sample at k (post-ΔT, post-transport)




# =============================================================================
# Temporal-matching primitives
# =============================================================================

def candidate_feature_set(p: Point,
                          p_Bk: jnp.ndarray, sigma: jnp.ndarray,
                          calib: dict, alg: dict, camera: str
                          ) -> list[CandidateSample]:
    """Candidate pixels around an ellipsoid centered at p_Bk (body frame).

    p_Bk : (3,) ellipsoid centre (typically the EKF's body-frame estimate).
    sigma: (3, 3) covariance.

    For each principal axis of sigma, samples ±sqrt(χ²_{α_sr,3} · λ_i)
    extremes. Projects all samples to pixels, deduplicates on integer
    pixel coords, and returns those in-image.

    Used by the EKF-state search path: feature whose state-frame
    position and covariance come from the filter.
    """
    assert camera in ["L", "R"], "Camera must be 'L' or 'R'"
    alpha_sr = float(alg["signif"]["alpha_sr"])
    size = tuple(calib["camera_intrinsics"][_camera_key(camera)]["size"])

    p_Bk_np = np.asarray(p_Bk, dtype=np.float64).reshape(3)
    sigma_np = np.asarray(sigma, dtype=np.float64).reshape(3, 3)

    body_samples = _ellipsoid_samples(p_Bk_np, sigma_np, alpha_sr)  # (M, 3)
    pixels = np.asarray(
        project_point(jnp.asarray(body_samples), calib, camera)      # (M, 2)
    )

    candidates: list[CandidateSample] = []
    seen: set[tuple[int, int]] = set()
    for i in range(pixels.shape[0]):
        pix = pixels[i]
        if not _pixel_in_bounds(pix, size):
            continue
        key = (int(round(pix[0])), int(round(pix[1])))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(CandidateSample(
            pixel=pix.copy(),
            p_Bk=body_samples[i].copy(),
            v_p=np.zeros(3, dtype=np.float64),
        ))
    return candidates


def _candidates_pv(p: Point, delta_T_hat: jaxlie.SE3,
                   Sigma_xi: np.ndarray, dt: float,
                   calib: dict, alg: dict, camera: str
                   ) -> list[CandidateSample]:
    """Branch: p̂ and v̂ both available (post-solver point).

    Domain: E_ξ × (E_{p,v} ⊕ S_a) per §search_region.
    """
    alpha_sr = float(alg["signif"]["alpha_sr"])
    a_max = float(alg["cv"]["a_max"])
    size = tuple(calib["camera_intrinsics"][_camera_key(camera)]["size"])

    p_hat = np.asarray(p.p_prev, dtype=np.float64).reshape(3)
    v_hat = np.asarray(p.v_prev, dtype=np.float64).reshape(3)
    Sigma_pv = np.asarray(p.Sigma_prev, dtype=np.float64)
    if Sigma_pv.shape != (6, 6):
        # Fallback: use only position covariance if Sigma_prev is 4x4
        return _candidates_p(p, delta_T_hat, Sigma_xi, dt, calib, alg, camera)

    pv_mean = np.concatenate([p_hat, v_hat])
    pv_samples = _ellipsoid_samples(pv_mean, Sigma_pv, alpha_sr)    # (N_pv, 6)
    a_samples = _ball_samples(a_max, 3)                             # (N_a, 3)

    # Expand: for each (p, v) and a, get (p, v + a·dt). (N_pv·N_a, 6)
    pv_exp = np.repeat(pv_samples, a_samples.shape[0], axis=0)      # (N_pv·N_a, 6)
    a_exp  = np.tile(a_samples, (pv_samples.shape[0], 1))           # (N_pv·N_a, 3)
    pv_exp[:, 3:] = pv_exp[:, 3:] + a_exp * dt                      # add a·dt to v

    # ΔT samples
    xi_samples = _se3_perturbation_samples(np.asarray(Sigma_xi), alpha_sr)
    se3_list, R_stack, t_stack = _build_deltaT_arrays(delta_T_hat, xi_samples)

    return _bulk_transform_project_collect(
        pv_exp, se3_list, R_stack, t_stack, dt, calib, camera, size)


def _candidates_p(p: Point, delta_T_hat: jaxlie.SE3,
                  Sigma_xi: np.ndarray, dt: float,
                  calib: dict, alg: dict, camera: str
                  ) -> list[CandidateSample]:
    """Branch: p̂ only (e.g., fresh stereo, no velocity estimate yet).

    Domain: E_ξ × E_p × S_v.
    """
    alpha_sr = float(alg["signif"]["alpha_sr"])
    v_max = float(alg["cv"]["v_max"])
    size = tuple(calib["camera_intrinsics"][_camera_key(camera)]["size"])

    p_hat = np.asarray(p.p_prev, dtype=np.float64).reshape(3)
    Sigma_p = np.asarray(p.Sigma_prev, dtype=np.float64)
    if Sigma_p.shape != (3, 3):
        Sigma_p = Sigma_p[:3, :3]
    
    p_samples = _ellipsoid_samples(p_hat, Sigma_p, alpha_sr)        # (N_p, 3)
    v_samples = _ball_samples(v_max, 3)                             # (N_v, 3)

    # Cartesian product: (N_p · N_v, 6) where col 0:3 = p, col 3:6 = v
    p_exp = np.repeat(p_samples, v_samples.shape[0], axis=0)        # (N_p·N_v, 3)
    v_exp = np.tile(v_samples, (p_samples.shape[0], 1))             # (N_p·N_v, 3)
    pv_exp = np.concatenate([p_exp, v_exp], axis=1)                 # (N_p·N_v, 6)

    xi_samples = _se3_perturbation_samples(np.asarray(Sigma_xi), alpha_sr)
    se3_list, R_stack, t_stack = _build_deltaT_arrays(delta_T_hat, xi_samples)

    return _bulk_transform_project_collect(
        pv_exp, se3_list, R_stack, t_stack, dt, calib, camera, size)


def _candidates_none(p: Point, delta_T_hat: jaxlie.SE3,
                     Sigma_xi: np.ndarray, dt: float,
                     calib: dict, alg: dict, camera: str
                     ) -> list[CandidateSample]:
    """Branch: neither p̂ nor v̂ (fresh mono, no 3D state).

    Domain: E_ξ × S_Z × S_v. p_Bkm1 reconstructed by back-projecting
    the previous pixel observation at each sampled depth.
    """
    alpha_sr = float(alg["signif"]["alpha_sr"])
    v_max = float(alg["cv"]["v_max"])
    Z_min = float(alg["cv"]["Z_min"])
    Z_max = float(alg["cv"]["Z_max"])
    size = tuple(calib["camera_intrinsics"][_camera_key(camera)]["size"])

    u_prev = p.uL_prev if camera == "L" else p.uR_prev
    if u_prev is None:
        return []

    Z_samples = _depth_samples(Z_min, Z_max, n=5)                   # (N_z,)
    # Back-project the previous pixel at each depth → N_z body-frame points
    u_pix = np.array(u_prev.pt, dtype=np.float64)
    p_samples = np.stack(
        [_backproject_pixel(u_pix, Z, calib, camera) for Z in Z_samples],
        axis=0
    )                                                                # (N_z, 3)

    v_samples = _ball_samples(v_max, 3)                              # (N_v, 3)

    # Cartesian: (N_z · N_v, 6)
    p_exp = np.repeat(p_samples, v_samples.shape[0], axis=0)
    v_exp = np.tile(v_samples, (p_samples.shape[0], 1))
    pv_exp = np.concatenate([p_exp, v_exp], axis=1)

    xi_samples = _se3_perturbation_samples(np.asarray(Sigma_xi), alpha_sr)
    se3_list, R_stack, t_stack = _build_deltaT_arrays(delta_T_hat, xi_samples)

    return _bulk_transform_project_collect(
        pv_exp, se3_list, R_stack, t_stack, dt, calib, camera, size)


def _bulk_transform_project_collect(pv_exp: np.ndarray,
                                    se3_list: list[jaxlie.SE3],
                                    R_stack: np.ndarray,
                                    t_stack: np.ndarray,
                                    dt: float,
                                    calib: dict, camera: str,
                                    size: tuple[int, int]
                                    ) -> list[CandidateSample]:
    """Apply each ΔT (R, t) to each (p, v) sample, project, dedupe.

    pv_exp  : (N_pv, 6) of [p (3), v (3)]
    R_stack : (N_dT, 3, 3), t_stack : (N_dT, 3)
    se3_list: N_dT jaxlie SE3 objects matching R_stack/t_stack
    """
    N_dT = R_stack.shape[0]
    N_pv = pv_exp.shape[0]
    if N_pv == 0 or N_dT == 0:
        return []

    p_arr = pv_exp[:, :3]                                            # (N_pv, 3)
    v_arr = pv_exp[:, 3:]                                            # (N_pv, 3)

    p_Bk = (R_stack @ p_arr.T).transpose(0, 2, 1) + t_stack[:, None, :]
    v_Bk = (R_stack @ v_arr.T).transpose(0, 2, 1)
    p_Bk_transported = p_Bk + v_Bk * dt                              # (N_dT, N_pv, 3)

    # Bulk project
    all_p_Bk = p_Bk_transported.reshape(-1, 3)                       # (N_dT·N_pv, 3)
    pixels = np.asarray(
    project_point(jnp.asarray(all_p_Bk), calib, camera)              # (N_dT·N_pv, 2)
    ).reshape(N_dT, N_pv, 2)
    
    # Build CandidateSamples, dedupe on integer pixel coords
    candidates: list[CandidateSample] = []
    seen: set[tuple[int, int]] = set()
    for a in range(N_dT):
        dT = se3_list[a]
        for n in range(N_pv):
            pix = pixels[a, n]
            if not _pixel_in_bounds(pix, size):
                continue
            key = (int(round(pix[0])), int(round(pix[1])))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(CandidateSample(
                delta_T=dT,
                p_Bkm1=p_arr[n].copy(),
                v_p=v_arr[n].copy(),
                p_Bk=p_Bk_transported[a, n].copy(),
                pixel=pix.copy(),
            ))
    return candidates


def candidate_set(p: Point,
                  delta_T_hat: jaxlie.SE3, Sigma_xi: np.ndarray,
                  dt: float, calib: dict, alg: dict, camera: str
                  ) -> list[CandidateSample]:
    """Dispatch to the right uncertainty-product branch (§search_region).

    Three branches by available state on `p`:
      - p̂ and v̂        : E_ξ × (E_{p,v} ⊕ S_a)
      - p̂ only         : E_ξ × E_p × S_v
      - neither        : E_ξ × S_Z × S_v   (uses u_prev for back-projection)
    """
    assert camera in ["L", "R"], "Camera must be 'L' or 'R'"
    has_p = p.p_prev is not None
    has_v = p.v_prev is not None
    if has_p and has_v:
        return _candidates_pv(p, delta_T_hat, Sigma_xi, dt, calib, alg, camera)
    if has_p:
        return _candidates_p(p, delta_T_hat, Sigma_xi, dt, calib, alg, camera)
    return _candidates_none(p, delta_T_hat, Sigma_xi, dt, calib, alg, camera)


# =============================================================================
# SSD coarse match
# =============================================================================

def ssd_coarse_match(image_src: MatLike, image_dst: MatLike,
                     u_src: cv.KeyPoint,
                     candidates: list[CandidateSample],
                     calib: dict, alg: dict
                     ) -> CandidateSample | None:
    """Pick the candidate in `image_dst` whose patch minimises SSD to the
    reference patch at u_src in image_src.

    Boundary handling: when u_src.pt is too close to an edge, the patch
    center is clamped to keep the patch in-image. The same (shift) is
    applied to each candidate, so patches in src and dst represent the
    same image-neighbourhood structure relative to (u_src ↔ u_cand).
    Candidates whose shifted center falls outside image_dst are skipped.

    Patch size and parameters from algorithm.yaml. Sets best.keypoint to
    a cv.KeyPoint at best.pixel.
    """
    patch_size = tuple(alg["cv"]["ssd_patch_size"])
    if u_src is None or len(candidates) == 0:
        return None

    # Reference patch with clamped center
    src_center = _clamp_center(u_src.pt, patch_size, image_src.shape)
    shift = (src_center[0] - float(u_src.pt[0]),
             src_center[1] - float(u_src.pt[1]))
    ref_patch = cv.getRectSubPix(image_src, patch_size, src_center)
    if ref_patch is None:
        return None
    ref_patch = ref_patch.astype(np.float32)

    h_dst, w_dst = image_dst.shape[:2]
    pw, ph = patch_size
    half_w = (pw - 1) / 2.0
    half_h = (ph - 1) / 2.0

    best: CandidateSample | None = None
    best_ssd = float("inf")
    for cand in candidates:
        if cand.pixel is None:
            continue
        cand_cx = float(cand.pixel[0]) + shift[0]
        cand_cy = float(cand.pixel[1]) + shift[1]
        if (cand_cx < half_w or cand_cx > w_dst - 1 - half_w
            or cand_cy < half_h or cand_cy > h_dst - 1 - half_h):
            continue
        patch = cv.getRectSubPix(image_dst, patch_size, (cand_cx, cand_cy))
        if patch is None:
            continue
        diff = ref_patch - patch.astype(np.float32)
        ssd = float(np.sum(diff * diff))
        if ssd < best_ssd:
            best_ssd = ssd
            best = cand

    if best is None:
        return None

    x, y = float(best.pixel[0]), float(best.pixel[1])
    best.keypoint = cv.KeyPoint(x, y, 1.0)
    return best


# =============================================================================
# LK refinement
# =============================================================================

def lk_refine(image_src: MatLike, image_dst: MatLike,
              u_src: cv.KeyPoint, u_init: cv.KeyPoint,
              alg: dict
              ) -> cv.KeyPoint | None:
    """Pyramidal Lucas-Kanade refinement from coarse seed `u_init`.

    Returns refined cv.KeyPoint on success, None on LK failure
    (status != 1 or invalid inputs). Callers should treat None as
    "could not refine" — typically pair with fb_check downstream.

    Parameters (winSize, maxLevel, max_iter, eps) from algorithm.yaml.
    """
    if u_src is None or u_init is None:
        return None

    win_size  = int(alg["cv"]["klt_window"])
    max_level = int(alg["cv"]["klt_pyramid"])
    max_iter  = int(alg["cv"]["klt_max_iter"])
    eps       = float(alg["cv"]["klt_eps"])

    prev_pts = np.array([[u_src.pt]], dtype=np.float32)
    next_pts = np.array([[u_init.pt]], dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, eps)

    next_refined, status, _ = cv.calcOpticalFlowPyrLK(
        image_src, image_dst,
        prev_pts, next_pts,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=criteria,
        flags=cv.OPTFLOW_USE_INITIAL_FLOW,
    )

    if (next_refined is None or status is None or status.shape[0] == 0
            or int(status[0, 0]) != 1):
        return None

    x = float(next_refined[0, 0, 0])
    y = float(next_refined[0, 0, 1])
    return cv.KeyPoint(x, y, u_init.size, u_init.angle,
                       u_init.response, u_init.octave, u_init.class_id)


# =============================================================================
# Forward-backward consistency check
# =============================================================================

def fb_check(image_src: MatLike, image_dst: MatLike,
             u_src: cv.KeyPoint, u_dst: cv.KeyPoint,
             alg: dict
             ) -> bool:
    """Forward-backward consistency.

    Back-tracks from u_dst (in image_dst) into image_src, initialised at
    u_src (informed prior — fast convergence when forward LK was right,
    drifts when it was wrong). Passes iff round-trip pixel error
    ≤ ε_fb (algorithm.yaml).
    """
    if u_src is None or u_dst is None:
        return False

    win_size  = int(alg["cv"]["klt_window"])
    max_level = int(alg["cv"]["klt_pyramid"])
    max_iter  = int(alg["cv"]["klt_max_iter"])
    eps       = float(alg["cv"]["klt_eps"])
    eps_fb    = float(alg["cv"]["fb_check"])

    prev_pts = np.array([[u_dst.pt]], dtype=np.float32)
    init_pts = np.array([[u_src.pt]], dtype=np.float32)   # informed init
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, eps)

    next_back, status, _ = cv.calcOpticalFlowPyrLK(
        image_dst, image_src,
        prev_pts, init_pts,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=criteria,
        flags=cv.OPTFLOW_USE_INITIAL_FLOW,
    )

    if (next_back is None or status is None or status.shape[0] == 0
            or int(status[0, 0]) != 1):
        return False

    x_back = float(next_back[0, 0, 0])
    y_back = float(next_back[0, 0, 1])
    x_src  = float(u_src.pt[0])
    y_src  = float(u_src.pt[1])
    error  = float(np.hypot(x_back - x_src, y_back - y_src))
    return error <= eps_fb


# =============================================================================
# Convenience wrappers: candidate_set → SSD → LK → FB
# =============================================================================

def temporal_match_one(p: Point,
                       image_src: MatLike, image_dst: MatLike,
                       delta_T: jaxlie.SE3, Sigma_xi: np.ndarray,
                       dt: float, calib: dict, alg: dict,
                       camera: str
                       ) -> CandidateSample | None:
    """Full temporal-matching pipeline for one (non-EKF) point in one camera.

    candidate_set → ssd_coarse_match → lk_refine → fb_check. Returns the
    winning CandidateSample on success, None on any failure.
    """
    assert camera in ["L", "R"]
    kpt_src = p.uL_prev if camera == "L" else p.uR_prev

    cands = candidate_set(p, delta_T, Sigma_xi, dt, calib, alg, camera)
    if len(cands) == 0:
        return None

    cand = ssd_coarse_match(image_src, image_dst, kpt_src, cands, calib, alg)
    if cand is None:
        return None

    refined = lk_refine(image_src, image_dst, kpt_src, cand.keypoint, alg)
    if refined is None:
        return None
    cand.keypoint = refined

    if not fb_check(image_src, image_dst, kpt_src, cand.keypoint, alg):
        return None
    return cand


def temporal_match_one_feature(p: Point,
                               image_src: MatLike, image_dst: MatLike,
                               p_Bk: jnp.ndarray, sigma: jnp.ndarray,
                               calib: dict, alg: dict,
                               camera: str
                               ) -> CandidateSample | None:
    """Full temporal-matching pipeline for one EKF-feature point.

    candidate_feature_set → ssd_coarse_match → lk_refine → fb_check.
    """
    assert camera in ["L", "R"]
    kpt_src = p.uL_prev if camera == "L" else p.uR_prev

    cands = candidate_feature_set(p, p_Bk, sigma, calib, alg, camera)
    if len(cands) == 0:
        return None

    cand = ssd_coarse_match(image_src, image_dst, kpt_src, cands, calib, alg)
    if cand is None:
        return None

    refined = lk_refine(image_src, image_dst, kpt_src, cand.keypoint, alg)
    if refined is None:
        return None
    cand.keypoint = refined

    if not fb_check(image_src, image_dst, kpt_src, cand.keypoint, alg):
        return None
    return cand
