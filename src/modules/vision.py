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
from scipy.stats import chi2

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