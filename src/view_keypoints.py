"""
view_keypoints.py — Visualise the keypoints the estimator actually tracks.

Runs the same Algo pipeline as main.py and overlays the three live point
sets on the left-camera image after each frame:

    Green  — F      : EKF feature points (drive the filter update)
    Yellow — F_pre  : pre-admission candidates (in the GN solver)
    Blue   — I      : interest points (focus-weighted pool)

Reads images and motor data directly from the bag — no preprocessing needed.

Note: the first frame takes ~30 s while JAX JIT-compiles the estimator.

Usage:
    python src/view_keypoints.py
    python src/view_keypoints.py --bag src/bags/my_flight.bag --limit 200

Controls:
    Space / any key  — advance one frame
    p                — toggle play/pause (auto-advance)
    q / Esc          — quit
"""

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

SRC = Path(__file__).parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import modules.utils as util
import modules.algo as algo
from modules.bag_loader import (
    get_bag_path,
    load_motor_data,
    iter_left_images,
    iter_right_images,
)


# ---------------------------------------------------------------------------
# Colours and legend
# ---------------------------------------------------------------------------

COLORS = {
    "F":     (0,   200,   0),    # green
    "F_pre": (0,   200, 200),    # yellow
    "I":     (200,  80,   0),    # blue
}

LEGEND = [
    ("F      EKF features",   COLORS["F"]),
    ("F_pre  pre-admission",  COLORS["F_pre"]),
    ("I      interest pts",   COLORS["I"]),
]


# ---------------------------------------------------------------------------
# Motor data helpers
# ---------------------------------------------------------------------------

def build_thrust_table(motor_df, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, thrusts) arrays for linear interpolation.

    thrusts has shape (N, 4) — columns match motor_cols order [m1, m4, m3, m2].
    """
    drn = calib["drone_parameters"]
    motor_cols = ["m1", "m4", "m3", "m2"]
    coeffs = np.array([
        drn["rotor_1"]["rpm_thr_coeff"],
        drn["rotor_4"]["rpm_thr_coeff"],
        drn["rotor_3"]["rpm_thr_coeff"],
        drn["rotor_2"]["rpm_thr_coeff"],
    ])

    pivot = (motor_df
             .pivot(index="timestamp_s", columns="motor", values="rpm")
             .ffill().bfill()
             .reset_index())
    pivot[motor_cols] = pivot[motor_cols].clip(0, 8000)

    times   = pivot["timestamp_s"].to_numpy(dtype=np.float64)
    thrusts = coeffs * pivot[motor_cols].to_numpy(dtype=np.float64) ** 2
    return times, thrusts


def interp_motor(t_s: float, times: np.ndarray, thrusts: np.ndarray) -> np.ndarray:
    return np.array([np.interp(t_s, times, thrusts[:, i]) for i in range(4)])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_frame(image: np.ndarray,
               accum: algo.Accumulator,
               frame_idx: int) -> np.ndarray:
    vis = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    for role, pset in (("F", accum.F), ("F_pre", accum.F_pre), ("I", accum.I)):
        color = COLORS[role]
        for pt in pset:
            kp = pt.uL_prev       # current-frame pixel (rolled to _prev after iter)
            if kp is None:
                continue
            cx, cy = int(round(kp.pt[0])), int(round(kp.pt[1]))
            cv.circle(vis, (cx, cy), 4, color,         -1, cv.LINE_AA)
            cv.circle(vis, (cx, cy), 4, (255, 255, 255), 1, cv.LINE_AA)

    n_F, n_Fp, n_I = len(accum.F), len(accum.F_pre), len(accum.I)
    hud = [
        f"Frame {frame_idx}",
        f"F={n_F}  F_pre={n_Fp}  I={n_I}",
        "Space=step  p=play/pause  q=quit",
    ]
    x0, y0, dy = 8, 18, 20
    for i, text in enumerate(hud):
        y = y0 + i * dy
        cv.putText(vis, text, (x0+1, y+1), cv.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0),       1, cv.LINE_AA)
        cv.putText(vis, text, (x0,   y),   cv.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1, cv.LINE_AA)

    h = vis.shape[0]
    for i, (label, color) in enumerate(LEGEND):
        y = h - 12 - i * 18
        cv.circle(vis, (x0+4, y-4), 4, color, -1, cv.LINE_AA)
        cv.putText(vis, label, (x0+12, y), cv.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag",   type=Path, default=None,
                   help="Bag file path. Auto-detected from src/bags/ if omitted.")
    p.add_argument("--limit", type=int,  default=None,
                   help="Stop after N frames.")
    p.add_argument("--fps",   type=float, default=10.0,
                   help="Frames per second in play mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    alg   = util.load_yaml(SRC / "constants" / "algorithm.yaml")
    calib = util.load_yaml(SRC / "constants" / "calibration.yaml")

    bag_path = get_bag_path(args.bag, bags_dir=SRC / "bags")

    print("Loading motor data from bag...")
    motor_df = load_motor_data(bag_path)
    thrust_times, thrust_values = build_thrust_table(motor_df, calib)

    F_default = np.array(alg["focus"]["point"])
    sigma_F   = float(alg["focus"]["sigma"])

    estimator = algo.Algo(calib, alg)

    window = "Keypoint Viewer"
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    cv.resizeWindow(window, 800, 600)

    playing  = False
    delay_ms = max(1, int(1000.0 / args.fps))

    left_iter  = iter_left_images(bag_path)
    right_iter = iter_right_images(bag_path)

    t_prev_s = None
    for k, ((t_ns, L), (_, R)) in enumerate(zip(left_iter, right_iter)):
        if args.limit is not None and k >= args.limit:
            break

        t_k = t_ns * 1e-9

        if k == 0:
            print("First frame — JAX is compiling, please wait (~30 s)...")
            estimator.init(L, R, t_k, F_default, sigma_F)
        else:
            u_km1 = interp_motor(t_prev_s, thrust_times, thrust_values)
            estimator.iter(L, R, t_k, u_km1, F_default, sigma_F)

        t_prev_s = t_k

        vis = draw_frame(L, estimator.accum, k)
        cv.imshow(window, vis)

        while True:
            wait = delay_ms if playing else 0
            key  = cv.waitKey(wait) & 0xFF

            if key in (ord('q'), 27):
                cv.destroyAllWindows()
                return
            if key == ord('p'):
                playing = not playing
                continue
            if playing or key != 255:
                break

    print(f"Done — {k + 1} frames.")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
