"""
fetch_vid.py — Preprocess a VID-Dataset rosbag into algorithm-ready inputs.

When called from main.py (the normal path), config is passed in via run().
Can also be run standalone for quick preprocessing:

    python fetch_vid.py [--bag <filename>]

Output is written to:

    <data_dir>/
        motor_data.csv
        body_pose.csv
        images/
            left/           # left_<idx>_<timestamp_ns>.png
            right/          # right_<idx>_<timestamp_ns>.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import make_interp_spline
from rosbags.typesys import get_typestore
import jax.numpy as jnp
import jaxlie
from scipy.signal import savgol_filter

from modules.bag_loader import (
    get_bag_path,
    list_topics,
    load_body_pose,
    load_motor_data,
    iter_left_images,
    iter_right_images,
)

REPO_ROOT = Path(__file__).resolve().parent
BAGS_DIR = REPO_ROOT / "bags"


def interpolate_to_camera_times(
    data_times: np.ndarray,
    data_values: np.ndarray,
    cols: list[str],
    camera_times_s: np.ndarray,
) -> pd.DataFrame:
    arr = np.column_stack([
        np.interp(camera_times_s, data_times, data_values[:, i])
        for i in range(len(cols))
    ])
    return pd.DataFrame(arr, columns=cols)


def preprocessing(
    motor: pd.DataFrame,
    pose: pd.DataFrame,
    bag_path: str,
    calib: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert each rosbag into time-aligned arrays at the camera frame times
    t_0, t_1, ..., t_N:

    - Stereo images (I^L_k, I^R_k)                         — algorithm input
    - Per-rotor thrusts u_{k-1} = (T1, T2, T3, T4)         — algorithm input
    - Pose S_k + body-frame velocities                      — for evaluation
    """

    # ── Camera timeline ───────────────────────────────────────────────────────
    camera_timestamps_ns = sorted(set(t for t, _ in iter_left_images(bag_path)))
    camera_timestamps_s  = np.array(camera_timestamps_ns) * 1e-9

    # ── Pose: position via linear interp, rotation via SLERP ─────────────────
    R_MB = jnp.array(calib["vicon_params"]["body_to_marker"]["rotation"])
    t_MB = jnp.array(calib["vicon_params"]["body_to_marker"]["translation"])
    R_M_B = jaxlie.SO3.from_matrix(R_MB)
    T_M_B = jaxlie.SE3.from_rotation_and_translation(R_M_B, t_MB)

    T_W_B_list = []
    for _, row in pose.iterrows():
        R = jaxlie.SO3.from_quaternion_xyzw(jnp.array([row.qx, row.qy, row.qz, row.qw]))
        T_W_M = jaxlie.SE3.from_rotation_and_translation(R, jnp.array([row.x, row.y, row.z]))
        T_W_B_list.append(T_W_M @ T_M_B)

    T_W_B_0 = T_W_B_list[0]
    S_vicon = [T_W_B_k.inverse() @ T_W_B_0 for T_W_B_k in T_W_B_list]
    pose_times = pose['timestamp_s'].values

    S_list = []
    for t_k in camera_timestamps_s:
        idx = np.searchsorted(pose_times, t_k)
        idx = np.clip(idx, 1, len(pose_times) - 1)
        t_l, t_s = pose_times[idx - 1], pose_times[idx]
        S_l, S_s = S_vicon[idx - 1], S_vicon[idx]
        alpha = (t_k - t_l) / (t_s - t_l)
        xi = (S_s @ S_l.inverse()).log()
        S_list.append(jaxlie.SE3.exp(alpha * xi) @ S_l)

    dt = np.diff(camera_timestamps_s)
    xi_raw = np.stack([
        (S_list[k] @ S_list[k+1].inverse()).log() / dt[k]
        for k in range(len(S_list) - 1)
    ])
    xi_smooth = savgol_filter(xi_raw, window_length=11, polyorder=3, axis=0)
    v_B     = xi_smooth[:, :3]
    omega_B = xi_smooth[:, 3:]

    translations = np.stack([s.translation() for s in S_list])
    quaternions  = np.stack([s.rotation().as_quaternion_xyzw() for s in S_list])

    aligned_pose = pd.DataFrame(
        np.hstack([translations[:-1], quaternions[:-1], v_B, omega_B]),
        columns=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    )
    aligned_pose.insert(0, 'timestamp_s',  camera_timestamps_s[:-1])
    aligned_pose.insert(0, 'timestamp_ns', camera_timestamps_ns[:-1])

    # ── Motor: interpolate then apply u_{k-1} lag ─────────────────────────────
    drn_params = calib["drone_parameters"]
    rpm_thr_coeff = [
        drn_params['rotor_1']['rpm_thr_coeff'],
        drn_params['rotor_4']['rpm_thr_coeff'],
        drn_params['rotor_3']['rpm_thr_coeff'],
        drn_params['rotor_2']['rpm_thr_coeff'],
    ]

    motor_cols = ['m1', 'm4', 'm3', 'm2']

    motor_pivot = motor.pivot(index='timestamp_s', columns='motor', values='rpm')
    motor_pivot = motor_pivot.ffill().bfill()
    motor_pivot = motor_pivot.reset_index()

    motor_pivot[motor_cols] = motor_pivot[motor_cols].clip(lower=0, upper=8000)

    aligned_motor = interpolate_to_camera_times(
        motor_pivot['timestamp_s'].values,
        np.array(rpm_thr_coeff) * np.array(motor_pivot[motor_cols].values)**2,
        motor_cols,
        camera_timestamps_s,
    )
    aligned_motor.insert(0, 'timestamp_s',  camera_timestamps_s)
    aligned_motor.insert(0, 'timestamp_ns', camera_timestamps_ns)

    return aligned_pose, aligned_motor


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise RuntimeError(f"Could not save image: {path}")


def export_csv_data(bag_path: Path, output_dir: Path, calib: dict) -> None:
    """Export motor and pose data to CSV under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    motor = load_motor_data(bag_path).sort_values("timestamp_ns").reset_index(drop=True)
    pose  = load_body_pose(bag_path).sort_values("timestamp_ns").reset_index(drop=True)

    aligned_pose, aligned_motor = preprocessing(motor, pose, bag_path, calib)

    aligned_motor.to_csv(output_dir / "motor_data.csv", index=False)
    aligned_pose.to_csv(output_dir / "body_pose.csv",   index=False)

    print(f"Saved {output_dir / 'motor_data.csv'}")
    print(f"Saved {output_dir / 'body_pose.csv'}")


def export_images(
    bag_path: Path,
    image_dir: Path,
    max_images: int | None = None,
) -> None:
    """Export left and right camera images to PNG files under image_dir."""
    left_dir  = image_dir / "left"
    right_dir = image_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    print("Saving left images...")
    for index, (timestamp_ns, image) in enumerate(iter_left_images(bag_path)):
        if max_images is not None and index >= max_images:
            break
        save_image(left_dir / f"left_{index:06d}_{timestamp_ns}.png", image)

    print("Saving right images...")
    for index, (timestamp_ns, image) in enumerate(iter_right_images(bag_path)):
        if max_images is not None and index >= max_images:
            break
        save_image(right_dir / f"right_{index:06d}_{timestamp_ns}.png", image)

    print(f"Saved images under {image_dir}")


def run(cfg, bag_path: Path, data_dir: Path, calib: dict) -> None:
    """Called by main.py when cfg.fetch_vid is true."""
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPreprocessing bag: {bag_path.name}")
    print(f"Output dir:        {data_dir}\n")
    export_csv_data(bag_path, data_dir, calib)
    print("\nExporting images...")
    export_images(bag_path, data_dir / "images", max_images=cfg.max_images)
    print("Preprocessing done.\n")


def main() -> None:
    """Standalone entry point for direct invocation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag", type=str, default=None,
                   help="Bag filename (relative to src/bags/). Auto-detected if only one bag exists.")
    args = p.parse_args()

    bag_path = get_bag_path(
        BAGS_DIR / args.bag if args.bag else None,
        bags_dir=BAGS_DIR,
    )
    print(f"Using bag: {bag_path}")
    print("\nTopics:")
    list_topics(bag_path)

    # Load calibration directly when running standalone
    calib_path = REPO_ROOT / "conf" / "calibration" / "default.yaml"
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    data_dir = REPO_ROOT / "output" / bag_path.stem.split(".")[0]
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\nExporting CSV data...")
    export_csv_data(bag_path, data_dir, calib)

    print("\nExporting images...")
    export_images(bag_path, data_dir / "images")

    print("\nDone.")


if __name__ == "__main__":
    main()
