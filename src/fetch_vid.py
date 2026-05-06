"""Example script for exporting drone bag data.

This script:
1. Finds the bag under the repository's ./bags folder
2. Loads motor data, preprocesses it and saves it as CSV
3. Loads body pose data, preprocesses it and saves it as CSV
4. Loads image data and saves it as PNG files

Run:

    python fetch_vid.py

Output is written to:

    output/
        motor_data.csv
        body_pose.csv
        images/
            left/
            right/
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
# from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, Slerp
from rosbags.typesys import get_typestore
import jax.numpy as jnp
import jaxlie
from scipy.signal import savgol_filter
from rosbags.highlevel import AnyReader



from modules.bag_loader import (
    get_bag_path,
    list_topics,
    load_body_pose,
    load_motor_data,
    iter_left_images,
    iter_right_images,
    read_topic,
)


# This makes paths work even if VS Code runs the script from another directory.
REPO_ROOT = Path(__file__).resolve().parent
BAGS_DIR = REPO_ROOT / "bags"
OUTPUT_DIR = REPO_ROOT / "output"
IMAGE_DIR = OUTPUT_DIR / "images"

# Use None to automatically find the bag inside ./bags.
# If you have several bag files, set this explicitly, for example:
BAG_PATH = BAGS_DIR / "indoor_loadless_hovor_3096.1g_79.04s.bag"
# BAG_PATH = "blackbird/blackbird-vio/src/bags/indoor_loadless_hovor_3096.1g_79.04s.bag"

# Set this to None to export all images.
# Keep it small while testing so you do not write thousands of images by accident.
MAX_IMAGES_PER_CAMERA = 20

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert each rosbag into time-aligned arrays at the camera frame times
    t_0, t_1, ..., t_N:

    - Stereo images (I^L_k, I^R_k)                         — algorithm input
    - Per-rotor thrusts u_{k-1} = (T1, T2, T3, T4)         — algorithm input
    - Pose S_k + body-frame velocities                      — for evaluation
    """
    
    with open ("calibration.yaml", "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # ── Camera timeline ───────────────────────────────────────────────────────
    camera_timestamps_ns = sorted(set(t for t, _ in iter_left_images(bag_path)))
    camera_timestamps_s  = np.array(camera_timestamps_ns) * 1e-9

    # ── Pose: position via linear interp, rotation via SLERP ─────────────────
    # 3a+3b: build T_W_B for each vicon sample
    R_MB = jnp.array(data["vicon_params"]["body_to_marker"]["rotation"])      # (3, 3)
    t_MB = jnp.array(data["vicon_params"]["body_to_marker"]["translation"]) 
    R_M_B = jaxlie.SO3.from_matrix(R_MB)
    T_M_B = jaxlie.SE3.from_rotation_and_translation(R_M_B, t_MB)

    T_W_B_list = []
    for _, row in pose.iterrows():
        R = jaxlie.SO3.from_quaternion_xyzw(jnp.array([row.qx, row.qy, row.qz, row.qw]))
        T_W_M = jaxlie.SE3.from_rotation_and_translation(R, jnp.array([row.x, row.y, row.z]))
        T_W_B_list.append(T_W_M @ T_M_B)

    # 3c: re-reference to initial frame
    T_W_B_0 = T_W_B_list[0]
    S_vicon = [T_W_B_k.inverse() @ T_W_B_0 for T_W_B_k in T_W_B_list]
    pose_times = pose['timestamp_s'].values

    # 3d: interpolate on SE(3) at camera timestamps
    S_list = []
    for t_k in camera_timestamps_s:
        idx = np.searchsorted(pose_times, t_k)
        idx = np.clip(idx, 1, len(pose_times) - 1)
        t_l, t_s = pose_times[idx - 1], pose_times[idx]
        S_l, S_s = S_vicon[idx - 1], S_vicon[idx]
        alpha = (t_k - t_l) / (t_s - t_l)
        xi = (S_s @ S_l.inverse()).log()
        S_list.append(jaxlie.SE3.exp(alpha * xi) @ S_l)

    # 3e: differentiate to get body-frame velocities
    dt = np.diff(camera_timestamps_s)
    xi_raw = np.stack([
        (S_list[k] @ S_list[k+1].inverse()).log() / dt[k]
        for k in range(len(S_list) - 1)
    ])
    xi_smooth = savgol_filter(xi_raw, window_length=11, polyorder=3, axis=0)
    v_B     = xi_smooth[:, :3]
    omega_B = xi_smooth[:, 3:]

    # Build aligned_pose from SE(3) results
    translations = np.stack([s.translation() for s in S_list])
    quaternions  = np.stack([s.rotation().as_quaternion_xyzw() for s in S_list])

    aligned_pose = pd.DataFrame(
        np.hstack([translations[:-1], quaternions[:-1], v_B, omega_B]),
        columns=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    )
    aligned_pose.insert(0, 'timestamp_s',  camera_timestamps_s[:-1])
    aligned_pose.insert(0, 'timestamp_ns', camera_timestamps_ns[:-1])

    # ── Motor: interpolate then apply u_{k-1} lag ─────────────────────────────
    # Pivot to wide format: one row per timestamp, one column per motor

    rpm_thr_coeff = [
    data['rotor_1']['rpm_thr_coeff'],  # m1
    data['rotor_4']['rpm_thr_coeff'],  # m4 → rotor 2
    data['rotor_3']['rpm_thr_coeff'],  # m3 → rotor 3
    data['rotor_2']['rpm_thr_coeff'],  # m2 → rotor 4
    ]

    motor_cols  = ['m1', 'm4', 'm3', 'm2']

    motor_pivot = motor.pivot(index='timestamp_s', columns='motor', values='rpm')
    motor_pivot = motor_pivot.ffill().bfill()  # fill gaps per motor column
    motor_pivot = motor_pivot.reset_index()

    valid_rpm_min, valid_rpm_max = 0, 8000   # adjust motor spec

    motor_pivot[motor_cols] = motor_pivot[motor_cols].clip(
        lower=valid_rpm_min, upper=valid_rpm_max
    )

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
    """Save one image array to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"Could not save image: {path}")


def export_csv_data(bag_path: Path) -> None:
    """Export motor and pose data to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    motor = load_motor_data(bag_path).sort_values("timestamp_ns").reset_index(drop=True)
    pose  = load_body_pose(bag_path).sort_values("timestamp_ns").reset_index(drop=True)

    aligned_pose, aligned_motor_melt = preprocessing(motor, pose, bag_path)

    motor_path = OUTPUT_DIR / "motor_data.csv"
    pose_path = OUTPUT_DIR / "body_pose.csv"

    aligned_motor_melt.to_csv(motor_path, index=False)
    aligned_pose.to_csv(pose_path, index=False)

    print(f"Saved {motor_path}")
    print(f"Saved {pose_path}")


def export_images(bag_path: Path) -> None:
    """Export left and right camera images to PNG files."""
    left_dir = IMAGE_DIR / "left"
    right_dir = IMAGE_DIR / "right"

    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    print("Saving left images...")
    for index, (timestamp_ns, image) in enumerate(iter_left_images(bag_path)):
        if MAX_IMAGES_PER_CAMERA is not None and index >= MAX_IMAGES_PER_CAMERA:
            break

        image_path = left_dir / f"left_{index:06d}_{timestamp_ns}.png"
        save_image(image_path, image)

    print("Saving right images...")
    for index, (timestamp_ns, image) in enumerate(iter_right_images(bag_path)):
        if MAX_IMAGES_PER_CAMERA is not None and index >= MAX_IMAGES_PER_CAMERA:
            break

        image_path = right_dir / f"right_{index:06d}_{timestamp_ns}.png"
        save_image(image_path, image)

    print(f"Saved images under {IMAGE_DIR}")


def main() -> None:
    bag_path = get_bag_path(BAG_PATH, bags_dir=BAGS_DIR)

    print(f"Using bag: {bag_path}")
    print("\nTopics:")
    list_topics(bag_path)

    print("\nExporting CSV data...")
    export_csv_data(bag_path)

    print("\nExporting images...")
    export_images(bag_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
