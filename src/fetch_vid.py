"""Example script for exporting drone bag data.

This script:
1. Finds the bag under the repository's ./bags folder
2. Loads motor data and saves it as CSV
3. Loads body pose data and saves it as CSV
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
# from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, Slerp
from rosbags.typesys import get_typestore


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
    # ── Camera timeline ───────────────────────────────────────────────────────
    camera_timestamps_ns = sorted(set(t for t, _ in iter_left_images(bag_path)))
    camera_timestamps_s  = np.array(camera_timestamps_ns) * 1e-9

    # ── Pose: position via linear interp, rotation via SLERP ─────────────────
    pose_times = pose['timestamp_s'].values

    # x, y, z — standard linear interpolation
    aligned_pos = interpolate_to_camera_times(
        pose_times,
        pose[['x', 'y', 'z']].values,
        ['x', 'y', 'z'],
        camera_timestamps_s,
    )

    # qx, qy, qz, qw — SLERP to preserve unit-quaternion constraint
    slerp = Slerp(pose_times, Rotation.from_quat(pose[['qx', 'qy', 'qz', 'qw']].values))
    interp_quats = slerp(camera_timestamps_s).as_quat()   # (N, 4) array: x y z w
    aligned_rot = pd.DataFrame(interp_quats, columns=['qx', 'qy', 'qz', 'qw'])

    aligned_pose = pd.concat([aligned_pos, aligned_rot], axis=1)
    aligned_pose.insert(0, 'timestamp_s',  camera_timestamps_s)
    aligned_pose.insert(0, 'timestamp_ns', camera_timestamps_ns)

    # ── Motor: interpolate then apply u_{k-1} lag ─────────────────────────────
    # Pivot to wide format: one row per timestamp, one column per motor

    motor_cols  = ['m1', 'm2', 'm3', 'm4']

    motor_pivot = motor.pivot(index='timestamp_s', columns='motor', values='rpm')
    motor_pivot = motor_pivot.ffill().bfill()  # fill gaps per motor column
    motor_pivot = motor_pivot.reset_index()

    aligned_motor = interpolate_to_camera_times(
        motor_pivot['timestamp_s'].values,
        motor_pivot[motor_cols].values,
        motor_cols,
        camera_timestamps_s,
    )
    aligned_motor.insert(0, 'timestamp_s',  camera_timestamps_s)
    aligned_motor.insert(0, 'timestamp_ns', camera_timestamps_ns)

    # u_{k-1}: motor command at step k-1 pairs with camera frame at step k.
    # Drop the first camera frame (no prior motor command) and the last motor row.
    aligned_motor = aligned_motor.iloc[:-1].reset_index(drop=True)   # u_0 … u_{N-1}
    aligned_pose  = aligned_pose.iloc[1:].reset_index(drop=True)     # S_1 … S_N

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

    motor = load_motor_data(bag_path)
    pose = load_body_pose(bag_path)

    # ── DEBUG ────────────────────────────────────────────────────────
    for timestamp, msg in read_topic("/m100withm3508/m3508_m1", bag_path):
        if msg.rpm != 0:
            print(f"First non-zero RPM at t={timestamp * 1e-9:.3f}: rpm={msg.rpm}")
            break
    else:
        print("All RPM values are zero!")
    return

    # # ── DEBUG ────────────────────────────────────────────────────────
    # print(motor.head(20))
    # print("motor time range:", motor['timestamp_s'].min(), "→", motor['timestamp_s'].max())

    # camera_timestamps_ns = sorted(set(t for t, _ in iter_left_images(bag_path)))
    # camera_timestamps_s = np.array(camera_timestamps_ns) * 1e-9
    # print("camera time range:", camera_timestamps_s.min(), "→", camera_timestamps_s.max())
    # return  # stop here for now
    # # ─────────────────────────────────────────────────────────────────

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
