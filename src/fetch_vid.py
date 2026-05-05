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
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


from modules.bag_loader import (
    get_bag_path,
    list_topics,
    load_body_pose,
    load_motor_data,
    iter_left_images,
    iter_right_images,
)


# This makes paths work even if VS Code runs the script from another directory.
REPO_ROOT = Path(__file__).resolve().parent
BAGS_DIR = REPO_ROOT / "bags"
OUTPUT_DIR = REPO_ROOT / "output"
IMAGE_DIR = OUTPUT_DIR / "images"

# Use None to automatically find the bag inside ./bags.
# If you have several bag files, set this explicitly, for example:
# BAG_PATH = BAGS_DIR / "indoor_loadless_hovor_3096.1g_79.04s.bag"
BAG_PATH = None

# Set this to None to export all images.
# Keep it small while testing so you do not write thousands of images by accident.
MAX_IMAGES_PER_CAMERA = 20

def interpolate_to_camera_times(data_times, data_values, cols, camera_times_s):
    funcs = [interp1d(data_times, data_values[:, i], kind='linear',
                      bounds_error=False, fill_value='extrapolate')
             for i in range(len(cols))]
    arr = np.column_stack([f(camera_times_s) for f in funcs])
    return pd.DataFrame(arr, columns=cols)

def preprocessing(motor, pose, bag_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert each rosbag into time-aligned arrays at the camera frame times
    $t_0, t_1, \ldots, t_N$:
    
- Stereo images $(I^L_k, I^R_k)$ — algorithm input
- Per-rotor thrusts $\mathbf{u}_{k-1} = (T_1, T_2, T_3, T_4)$ — algorithm input
- Pose $S_k$ + body-frame velocities — for evaluation
        
    """
    # Get camera timestamps (assuming left and right are synchronized)
    camera_timestamps_ns = sorted(set(t for t, _ in iter_left_images(bag_path)))
    camera_timestamps_s = [t * 1e-9 for t in camera_timestamps_ns]
    
    # Align pose data to camera timestamps
    pose_times = pose['timestamp_s'].values
    pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    pose_data = pose[pose_cols].values
    
    # Create interpolation functions for each pose component
    # interp_funcs_pose = [
    #     interp1d(pose_times, pose_data[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
    #     for i in range(len(pose_cols))
    # ]

    rotations = Rotation.from_quat(pose[['qx','qy','qz','qw']].values)
    slerp = Slerp(pose_times, rotations)
    interp_funcs_pose = slerp(camera_timestamps_s)
    
    # Interpolate pose to camera timestamps
    aligned_pose_data = np.array([[f(t) for f in interp_funcs_pose] for t in camera_timestamps_s])
    aligned_pose = pd.DataFrame(aligned_pose_data, columns=pose_cols)
    aligned_pose['timestamp_ns'] = camera_timestamps_ns
    aligned_pose['timestamp_s'] = camera_timestamps_s
    aligned_pose = aligned_pose[['timestamp_ns', 'timestamp_s'] + pose_cols]
    
    # Align motor data to camera timestamps
    # Pivot motor data to have one row per timestamp with columns for each motor's rpm
    motor_pivot = motor.pivot(index='timestamp_s', columns='motor', values='rpm').reset_index()
    motor_times = motor_pivot['timestamp_s'].values
    motor_cols = ['m1', 'm2', 'm3', 'm4']
    motor_data = motor_pivot[motor_cols].values
    
    # Create interpolation functions for each motor's rpm
    interp_funcs_motor = [
        interp1d(motor_times, motor_data[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        for i in range(len(motor_cols))
    ]
    
    # Interpolate motor data to camera timestamps
    aligned_motor_data = np.array([[f(t) for f in interp_funcs_motor] for t in camera_timestamps_s])

    # After interpolation, shift by one (u_{k-1})
    aligned_motor_data = aligned_motor_data[:-1]   # drop last

    aligned_motor = pd.DataFrame(aligned_motor_data, columns=motor_cols)
    aligned_motor['timestamp_ns'] = camera_timestamps_ns
    aligned_motor['timestamp_s'] = camera_timestamps_s
    
    # Melt back to original motor format for compatibility
    aligned_motor_melt = aligned_motor.melt(
        id_vars=['timestamp_ns', 'timestamp_s'], 
        value_vars=motor_cols, 
        var_name='motor', 
        value_name='rpm'
    )
    aligned_motor_melt['id'] = 0  # Dummy value
    aligned_motor_melt['current'] = 0.0  # Dummy value
    aligned_motor_melt = aligned_motor_melt[['timestamp_ns', 'timestamp_s', 'motor', 'id', 'current', 'rpm']]
    aligned_motor_melt = aligned_motor_melt.sort_values(['timestamp_ns', 'motor']).reset_index(drop=True)
    
    # Update the original DataFrames in place
    pose[:] = aligned_pose
    motor[:] = aligned_motor_melt

    return aligned_pose, aligned_motor_melt

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

    aligned_pose, aligned_motor_melt = preprocessing(motor, pose, bag_path)

    motor_path = OUTPUT_DIR / "motor_data.csv"
    pose_path = OUTPUT_DIR / "body_pose.csv"

    aligned_pose.to_csv(motor_path, index=False)
    aligned_motor_melt.to_csv(pose_path, index=False)

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
