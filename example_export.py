"""Example script for exporting drone bag data.

This script:
1. Finds the bag under the repository's ./bags folder
2. Loads motor data and saves it as CSV
3. Loads body pose data and saves it as CSV
4. Loads image data and saves it as PNG files

Run:

    python example_export.py

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

from bag_loader import (
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

    motor_path = OUTPUT_DIR / "motor_data.csv"
    pose_path = OUTPUT_DIR / "body_pose.csv"

    motor.to_csv(motor_path, index=False)
    pose.to_csv(pose_path, index=False)

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
