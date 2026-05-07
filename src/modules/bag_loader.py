"""
bag_loader.py — Functions for reading the drone ROS bag stored in the ./bags folder.

This file is intended to be placed in the root of your repository:

    your_repo/
        bags/
            your_bag_file.bag
        bag_loader.py
        fetch_vid.py

Notes:
- The image topics are loaded lazily with generator functions so you do not
  accidentally load every image into RAM.
- Pose and motor data are returned as pandas DataFrames.
"""

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


# ---------------------------------------------------------------------------
# Dataset topic names
# ---------------------------------------------------------------------------

LEFT_IMAGE_TOPIC = "/camera/infra1/image_rect_raw"
RIGHT_IMAGE_TOPIC = "/camera/infra2/image_rect_raw"
BODY_POSE_TOPIC = "/vicon/m100/m100"

MOTOR_TOPICS = {
    "m1": "/m100withm3508/m3508_m1",
    "m2": "/m100withm3508/m3508_m2",
    "m3": "/m100withm3508/m3508_m3",
    "m4": "/m100withm3508/m3508_m4",
}

TYPESTORE = get_typestore(Stores.ROS2_FOXY)


# ---------------------------------------------------------------------------
# Bag path helpers
# ---------------------------------------------------------------------------

def find_bags(bags_dir: str | Path = "bags") -> list[Path]:
    """Return bag files/folders found inside the bags directory.

    Supports:
    - ROS 1 bag files: *.bag
    - ROS 2 bag folders: folders containing metadata.yaml
    """
    bags_dir = Path(bags_dir)

    if not bags_dir.exists():
        raise FileNotFoundError(f"Could not find bags directory: {bags_dir}")

    bag_paths: list[Path] = []

    # ROS 1 .bag files.
    bag_paths.extend(sorted(bags_dir.glob("*.bag")))

    # ROS 2 bag folders.
    for path in sorted(bags_dir.iterdir()):
        if path.is_dir() and (path / "metadata.yaml").exists():
            bag_paths.append(path)

    return bag_paths


def get_bag_path(bag_path: str | Path | None = None, bags_dir: str | Path = "bags") -> Path:
    """Return the bag path to use.

    If bag_path is given, that path is used.
    If bag_path is None, this function looks in ./bags.

    If there is exactly one bag in ./bags, it is used automatically.
    If there are multiple bags, pass the one you want explicitly.
    """
    if bag_path is not None:
        path = Path(bag_path)
        if not path.exists():
            raise FileNotFoundError(f"Bag path does not exist: {path}")
        return path

    bags = find_bags(bags_dir)

    if not bags:
        raise FileNotFoundError(f"No .bag files or ROS 2 bag folders found in: {bags_dir}")

    if len(bags) > 1:
        found = "\n".join(f"  {path}" for path in bags)
        raise ValueError(
            "More than one bag found. Pass the bag path explicitly.\n"
            f"Found:\n{found}"
        )

    return bags[0]


# ---------------------------------------------------------------------------
# Basic reading functions
# ---------------------------------------------------------------------------

def list_topics(bag_path: str | Path | None = None) -> None:
    """Print all topics and message types in the bag."""
    bag_path = get_bag_path(bag_path)

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        for connection in reader.connections:
            print(connection.topic, "--", connection.msgtype)


def read_topic(topic: str, bag_path: str | Path | None = None) -> list[tuple[int, object]]:
    """Read one topic and return a list of (timestamp_ns, message)."""
    bag_path = get_bag_path(bag_path)
    data = []

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic == topic]

        if not connections:
            available = "\n".join(sorted(c.topic for c in reader.connections))
            raise ValueError(f"Topic not found: {topic}\n\nAvailable topics:\n{available}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            data.append((timestamp, msg))

    return data


def iter_topic(topic: str, bag_path: str | Path | None = None) -> Generator[tuple[int, object], None, None]:
    """Iterate through one topic lazily as (timestamp_ns, message)."""
    bag_path = get_bag_path(bag_path)

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic == topic]

        if not connections:
            available = "\n".join(sorted(c.topic for c in reader.connections))
            raise ValueError(f"Topic not found: {topic}\n\nAvailable topics:\n{available}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            yield timestamp, msg


# ---------------------------------------------------------------------------
# Pose and motor loading
# ---------------------------------------------------------------------------

def load_body_pose(bag_path: str | Path | None = None) -> pd.DataFrame:
    """Load body pose from /vicon/m100/m100.

    Returns columns:
        timestamp_ns, timestamp_s, x, y, z, qx, qy, qz, qw
    """
    rows = []

    for timestamp, msg in read_topic(BODY_POSE_TOPIC, bag_path):
        pose = msg.pose.pose
        position = pose.position
        orientation = pose.orientation

        rows.append({
            "timestamp_ns": timestamp,
            "timestamp_s": timestamp * 1e-9,
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "qx": orientation.x,
            "qy": orientation.y,
            "qz": orientation.z,
            "qw": orientation.w,
        })

    return pd.DataFrame(rows)


def load_motor_data(bag_path: str | Path | None = None) -> pd.DataFrame:
    """Load motor data from m1, m2, m3 and m4 topics.

    Returns columns:
        timestamp_ns, timestamp_s, motor, id, current, rpm
    """
    rows = []

    for motor_name, topic in MOTOR_TOPICS.items():
        for timestamp, msg in read_topic(topic, bag_path):
            rows.append({
                "timestamp_ns": timestamp,
                "timestamp_s": timestamp * 1e-9,
                "motor": motor_name,
                "id": msg.id,
                "current": msg.current,
                "rpm": msg.rpm,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # This makes m1, m2, m3, m4 appear together by time.
    return df.sort_values(["timestamp_ns", "motor"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def image_to_numpy(msg) -> np.ndarray:
    """Convert a sensor_msgs/msg/Image message to a numpy array.

    Supports common encodings:
    - mono8 / 8UC1
    - mono16 / 16UC1
    - rgb8
    - bgr8
    """
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()

    if encoding in ["rgb8", "bgr8"]:
        channels = 3
        dtype = np.uint8
    elif encoding in ["mono8", "8uc1"]:
        channels = 1
        dtype = np.uint8
    elif encoding in ["mono16", "16uc1"]:
        channels = 1
        dtype = np.uint16
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    image = np.frombuffer(np.asarray(msg.data, dtype=np.uint8).tobytes(), dtype=dtype)

    if channels == 1:
        expected = height * width
        return image[:expected].reshape(height, width)

    expected = height * width * channels
    return image[:expected].reshape(height, width, channels)


def iter_left_images(bag_path: str | Path | None = None) -> Generator[tuple[int, np.ndarray], None, None]:
    """Iterate through left camera images as (timestamp_ns, image_array)."""
    for timestamp, msg in iter_topic(LEFT_IMAGE_TOPIC, bag_path):
        yield timestamp, image_to_numpy(msg)


def iter_right_images(bag_path: str | Path | None = None) -> Generator[tuple[int, np.ndarray], None, None]:
    """Iterate through right camera images as (timestamp_ns, image_array)."""
    for timestamp, msg in iter_topic(RIGHT_IMAGE_TOPIC, bag_path):
        yield timestamp, image_to_numpy(msg)


def load_left_images(bag_path: str | Path | None = None) -> list[tuple[int, np.ndarray]]:
    """Load all left images into memory.

    For large bags, prefer iter_left_images().
    """
    return list(iter_left_images(bag_path))


def load_right_images(bag_path: str | Path | None = None) -> list[tuple[int, np.ndarray]]:
    """Load all right images into memory.

    For large bags, prefer iter_right_images().
    """
    return list(iter_right_images(bag_path))
