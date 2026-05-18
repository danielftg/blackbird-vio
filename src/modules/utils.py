"""
utils.py - Generic functionality 
"""
from pathlib import Path
from cv2.typing import MatLike
import cv2 as cv
import jax.numpy as jnp
import numpy as np
import yaml




def load_yaml(path: Path) -> dict:
    """Load a YAML file with explicit UTF-8 encoding (Windows-safe)."""
    with open(path, "r" , encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image(path: Path) -> MatLike:
    """Read an image as grayscale uint8."""
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def skew(v: jnp.ndarray) -> jnp.ndarray:
    """[v]_x : 3-vector → 3x3 skew-symmetric matrix."""
    return jnp.array([
        [   0.0, -v[2],  v[1]],
        [ v[2],    0.0, -v[0]],
        [-v[1],  v[0],    0.0],
    ])


def np_skew(v: np.ndarray) -> np.ndarray:
    """3-vector to 3x3 skew."""
    return np.array([[ 0.0,  -v[2],  v[1]],
                     [ v[2],   0.0, -v[0]],
                     [-v[1],  v[0],  0.0]])