"""
eval.py — Performance Evaluation.

Evaluates the performance of the algorithm against ground truth.
Visualizes and presents data for inspection and reporting.
"""

from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd


log = logging.getLogger(__name__)


def evaluate(results_path: Path,
             ground_truth_path: Path) -> dict:
    """Compare estimator output against Vicon ground truth.

    Reads the per-frame results written by main.py and the time-aligned
    ground-truth pose/velocity table from preprocessing. Aligns the two
    by camera timestamp, computes error metrics, and emits plots.

    Args:
        results_path      : CSV from main.py with per-frame
                            estimator output (timestamp_ns, pose, v_B,
                            ω_B, plus optional diagnostics).
        ground_truth_path : body_pose.csv from fetch_vid.py with
                            time-aligned ground truth.


    Notes:
        - Suggestions: translation magnitude RMSE, rotation
           magnitude RMSE (In tangent space), body-frame velocity RMSE per axis, body-frame
           angular velocity RMSE per axis. Per-frame errors and plots
           written to disk alongside results_path.

        - The vicon-marker → body offset T_{M,B} is unknown; per-axis
          comparisons in body frame are therefore biased by a constant if our guess was wrong.
          Compare rotation-invariant magnitudes (‖t‖, rotation
          angle, ‖v‖) for honest accuracy numbers; per-axis components
          are useful for trend-spotting only.
        - Filter output is in T_{B_k, B_0} convention; ground truth uses
          the same convention (preprocessing writes S_k that way).
          No conversion needed before comparison.
    """
    ...