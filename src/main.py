"""
main.py — Entry point. No logic; only orchestration.

Creates/reads preprocessed data from output/, runs the estimator one frame at
a time, writes per-frame estimates to a results file. Evaluation is a
separate post-hoc step (eval.py) that consumes the same file.

Usage example:
    python main.py [--data DIR] [--results PATH] [--limit N]
"""

import os

N = "4"
os.environ["OMP_NUM_THREADS"]      = N
os.environ["OPENBLAS_NUM_THREADS"] = N
os.environ["MKL_NUM_THREADS"]      = N
os.environ["NUMEXPR_NUM_THREADS"]  = N
os.environ["VECLIB_MAXIMUM_THREADS"] = N
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np

import modules.utils as util
import modules.algo as algo
import eval
import fetch_vid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",        type=Path, default=Path("output"))
    p.add_argument("--results",     type=Path, default=Path("output/results.csv"))
    p.add_argument("--limit",       type=int,  default=None,
                   help="Process only the first N frames (debugging)")
    p.add_argument("--fetch_vid",    action="store_true", default=False,
                   help="Run fetch_vid before estimation starts")
    p.add_argument("--evaluate",    action="store_true", default=False, 
                   help="Run eval after estimation completes")
    p.add_argument("--log-level",   default="INFO")
    return p.parse_args()


def list_image_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Enumerate (left, right) image pairs in chronological order.

    Filenames are formatted as <prefix>_<frame_idx>_<timestamp_ns>.png;
    sorting lexicographically gives chronological order.
    """
    left_dir  = data_dir / "images" / "left"
    right_dir = data_dir / "images" / "right"
    lefts  = sorted(left_dir.glob("*.png"))
    rights = sorted(right_dir.glob("*.png"))
    if len(lefts) != len(rights):
        raise RuntimeError(
            f"image count mismatch: {len(lefts)} L vs {len(rights)} R")
    return list(zip(lefts, rights))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("main")

    # ---- optional fetch dataset ----------------------------------------
    if args.fetch_vid:
        fetch_vid.main()

    # ---- load config ---------------------------------------------------
    alg = util.load_yaml("constants/algorithm.yaml")
    calib = util.load_yaml("constants/calibration.yaml")
    
    # ---- load preprocessed data -----------------------------------------
    motor = pd.read_csv(args.data / "motor_data.csv")
    pose  = pd.read_csv(args.data / "body_pose.csv")    # GT, used by eval only
    pairs = list_image_pairs(args.data)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    motor_cols = ["m1", "m4", "m3", "m2"]   # in our rotor order: 1, 2, 3, 4
    log.info("loaded %d frames, %d motor rows, %d pose rows",
             len(pairs), len(motor), len(pose))

    # ---- focus stub -----------------------------------------------------
    # Focus mechanism not yet driven by any external signal
    # Replace when we have a focus source.
    F_default     = np.array(alg["focus"]["point"]) 
    sigma_F       = float(alg["focus"]["sigma"])

    # ---- estimator ------------------------------------------------------
    estimator = algo.Algo(calib, alg)
   
    # ---- prepare results file ------------------------------------------------
    args.results.parent.mkdir(parents=True, exist_ok=True)
    if args.results.exists():
        args.results.unlink()                  # fresh start each run

    for k, (path_L, path_R) in enumerate(pairs):
        log.info("frame %d / %d", k, len(pairs))
        L = util.load_image(path_L)
        R = util.load_image(path_R)

        # timestamp from filename: <prefix>_<idx>_<timestamp_ns>.png
        t_k_ns = int(path_L.stem.split("_")[-1])
        t_k    = t_k_ns * 1e-9

        if k == 0:
            output = estimator.init(L, R, t_k, F_default, sigma_F)
        else:
            # control input is u_{k-1}: row k-1 of motor_data.csv
            u_km1 = motor.iloc[k - 1][motor_cols].to_numpy(dtype=np.float64)
            output = estimator.iter(L, R, t_k, u_km1, F_default, sigma_F)

        
        
        # ---- write results --------------------------------------------------
        row = serialize_output(k, t_k_ns, output)
        pd.DataFrame([row]).to_csv(
                args.results,
                mode="a",
                header=(k == 0),                   # header only on first row
                index=False,
        )
        
    # ---- optional evaluation --------------------------------------------
    if args.evaluate:
        eval.evaluate(
            results_path=args.results,
            ground_truth_path=args.data / "body_pose.csv",
        )


def serialize_output(k: int, t_k_ns: int, output: algo.IterOutput) -> dict:
    """Flatten one IterOutput into a flat dict for DataFrame storage.

    Per-frame columns:
        k, timestamp_ns, timestamp_s
        x, y, z, qx, qy, qz, qw                 — pose T_{B_k, B_0}
        vx, vy, vz                              — body-frame linear velocity
        wx, wy, wz                              — body-frame angular velocity
        gx, gy, gz                              — body-frame gravity
        dx, dy, dz                              — body-frame disturbance
        n_F, n_F_pre, n_I                       — point-set sizes

    Point cloud not serialised here — too large for one CSV row. Persist
    separately if needed (per-frame parquet keyed by k).
    """
    X = output.X_core
    t = X.T.translation()                  # (3,) jnp / np array
    q = X.T.rotation().as_quaternion_xyzw()  # (4,) qx qy qz qw

    # Coerce to plain Python floats so pandas writes scalars, not array reprs
    t = np.asarray(t).flatten()
    q = np.asarray(q).flatten()
    v = np.asarray(X.v).flatten()
    w = np.asarray(X.omega).flatten()
    g = np.asarray(X.g_B).flatten()
    d = np.asarray(X.d_B).flatten()

    # Count by role from the cloud
    roles = [cp.role for cp in output.point_cloud.values()]
    n_F     = roles.count("F")
    n_F_pre = roles.count("F_pre")
    n_I     = roles.count("I")

    return {
        "k":            k,
        "timestamp_ns": t_k_ns,
        "timestamp_s":  t_k_ns * 1e-9,
        "x":  float(t[0]), "y":  float(t[1]), "z":  float(t[2]),
        "qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]), "qw": float(q[3]),
        "vx": float(v[0]), "vy": float(v[1]), "vz": float(v[2]),
        "wx": float(w[0]), "wy": float(w[1]), "wz": float(w[2]),
        "gx": float(g[0]), "gy": float(g[1]), "gz": float(g[2]),
        "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]),
        "n_F":     n_F,
        "n_F_pre": n_F_pre,
        "n_I":     n_I,
    }

if __name__ == "__main__":
    main()