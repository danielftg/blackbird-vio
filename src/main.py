"""
main.py — Entry point. No logic; only orchestration.

Creates/reads preprocessed data from output/<bag_name>/, runs the estimator
one frame at a time, writes per-frame estimates to a results file in the
Hydra-managed run directory. Evaluation is a separate post-hoc step (eval.py).

Usage (run from src/):
    python main.py                                  # default bag, no eval
    python main.py bag=indoor_loadless_8            # select bag
    python main.py fetch_vid=true evaluate=true     # full pipeline
    python main.py offset=200 limit=50              # frames 200-249 only
    python main.py profile=true limit=20            # profile the loop
    python main.py algorithm.cv.N_F_max=30          # override any config value
    python main.py --multirun bag=indoor_loadless_hover,indoor_loadless_8
"""

import os

N = "4"
os.environ["OMP_NUM_THREADS"]        = N
os.environ["OPENBLAS_NUM_THREADS"]   = N
os.environ["MKL_NUM_THREADS"]        = N
os.environ["NUMEXPR_NUM_THREADS"]    = N
os.environ["VECLIB_MAXIMUM_THREADS"] = N
os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={N}"

import cProfile
import pstats
import time
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import modules.algo as algo
import modules.utils as util
import eval
import fetch_vid


log = logging.getLogger("main")


def list_image_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Enumerate (left, right) image pairs in chronological order."""
    left_dir  = data_dir / "images" / "left"
    right_dir = data_dir / "images" / "right"
    lefts  = sorted(left_dir.glob("*.png"))
    rights = sorted(right_dir.glob("*.png"))
    if len(lefts) != len(rights):
        raise RuntimeError(
            f"image count mismatch: {len(lefts)} L vs {len(rights)} R")
    return list(zip(lefts, rights))


def serialize_output(
    k: int,
    t_k_ns: int,
    frame_time_s: float,
    output: algo.IterOutput,
) -> dict:
    """Flatten one IterOutput into a flat dict for DataFrame storage."""
    X = output.X_core
    t = X.T.translation()
    q = X.T.rotation().as_quaternion_xyzw()

    t = np.asarray(t).flatten()
    q = np.asarray(q).flatten()
    v = np.asarray(X.v).flatten()
    w = np.asarray(X.omega).flatten()
    g = np.asarray(X.g_B).flatten()
    d = np.asarray(X.d_B).flatten()

    roles = [cp.role for cp in output.point_cloud.values()]
    n_F     = roles.count("F")
    n_F_pre = roles.count("F_pre")
    n_I     = roles.count("I")

    return {
        "k":            k,
        "timestamp_ns": t_k_ns,
        "timestamp_s":  t_k_ns * 1e-9,
        "frame_time_s": frame_time_s,
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


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    src_root   = Path(__file__).resolve().parent
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    data_dir   = output_dir.parent          # output/<bag_name>/
    bag_path   = src_root / "bags" / cfg.bag.filename
    results_path = output_dir / "results.csv"

    log.info("bag:        %s", cfg.bag.name)
    log.info("data_dir:   %s", data_dir)
    log.info("output_dir: %s", output_dir)
    log.info("bag_path:   %s", bag_path)

    # ---- convert Hydra DictConfig to plain dicts for algorithm modules ------
    alg   = OmegaConf.to_container(cfg.algorithm,   resolve=True)
    calib = OmegaConf.to_container(cfg.calibration, resolve=True)

    # ---- optional preprocessing ---------------------------------------------
    if cfg.fetch_vid:
        fetch_vid.run(cfg, bag_path, data_dir, calib)

    # ---- load preprocessed data ---------------------------------------------
    motor = pd.read_csv(data_dir / "motor_data.csv")
    pose  = pd.read_csv(data_dir / "body_pose.csv")
    pairs = list_image_pairs(data_dir)

    # ---- frame slice (offset / limit) ---------------------------------------
    start = int(cfg.offset) if cfg.offset is not None else 0
    if cfg.limit is not None:
        end = start + int(cfg.limit)
    else:
        end = len(pairs)
    pairs = pairs[start:end]

    motor_cols = ["m1", "m4", "m3", "m2"]
    log.info("frames %d–%d (%d total), %d motor rows, %d pose rows",
             start, start + len(pairs) - 1, len(pairs), len(motor), len(pose))

    # ---- focus stub ---------------------------------------------------------
    F_default = np.array(alg["focus"]["point"])
    sigma_F   = float(alg["focus"]["sigma"])

    # ---- estimator ----------------------------------------------------------
    estimator = algo.Algo(calib, alg)

    # ---- prepare results file -----------------------------------------------
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()

    # ---- optional profiler --------------------------------------------------
    profiler = cProfile.Profile() if cfg.profile else None
    if profiler:
        profiler.enable()

    # ---- main loop ----------------------------------------------------------
    for k, (path_L, path_R) in enumerate(pairs):
        frame_idx = start + k
        log.info("frame %d / %d  (global index %d)", k, len(pairs), frame_idx)
        t0 = time.perf_counter()

        L = util.load_image(path_L)
        R = util.load_image(path_R)

        t_k_ns = int(path_L.stem.split("_")[-1])
        t_k    = t_k_ns * 1e-9

        if k == 0:
            output = estimator.init(L, R, t_k, F_default, sigma_F)
        else:
            u_km1  = motor.iloc[frame_idx - 1][motor_cols].to_numpy(dtype=np.float64)
            output = estimator.iter(L, R, t_k, u_km1, F_default, sigma_F)

        frame_time_s = time.perf_counter() - t0

        row = serialize_output(k, t_k_ns, frame_time_s, output)
        pd.DataFrame([row]).to_csv(
            results_path,
            mode="a",
            header=(k == 0),
            index=False,
        )

    # ---- save profiling results ---------------------------------------------
    if profiler:
        profiler.disable()
        prof_path = output_dir / "profile.prof"
        profiler.dump_stats(prof_path)
        with open(output_dir / "profile.txt", "w") as f:
            pstats.Stats(profiler, stream=f).sort_stats("cumulative").print_stats(40)
        log.info("profile saved to %s", prof_path)

    # ---- optional evaluation ------------------------------------------------
    if cfg.evaluate:
        eval.evaluate(
            results_path=results_path,
            ground_truth_path=data_dir / "body_pose.csv",
        )


if __name__ == "__main__":
    main()
