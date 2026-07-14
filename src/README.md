# src/ — running the estimator

What's here today: a stereo-camera, vision-only state estimator (no IMU yet),
plus the tooling around it — preprocessing flight recordings (`fetch_vid.py`)
and evaluating output against ground truth (`eval.py`). This will keep growing;
this file describes it as of now.

All commands below are run from this directory (`cd src`).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
```

Download one or more flight recordings ("bags") from the
[ZJU FAST-Lab VID-Dataset](https://github.com/ZJU-FAST-Lab/VID-Dataset)
and place them under `bags/`.

## Quickstart

```bash
python main.py limit=5            # run 5 frames on the default bag (indoor_loadless_hover)
```

Output lands in `output/<bag_name>/<timestamp>/`. Each run directory contains:
- `results.csv` — per-frame pose, velocity, point counts, and wall-clock timing
- `.hydra/` — full config snapshot (`config.yaml`) and CLI overrides (`overrides.yaml`)

## Select a bag

```bash
python main.py bag=indoor_loadless_8
python main.py bag=indoor_loadless_round
```

Bag names map to `conf/bag/<name>.yaml`. Add a new YAML there to register a new bag.

## Preprocess a dataset

Before the first run on a bag, extract images and aligned CSVs from the rosbag:

```bash
python main.py bag=indoor_loadless_8 fetch_vid=true limit=5
```

`fetch_vid=true` writes `motor_data.csv`, `body_pose.csv`, and `images/` directly into
`output/<bag_name>/` (shared across all runs on that bag).

## Full pipeline (with evaluation against ground truth)

```bash
python main.py bag=indoor_loadless_hover fetch_vid=true evaluate=true
```

## Frame slicing

Use `offset` and `limit` to run a specific window of frames — useful for skipping
takeoff/landing transients or isolating a segment.

```bash
python main.py offset=200 limit=100   # frames 200-299 only
python main.py limit=50               # first 50 frames
```

## Profiling

```bash
python main.py profile=true limit=20
```

Saves `profile.prof` and `profile.txt` (top-40 cumulative time) in the run directory.
Open the `.prof` file with `snakeviz profile.prof` for an interactive flamegraph.

## Override any config value inline

Any value in `conf/` can be overridden on the command line:

```bash
python main.py algorithm.cv.N_F_max=30
python main.py algorithm.signif.alpha_NIS=0.95
python main.py hydra.verbose=true        # verbose logging
```

## Multirun (sweep over bags or parameters)

```bash
python main.py --multirun \
    bag=indoor_loadless_hover,indoor_loadless_8,indoor_loadless_round \
    limit=5
```

Each job's results land in that bag's own directory (`output/<bag_name>/<timestamp>/`),
not a shared `multirun/` folder.

## Custom output root

```bash
python main.py data_dir=/mnt/ssd/blackbird
```

## Output structure

```
output/
  indoor_loadless_hover/
    motor_data.csv          shared preprocessing (fetch_vid output)
    body_pose.csv
    images/left/, images/right/
    2026-05-24_10-30-00/    per-run results
      results.csv
      profile.prof          if profile=true
      profile.txt
      results_errors.svg    if evaluate=true
      results_trajectory.svg
      results_diagnostics.svg
      .hydra/
        config.yaml         full merged config
        overrides.yaml      CLI overrides used for this run
```

## Configuration

Configuration lives in `conf/`:

| File | Purpose |
|------|---------|
| `config.yaml` | Root config with run-level flags (`fetch_vid`, `evaluate`, `profile`, `limit`, `offset`) |
| `algorithm/default.yaml` | CV, EKF, and solver tuning parameters |
| `calibration/default.yaml` | Camera intrinsics/extrinsics, noise parameters, drone specs |
| `bag/<name>.yaml` | Bag filename and display name |

## Source layout

```
main.py              entry point; orchestrates bag loading -> estimator -> 
fetch_vid.py         preprocess VID-Dataset rosbags into algorithm-ready 
eval.py              post-hoc evaluation against ground truth (plots, diagnostics)
conf/
  config.yaml               top-level Hydra config
  algorithm/default.yaml    CV/EKF/solver tuning parameters
  calibration/default.yaml  camera intrinsics/extrinsics, noise spectral densities
  bag/*.yaml                per-dataset configs (bag path, topic names)
modules/
  algo.py            top-level orchestration: init + iter state-machine
  ekf.py             continuous-discrete EKF on composite manifold
  solver.py          two-frame Gauss-Newton joint MAP solver
  vision.py          stateless CV primitives (detection, tracking, stereo, depth)
  points.py          point data structures and set bookkeeping (F, F_pre, I)
  stats.py           chi-squared statistical gates (NIS, joint consistency, admission)
  utils.py           shared utilities (skew matrices, make_psd, image I/O)
  bag_loader.py      ROS bag reading and image topic loading
```

For what the algorithm is actually doing — the EKF, the solver, the point
lifecycle — see [journal/README.md](../journal/README.md) and the
[paper](../journal/paper/main.pdf).

No formal test suite exists in this project (no pytest, no pyproject.toml).
