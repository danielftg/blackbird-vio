# blackbird-vio

Vision-only body-frame state estimator for quadcopters. Stereo cameras, no IMU.
See `paper.pdf` for details. *(link tbd)*

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download a flight bag (e.g. `indoor_loadless_hovor_3096.1g_79.04s.bag`) from
the [ZJU FAST-Lab VID-Dataset](https://github.com/ZJU-FAST-Lab/VID-Dataset)
and place it under `src/bags/`.

## Run

```bash
python src/main.py --fetch_vid --evaluate
```

Results appear in `src/output/`.