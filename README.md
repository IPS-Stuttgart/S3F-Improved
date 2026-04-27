# SE3PlusPlusS3F

Experiments for multiresolution Rao-Blackwellized grid filters for `SE(3)` and
`SE(3)++`, starting with the WP1 `S1 x R2` relaxed S3F pilot.

This repository is intentionally experiment-specific. Reusable filtering
infrastructure should move upstream to
[PyRecEst](https://github.com/FlorianPfaff/PyRecEst) once the API is stable.

## Setup

```bash
python -m pip install -e ".[dev]"
```

For a minimal runtime install:

```bash
python -m pip install -e .
```

## Run the WP1 Pilot

```bash
se3plusplus-s3f wp1-relaxed-s3f
```

This writes a metrics CSV, plots, and a short note to
`results/wp1_s1_r2_relaxed_s3f/`.

## Test

```bash
python -m pytest
```
