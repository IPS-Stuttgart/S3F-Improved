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

This writes a metrics CSV, plots, a run metadata JSON file, and a short note to
`results/wp1_s1_r2_relaxed_s3f/`.

To reproduce the committed WP1 result configuration exactly:

```bash
python scripts/reproduce_wp1_results.py
```

The fixed configuration is stored in
`configs/wp1_relaxed_s3f_pilot.json`. The generated
`run_metadata.json` records the configuration, Python/platform details, package
versions, and metrics schema for the run. Runtime columns are expected to vary
by machine; the reference test checks deterministic behavioral metrics only.

The same configuration can also be passed through the package CLI:

```bash
se3plusplus-s3f wp1-relaxed-s3f --config configs/wp1_relaxed_s3f_pilot.json
```

## Test

```bash
python -m pytest
```
