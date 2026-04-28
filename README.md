# SE3PlusPlusS3F

Experiments for multiresolution Rao-Blackwellized grid filters for `SE(3)` and
`SE(3)++`, starting with the `S1 x R2` relaxed S3F pilot.

This repository is intentionally experiment-specific. Reusable filtering
infrastructure should move upstream to
[PyRecEst](https://github.com/FlorianPfaff/PyRecEst) once the API is stable.

## Setup

```bash
python -m pip install -e ".[dev]"
```

The package currently pins PyRecEst to the merge commit that contains the
relaxed circular S3F helper. Switch back to a versioned PyPI dependency after
the next PyRecEst release includes that helper.

For a minimal runtime install:

```bash
python -m pip install -e .
```

## Run the Relaxed S3F Pilot

```bash
se3plusplus-s3f relaxed-s3f
```

This writes a metrics CSV, plots, a run metadata JSON file, and a short note to
`results/relaxed_s3f_pilot/`.

To reproduce the fixed pilot result configuration exactly:

```bash
python scripts/reproduce_relaxed_s3f_results.py
```

The fixed configuration is stored in
`configs/relaxed_s3f_pilot.json`. The generated
`run_metadata.json` records the configuration, Python/platform details, package
versions, and metrics schema for the run. Runtime columns are expected to vary
by machine; the reference test checks deterministic behavioral metrics only.

The same configuration can also be passed through the package CLI:

```bash
se3plusplus-s3f relaxed-s3f --config configs/relaxed_s3f_pilot.json
```

## Compare Against a High-Resolution S3F Reference

Before comparing against unrelated filters, the first controlled comparison is
coarse relaxed S3F against an expensive high-resolution baseline S3F reference.

```bash
se3plusplus-s3f highres-reference
```

This writes a metrics CSV, plots, metadata, and a short note to
`results/highres_reference/`. The main approximation metrics are
translation RMSE and circular orientation error relative to the high-resolution
reference.

## Run a EuRoC Planar Smoke Test

The EuRoC smoke path uses a single ground-truth trajectory file and projects it
to the `S1 x R2` model setting. It is a trajectory-geometry check, not a
visual-inertial frontend.

```bash
se3plusplus-s3f euroc-planar --groundtruth-path path/to/MH_01_easy.txt
```

The GitHub workflow `.github/workflows/euroc-mh01-smoke.yml` downloads the
`MH_01_easy` ground-truth text file from the DROID-SLAM EuRoC ground-truth
mirror, verifies its SHA-256 checksum, caches it, runs the planar relaxed-S3F
smoke test, and validates finite nontrivial metrics.

## Test

```bash
python -m pytest
```
