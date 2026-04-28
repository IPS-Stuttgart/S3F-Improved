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

## Run a EuRoC Planar Smoke Test

The EuRoC smoke path uses a single ground-truth trajectory file and projects it
to the WP1 `S1 x R2` setting. It is a trajectory-geometry check, not a
visual-inertial frontend.

```bash
se3plusplus-s3f wp1-euroc-planar --groundtruth-path path/to/MH_01_easy.txt
```

The GitHub workflow `.github/workflows/euroc-mh01-smoke.yml` downloads the
`MH_01_easy` ground-truth text file from the DROID-SLAM EuRoC ground-truth
mirror, verifies its SHA-256 checksum, caches it, runs the planar relaxed-S3F
smoke test, and validates finite nontrivial metrics.

## Test

```bash
python -m pytest
```
