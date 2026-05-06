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

## Compare Against Baseline Filters

The next controlled comparison uses the same generated trials for every method
and adds a single-Gaussian EKF and bootstrap particle filter baseline.

```bash
se3plusplus-s3f compare-baselines
```

This writes a metrics CSV, plots, metadata, and a short note to
`results/baseline_comparison/`.

For a larger reproducible run without committing generated files, use the
manual GitHub Actions workflow `Baseline Comparison Benchmark`. It uploads the
generated CSV, plots, metadata, and note as a workflow artifact attached to the
Actions run.

## Report Quality vs Cost

To combine relaxed S3F accuracy, consistency, runtime, and distance to a
high-resolution S3F reference in one report:

```bash
se3plusplus-s3f quality-cost
```

This writes S3F metrics, grid-saving claims, a combined S3F-vs-particle Pareto
table, plots, metadata, and a short note to `results/quality_cost/`. For a
reproducible runner-side report without committing generated files, use the
manual GitHub Actions workflow `Quality Cost Report`.

Use `--repeats N` or the workflow repeat input to rerun the quality-cost
comparison with independent seeds. Repeated runs add
`quality_cost_repeat_pareto.csv` and `quality_cost_summary.csv` to the report
artifact, including mean, standard deviation, and approximate 95% CI columns.
When selected candidate/comparator resources are present, repeated runs also add
`quality_cost_pairwise.csv` with paired candidate-minus-comparator deltas.

## Sweep Particle Counts

To study the accuracy-runtime tradeoff between relaxed S3F grids and bootstrap
particle-filter sample counts:

```bash
se3plusplus-s3f particle-sensitivity
```

This writes outputs to `results/particle_sensitivity/`. For a larger run that
keeps generated files out of git, use the manual GitHub Actions workflow
`Particle Sensitivity Benchmark`.

## Run the S3+ x R3 Prototype

The first 3-D orientation prototype keeps the same S3F structure but uses a
PyRecEst hyperhemispherical quaternion grid and one Gaussian `R3` position
component per grid point. Its `R1` and `R1+R2` terms are estimated from
deterministic local tangent samples around each grid quaternion, so this is a
numerical prototype rather than an exact `S3+` cell-integral implementation.

```bash
se3plusplus-s3f s3r3-relaxed
```

This writes outputs to `results/s3r3_relaxed/`. For a reproducible runner-side
report without committing generated files, use the manual GitHub Actions
workflow `S3R3 Relaxed Report`.

## Check the S3+ Orientation Basis

To verify that PyRecEst's `HyperhemisphericalGridFilter` can serve as the
quaternion orientation marginal for the coupled `S3+ x R3` S3F prototype:

```bash
se3plusplus-s3f s3r3-orientation-basis
```

This writes grid-matching diagnostics, PyRecEst orientation point-estimate
errors, effective orientation-cell counts, plots, metadata, and a short note to
`results/s3r3_orientation_basis/`.

## Run a Dynamic S3+ x R3 Pose Benchmark

To test the relaxed prediction when orientation itself evolves with a known
quaternion increment:

```bash
se3plusplus-s3f s3r3-dynamic-pose
```

This uses `q_next = q_current * delta_q` and
`p_next = p_current + R(q_current) u + noise`, with a soft PyRecEst
hyperhemispherical transition density for the S3+ grid. It writes metrics,
claim rows, plots, metadata, and a short note to `results/s3r3_dynamic_pose/`.

## Sweep Dynamic S3+ x R3 Robustness

To check whether the dynamic pose result survives different random seeds and
orientation-increment magnitudes:

```bash
se3plusplus-s3f s3r3-dynamic-robustness
```

This repeats the dynamic pose benchmark across seed and turn-rate scenarios,
then writes raw metrics, raw claim rows, aggregate win-rate/RMSE/NEES tables,
plots, metadata, and a short note to `results/s3r3_dynamic_robustness/`.

## Compare S3R3 Against a High-Resolution Reference

To test whether coarse S3R3 relaxed propagation follows a denser S3F reference:

```bash
se3plusplus-s3f s3r3-highres-reference
```

This writes outputs to `results/s3r3_highres_reference/`. The reference is a
denser baseline S3F quaternion grid, not an external filter.

## Compare Dynamic S3R3 Against a High-Resolution Reference

To test whether dynamic coarse relaxed propagation follows a denser dynamic
baseline S3F reference:

```bash
se3plusplus-s3f s3r3-dynamic-highres-reference
```

This uses the same dynamic model as `s3r3-dynamic-pose` and compares each
coarse baseline, `R1`, and `R1+R2` row against a denser dynamic baseline S3F.
It writes metrics, claim rows, plots, metadata, and a short note to
`results/s3r3_dynamic_highres_reference/`.

## Summarize S3R3 Evidence

To combine direct relaxed S3R3 metrics and the high-resolution reference
comparison into explicit claim rows:

```bash
se3plusplus-s3f s3r3-evidence-summary
```

This writes relaxed metrics, high-resolution reference metrics, claim rows,
plots, metadata, and a short note to `results/s3r3_evidence_summary/`.

## Sweep S3R3 Stress Conditions

To map where `R1+R2` helps as orientation uncertainty and motion-induced
translation bias change:

```bash
se3plusplus-s3f s3r3-stress-sweep
```

This writes scenario metrics, comparison claim rows, summary tables, plots,
metadata, and a short note to `results/s3r3_stress_sweep/`.

## Compare S3R3 Against a Particle Filter

To compare the best `R1+R2` relaxed S3F row in each S3R3 stress scenario
against bootstrap particle-filter particle counts:

```bash
se3plusplus-s3f s3r3-particle-comparison
```

This writes combined S3F/PF metrics, per-scenario comparison summaries, plots,
metadata, and a short note to `results/s3r3_particle_comparison/`. For larger
artifact-only runs, use the manual GitHub Actions workflow
`S3R3 Particle Comparison`.

## Profile S3F Runtime

To split relaxed S3F runtime into likelihood construction, cell-statistics,
`predict_linear`, update, and metric-bookkeeping phases:

```bash
se3plusplus-s3f profile-s3f-runtime
```

This writes outputs to `results/s3f_runtime_profile/`. For a reproducible CI
runner profile, use the manual GitHub Actions workflow `S3F Runtime Profile`.

## Run a EuRoC Planar Smoke Test

The EuRoC smoke path uses a single ground-truth trajectory file and projects it
to the `S1 x R2` model setting. It is a trajectory-geometry check, not a
visual-inertial frontend.

```bash
se3plusplus-s3f euroc-planar --groundtruth-path path/to/MH_01_easy.txt
```

The GitHub workflow `.github/workflows/euroc-mh01-smoke.yml` downloads the
`MH_01_easy` ground-truth text file from the DROID-SLAM EuRoC ground-truth
mirror, verifies its SHA-256 checksum, caches it, runs the planar and S3+ x R3
relaxed-S3F smoke tests, and validates finite nontrivial metrics.

## Run a EuRoC S3+ x R3 Pose Smoke Test

To use an existing 3-D trajectory before adding a full visual-inertial frontend,
the EuRoC S3R3 smoke path reads the same ground-truth pose file and derives
known body-frame translation controls plus quaternion increments from successive
poses. Position measurements are simulated by adding Gaussian noise to the
ground-truth positions. The report includes baseline S3F, `R1`, `R1+R2`, and a
PyRecEst `UKFOnManifolds` baseline on `SO3 x R3` with a 6-D tangent covariance.

```bash
se3plusplus-s3f euroc-s3r3-pose --groundtruth-path path/to/MH_01_easy.txt
```

To stress the non-Gaussian orientation case, pass multiple initial yaw-offset
modes. For example, this gives S3F a two-mode prior while the manifold UKF uses
one tangent-Gaussian approximation to the same yaw ambiguity:

```bash
se3plusplus-s3f euroc-s3r3-pose --groundtruth-path path/to/MH_01_easy.txt --prior-yaw-offsets 0 3.141592653589793 --prior-weights 0.5 0.5
```

This writes variant metrics, claim rows, plots, metadata, and a short note to
`results/euroc_s3r3_pose/`. The GitHub workflow
`.github/workflows/euroc-mh01-smoke.yml` also runs this S3+ x R3 pose smoke
test on the cached `MH_01_easy` file and uploads the generated outputs as an
Actions artifact.

## Run a EuRoC S3+ x R3 Comparison Report

For paper-facing evidence, the EuRoC comparison report runs coarse baseline,
`R1`, and `R1+R2` S3F rows against a denser dynamic baseline S3F reference on
the same EuRoC controls and simulated position measurements. It also includes
the manifold UKF and an optional bootstrap particle filter row for runtime and
accuracy context.

```bash
se3plusplus-s3f euroc-s3r3-comparison-report --groundtruth-path path/to/MH_01_easy.txt
```

This writes metrics, claim rows, five plots, metadata, and a note to
`results/euroc_s3r3_comparison/`. The manual GitHub workflow
`.github/workflows/euroc-s3r3-comparison-report.yml` downloads and caches the
same EuRoC ground-truth file, validates the report outputs, and uploads the
CSV/PNG/Markdown files as an Actions artifact.

## Test

```bash
python -m pytest
```
