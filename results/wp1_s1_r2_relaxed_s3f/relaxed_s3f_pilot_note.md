# Relaxed S3F Pilot Note

## What Was Run

Synthetic S1 x R2 tracking benchmark with a broad, two-mode orientation prior,
known body-frame translation increments, and noisy position measurements.
The compared variants are baseline S3F, S3F + R1, and S3F + R1 + R2.

- trials: 48
- steps per trial: 24
- grid sizes: [8, 16, 32, 64]
- metrics: `relaxed_s3f_metrics.csv`

## First Result

Lowest translation RMSE in this run:
`r1_r2` at grid size `64` with RMSE
`0.1680`.

Closest empirical 95% coverage:
`r1_r2` at grid size `64` with
coverage `0.938` and mean NEES
`2.066`.

## Plots

- `translation_rmse_vs_grid.png`
- `orientation_error_vs_grid.png`
- `mean_nees_vs_grid.png`
- `runtime_vs_grid.png`

## Interpretation

This pilot tests the WP1 claim that replacing representative-cell motion by
cell-averaged motion and adding within-cell covariance can reduce coarse-grid
artifacts in the S3F model problem. It is intentionally limited to S1 x R2 and
synthetic data. It does not yet validate S3+, SE(3)+, adaptive grids, or
visual-inertial odometry.
