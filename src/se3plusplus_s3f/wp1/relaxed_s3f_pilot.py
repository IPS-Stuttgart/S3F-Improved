"""Synthetic WP1 benchmark for relaxed S3F on S1 x R2."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0

from .relaxed_s3f_circular import (
    SUPPORTED_RELAXED_S3F_VARIANTS,
    circular_error,
    rotation_matrix,
)
from .s3f_common import (
    linear_position_error_stats,
    make_linear_likelihood,
    make_s3f_filter,
    orientation_mode_and_mean,
    predict_update_linear_position,
)


VARIANT_LABELS = {
    "baseline": "Baseline S3F",
    "r1": "S3F + R1",
    "r1_r2": "S3F + R1 + R2",
}


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for the relaxed-S3F pilot benchmark."""

    grid_sizes: tuple[int, ...] = (8, 16, 32, 64)
    variants: tuple[str, ...] = SUPPORTED_RELAXED_S3F_VARIANTS
    n_trials: int = 48
    n_steps: int = 24
    seed: int = 7
    body_increment: tuple[float, float] = (0.45, 0.10)
    measurement_noise_std: float = 0.25
    process_noise_std: float = 0.02
    initial_position_std: float = 0.20
    prior_modes: tuple[float, float] = (0.65, 3.85)
    prior_weights: tuple[float, float] = (0.55, 0.45)
    prior_kappa: float = 1.2


def run_relaxed_s3f_pilot(config: PilotConfig = PilotConfig()) -> list[dict[str, float | int | str]]:
    """Run the relaxed-S3F benchmark and return one metrics row per variant/grid."""

    trials = _generate_trials(config)
    rows: list[dict[str, float | int | str]] = []

    for n_cells in config.grid_sizes:
        for variant in config.variants:
            if variant not in SUPPORTED_RELAXED_S3F_VARIANTS:
                raise ValueError(f"Unknown variant {variant!r}.")
            rows.append(_run_variant(config, trials, n_cells, variant))

    return rows


def write_relaxed_s3f_pilot_outputs(
    output_dir: Path,
    config: PilotConfig = PilotConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the pilot and write CSV, optional plots, and a short note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_relaxed_s3f_pilot(config)

    metrics_path = output_dir / "relaxed_s3f_metrics.csv"
    _write_csv(metrics_path, rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "relaxed_s3f_pilot_note.md"
    _write_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path
    return outputs


def make_initial_filter(config: PilotConfig, n_cells: int):
    """Construct the shared broad/multimodal initial S3F state."""

    grid = np.linspace(0.0, 2.0 * np.pi, n_cells, endpoint=False)
    prior_values = _orientation_prior_pdf(grid, config)
    initial_cov = np.eye(2) * config.initial_position_std**2
    return make_s3f_filter(grid, prior_values, np.zeros(2), initial_cov)


def _run_variant(
    config: PilotConfig,
    trials: list[dict[str, np.ndarray | float]],
    n_cells: int,
    variant: str,
) -> dict[str, float | int | str]:
    measurement_cov = np.eye(2) * config.measurement_noise_std**2
    process_noise_cov = np.eye(2) * config.process_noise_std**2

    sum_position_sq_error = 0.0
    sum_orientation_mode_error = 0.0
    sum_orientation_mean_error = 0.0
    sum_nees = 0.0
    coverage_hits = 0
    n_metrics = 0
    runtime = 0.0

    for trial in trials:
        filter_ = make_initial_filter(config, n_cells)
        true_angle = float(trial["angle"])
        true_positions = np.asarray(trial["positions"])
        measurements = np.asarray(trial["measurements"])

        for step, measurement in enumerate(measurements):
            likelihood = make_linear_likelihood(measurement, measurement_cov)
            runtime += predict_update_linear_position(
                filter_,
                np.asarray(config.body_increment),
                variant,
                process_noise_cov,
                likelihood,
            )

            error, nees = linear_position_error_stats(filter_, true_positions[step + 1])
            sq_error = float(error @ error)
            sum_position_sq_error += sq_error
            sum_nees += nees
            coverage_hits += int(nees <= 5.991464547107979)

            mode_angle, mean_angle = orientation_mode_and_mean(filter_)
            sum_orientation_mode_error += circular_error(mode_angle, true_angle)
            sum_orientation_mean_error += circular_error(mean_angle, true_angle)
            n_metrics += 1

    return {
        "grid_size": n_cells,
        "variant": variant,
        "position_rmse": float(np.sqrt(sum_position_sq_error / n_metrics)),
        "orientation_mode_error_rad": sum_orientation_mode_error / n_metrics,
        "orientation_mean_error_rad": sum_orientation_mean_error / n_metrics,
        "mean_nees": sum_nees / n_metrics,
        "coverage_95": coverage_hits / n_metrics,
        "runtime_ms_per_step": 1000.0 * runtime / n_metrics,
        "n_trials": config.n_trials,
        "n_steps": config.n_steps,
    }


def _generate_trials(config: PilotConfig) -> list[dict[str, np.ndarray | float]]:
    rng = np.random.default_rng(config.seed)
    trials = []
    body_increment = np.asarray(config.body_increment, dtype=float)
    prior_weights = np.asarray(config.prior_weights, dtype=float)
    prior_weights = prior_weights / np.sum(prior_weights)

    for _ in range(config.n_trials):
        component = int(rng.choice(len(config.prior_modes), p=prior_weights))
        angle = float(rng.vonmises(config.prior_modes[component], config.prior_kappa))
        displacement = rotation_matrix(angle) @ body_increment

        positions = np.zeros((config.n_steps + 1, 2), dtype=float)
        for step in range(config.n_steps):
            positions[step + 1] = positions[step] + displacement

        measurements = positions[1:] + rng.normal(
            0.0,
            config.measurement_noise_std,
            size=(config.n_steps, 2),
        )
        trials.append({"angle": angle, "positions": positions, "measurements": measurements})

    return trials


def _orientation_prior_pdf(angles: np.ndarray, config: PilotConfig) -> np.ndarray:
    angles = np.asarray(angles, dtype=float)
    values = np.zeros_like(angles, dtype=float)
    weights = np.asarray(config.prior_weights, dtype=float)
    weights = weights / np.sum(weights)

    norm_const = 2.0 * np.pi * i0(config.prior_kappa)
    for weight, mode in zip(weights, config.prior_modes, strict=True):
        values += weight * np.exp(config.prior_kappa * np.cos(angles - mode)) / norm_const
    return values


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "grid_size",
        "variant",
        "position_rmse",
        "orientation_mode_error_rad",
        "orientation_mean_error_rad",
        "mean_nees",
        "coverage_95",
        "runtime_ms_per_step",
        "n_trials",
        "n_steps",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    plot_specs = [
        ("position_rmse", "Translation RMSE", "translation_rmse_vs_grid.png"),
        ("orientation_mode_error_rad", "Orientation Mode Error [rad]", "orientation_error_vs_grid.png"),
        ("mean_nees", "Mean Position NEES", "mean_nees_vs_grid.png"),
        ("runtime_ms_per_step", "Runtime [ms/step]", "runtime_vs_grid.png"),
    ]

    paths = []
    for metric, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for variant in SUPPORTED_RELAXED_S3F_VARIANTS:
            variant_rows = sorted(
                [row for row in rows if row["variant"] == variant],
                key=lambda row: int(row["grid_size"]),
            )
            xs = [int(row["grid_size"]) for row in variant_rows]
            ys = [float(row[metric]) for row in variant_rows]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=VARIANT_LABELS[variant])

        ax.set_xlabel("Number of circular cells")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    metrics_path: Path,
    plot_paths: list[Path],
    config: PilotConfig,
) -> None:
    best_rmse = min(rows, key=lambda row: float(row["position_rmse"]))
    best_coverage = min(rows, key=lambda row: abs(float(row["coverage_95"]) - 0.95))

    plot_lines = "\n".join(f"- `{plot_path.name}`" for plot_path in plot_paths)
    if not plot_lines:
        plot_lines = "- plots were disabled for this run"

    content = f"""# Relaxed S3F Pilot Note

## What Was Run

Synthetic S1 x R2 tracking benchmark with a broad, two-mode orientation prior,
known body-frame translation increments, and noisy position measurements.
The compared variants are baseline S3F, S3F + R1, and S3F + R1 + R2.

- trials: {config.n_trials}
- steps per trial: {config.n_steps}
- grid sizes: {list(config.grid_sizes)}
- metrics: `{metrics_path.name}`

## First Result

Lowest translation RMSE in this run:
`{best_rmse["variant"]}` at grid size `{best_rmse["grid_size"]}` with RMSE
`{float(best_rmse["position_rmse"]):.4f}`.

Closest empirical 95% coverage:
`{best_coverage["variant"]}` at grid size `{best_coverage["grid_size"]}` with
coverage `{float(best_coverage["coverage_95"]):.3f}` and mean NEES
`{float(best_coverage["mean_nees"]):.3f}`.

## Plots

{plot_lines}

## Interpretation

This pilot tests the WP1 claim that replacing representative-cell motion by
cell-averaged motion and adding within-cell covariance can reduce coarse-grid
artifacts in the S3F model problem. It is intentionally limited to S1 x R2 and
synthetic data. It does not yet validate S3+, SE(3)+, adaptive grids, or
visual-inertial odometry.
"""
    path.write_text(content, encoding="utf-8")
