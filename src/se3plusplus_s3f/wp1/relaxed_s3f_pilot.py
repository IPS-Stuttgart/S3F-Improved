"""Synthetic WP1 benchmark for relaxed S3F on S1 x R2."""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from pyrecest.filters.relaxed_s3f_circular import (
    SUPPORTED_RELAXED_S3F_VARIANTS,
    circular_error,
    rotation_matrix,
)
from scipy.special import i0

from .plotting import format_plot_list, write_metric_line_plots
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

METRIC_FIELDNAMES = [
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


def pilot_config_to_dict(config: PilotConfig) -> dict[str, Any]:
    """Return a JSON-serializable representation of a pilot config."""

    return {
        "grid_sizes": list(config.grid_sizes),
        "variants": list(config.variants),
        "n_trials": config.n_trials,
        "n_steps": config.n_steps,
        "seed": config.seed,
        "body_increment": list(config.body_increment),
        "measurement_noise_std": config.measurement_noise_std,
        "process_noise_std": config.process_noise_std,
        "initial_position_std": config.initial_position_std,
        "prior_modes": list(config.prior_modes),
        "prior_weights": list(config.prior_weights),
        "prior_kappa": config.prior_kappa,
    }


def pilot_config_from_dict(data: Mapping[str, Any]) -> PilotConfig:
    """Build a pilot config from a JSON-like mapping."""

    allowed_keys = set(pilot_config_to_dict(PilotConfig()))
    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown pilot config keys: {unknown_keys}.")

    default = PilotConfig()
    return PilotConfig(
        grid_sizes=tuple(int(value) for value in data.get("grid_sizes", default.grid_sizes)),
        variants=tuple(str(value) for value in data.get("variants", default.variants)),
        n_trials=int(data.get("n_trials", default.n_trials)),
        n_steps=int(data.get("n_steps", default.n_steps)),
        seed=int(data.get("seed", default.seed)),
        body_increment=tuple(float(value) for value in data.get("body_increment", default.body_increment)),
        measurement_noise_std=float(data.get("measurement_noise_std", default.measurement_noise_std)),
        process_noise_std=float(data.get("process_noise_std", default.process_noise_std)),
        initial_position_std=float(data.get("initial_position_std", default.initial_position_std)),
        prior_modes=tuple(float(value) for value in data.get("prior_modes", default.prior_modes)),
        prior_weights=tuple(float(value) for value in data.get("prior_weights", default.prior_weights)),
        prior_kappa=float(data.get("prior_kappa", default.prior_kappa)),
    )


def load_pilot_config(path: Path) -> PilotConfig:
    """Load a pilot config from a JSON file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected a JSON object in {path}.")
    return pilot_config_from_dict(data)


def run_relaxed_s3f_pilot(config: PilotConfig = PilotConfig()) -> list[dict[str, float | int | str]]:
    """Run the relaxed-S3F benchmark and return one metrics row per variant/grid."""

    trials = generate_pilot_trials(config)
    rows: list[dict[str, float | int | str]] = []

    for n_cells in config.grid_sizes:
        for variant in config.variants:
            if variant not in SUPPORTED_RELAXED_S3F_VARIANTS:
                raise ValueError(f"Unknown variant {variant!r}.")
            rows.append(_run_variant(config, trials, n_cells, variant))

    return rows


def generate_pilot_trials(config: PilotConfig) -> list[dict[str, np.ndarray | float]]:
    """Generate deterministic synthetic trials for the WP1 pilot config."""

    return _generate_trials(config)


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

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
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
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: Path, rows: list[dict[str, float | int | str]], config: PilotConfig) -> None:
    content = {
        "experiment": "wp1_s1_r2_relaxed_s3f",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": pilot_config_to_dict(config),
        "metrics_schema": METRIC_FIELDNAMES,
        "metrics_rows": len(rows),
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "packages": {
            "SE3PlusPlusS3F": _package_version("SE3PlusPlusS3F"),
            "matplotlib": _package_version("matplotlib"),
            "numpy": _package_version("numpy"),
            "pyrecest": _package_version("pyrecest"),
            "scipy": _package_version("scipy"),
        },
    }
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    plot_specs = [
        ("position_rmse", "Translation RMSE", "translation_rmse_vs_grid.png"),
        ("orientation_mode_error_rad", "Orientation Mode Error [rad]", "orientation_error_vs_grid.png"),
        ("mean_nees", "Mean Position NEES", "mean_nees_vs_grid.png"),
        ("runtime_ms_per_step", "Runtime [ms/step]", "runtime_vs_grid.png"),
    ]
    return write_metric_line_plots(output_dir, rows, plot_specs, SUPPORTED_RELAXED_S3F_VARIANTS, VARIANT_LABELS)


def _write_note(path: Path, rows: list[dict[str, float | int | str]], metrics_path: Path, plot_paths: list[Path], config: PilotConfig) -> None:
    best_rmse = min(rows, key=lambda row: float(row["position_rmse"]))
    best_coverage = min(rows, key=lambda row: abs(float(row["coverage_95"]) - 0.95))

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

{format_plot_list(plot_paths)}

## Interpretation

This pilot tests the WP1 claim that replacing representative-cell motion by
cell-averaged motion and adding within-cell covariance can reduce coarse-grid
artifacts in the S3F model problem. It is intentionally limited to S1 x R2 and
synthetic data. It does not yet validate S3+, SE(3)+, adaptive grids, or
visual-inertial odometry.
"""
    path.write_text(content, encoding="utf-8")
