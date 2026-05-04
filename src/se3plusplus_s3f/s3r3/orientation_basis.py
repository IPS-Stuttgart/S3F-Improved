"""Diagnostics for using PyRecEst's S3+ grid filter as a 3D-pose orientation basis."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution

from ..s1r2.plotting import save_figure
from .relaxed_s3f_prototype import (
    SUPPORTED_S3R3_VARIANTS,
    S3R3PrototypeConfig,
    generate_s3r3_trials,
    make_s3r3_filter,
    make_s3r3_orientation_filter,
    predict_s3r3_relaxed,
    s3r3_orientation_distance,
    s3r3_orientation_filter_from_s3f,
    s3r3_orientation_mode,
    s3r3_orientation_point_estimate,
    validate_s3r3_prototype_config,
)

ORIENTATION_BASIS_FIELDNAMES = [
    "grid_size",
    "variant",
    "basis_grid_matches_s3f",
    "basis_prior_max_abs_diff",
    "mean_mode_error_rad",
    "mean_point_error_rad",
    "mean_mode_point_gap_rad",
    "mean_effective_cells",
    "runtime_ms_per_step",
    "cell_radius_rad",
    "cell_sample_count",
    "n_trials",
    "n_steps",
]


@dataclass(frozen=True)
class S3R3OrientationBasisConfig:
    """Configuration for the S3+ orientation-basis diagnostic."""

    prototype: S3R3PrototypeConfig = field(
        default_factory=lambda: S3R3PrototypeConfig(
            grid_sizes=(8, 16, 32),
            n_trials=4,
            n_steps=5,
            seed=43,
            cell_sample_count=27,
        )
    )
    variant: str = "r1_r2"


def s3r3_orientation_basis_config_to_dict(config: S3R3OrientationBasisConfig) -> dict[str, Any]:
    """Return a JSON-serializable S3+ orientation-basis config."""

    return json.loads(json.dumps(asdict(config)))


def run_s3r3_orientation_basis_diagnostic(
    config: S3R3OrientationBasisConfig = S3R3OrientationBasisConfig(),
) -> list[dict[str, float | int | str]]:
    """Check that PyRecEst's hyperhemispherical grid filter can carry the S3F orientation marginal."""

    _validate_config(config)
    prototype = config.prototype
    trials = generate_s3r3_trials(prototype)
    rows: list[dict[str, float | int | str]] = []
    for grid_size in prototype.grid_sizes:
        rows.append(_run_grid_diagnostic(config, trials, grid_size))
    return rows


def write_s3r3_orientation_basis_outputs(
    output_dir: Path,
    config: S3R3OrientationBasisConfig = S3R3OrientationBasisConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the S3+ orientation-basis diagnostic and write CSV, metadata, note, and optional plots."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_s3r3_orientation_basis_diagnostic(config)

    metrics_path = output_dir / "s3r3_orientation_basis_metrics.csv"
    _write_csv(metrics_path, rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_orientation_basis_note.md"
    _write_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: S3R3OrientationBasisConfig) -> None:
    validate_s3r3_prototype_config(config.prototype)
    if config.variant not in SUPPORTED_S3R3_VARIANTS:
        raise ValueError(f"Unknown variant {config.variant!r}.")


def _run_grid_diagnostic(
    config: S3R3OrientationBasisConfig,
    trials: list[dict[str, np.ndarray]],
    grid_size: int,
) -> dict[str, float | int | str]:
    prototype = config.prototype
    orientation_filter = make_s3r3_orientation_filter(prototype, grid_size)
    s3f_filter = make_s3r3_filter(prototype, grid_size)
    basis_grid = np.asarray(orientation_filter.filter_state.get_grid(), dtype=float)
    s3f_grid = np.asarray(s3f_filter.filter_state.gd.get_grid(), dtype=float)
    basis_weights = _normalized_grid_values(np.asarray(orientation_filter.filter_state.grid_values, dtype=float))
    s3f_weights = _normalized_grid_values(np.asarray(s3f_filter.filter_state.gd.grid_values, dtype=float))

    process_noise_cov = np.eye(3) * prototype.process_noise_std**2
    measurement_cov = np.eye(3) * prototype.measurement_noise_std**2
    body_increment = np.asarray(prototype.body_increment, dtype=float)

    sum_mode_error = 0.0
    sum_point_error = 0.0
    sum_mode_point_gap = 0.0
    sum_effective_cells = 0.0
    runtime = 0.0
    n_metrics = 0
    last_cell_radius = 0.0

    for trial in trials:
        filter_ = make_s3r3_filter(prototype, grid_size)
        true_orientation = np.asarray(trial["orientation"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)
        for measurement in measurements:
            likelihood = GaussianDistribution(measurement, measurement_cov, check_validity=False)
            start = perf_counter()
            stats = predict_s3r3_relaxed(
                filter_,
                body_increment,
                variant=config.variant,
                process_noise_cov=process_noise_cov,
                cell_sample_count=prototype.cell_sample_count,
            )
            filter_.update(likelihoods_linear=[likelihood])
            orientation_marginal = s3r3_orientation_filter_from_s3f(filter_)
            point_estimate = s3r3_orientation_point_estimate(filter_)
            runtime += perf_counter() - start

            mode = s3r3_orientation_mode(filter_)
            marginal_weights = _normalized_grid_values(np.asarray(orientation_marginal.filter_state.grid_values, dtype=float))
            sum_mode_error += s3r3_orientation_distance(mode, true_orientation)
            sum_point_error += s3r3_orientation_distance(point_estimate, true_orientation)
            sum_mode_point_gap += s3r3_orientation_distance(mode, point_estimate)
            sum_effective_cells += _effective_cell_count(marginal_weights)
            last_cell_radius = stats.cell_radius_rad
            n_metrics += 1

    return {
        "grid_size": grid_size,
        "variant": config.variant,
        "basis_grid_matches_s3f": int(np.allclose(basis_grid, s3f_grid, atol=1e-12)),
        "basis_prior_max_abs_diff": float(np.max(np.abs(basis_weights - s3f_weights))),
        "mean_mode_error_rad": sum_mode_error / n_metrics,
        "mean_point_error_rad": sum_point_error / n_metrics,
        "mean_mode_point_gap_rad": sum_mode_point_gap / n_metrics,
        "mean_effective_cells": sum_effective_cells / n_metrics,
        "runtime_ms_per_step": 1000.0 * runtime / n_metrics,
        "cell_radius_rad": last_cell_radius,
        "cell_sample_count": prototype.cell_sample_count,
        "n_trials": prototype.n_trials,
        "n_steps": prototype.n_steps,
    }


def _normalized_grid_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("grid values must have positive total mass.")
    return values / total


def _effective_cell_count(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights**2))


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ORIENTATION_BASIS_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in ORIENTATION_BASIS_FIELDNAMES})


def _write_metadata(
    path: Path,
    rows: list[dict[str, float | int | str]],
    config: S3R3OrientationBasisConfig,
) -> None:
    metadata = {
        "experiment": "s3r3_orientation_basis",
        "config": s3r3_orientation_basis_config_to_dict(config),
        "metrics_schema": ORIENTATION_BASIS_FIELDNAMES,
        "metrics_rows": len(rows),
        "orientation_basis": "pyrecest.filters.HyperhemisphericalGridFilter",
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    metrics_path: Path,
    plot_paths: list[Path],
    config: S3R3OrientationBasisConfig,
) -> None:
    worst_prior_gap = max(float(row["basis_prior_max_abs_diff"]) for row in rows)
    mean_point_error = float(np.mean([float(row["mean_point_error_rad"]) for row in rows]))
    lines = [
        "# S3+ Orientation Basis Diagnostic",
        "",
        "This diagnostic checks whether PyRecEst's HyperhemisphericalGridFilter can act as the quaternion orientation marginal for the coupled S3+ x R3 S3F prototype.",
        "The coupled filter still owns the Rao-Blackwellized R3 Gaussian components; the hyperhemispherical filter is used as the orientation grid and marginal container.",
        "",
        f"Variant: `{config.variant}`",
        f"Trials: {config.prototype.n_trials}",
        f"Steps per trial: {config.prototype.n_steps}",
        f"Grid sizes: {list(config.prototype.grid_sizes)}",
        f"Cell sample count: {config.prototype.cell_sample_count}",
        f"Metrics file: `{metrics_path.name}`",
        "",
        "## Headline",
        "",
        f"- Maximum normalized prior difference between the standalone orientation basis and coupled S3F grid: {worst_prior_gap:.3e}.",
        f"- Mean PyRecEst orientation point-estimate error across rows: {mean_point_error:.3f} rad.",
        "",
    ]
    if plot_paths:
        lines.extend(
            [
                "## Plots",
                "",
                *[f"- `{plot_path.name}`" for plot_path in plot_paths],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    grid_sizes = [int(row["grid_size"]) for row in rows]
    mode_errors = [float(row["mean_mode_error_rad"]) for row in rows]
    point_errors = [float(row["mean_point_error_rad"]) for row in rows]
    effective_cells = [float(row["mean_effective_cells"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid_sizes, mode_errors, marker="o", label="Grid mode")
    ax.plot(grid_sizes, point_errors, marker="s", label="PyRecEst point estimate")
    ax.set_xlabel("Number of quaternion grid cells")
    ax.set_ylabel("Mean orientation error [rad]")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    error_path = save_figure(fig, output_dir, "s3r3_orientation_basis_errors.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid_sizes, effective_cells, marker="o")
    ax.set_xlabel("Number of quaternion grid cells")
    ax.set_ylabel("Mean effective orientation cells")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    effective_path = save_figure(fig, output_dir, "s3r3_orientation_basis_effective_cells.png")
    return [error_path, effective_path]
