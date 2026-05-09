"""S3+ x R3 relaxed-S3F prototype experiments using PyRecEst helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from pyrecest.filters.hyperhemispherical_grid_filter import HyperhemisphericalGridFilter
from pyrecest.filters.relaxed_s3f_so3 import (
    S3R3CellStatistics,
    SUPPORTED_RELAXED_S3F_SO3_VARIANTS,
    predict_s3r3_relaxed,
    s3r3_cell_statistics,
    s3r3_orientation_distance,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter

from ..s1r2.plotting import format_plot_list, save_figure
from .so3_helpers import (
    canonical_quaternions as _canonical_quaternions,
    exp_map_identity as _exp_map_identity,
    geodesic_distance as _geodesic_distance,
    quaternion_distance_matrix as _quaternion_distance_matrix,
    quaternion_multiply as _quaternion_multiply,
    quaternion_to_rotation_matrices as _quaternion_to_rotation_matrices,
    rotate_vectors as _rotate_vectors,
)

SUPPORTED_S3R3_VARIANTS = SUPPORTED_RELAXED_S3F_SO3_VARIANTS
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
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "cell_radius_rad",
    "cell_sample_count",
    "n_trials",
    "n_steps",
]


@dataclass(frozen=True)
class S3R3PrototypeConfig:
    """Configuration for the S3+ x R3 relaxed-S3F prototype benchmark."""

    grid_sizes: tuple[int, ...] = (16, 32)
    variants: tuple[str, ...] = SUPPORTED_S3R3_VARIANTS
    n_trials: int = 8
    n_steps: int = 8
    seed: int = 23
    body_increment: tuple[float, float, float] = (0.35, 0.05, 0.12)
    measurement_noise_std: float = 0.30
    process_noise_std: float = 0.03
    initial_position_std: float = 0.25
    prior_modes: tuple[tuple[float, float, float, float], ...] = (
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.7833269096274834, 0.0, 0.6216099682706644),
    )
    prior_weights: tuple[float, ...] = (0.58, 0.42)
    prior_kappa: float = 3.0
    orientation_noise_std: float = 0.55
    cell_sample_count: int = 27


def s3r3_prototype_config_to_dict(config: S3R3PrototypeConfig) -> dict[str, Any]:
    """Return a JSON-serializable S3R3 prototype config."""

    return json.loads(json.dumps(asdict(config)))


def validate_s3r3_prototype_config(
    config: S3R3PrototypeConfig,
    *,
    reference_grid_size: int | None = None,
    required_variants: tuple[str, ...] = (),
) -> None:
    """Validate shared S3R3 prototype and reference-run settings."""

    if not config.grid_sizes:
        raise ValueError("grid_sizes must not be empty.")
    if min(config.grid_sizes) <= 0:
        raise ValueError("all grid sizes must be positive.")
    if reference_grid_size is not None and reference_grid_size <= max(config.grid_sizes):
        raise ValueError("reference_grid_size must be greater than every coarse grid size.")
    if config.n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if config.cell_sample_count <= 0:
        raise ValueError("cell_sample_count must be positive.")

    unknown_variants = [variant for variant in config.variants if variant not in SUPPORTED_S3R3_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown variant {unknown_variants[0]!r}.")

    missing_variants = sorted(set(required_variants) - set(config.variants))
    if missing_variants:
        raise ValueError(f"evidence summary requires variants: {missing_variants}.")


def run_s3r3_relaxed_prototype(config: S3R3PrototypeConfig = S3R3PrototypeConfig()) -> list[dict[str, float | int | str]]:
    """Run the S3+ x R3 relaxed-S3F prototype and return one metrics row per variant/grid."""

    validate_s3r3_prototype_config(config)
    trials = generate_s3r3_trials(config)
    rows: list[dict[str, float | int | str]] = []
    for grid_size in config.grid_sizes:
        for variant in config.variants:
            rows.append(_run_variant(config, trials, grid_size, variant))
    return rows


def write_s3r3_relaxed_outputs(
    output_dir: Path,
    config: S3R3PrototypeConfig = S3R3PrototypeConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the S3+ x R3 prototype and write CSV, optional plots, metadata, and a note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_s3r3_relaxed_prototype(config)

    metrics_path = output_dir / "s3r3_relaxed_metrics.csv"
    _write_csv(metrics_path, rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_relaxed_note.md"
    _write_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
    return outputs


def make_s3r3_orientation_filter(config: S3R3PrototypeConfig, grid_size: int) -> HyperhemisphericalGridFilter:
    """Construct the PyRecEst S3+ orientation marginal used by the coupled S3F state."""

    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    orientation_filter = HyperhemisphericalGridFilter(grid_size, 3, "leopardi_symm")
    grid = _canonical_quaternions(np.asarray(orientation_filter.filter_state.get_grid(), dtype=float))
    grid_values = _orientation_prior_values(grid, config)
    gd = HyperhemisphericalGridDistribution(grid, grid_values, enforce_pdf_nonnegative=True)
    gd.normalize_in_place(warn_unnorm=False)
    orientation_filter.filter_state = gd
    return orientation_filter


def make_s3r3_filter(config: S3R3PrototypeConfig, grid_size: int) -> StateSpaceSubdivisionFilter:
    """Construct an S3+ x R3 S3F state with one Gaussian position component per quaternion grid point."""

    orientation_filter = make_s3r3_orientation_filter(config, grid_size)
    gd = orientation_filter.filter_state

    initial_covariance = np.eye(3) * config.initial_position_std**2
    gaussians = [
        GaussianDistribution(np.zeros(3), initial_covariance.copy(), check_validity=False)
        for _ in range(gd.get_grid().shape[0])
    ]
    return StateSpaceSubdivisionFilter(StateSpaceSubdivisionGaussianDistribution(gd, gaussians))


def s3r3_orientation_filter_from_s3f(filter_: StateSpaceSubdivisionFilter) -> HyperhemisphericalGridFilter:
    """Return the current S3F orientation marginal as a PyRecEst hyperhemispherical grid filter."""

    state = filter_.filter_state
    if state.lin_dim != 3:
        raise ValueError("s3r3_orientation_filter_from_s3f requires a 3-D linear state.")

    grid = _canonical_quaternions(np.asarray(state.gd.get_grid(), dtype=float))
    grid_values = np.asarray(state.gd.grid_values, dtype=float)
    gd = HyperhemisphericalGridDistribution(grid, grid_values, enforce_pdf_nonnegative=True)
    gd.normalize_in_place(warn_unnorm=False)
    orientation_filter = HyperhemisphericalGridFilter(grid.shape[0], 3, "leopardi_symm")
    orientation_filter.filter_state = gd
    return orientation_filter


def s3r3_orientation_point_estimate(filter_: StateSpaceSubdivisionFilter) -> np.ndarray:
    """Return PyRecEst's orientation point estimate from the S3F orientation marginal."""

    orientation_filter = s3r3_orientation_filter_from_s3f(filter_)
    return _canonical_quaternions(np.asarray(orientation_filter.get_point_estimate(), dtype=float))


def generate_s3r3_trials(config: S3R3PrototypeConfig) -> list[dict[str, np.ndarray]]:
    """Generate reproducible synthetic S3+ x R3 tracking trials."""

    return _generate_trials(config)


def s3r3_linear_position_mean(filter_: StateSpaceSubdivisionFilter) -> np.ndarray:
    """Return the current R3 position mean for an S3+ x R3 S3F state."""

    return np.asarray(filter_.filter_state.linear_mean(), dtype=float)


def s3r3_linear_position_error_stats(filter_: StateSpaceSubdivisionFilter, true_position: np.ndarray) -> tuple[np.ndarray, float]:
    """Return R3 position error and NEES for an S3+ x R3 S3F state."""

    return _linear_position_error_stats(filter_, true_position)


def s3r3_orientation_mode(filter_: StateSpaceSubdivisionFilter) -> np.ndarray:
    """Return the modal quaternion grid point for an S3+ x R3 S3F state."""

    state = filter_.filter_state
    weights = np.asarray(state.gd.grid_values, dtype=float)
    grid = _canonical_quaternions(np.asarray(state.gd.get_grid(), dtype=float))
    return grid[int(np.argmax(weights))]


def _run_variant(
    config: S3R3PrototypeConfig,
    trials: list[dict[str, np.ndarray]],
    grid_size: int,
    variant: str,
) -> dict[str, float | int | str]:
    measurement_cov = np.eye(3) * config.measurement_noise_std**2
    process_noise_cov = np.eye(3) * config.process_noise_std**2
    body_increment = np.asarray(config.body_increment, dtype=float)

    sum_position_sq_error = 0.0
    sum_orientation_mode_error = 0.0
    sum_nees = 0.0
    coverage_hits = 0
    n_metrics = 0
    runtime = 0.0
    last_cell_radius = 0.0

    for trial in trials:
        filter_ = make_s3r3_filter(config, grid_size)
        true_orientation = np.asarray(trial["orientation"], dtype=float)
        true_positions = np.asarray(trial["positions"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)

        for step, measurement in enumerate(measurements):
            likelihood = GaussianDistribution(measurement, measurement_cov, check_validity=False)
            start = perf_counter()
            stats = predict_s3r3_relaxed(
                filter_,
                body_increment,
                variant=variant,
                process_noise_cov=process_noise_cov,
                cell_sample_count=config.cell_sample_count,
            )
            filter_.update(likelihoods_linear=[likelihood])
            runtime += perf_counter() - start
            last_cell_radius = stats.cell_radius_rad

            error, nees = _linear_position_error_stats(filter_, true_positions[step + 1])
            sum_position_sq_error += float(error @ error)
            sum_nees += nees
            coverage_hits += int(nees <= 7.814727903251179)
            sum_orientation_mode_error += _orientation_mode_error(filter_, true_orientation)
            n_metrics += 1

    return {
        "grid_size": grid_size,
        "variant": variant,
        "position_rmse": float(np.sqrt(sum_position_sq_error / n_metrics)),
        "orientation_mode_error_rad": sum_orientation_mode_error / n_metrics,
        "mean_nees": sum_nees / n_metrics,
        "coverage_95": coverage_hits / n_metrics,
        "runtime_ms_per_step": 1000.0 * runtime / n_metrics,
        "cell_radius_rad": last_cell_radius,
        "cell_sample_count": config.cell_sample_count,
        "n_trials": config.n_trials,
        "n_steps": config.n_steps,
    }


def _generate_trials(config: S3R3PrototypeConfig) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(config.seed)
    modes = _canonical_quaternions(np.asarray(config.prior_modes, dtype=float))
    weights = np.asarray(config.prior_weights, dtype=float)
    weights = weights / np.sum(weights)
    body_increment = np.asarray(config.body_increment, dtype=float)
    process_noise_std = float(config.process_noise_std)
    measurement_noise_std = float(config.measurement_noise_std)

    trials = []
    for _ in range(config.n_trials):
        component = int(rng.choice(len(modes), p=weights))
        local_noise = rng.normal(scale=config.orientation_noise_std, size=3)
        orientation = _quaternion_multiply(modes[component], _exp_map_identity(local_noise)[0])
        displacement = _rotate_vectors(orientation, body_increment)[0]
        positions = [np.zeros(3)]
        measurements = []
        for _step in range(config.n_steps):
            process_noise = rng.normal(scale=process_noise_std, size=3)
            next_position = positions[-1] + displacement + process_noise
            positions.append(next_position)
            measurements.append(next_position + rng.normal(scale=measurement_noise_std, size=3))
        trials.append(
            {
                "orientation": orientation,
                "positions": np.asarray(positions),
                "measurements": np.asarray(measurements),
            }
        )
    return trials


def _orientation_prior_values(grid: np.ndarray, config: S3R3PrototypeConfig) -> np.ndarray:
    modes = _canonical_quaternions(np.asarray(config.prior_modes, dtype=float))
    weights = np.asarray(config.prior_weights, dtype=float)
    weights = weights / np.sum(weights)
    values = np.zeros(grid.shape[0], dtype=float)
    for weight, mode in zip(weights, modes, strict=True):
        antipodal_inner = np.abs(grid @ mode)
        values += weight * np.exp(config.prior_kappa * (antipodal_inner**2 - 1.0))
    return values + 1e-12


def _linear_position_error_stats(filter_: StateSpaceSubdivisionFilter, true_position: np.ndarray) -> tuple[np.ndarray, float]:
    state = filter_.filter_state
    error = np.subtract(
        np.asarray(state.linear_mean(), dtype=float),
        np.asarray(true_position, dtype=float),
    )
    covariance = _regularized_covariance(np.asarray(state.linear_covariance(), dtype=float), error.size)
    return error, _quadratic_form(error, covariance)


def _regularized_covariance(covariance: np.ndarray, dimension: int) -> np.ndarray:
    return covariance + 1e-10 * np.eye(dimension, dtype=float)


def _quadratic_form(vector: np.ndarray, matrix: np.ndarray) -> float:
    return float(np.dot(vector, np.linalg.solve(matrix, vector)))


def _orientation_mode_error(filter_: StateSpaceSubdivisionFilter, true_orientation: np.ndarray) -> float:
    return s3r3_orientation_distance(s3r3_orientation_mode(filter_), true_orientation)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in METRIC_FIELDNAMES})


def _write_metadata(
    path: Path,
    rows: list[dict[str, float | int | str]],
    config: S3R3PrototypeConfig,
) -> None:
    metadata = {
        "experiment": "s3r3_relaxed_prototype",
        "config": s3r3_prototype_config_to_dict(config),
        "metrics_schema": METRIC_FIELDNAMES,
        "metrics_rows": len(rows),
        "cell_model": "pyrecest.relaxed_s3f_so3.local_tangent_samples",
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    metrics_path: Path,
    plot_paths: list[Path],
    config: S3R3PrototypeConfig,
) -> None:
    best_row = min(rows, key=lambda row: float(row["position_rmse"]))
    lines = [
        "# S3+ x R3 Relaxed S3F Prototype",
        "",
        "This prototype uses PyRecEst's S3+ x R3 relaxed S3F prediction helpers with a hyperhemispherical quaternion grid and one Gaussian R3 position component per grid point.",
        "R1/R2 statistics are provided by `pyrecest.filters.relaxed_s3f_so3` using deterministic local tangent samples around each grid quaternion, not exact S3+ Voronoi-cell integrals.",
        "",
        f"Trials: {config.n_trials}",
        f"Steps per trial: {config.n_steps}",
        f"Grid sizes: {list(config.grid_sizes)}",
        f"Cell sample count: {config.cell_sample_count}",
        f"Metrics file: `{metrics_path.name}`",
        "",
        "## Best Row",
        "",
        (
            f"`{VARIANT_LABELS[str(best_row['variant'])]}` at `{best_row['grid_size']}` cells has RMSE "
            f"`{float(best_row['position_rmse']):.4f}`, NEES `{float(best_row['mean_nees']):.3f}`, "
            f"and runtime `{float(best_row['runtime_ms_per_step']):.3f}` ms/step."
        ),
        "",
        "## Metrics",
        "",
        _format_metrics_table(rows),
        "",
        "Plots:",
        format_plot_list(plot_paths),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metrics_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Variant | Cells | RMSE | Mode error rad | NEES | Coverage | Runtime ms/step |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in sorted(rows, key=lambda item: (int(item["grid_size"]), str(item["variant"]))):
        body.append(
            "| "
            f"{VARIANT_LABELS[str(row['variant'])]} | "
            f"{int(row['grid_size'])} | "
            f"{float(row['position_rmse']):.4f} | "
            f"{float(row['orientation_mode_error_rad']):.3f} | "
            f"{float(row['mean_nees']):.3f} | "
            f"{float(row['coverage_95']):.3f} | "
            f"{float(row['runtime_ms_per_step']):.3f} |"
        )
    return "\n".join([header, separator, *body])


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    specs = [
        ("position_rmse", "Translation RMSE", "s3r3_relaxed_position_rmse.png"),
        ("mean_nees", "Mean Position NEES", "s3r3_relaxed_nees.png"),
    ]
    paths = []
    for metric, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for variant in SUPPORTED_S3R3_VARIANTS:
            grid_sizes, metric_values = _metric_series(rows, variant, metric)
            if not grid_sizes:
                continue
            ax.plot(grid_sizes, metric_values, marker="o", linewidth=1.8, label=VARIANT_LABELS[variant])
        ax.set_xlabel("Number of quaternion grid cells")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        paths.append(save_figure(fig, output_dir, filename))
    return paths


def _metric_series(
    rows: list[dict[str, float | int | str]],
    variant: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    ordered_rows = sorted(
        (row for row in rows if row["variant"] == variant),
        key=lambda row: int(row["grid_size"]),
    )
    return (
        [int(row["grid_size"]) for row in ordered_rows],
        [float(row[metric]) for row in ordered_rows],
    )
