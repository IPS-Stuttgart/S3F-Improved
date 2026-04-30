"""Numerical relaxed S3F prototype for S3+ x R3."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
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
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter

from ..s1r2.plotting import format_plot_list, save_figure


SUPPORTED_S3R3_VARIANTS = ("baseline", "r1", "r1_r2")
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
class S3R3CellStatistics:
    """Numerical R1/R2 statistics for local tangent cells around S3+ grid points."""

    grid: np.ndarray
    cell_radius_rad: float
    body_increment: np.ndarray
    representative_displacements: np.ndarray
    mean_displacements: np.ndarray
    covariance_inflations: np.ndarray


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


def make_s3r3_filter(config: S3R3PrototypeConfig, grid_size: int) -> StateSpaceSubdivisionFilter:
    """Construct an S3+ x R3 S3F state with one Gaussian position component per quaternion grid point."""

    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    gd_uniform = HyperhemisphericalGridDistribution.from_distribution(
        HyperhemisphericalUniformDistribution(3),
        grid_size,
        "leopardi_symm",
    )
    grid = _canonical_quaternions(np.asarray(gd_uniform.get_grid(), dtype=float))
    grid_values = _orientation_prior_values(grid, config)
    gd = HyperhemisphericalGridDistribution(grid, grid_values, enforce_pdf_nonnegative=True)
    gd.normalize_in_place(warn_unnorm=False)

    initial_covariance = np.eye(3) * config.initial_position_std**2
    gaussians = [
        GaussianDistribution(np.zeros(3), initial_covariance.copy(), check_validity=False)
        for _ in range(grid.shape[0])
    ]
    return StateSpaceSubdivisionFilter(StateSpaceSubdivisionGaussianDistribution(gd, gaussians))


def generate_s3r3_trials(config: S3R3PrototypeConfig) -> list[dict[str, np.ndarray]]:
    """Generate reproducible synthetic S3+ x R3 tracking trials."""

    return _generate_trials(config)


def s3r3_cell_statistics(
    grid: np.ndarray,
    body_increment: np.ndarray,
    cell_sample_count: int = 27,
) -> S3R3CellStatistics:
    """Compute numerical R1/R2 displacement statistics for local tangent cells on S3+."""

    if cell_sample_count <= 0:
        raise ValueError("cell_sample_count must be positive.")

    grid = np.ascontiguousarray(_canonical_quaternions(np.asarray(grid, dtype=np.float64)))
    body_increment = np.asarray(body_increment, dtype=np.float64).reshape(3)
    return _cached_s3r3_cell_statistics(
        grid.shape,
        grid.tobytes(),
        tuple(float(value) for value in body_increment),
        int(cell_sample_count),
    )


@lru_cache(maxsize=128)
def _cached_s3r3_cell_statistics(
    grid_shape: tuple[int, ...],
    grid_bytes: bytes,
    body_increment_values: tuple[float, float, float],
    cell_sample_count: int,
) -> S3R3CellStatistics:
    grid = np.frombuffer(grid_bytes, dtype=np.float64).reshape(grid_shape)
    body_increment = np.asarray(body_increment_values, dtype=np.float64)
    return _freeze_cell_statistics(_compute_s3r3_cell_statistics(grid, body_increment, cell_sample_count))


def _compute_s3r3_cell_statistics(
    grid: np.ndarray,
    body_increment: np.ndarray,
    cell_sample_count: int,
) -> S3R3CellStatistics:
    cell_radius = _estimate_cell_radius(grid)
    tangent_offsets = _tangent_cell_offsets(cell_radius, cell_sample_count)
    local_quaternions = _exp_map_identity(tangent_offsets)

    representative_displacements = _rotate_vectors(grid, body_increment)
    mean_displacements = []
    covariance_inflations = []
    for center in grid:
        center_batch = np.repeat(center.reshape(1, 4), local_quaternions.shape[0], axis=0)
        sample_quaternions = _quaternion_multiply(center_batch, local_quaternions)
        displacements = _rotate_vectors(sample_quaternions, body_increment)
        mean_displacement = np.mean(displacements, axis=0)
        centered = displacements - mean_displacement
        covariance = centered.T @ centered / displacements.shape[0]
        mean_displacements.append(mean_displacement)
        covariance_inflations.append(_symmetrize(covariance))

    return S3R3CellStatistics(
        grid=grid,
        cell_radius_rad=cell_radius,
        body_increment=body_increment,
        representative_displacements=representative_displacements,
        mean_displacements=np.asarray(mean_displacements),
        covariance_inflations=np.asarray(covariance_inflations),
    )


def _freeze_cell_statistics(stats: S3R3CellStatistics) -> S3R3CellStatistics:
    for array in (
        stats.grid,
        stats.body_increment,
        stats.representative_displacements,
        stats.mean_displacements,
        stats.covariance_inflations,
    ):
        array.setflags(write=False)
    return stats


def predict_s3r3_relaxed(
    filter_: StateSpaceSubdivisionFilter,
    body_increment: np.ndarray,
    variant: str = "r1_r2",
    process_noise_cov: np.ndarray | None = None,
    cell_sample_count: int = 27,
) -> S3R3CellStatistics:
    """Predict an S3+ x R3 S3F with baseline, R1, or R1+R2 displacement statistics."""

    if variant not in SUPPORTED_S3R3_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")

    state = filter_.filter_state
    if state.lin_dim != 3:
        raise ValueError("predict_s3r3_relaxed requires a 3-D linear state.")

    stats = s3r3_cell_statistics(np.asarray(state.gd.get_grid(), dtype=float), body_increment, cell_sample_count)
    if variant == "baseline":
        displacements = stats.representative_displacements
        covariance_inflations = np.zeros_like(stats.covariance_inflations)
    elif variant == "r1":
        displacements = stats.mean_displacements
        covariance_inflations = np.zeros_like(stats.covariance_inflations)
    else:
        displacements = stats.mean_displacements
        covariance_inflations = stats.covariance_inflations

    q_base = np.zeros((3, 3), dtype=float) if process_noise_cov is None else np.asarray(process_noise_cov, dtype=float)
    if q_base.shape != (3, 3):
        raise ValueError("process_noise_cov must have shape (3, 3).")

    covariance_matrices = np.stack([q_base + covariance_inflations[idx] for idx in range(displacements.shape[0])], axis=2)
    filter_.predict_linear(
        covariance_matrices=covariance_matrices,
        linear_input_vectors=displacements.T,
    )
    return stats


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


def s3r3_orientation_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the antipodal-invariant geodesic distance between two S3+ quaternions."""

    return _geodesic_distance(left, right)


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


def _estimate_cell_radius(grid: np.ndarray) -> float:
    if grid.shape[0] <= 1:
        return np.pi
    distances = _quaternion_distance_matrix(grid)
    distances[distances == 0.0] = np.inf
    nearest = np.min(distances, axis=1)
    return float(max(0.5 * np.median(nearest), 1e-6))


def _tangent_cell_offsets(cell_radius: float, sample_count: int) -> np.ndarray:
    levels = int(np.ceil(sample_count ** (1.0 / 3.0)))
    if levels % 2 == 0:
        levels += 1
    while levels**3 < sample_count:
        levels += 2
    axis_values = np.linspace(-cell_radius, cell_radius, levels)
    mesh = np.stack(np.meshgrid(axis_values, axis_values, axis_values, indexing="ij"), axis=-1).reshape(-1, 3)
    norms = np.linalg.norm(mesh, axis=1)
    order = np.lexsort((mesh[:, 2], mesh[:, 1], mesh[:, 0], norms))
    return mesh[order[:sample_count]]


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


def _canonical_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    original_shape = quaternions.shape
    quaternions = quaternions.reshape(-1, 4)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("quaternions must be nonzero.")
    normalized = quaternions / norms[:, None]
    normalized = np.where(normalized[:, 3:4] < 0.0, -normalized, normalized)
    return normalized.reshape(original_shape)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _canonical_quaternions(left)
    right = _canonical_quaternions(right)
    x1, y1, z1, w1 = left[..., 0], left[..., 1], left[..., 2], left[..., 3]
    x2, y2, z2, w2 = right[..., 0], right[..., 1], right[..., 2], right[..., 3]
    product = np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )
    return _canonical_quaternions(product)


def _exp_map_identity(tangent_vectors: np.ndarray) -> np.ndarray:
    tangent_vectors = np.asarray(tangent_vectors, dtype=float).reshape(-1, 3)
    angles = np.linalg.norm(tangent_vectors, axis=1)
    safe_angles = np.where(angles > 1e-12, angles, 1.0)
    vector_scale = np.where(
        angles > 1e-12,
        np.sin(0.5 * angles) / safe_angles,
        0.5 - angles**2 / 48.0,
    )
    quaternions = np.concatenate(
        (tangent_vectors * vector_scale[:, None], np.cos(0.5 * angles)[:, None]),
        axis=1,
    )
    return _canonical_quaternions(quaternions)


def _quaternion_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = _canonical_quaternions(quaternions).reshape(-1, 4)
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    row_0 = np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), axis=1)
    row_1 = np.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)), axis=1)
    row_2 = np.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)), axis=1)
    return np.stack((row_0, row_1, row_2), axis=1)


def _rotate_vectors(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrices = _quaternion_to_rotation_matrices(quaternions)
    return np.einsum("nij,j->ni", matrices, np.asarray(vector, dtype=float).reshape(3))


def _quaternion_distance_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = _canonical_quaternions(quaternions)
    inner = np.clip(np.abs(quaternions @ quaternions.T), 0.0, 1.0)
    return 2.0 * np.arccos(inner)


def _geodesic_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = _canonical_quaternions(left).reshape(4)
    right = _canonical_quaternions(right).reshape(4)
    inner = float(np.clip(abs(left @ right), 0.0, 1.0))
    return 2.0 * float(np.arccos(inner))


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


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
        "cell_model": "local_tangent_samples",
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
        "This prototype uses a PyRecEst hyperhemispherical quaternion grid and one Gaussian R3 position component per grid point.",
        "R1/R2 statistics are estimated from deterministic local tangent samples around each grid quaternion, not from exact S3+ Voronoi-cell integrals.",
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
    return [int(row["grid_size"]) for row in ordered_rows], [float(row[metric]) for row in ordered_rows]
