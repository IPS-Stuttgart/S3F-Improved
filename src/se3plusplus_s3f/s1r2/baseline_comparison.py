"""Controlled S1 x R2 comparison against non-S3F baseline filters."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.filters.relaxed_s3f_circular import circular_error, rotation_matrix
from scipy.special import i0, i1

from .plotting import format_plot_list, save_figure
from .relaxed_s3f_pilot import (
    VARIANT_LABELS,
    PilotConfig,
    generate_pilot_trials,
    pilot_config_to_dict,
    run_relaxed_s3f_pilot_on_trials,
)


BASELINE_COMPARISON_FIELDNAMES = [
    "filter",
    "variant",
    "grid_size",
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_mean_error_rad",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "particle_count",
    "n_trials",
    "n_steps",
]


@dataclass(frozen=True)
class BaselineComparisonConfig:
    """Configuration for a shared-trial comparison with non-S3F baselines."""

    pilot: PilotConfig = field(
        default_factory=lambda: PilotConfig(
            n_trials=32,
            n_steps=20,
        )
    )
    particle_count: int = 1024
    particle_seed: int = 101
    particle_resample_threshold: float = 0.5


@dataclass(frozen=True)
class ParticleSensitivityConfig:
    """Configuration for S3F-vs-particle accuracy/runtime sensitivity."""

    pilot: PilotConfig = field(
        default_factory=lambda: PilotConfig(
            n_trials=32,
            n_steps=20,
        )
    )
    particle_counts: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192)
    particle_seed: int = 101
    particle_resample_threshold: float = 0.5


@dataclass
class _MetricAccumulator:
    position_sq_error: float = 0.0
    orientation_mode_error: float = 0.0
    orientation_mean_error: float = 0.0
    nees: float = 0.0
    coverage_hits: int = 0
    runtime_s: float = 0.0
    n_metrics: int = 0


def baseline_comparison_config_to_dict(config: BaselineComparisonConfig) -> dict[str, Any]:
    """Return a JSON-serializable baseline comparison config."""

    return {
        "pilot": pilot_config_to_dict(config.pilot),
        "particle_count": config.particle_count,
        "particle_seed": config.particle_seed,
        "particle_resample_threshold": config.particle_resample_threshold,
    }


def particle_sensitivity_config_to_dict(config: ParticleSensitivityConfig) -> dict[str, Any]:
    """Return a JSON-serializable particle sensitivity config."""

    return {
        "pilot": pilot_config_to_dict(config.pilot),
        "particle_counts": list(config.particle_counts),
        "particle_seed": config.particle_seed,
        "particle_resample_threshold": config.particle_resample_threshold,
    }


def run_baseline_comparison(
    config: BaselineComparisonConfig = BaselineComparisonConfig(),
) -> list[dict[str, float | int | str]]:
    """Run S3F, EKF, and bootstrap particle baselines on identical synthetic trials."""

    trials = generate_pilot_trials(config.pilot)
    return run_baseline_comparison_on_trials(config, trials)


def run_baseline_comparison_on_trials(
    config: BaselineComparisonConfig,
    trials: list[dict[str, np.ndarray | float]],
) -> list[dict[str, float | int | str]]:
    """Run the baseline comparison on precomputed shared trials."""

    _validate_config(config)
    rows = [_comparison_row_from_s3f(row) for row in run_relaxed_s3f_pilot_on_trials(config.pilot, trials)]
    rows.append(_run_ekf_baseline(config.pilot, trials))
    rows.append(_run_particle_baseline(config, trials))
    return rows


def run_particle_sensitivity(
    config: ParticleSensitivityConfig = ParticleSensitivityConfig(),
) -> list[dict[str, float | int | str]]:
    """Compare S3F grid sizes against a sweep of bootstrap particle counts."""

    trials = generate_pilot_trials(config.pilot)
    return run_particle_sensitivity_on_trials(config, trials)


def run_particle_sensitivity_on_trials(
    config: ParticleSensitivityConfig,
    trials: list[dict[str, np.ndarray | float]],
) -> list[dict[str, float | int | str]]:
    """Run the particle sensitivity benchmark on precomputed shared trials."""

    _validate_particle_sensitivity_config(config)
    rows = [_comparison_row_from_s3f(row) for row in run_relaxed_s3f_pilot_on_trials(config.pilot, trials)]
    rows.extend(
        _run_particle_baseline(
            BaselineComparisonConfig(
                pilot=config.pilot,
                particle_count=particle_count,
                particle_seed=config.particle_seed,
                particle_resample_threshold=config.particle_resample_threshold,
            ),
            trials,
        )
        for particle_count in config.particle_counts
    )
    return rows


def write_baseline_comparison_outputs(
    output_dir: Path,
    config: BaselineComparisonConfig = BaselineComparisonConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the baseline comparison and write CSV, optional plots, metadata, and a short note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_baseline_comparison(config)

    metrics_path = output_dir / "baseline_comparison_metrics.csv"
    _write_csv(metrics_path, rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_bar_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "baseline_comparison_note.md"
    _write_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
    return outputs


def write_particle_sensitivity_outputs(
    output_dir: Path,
    config: ParticleSensitivityConfig = ParticleSensitivityConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run particle sensitivity and write CSV, optional plots, metadata, and a short note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_particle_sensitivity(config)

    metrics_path = output_dir / "particle_sensitivity_metrics.csv"
    _write_csv(metrics_path, rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_sensitivity_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "particle_sensitivity_note.md"
    _write_particle_sensitivity_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_particle_sensitivity_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: BaselineComparisonConfig) -> None:
    if config.particle_count <= 0:
        raise ValueError("particle_count must be positive.")
    if not 0.0 < config.particle_resample_threshold <= 1.0:
        raise ValueError("particle_resample_threshold must be in (0, 1].")


def _validate_particle_sensitivity_config(config: ParticleSensitivityConfig) -> None:
    if not config.particle_counts:
        raise ValueError("particle_counts must not be empty.")
    if min(config.particle_counts) <= 0:
        raise ValueError("particle_counts must be positive.")
    if not 0.0 < config.particle_resample_threshold <= 1.0:
        raise ValueError("particle_resample_threshold must be in (0, 1].")


def _comparison_row_from_s3f(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    return {
        "filter": "s3f",
        "variant": row["variant"],
        "grid_size": row["grid_size"],
        "position_rmse": row["position_rmse"],
        "orientation_mode_error_rad": row["orientation_mode_error_rad"],
        "orientation_mean_error_rad": row["orientation_mean_error_rad"],
        "mean_nees": row["mean_nees"],
        "coverage_95": row["coverage_95"],
        "runtime_ms_per_step": row["runtime_ms_per_step"],
        "particle_count": "",
        "n_trials": row["n_trials"],
        "n_steps": row["n_steps"],
    }


def _run_ekf_baseline(
    pilot_config: PilotConfig,
    trials: list[dict[str, np.ndarray | float]],
) -> dict[str, float | int | str]:
    measurement_cov = np.eye(2) * pilot_config.measurement_noise_std**2
    process_noise_cov = np.eye(2) * pilot_config.process_noise_std**2
    accumulator = _MetricAccumulator()

    for trial in trials:
        mean, covariance = _initial_ekf_state(pilot_config)
        true_angle, true_positions, measurements = _trial_contents(trial)

        for step, measurement in enumerate(measurements):
            start = perf_counter()
            mean, covariance = _ekf_predict(mean, covariance, np.asarray(pilot_config.body_increment, dtype=float), process_noise_cov)
            mean, covariance = _ekf_update(mean, covariance, measurement, measurement_cov)
            accumulator.runtime_s += perf_counter() - start

            position_error = mean[1:] - true_positions[step + 1]
            position_covariance = covariance[1:, 1:]
            _accumulate_metrics(accumulator, position_error, position_covariance, mean[0], mean[0], true_angle)

    return _row_from_accumulator(
        filter_name="ekf",
        variant="single_gaussian",
        grid_size="",
        particle_count="",
        accumulator=accumulator,
        config=pilot_config,
    )


def _run_particle_baseline(
    config: BaselineComparisonConfig,
    trials: list[dict[str, np.ndarray | float]],
) -> dict[str, float | int | str]:
    pilot_config = config.pilot
    rng = np.random.default_rng(config.particle_seed)
    measurement_cov = np.eye(2) * pilot_config.measurement_noise_std**2
    process_noise_cov = np.eye(2) * pilot_config.process_noise_std**2
    accumulator = _MetricAccumulator()

    for trial in trials:
        particles, weights = _initial_particles(pilot_config, config.particle_count, rng)
        true_angle, true_positions, measurements = _trial_contents(trial)

        for step, measurement in enumerate(measurements):
            start = perf_counter()
            particles[:, 1:] += _particle_displacements(particles[:, 0], pilot_config, rng, process_noise_cov)
            weights = _update_particle_weights(particles, weights, measurement, measurement_cov)
            particles, weights = _resample_particles(particles, weights, rng, config.particle_resample_threshold)
            accumulator.runtime_s += perf_counter() - start

            position_mean, position_covariance = _particle_position_stats(particles, weights)
            position_error = position_mean - true_positions[step + 1]
            mode_angle = _particle_orientation_mode(particles[:, 0], weights)
            mean_angle = _particle_orientation_mean(particles[:, 0], weights)
            _accumulate_metrics(accumulator, position_error, position_covariance, mode_angle, mean_angle, true_angle)

    return _row_from_accumulator(
        filter_name="particle_filter",
        variant="bootstrap",
        grid_size="",
        particle_count=config.particle_count,
        accumulator=accumulator,
        config=pilot_config,
    )


def _trial_contents(trial: dict[str, np.ndarray | float]) -> tuple[float, np.ndarray, np.ndarray]:
    return (
        float(trial["angle"]),
        np.asarray(trial["positions"], dtype=float),
        np.asarray(trial["measurements"], dtype=float),
    )


def _initial_ekf_state(config: PilotConfig) -> tuple[np.ndarray, np.ndarray]:
    orientation_mean, orientation_variance = _orientation_prior_moment_approximation(config)
    mean = np.array([orientation_mean, 0.0, 0.0], dtype=float)
    covariance = np.diag(
        [
            orientation_variance,
            config.initial_position_std**2,
            config.initial_position_std**2,
        ]
    )
    return mean, covariance


def _orientation_prior_moment_approximation(config: PilotConfig) -> tuple[float, float]:
    weights = np.asarray(config.prior_weights, dtype=float)
    weights = weights / np.sum(weights)
    modes = np.asarray(config.prior_modes, dtype=float)
    resultant = float(i1(config.prior_kappa) / i0(config.prior_kappa)) * np.sum(weights * np.exp(1j * modes))
    mean = float(np.angle(resultant))
    resultant_length = float(np.clip(np.abs(resultant), 1e-6, 1.0))
    variance = float(np.clip(-2.0 * np.log(resultant_length), 1e-4, np.pi**2))
    return mean, variance


def _ekf_predict(
    mean: np.ndarray,
    covariance: np.ndarray,
    body_increment: np.ndarray,
    process_noise_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta = float(mean[0])
    displacement = rotation_matrix(theta) @ body_increment
    derivative = np.array(
        [
            -np.sin(theta) * body_increment[0] - np.cos(theta) * body_increment[1],
            np.cos(theta) * body_increment[0] - np.sin(theta) * body_increment[1],
        ]
    )

    predicted_mean = mean.copy()
    predicted_mean[1:] += displacement
    predicted_mean[0] = _wrap_angle(predicted_mean[0])

    jacobian = np.eye(3)
    jacobian[1:, 0] = derivative
    process_covariance = np.zeros((3, 3))
    process_covariance[1:, 1:] = process_noise_cov
    predicted_covariance = jacobian @ covariance @ jacobian.T + process_covariance
    return predicted_mean, _symmetrize(predicted_covariance)


def _ekf_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    measurement_matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    innovation = np.asarray(measurement, dtype=float) - measurement_matrix @ mean
    innovation_covariance = measurement_matrix @ covariance @ measurement_matrix.T + measurement_cov
    gain = covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
    updated_mean = mean + gain @ innovation
    updated_mean[0] = _wrap_angle(updated_mean[0])

    identity = np.eye(3)
    residual_matrix = identity - gain @ measurement_matrix
    updated_covariance = residual_matrix @ covariance @ residual_matrix.T + gain @ measurement_cov @ gain.T
    return updated_mean, _symmetrize(updated_covariance)


def _initial_particles(
    config: PilotConfig,
    particle_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(config.prior_weights, dtype=float)
    weights = weights / np.sum(weights)
    components = rng.choice(len(config.prior_modes), size=particle_count, p=weights)
    orientations = rng.vonmises(np.asarray(config.prior_modes, dtype=float)[components], config.prior_kappa)
    positions = rng.normal(0.0, config.initial_position_std, size=(particle_count, 2))
    particles = np.column_stack([orientations, positions])
    particle_weights = np.full(particle_count, 1.0 / particle_count)
    return particles, particle_weights


def _particle_displacements(
    angles: np.ndarray,
    config: PilotConfig,
    rng: np.random.Generator,
    process_noise_cov: np.ndarray,
) -> np.ndarray:
    body_increment = np.asarray(config.body_increment, dtype=float)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    displacements = np.column_stack(
        [
            cos_angles * body_increment[0] - sin_angles * body_increment[1],
            sin_angles * body_increment[0] + cos_angles * body_increment[1],
        ]
    )
    return displacements + rng.multivariate_normal(np.zeros(2), process_noise_cov, size=angles.shape[0])


def _update_particle_weights(
    particles: np.ndarray,
    weights: np.ndarray,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
) -> np.ndarray:
    innovation = particles[:, 1:] - np.asarray(measurement, dtype=float)
    precision = np.linalg.inv(measurement_cov)
    log_likelihood = -0.5 * np.einsum("ni,ij,nj->n", innovation, precision, innovation)
    log_weights = np.log(weights + np.finfo(float).tiny) + log_likelihood
    log_weights -= np.max(log_weights)
    updated_weights = np.exp(log_weights)
    weight_sum = np.sum(updated_weights)
    if weight_sum <= 0.0:
        return np.full_like(weights, 1.0 / weights.shape[0])
    return updated_weights / weight_sum


def _resample_particles(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    effective_sample_size = 1.0 / float(np.sum(weights**2))
    if effective_sample_size >= threshold * weights.shape[0]:
        return particles, weights

    indices = rng.choice(weights.shape[0], size=weights.shape[0], replace=True, p=weights)
    resampled = particles[indices].copy()
    return resampled, np.full_like(weights, 1.0 / weights.shape[0])


def _particle_position_stats(particles: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = particles[:, 1:]
    mean = weights @ positions
    centered = positions - mean
    covariance = centered.T @ (centered * weights[:, None])
    return mean, _symmetrize(covariance)


def _particle_orientation_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    return float(np.arctan2(weights @ np.sin(angles), weights @ np.cos(angles)))


def _particle_orientation_mode(angles: np.ndarray, weights: np.ndarray) -> float:
    hist, edges = np.histogram(np.mod(angles, 2.0 * np.pi), bins=72, range=(0.0, 2.0 * np.pi), weights=weights)
    index = int(np.argmax(hist))
    return float(0.5 * (edges[index] + edges[index + 1]))


def _accumulate_metrics(
    accumulator: _MetricAccumulator,
    position_error: np.ndarray,
    position_covariance: np.ndarray,
    mode_angle: float,
    mean_angle: float,
    true_angle: float,
) -> None:
    covariance_reg = position_covariance + 1e-10 * np.eye(position_error.shape[0])
    nees = float(position_error @ np.linalg.solve(covariance_reg, position_error))
    accumulator.position_sq_error += float(position_error @ position_error)
    accumulator.orientation_mode_error += circular_error(mode_angle, true_angle)
    accumulator.orientation_mean_error += circular_error(mean_angle, true_angle)
    accumulator.nees += nees
    accumulator.coverage_hits += int(nees <= 5.991464547107979)
    accumulator.n_metrics += 1


def _row_from_accumulator(
    filter_name: str,
    variant: str,
    grid_size: int | str,
    particle_count: int | str,
    accumulator: _MetricAccumulator,
    config: PilotConfig,
) -> dict[str, float | int | str]:
    return {
        "filter": filter_name,
        "variant": variant,
        "grid_size": grid_size,
        "position_rmse": float(np.sqrt(accumulator.position_sq_error / accumulator.n_metrics)),
        "orientation_mode_error_rad": accumulator.orientation_mode_error / accumulator.n_metrics,
        "orientation_mean_error_rad": accumulator.orientation_mean_error / accumulator.n_metrics,
        "mean_nees": accumulator.nees / accumulator.n_metrics,
        "coverage_95": accumulator.coverage_hits / accumulator.n_metrics,
        "runtime_ms_per_step": 1000.0 * accumulator.runtime_s / accumulator.n_metrics,
        "particle_count": particle_count,
        "n_trials": config.n_trials,
        "n_steps": config.n_steps,
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BASELINE_COMPARISON_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: Path, rows: list[dict[str, float | int | str]], config: BaselineComparisonConfig) -> None:
    metadata = dict(
        config=baseline_comparison_config_to_dict(config),
        experiment="baseline_comparison",
        metrics_rows=len(rows),
        metrics_schema=BASELINE_COMPARISON_FIELDNAMES,
    )
    serialized_metadata = json.dumps(metadata, indent=2, sort_keys=True)
    path.write_text(f"{serialized_metadata}\n", encoding="utf-8")


def _write_particle_sensitivity_metadata(path: Path, rows: list[dict[str, float | int | str]], config: ParticleSensitivityConfig) -> None:
    metadata = dict(
        config=particle_sensitivity_config_to_dict(config),
        experiment="particle_sensitivity",
        metrics_rows=len(rows),
        metrics_schema=BASELINE_COMPARISON_FIELDNAMES,
    )
    serialized_metadata = json.dumps(metadata, indent=2, sort_keys=True)
    path.write_text(f"{serialized_metadata}\n", encoding="utf-8")


def _write_bar_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    plot_specs = [
        ("position_rmse", "Translation RMSE", "baseline_translation_rmse.png"),
        ("orientation_mean_error_rad", "Mean Orientation Error [rad]", "baseline_orientation_error.png"),
        ("mean_nees", "Mean Position NEES", "baseline_mean_nees.png"),
        ("runtime_ms_per_step", "Runtime [ms/step]", "baseline_runtime.png"),
    ]
    paths = []
    labels = [_row_label(row) for row in rows]
    for metric, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.bar(labels, [float(row[metric]) for row in rows])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)
        path = save_figure(fig, output_dir, filename)
        paths.append(path)
    return paths


def _write_sensitivity_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    plot_specs = [
        ("position_rmse", "Translation RMSE", "particle_sensitivity_rmse_runtime.png"),
        ("orientation_mean_error_rad", "Mean Orientation Error [rad]", "particle_sensitivity_orientation_runtime.png"),
        ("coverage_95", "Empirical 95% Coverage", "particle_sensitivity_coverage_runtime.png"),
        ("runtime_ms_per_step", "Runtime [ms/step]", "particle_sensitivity_runtime_resource.png"),
    ]
    paths = []
    for metric, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        if metric == "runtime_ms_per_step":
            _plot_runtime_by_resource(ax, rows)
            ax.set_ylabel(ylabel)
        else:
            _plot_metric_by_runtime(ax, rows, metric)
            ax.set_xlabel("Runtime [ms/step]")
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = save_figure(fig, output_dir, filename)
        paths.append(path)
    return paths


def _plot_metric_by_runtime(ax, rows: list[dict[str, float | int | str]], metric: str) -> None:
    for variant in VARIANT_LABELS:
        variant_rows = _s3f_variant_rows(rows, variant)
        ax.plot(
            [float(row["runtime_ms_per_step"]) for row in variant_rows],
            [float(row[metric]) for row in variant_rows],
            marker="o",
            linewidth=1.7,
            label=VARIANT_LABELS[variant],
        )

    particle_rows = _particle_rows(rows)
    ax.plot(
        [float(row["runtime_ms_per_step"]) for row in particle_rows],
        [float(row[metric]) for row in particle_rows],
        marker="s",
        linewidth=1.7,
        label="Particle filter",
    )


def _plot_runtime_by_resource(ax, rows: list[dict[str, float | int | str]]) -> None:
    for variant in VARIANT_LABELS:
        variant_rows = _s3f_variant_rows(rows, variant)
        ax.plot(
            [int(row["grid_size"]) for row in variant_rows],
            [float(row["runtime_ms_per_step"]) for row in variant_rows],
            marker="o",
            linewidth=1.7,
            label=f"{VARIANT_LABELS[variant]} by cells",
        )

    particle_rows = _particle_rows(rows)
    ax.plot(
        [int(row["particle_count"]) for row in particle_rows],
        [float(row["runtime_ms_per_step"]) for row in particle_rows],
        marker="s",
        linewidth=1.7,
        label="Particle filter by particles",
    )
    ax.set_xlabel("Cells or particles")


def _s3f_variant_rows(rows: list[dict[str, float | int | str]], variant: str) -> list[dict[str, float | int | str]]:
    return sorted(
        [row for row in rows if row["filter"] == "s3f" and row["variant"] == variant],
        key=lambda row: int(row["grid_size"]),
    )


def _particle_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    return sorted(
        [row for row in rows if row["filter"] == "particle_filter"],
        key=lambda row: int(row["particle_count"]),
    )


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    metrics_path: Path,
    plot_paths: list[Path],
    config: BaselineComparisonConfig,
) -> None:
    best_rmse = min(rows, key=lambda row: float(row["position_rmse"]))
    best_coverage = min(rows, key=lambda row: abs(float(row["coverage_95"]) - 0.95))
    lines = [
        "# Baseline Comparison Note",
        "",
        "Shared synthetic S1 x R2 trials were evaluated with relaxed S3F variants,",
        "a single-Gaussian EKF over `[theta, x, y]`, and a bootstrap particle filter.",
        "",
        f"Trials: {config.pilot.n_trials}",
        f"Steps per trial: {config.pilot.n_steps}",
        f"S3F grid sizes: {list(config.pilot.grid_sizes)}",
        f"Particle count: {config.particle_count}",
        f"Metrics file: `{metrics_path.name}`",
        "",
        f"Best translation RMSE: `{_row_label(best_rmse)}` at `{float(best_rmse['position_rmse']):.4f}`.",
        (
            "Coverage closest to 95%: "
            f"`{_row_label(best_coverage)}` at `{float(best_coverage['coverage_95']):.3f}` "
            f"with mean NEES `{float(best_coverage['mean_nees']):.3f}`."
        ),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "The EKF baseline is deliberately local and single-modal, while the particle",
        "filter can carry multiple orientation hypotheses at a higher sampling cost.",
        "This remains a synthetic model comparison rather than a drone or VIO benchmark.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_particle_sensitivity_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    metrics_path: Path,
    plot_paths: list[Path],
    config: ParticleSensitivityConfig,
) -> None:
    s3f_rows = [row for row in rows if row["filter"] == "s3f"]
    particle_rows = [row for row in rows if row["filter"] == "particle_filter"]
    best_s3f = min(s3f_rows, key=lambda row: float(row["position_rmse"]))
    best_particle = min(particle_rows, key=lambda row: float(row["position_rmse"]))
    best_coverage = min(rows, key=lambda row: abs(float(row["coverage_95"]) - 0.95))
    lines = [
        "# Particle Sensitivity Note",
        "",
        "Shared synthetic S1 x R2 trials were evaluated with relaxed S3F grids and",
        "a bootstrap particle filter over multiple particle counts.",
        "",
        f"Trials: {config.pilot.n_trials}",
        f"Steps per trial: {config.pilot.n_steps}",
        f"S3F grid sizes: {list(config.pilot.grid_sizes)}",
        f"Particle counts: {list(config.particle_counts)}",
        f"Metrics file: `{metrics_path.name}`",
        "",
        f"Best S3F RMSE: `{_row_label(best_s3f)}` at `{float(best_s3f['position_rmse']):.4f}`.",
        f"Best particle-filter RMSE: `{_row_label(best_particle)}` at `{float(best_particle['position_rmse']):.4f}`.",
        (
            "Coverage closest to 95%: "
            f"`{_row_label(best_coverage)}` at `{float(best_coverage['coverage_95']):.3f}` "
            f"with mean NEES `{float(best_coverage['mean_nees']):.3f}`."
        ),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "This benchmark is intended to show the accuracy-runtime tradeoff across",
        "particle counts, not to establish final dominance of one filter family.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_label(row: dict[str, float | int | str]) -> str:
    if row["filter"] == "s3f":
        return f"{VARIANT_LABELS[str(row['variant'])]} ({row['grid_size']} cells)"
    if row["filter"] == "ekf":
        return "EKF"
    if row["filter"] == "particle_filter":
        return f"Particle filter ({row['particle_count']})"
    return f"{row['filter']}:{row['variant']}"


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
