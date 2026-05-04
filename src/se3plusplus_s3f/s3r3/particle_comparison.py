"""S3+ x R3 comparison against a bootstrap particle filter."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..s1r2.plotting import format_plot_list, save_figure
from .relaxed_s3f_prototype import (
    SUPPORTED_S3R3_VARIANTS,
    VARIANT_LABELS,
    S3R3PrototypeConfig,
    _canonical_quaternions,
    _exp_map_identity,
    _quaternion_multiply,
    _rotate_vectors,
    _symmetrize,
    generate_s3r3_trials,
    s3r3_orientation_distance,
    validate_s3r3_prototype_config,
)
from .stress_sweep import S3R3StressSweepConfig, _scenario_config, _scenario_id, run_s3r3_stress_sweep


S3R3_PARTICLE_METRIC_FIELDNAMES = [
    "scenario_id",
    "prior_kappa",
    "body_increment_scale",
    "body_increment_norm",
    "filter",
    "variant",
    "grid_size",
    "particle_count",
    "resource_count",
    "position_rmse",
    "orientation_error_rad",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "cell_radius_rad",
    "cell_sample_count",
    "n_trials",
    "n_steps",
]

S3R3_PARTICLE_COMPARISON_FIELDNAMES = [
    "scenario_id",
    "prior_kappa",
    "body_increment_scale",
    "body_increment_norm",
    "best_s3f_grid_size",
    "best_s3f_position_rmse",
    "best_s3f_mean_nees",
    "best_s3f_runtime_ms_per_step",
    "best_particle_count",
    "best_particle_position_rmse",
    "best_particle_mean_nees",
    "best_particle_runtime_ms_per_step",
    "best_particle_rmse_ratio",
    "best_particle_runtime_ratio",
    "nearest_particle_count",
    "nearest_particle_position_rmse",
    "nearest_particle_runtime_ms_per_step",
    "nearest_particle_rmse_ratio",
    "nearest_particle_runtime_ratio",
    "no_slower_particle_count",
    "no_slower_particle_position_rmse",
    "no_slower_particle_runtime_ms_per_step",
    "no_slower_particle_rmse_ratio",
    "s3f_beats_best_particle_rmse",
    "s3f_faster_than_best_particle",
    "s3f_beats_nearest_particle_rmse",
    "no_slower_particle_dominates_s3f",
]

PARTICLE_VARIANT = "bootstrap"
PARTICLE_LABEL = "Bootstrap PF"
S3F_COMPARISON_VARIANT = "r1_r2"


@dataclass(frozen=True)
class S3R3ParticleComparisonResult:
    """Container for S3R3 S3F-vs-particle comparison outputs."""

    metrics: list[dict[str, float | int | str]]
    comparisons: list[dict[str, float | int | str | bool]]


@dataclass(frozen=True)
class S3R3ParticleComparisonConfig:
    """Configuration for comparing relaxed S3F grids with particle counts."""

    prototype: S3R3PrototypeConfig = field(
        default_factory=lambda: S3R3PrototypeConfig(
            grid_sizes=(8, 16, 32),
            variants=SUPPORTED_S3R3_VARIANTS,
            n_trials=4,
            n_steps=5,
            seed=37,
        )
    )
    prior_kappas: tuple[float, ...] = (1.5, 3.0, 8.0)
    body_increment_scales: tuple[float, ...] = (0.5, 1.0, 1.5)
    particle_counts: tuple[int, ...] = (128, 512, 2048)
    particle_seed: int = 211
    particle_resample_threshold: float = 0.5


@dataclass
class _MetricAccumulator:
    position_sq_error: float = 0.0
    orientation_error: float = 0.0
    nees: float = 0.0
    coverage_hits: int = 0
    runtime_s: float = 0.0
    n_metrics: int = 0


def s3r3_particle_comparison_config_to_dict(config: S3R3ParticleComparisonConfig) -> dict[str, Any]:
    """Return a JSON-serializable particle-comparison config."""

    return json.loads(json.dumps(asdict(config)))


def run_s3r3_particle_comparison(
    config: S3R3ParticleComparisonConfig = S3R3ParticleComparisonConfig(),
) -> S3R3ParticleComparisonResult:
    """Run relaxed S3F and bootstrap particle rows on the same S3R3 stress scenarios."""

    _validate_config(config)
    stress_result = run_s3r3_stress_sweep(
        S3R3StressSweepConfig(
            prototype=config.prototype,
            prior_kappas=config.prior_kappas,
            body_increment_scales=config.body_increment_scales,
        )
    )
    metrics = [_s3f_metric_row(row) for row in stress_result.metrics]

    for prior_kappa in config.prior_kappas:
        for body_increment_scale in config.body_increment_scales:
            scenario_config = _scenario_config(config.prototype, prior_kappa, body_increment_scale)
            scenario_id = _scenario_id(prior_kappa, body_increment_scale)
            trials = generate_s3r3_trials(scenario_config)
            for particle_count in config.particle_counts:
                metrics.append(_run_particle_row(config, scenario_config, trials, scenario_id, prior_kappa, body_increment_scale, particle_count))

    comparisons = _build_comparison_rows(metrics)
    return S3R3ParticleComparisonResult(metrics=metrics, comparisons=comparisons)


def write_s3r3_particle_comparison_outputs(
    output_dir: Path,
    config: S3R3ParticleComparisonConfig = S3R3ParticleComparisonConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the S3R3 particle comparison and write CSVs, plots, metadata, and a note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s3r3_particle_comparison(config)

    metrics_path = output_dir / "s3r3_particle_comparison_metrics.csv"
    _write_csv(metrics_path, result.metrics, S3R3_PARTICLE_METRIC_FIELDNAMES)

    comparisons_path = output_dir / "s3r3_particle_comparison_summary.csv"
    _write_csv(comparisons_path, result.comparisons, S3R3_PARTICLE_COMPARISON_FIELDNAMES)

    outputs = {"metrics": metrics_path, "summary": comparisons_path}
    plot_paths = _write_plots(output_dir, result.metrics, result.comparisons) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_particle_comparison_note.md"
    _write_note(note_path, result, metrics_path, comparisons_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: S3R3ParticleComparisonConfig) -> None:
    validate_s3r3_prototype_config(config.prototype, required_variants=SUPPORTED_S3R3_VARIANTS)
    if not config.prior_kappas:
        raise ValueError("prior_kappas must not be empty.")
    if min(config.prior_kappas) <= 0.0:
        raise ValueError("all prior_kappas must be positive.")
    if not config.body_increment_scales:
        raise ValueError("body_increment_scales must not be empty.")
    if min(config.body_increment_scales) <= 0.0:
        raise ValueError("all body_increment_scales must be positive.")
    if not config.particle_counts:
        raise ValueError("particle_counts must not be empty.")
    if min(config.particle_counts) <= 0:
        raise ValueError("particle_counts must be positive.")
    if not 0.0 < config.particle_resample_threshold <= 1.0:
        raise ValueError("particle_resample_threshold must be in (0, 1].")


def _s3f_metric_row(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    grid_size = int(row["grid_size"])
    return {
        "scenario_id": row["scenario_id"],
        "prior_kappa": row["prior_kappa"],
        "body_increment_scale": row["body_increment_scale"],
        "body_increment_norm": row["body_increment_norm"],
        "filter": "s3f",
        "variant": row["variant"],
        "grid_size": grid_size,
        "particle_count": "",
        "resource_count": grid_size,
        "position_rmse": row["position_rmse"],
        "orientation_error_rad": row["orientation_mode_error_rad"],
        "mean_nees": row["mean_nees"],
        "coverage_95": row["coverage_95"],
        "runtime_ms_per_step": row["runtime_ms_per_step"],
        "cell_radius_rad": row["cell_radius_rad"],
        "cell_sample_count": row["cell_sample_count"],
        "n_trials": row["n_trials"],
        "n_steps": row["n_steps"],
    }


def _run_particle_row(
    config: S3R3ParticleComparisonConfig,
    scenario_config: S3R3PrototypeConfig,
    trials: list[dict[str, np.ndarray]],
    scenario_id: str,
    prior_kappa: float,
    body_increment_scale: float,
    particle_count: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(config.particle_seed + int(particle_count) + _particle_scenario_seed_offset(prior_kappa, body_increment_scale))
    measurement_cov = np.eye(3) * scenario_config.measurement_noise_std**2
    process_noise_cov = np.eye(3) * scenario_config.process_noise_std**2
    body_increment = np.asarray(scenario_config.body_increment, dtype=float)
    accumulator = _MetricAccumulator()

    for trial in trials:
        orientations, positions, weights = _initial_particles(scenario_config, particle_count, rng)
        true_orientation = np.asarray(trial["orientation"], dtype=float)
        true_positions = np.asarray(trial["positions"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)

        for step, measurement in enumerate(measurements):
            start = perf_counter()
            positions += _rotate_vectors(orientations, body_increment) + rng.multivariate_normal(np.zeros(3), process_noise_cov, size=particle_count)
            weights = _update_particle_weights(positions, weights, measurement, measurement_cov)
            _accumulate_particle_metrics(accumulator, orientations, positions, weights, true_positions[step + 1], true_orientation)
            orientations, positions, weights = _resample_particles(orientations, positions, weights, rng, config.particle_resample_threshold)
            accumulator.runtime_s += perf_counter() - start

    body_increment_norm = float(np.linalg.norm(body_increment))
    return {
        "scenario_id": scenario_id,
        "prior_kappa": prior_kappa,
        "body_increment_scale": body_increment_scale,
        "body_increment_norm": body_increment_norm,
        "filter": "particle_filter",
        "variant": PARTICLE_VARIANT,
        "grid_size": "",
        "particle_count": particle_count,
        "resource_count": particle_count,
        "position_rmse": float(np.sqrt(accumulator.position_sq_error / accumulator.n_metrics)),
        "orientation_error_rad": accumulator.orientation_error / accumulator.n_metrics,
        "mean_nees": accumulator.nees / accumulator.n_metrics,
        "coverage_95": accumulator.coverage_hits / accumulator.n_metrics,
        "runtime_ms_per_step": 1000.0 * accumulator.runtime_s / accumulator.n_metrics,
        "cell_radius_rad": "",
        "cell_sample_count": "",
        "n_trials": scenario_config.n_trials,
        "n_steps": scenario_config.n_steps,
    }


def _particle_scenario_seed_offset(prior_kappa: float, body_increment_scale: float) -> int:
    return int(round(10_000.0 * prior_kappa + 1000.0 * body_increment_scale))


def _initial_particles(
    config: S3R3PrototypeConfig,
    particle_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = _canonical_quaternions(np.asarray(config.prior_modes, dtype=float))
    prior_weights = np.asarray(config.prior_weights, dtype=float)
    prior_weights = prior_weights / np.sum(prior_weights)
    components = rng.choice(len(modes), size=particle_count, p=prior_weights)

    tangent_std = 1.0 / np.sqrt(max(float(config.prior_kappa), 1e-9))
    local_quaternions = _exp_map_identity(rng.normal(scale=tangent_std, size=(particle_count, 3)))
    orientations = _quaternion_multiply(modes[components], local_quaternions)
    positions = rng.normal(0.0, config.initial_position_std, size=(particle_count, 3))
    weights = np.full(particle_count, 1.0 / particle_count)
    return orientations, positions, weights


def _update_particle_weights(
    positions: np.ndarray,
    weights: np.ndarray,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
) -> np.ndarray:
    innovation = positions - np.asarray(measurement, dtype=float)
    precision = np.linalg.inv(measurement_cov)
    log_likelihood = -0.5 * np.einsum("ni,ij,nj->n", innovation, precision, innovation)
    log_weights = np.log(weights + np.finfo(float).tiny) + log_likelihood
    log_weights -= np.max(log_weights)
    updated = np.exp(log_weights)
    weight_sum = float(np.sum(updated))
    if weight_sum <= 0.0:
        return np.full_like(weights, 1.0 / weights.shape[0])
    return updated / weight_sum


def _accumulate_particle_metrics(
    accumulator: _MetricAccumulator,
    orientations: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    true_position: np.ndarray,
    true_orientation: np.ndarray,
) -> None:
    position_mean, position_covariance = _weighted_position_stats(positions, weights)
    position_error = position_mean - np.asarray(true_position, dtype=float)
    covariance_reg = position_covariance + 1e-10 * np.eye(3)
    nees = float(position_error @ np.linalg.solve(covariance_reg, position_error))

    accumulator.position_sq_error += float(position_error @ position_error)
    accumulator.orientation_error += s3r3_orientation_distance(_weighted_quaternion_mean(orientations, weights), true_orientation)
    accumulator.nees += nees
    accumulator.coverage_hits += int(nees <= 7.814727903251179)
    accumulator.n_metrics += 1


def _weighted_position_stats(positions: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = weights @ positions
    centered = positions - mean
    covariance = centered.T @ (centered * weights[:, None])
    return mean, _symmetrize(covariance)


def _weighted_quaternion_mean(orientations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    canonical = _canonical_quaternions(orientations).reshape(-1, 4)
    scatter = canonical.T @ (canonical * weights[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    return _canonical_quaternions(eigenvectors[:, int(np.argmax(eigenvalues))])


def _resample_particles(
    orientations: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effective_sample_size = 1.0 / float(np.sum(weights**2))
    if effective_sample_size >= threshold * weights.shape[0]:
        return orientations, positions, weights

    indices = rng.choice(weights.shape[0], size=weights.shape[0], replace=True, p=weights)
    return orientations[indices].copy(), positions[indices].copy(), np.full_like(weights, 1.0 / weights.shape[0])


def _build_comparison_rows(
    metrics: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str | bool]]:
    comparisons = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in metrics})
    for scenario_id in scenario_ids:
        scenario_rows = [row for row in metrics if str(row["scenario_id"]) == scenario_id]
        r1_r2_rows = [row for row in scenario_rows if row["filter"] == "s3f" and row["variant"] == S3F_COMPARISON_VARIANT]
        particle_rows = [row for row in scenario_rows if row["filter"] == "particle_filter"]
        best_s3f = min(r1_r2_rows, key=lambda row: float(row["position_rmse"]))
        best_particle = min(particle_rows, key=lambda row: float(row["position_rmse"]))
        nearest_particle = min(particle_rows, key=lambda row: abs(float(row["runtime_ms_per_step"]) - float(best_s3f["runtime_ms_per_step"])))
        no_slower_particles = [row for row in particle_rows if float(row["runtime_ms_per_step"]) <= float(best_s3f["runtime_ms_per_step"])]
        best_no_slower = min(no_slower_particles, key=lambda row: float(row["position_rmse"])) if no_slower_particles else None
        comparisons.append(_comparison_row(best_s3f, best_particle, nearest_particle, best_no_slower))
    return comparisons


def _comparison_row(
    best_s3f: dict[str, float | int | str],
    best_particle: dict[str, float | int | str],
    nearest_particle: dict[str, float | int | str],
    best_no_slower: dict[str, float | int | str] | None,
) -> dict[str, float | int | str | bool]:
    best_s3f_rmse = float(best_s3f["position_rmse"])
    best_s3f_runtime = float(best_s3f["runtime_ms_per_step"])
    best_particle_rmse = float(best_particle["position_rmse"])
    best_particle_runtime = float(best_particle["runtime_ms_per_step"])
    nearest_rmse = float(nearest_particle["position_rmse"])
    nearest_runtime = float(nearest_particle["runtime_ms_per_step"])

    no_slower_rmse = "" if best_no_slower is None else float(best_no_slower["position_rmse"])
    no_slower_runtime = "" if best_no_slower is None else float(best_no_slower["runtime_ms_per_step"])
    no_slower_ratio = "" if best_no_slower is None else _ratio(best_s3f_rmse, float(best_no_slower["position_rmse"]))
    no_slower_dominates = bool(
        best_no_slower is not None
        and float(best_no_slower["position_rmse"]) <= best_s3f_rmse
        and float(best_no_slower["runtime_ms_per_step"]) <= best_s3f_runtime
    )

    return {
        "scenario_id": best_s3f["scenario_id"],
        "prior_kappa": best_s3f["prior_kappa"],
        "body_increment_scale": best_s3f["body_increment_scale"],
        "body_increment_norm": best_s3f["body_increment_norm"],
        "best_s3f_grid_size": best_s3f["grid_size"],
        "best_s3f_position_rmse": best_s3f_rmse,
        "best_s3f_mean_nees": best_s3f["mean_nees"],
        "best_s3f_runtime_ms_per_step": best_s3f_runtime,
        "best_particle_count": best_particle["particle_count"],
        "best_particle_position_rmse": best_particle_rmse,
        "best_particle_mean_nees": best_particle["mean_nees"],
        "best_particle_runtime_ms_per_step": best_particle_runtime,
        "best_particle_rmse_ratio": _ratio(best_s3f_rmse, best_particle_rmse),
        "best_particle_runtime_ratio": _ratio(best_s3f_runtime, best_particle_runtime),
        "nearest_particle_count": nearest_particle["particle_count"],
        "nearest_particle_position_rmse": nearest_rmse,
        "nearest_particle_runtime_ms_per_step": nearest_runtime,
        "nearest_particle_rmse_ratio": _ratio(best_s3f_rmse, nearest_rmse),
        "nearest_particle_runtime_ratio": _ratio(best_s3f_runtime, nearest_runtime),
        "no_slower_particle_count": "" if best_no_slower is None else best_no_slower["particle_count"],
        "no_slower_particle_position_rmse": no_slower_rmse,
        "no_slower_particle_runtime_ms_per_step": no_slower_runtime,
        "no_slower_particle_rmse_ratio": no_slower_ratio,
        "s3f_beats_best_particle_rmse": best_s3f_rmse <= best_particle_rmse,
        "s3f_faster_than_best_particle": best_s3f_runtime <= best_particle_runtime,
        "s3f_beats_nearest_particle_rmse": best_s3f_rmse <= nearest_rmse,
        "no_slower_particle_dominates_s3f": no_slower_dominates,
    }


def _ratio(candidate: float, comparator: float) -> float:
    return float(candidate / comparator) if comparator > 0.0 else float("inf")


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _write_metadata(
    path: Path,
    result: S3R3ParticleComparisonResult,
    config: S3R3ParticleComparisonConfig,
) -> None:
    metadata = {
        "config": s3r3_particle_comparison_config_to_dict(config),
        "experiment": "s3r3_particle_comparison",
        "metrics_rows": len(result.metrics),
        "summary_rows": len(result.comparisons),
        "metrics_schema": S3R3_PARTICLE_METRIC_FIELDNAMES,
        "summary_schema": S3R3_PARTICLE_COMPARISON_FIELDNAMES,
        "particle_prior_model": "local_tangent_gaussian_std_1_over_sqrt_kappa",
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    result: S3R3ParticleComparisonResult,
    metrics_path: Path,
    comparisons_path: Path,
    plot_paths: list[Path],
    config: S3R3ParticleComparisonConfig,
) -> None:
    comparisons = result.comparisons
    nearest_wins = sum(bool(row["s3f_beats_nearest_particle_rmse"]) for row in comparisons)
    best_wins = sum(bool(row["s3f_beats_best_particle_rmse"]) for row in comparisons)
    no_slower_dominance = sum(bool(row["no_slower_particle_dominates_s3f"]) for row in comparisons)
    best_scenario = min(comparisons, key=lambda row: float(row["best_particle_rmse_ratio"]))
    worst_scenario = max(comparisons, key=lambda row: float(row["best_particle_rmse_ratio"]))
    lines = [
        "# S3+ x R3 Particle Comparison",
        "",
        "This report compares the best R1+R2 relaxed S3F row in each S3R3 stress scenario against a bootstrap particle filter over several particle counts.",
        "The particle filter samples the continuous orientation prior with a local tangent Gaussian approximation whose standard deviation is `1 / sqrt(kappa)`.",
        "",
        f"Trials per scenario: {config.prototype.n_trials}",
        f"Steps per trial: {config.prototype.n_steps}",
        f"S3F grid sizes: {list(config.prototype.grid_sizes)}",
        f"Particle counts: {list(config.particle_counts)}",
        f"Prior kappas: {list(config.prior_kappas)}",
        f"Body-increment scales: {list(config.body_increment_scales)}",
        f"Metrics file: `{metrics_path.name}`",
        f"Summary file: `{comparisons_path.name}`",
        "",
        "## Headline",
        "",
        f"Best R1+R2 beats the nearest-runtime particle row on RMSE in `{nearest_wins}/{len(comparisons)}` scenarios.",
        f"Best R1+R2 beats the best-RMSE particle row on RMSE in `{best_wins}/{len(comparisons)}` scenarios.",
        f"A no-slower particle row dominates best R1+R2 in `{no_slower_dominance}/{len(comparisons)}` scenarios.",
        (
            f"Strongest R1+R2-vs-best-particle scenario: `{best_scenario['scenario_id']}` "
            f"with RMSE ratio `{float(best_scenario['best_particle_rmse_ratio']):.3f}`."
        ),
        (
            f"Weakest R1+R2-vs-best-particle scenario: `{worst_scenario['scenario_id']}` "
            f"with RMSE ratio `{float(worst_scenario['best_particle_rmse_ratio']):.3f}`."
        ),
        "",
        "## Scenario Summary",
        "",
        _format_summary_table(comparisons),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "This is still a synthetic benchmark. It is intended to show whether the relaxed S3F result survives a non-grid filter comparison, not to replace real-data validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_summary_table(rows: list[dict[str, float | int | str | bool]]) -> str:
    header = "| Scenario | Best S3F cells | Best PF particles | S3F/PF RMSE | S3F/PF runtime | Nearest PF particles | S3F/nearest RMSE |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in rows:
        body.append(
            "| "
            f"{row['scenario_id']} | "
            f"{int(row['best_s3f_grid_size'])} | "
            f"{int(row['best_particle_count'])} | "
            f"{float(row['best_particle_rmse_ratio']):.3f} | "
            f"{float(row['best_particle_runtime_ratio']):.3f} | "
            f"{int(row['nearest_particle_count'])} | "
            f"{float(row['nearest_particle_rmse_ratio']):.3f} |"
        )
    return "\n".join([header, separator, *body])


def _write_plots(
    output_dir: Path,
    metrics: list[dict[str, float | int | str]],
    comparisons: list[dict[str, float | int | str | bool]],
) -> list[Path]:
    return [
        _write_rmse_runtime_plot(output_dir, metrics),
        _write_ratio_heatmap(output_dir, comparisons, "best_particle_rmse_ratio", "Best R1+R2 RMSE / best PF RMSE", "s3r3_particle_best_rmse_ratio.png"),
        _write_ratio_heatmap(output_dir, comparisons, "nearest_particle_rmse_ratio", "Best R1+R2 RMSE / nearest-runtime PF RMSE", "s3r3_particle_nearest_rmse_ratio.png"),
    ]


def _write_rmse_runtime_plot(output_dir: Path, metrics: list[dict[str, float | int | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    mean_rows = _mean_rows_by_method(metrics)
    for variant in SUPPORTED_S3R3_VARIANTS:
        rows = sorted(
            [row for row in mean_rows if row["filter"] == "s3f" and row["variant"] == variant],
            key=lambda row: int(row["resource_count"]),
        )
        ax.plot(
            [float(row["runtime_ms_per_step"]) for row in rows],
            [float(row["position_rmse"]) for row in rows],
            marker="o",
            linewidth=1.7,
            label=VARIANT_LABELS[variant],
        )
    particle_rows = sorted([row for row in mean_rows if row["filter"] == "particle_filter"], key=lambda row: int(row["resource_count"]))
    ax.plot(
        [float(row["runtime_ms_per_step"]) for row in particle_rows],
        [float(row["position_rmse"]) for row in particle_rows],
        marker="s",
        linewidth=1.7,
        label=PARTICLE_LABEL,
    )
    ax.set_xlabel("Runtime [ms/step]")
    ax.set_ylabel("Position RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, "s3r3_particle_rmse_runtime.png")


def _mean_rows_by_method(metrics: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, int], list[dict[str, float | int | str]]] = {}
    for row in metrics:
        key = (str(row["filter"]), str(row["variant"]), int(row["resource_count"]))
        groups.setdefault(key, []).append(row)
    mean_rows = []
    for (filter_name, variant, resource_count), rows in groups.items():
        mean_rows.append(
            {
                "filter": filter_name,
                "variant": variant,
                "resource_count": resource_count,
                "position_rmse": float(np.mean([float(row["position_rmse"]) for row in rows])),
                "runtime_ms_per_step": float(np.mean([float(row["runtime_ms_per_step"]) for row in rows])),
            }
        )
    return mean_rows


def _write_ratio_heatmap(
    output_dir: Path,
    rows: list[dict[str, float | int | str | bool]],
    metric_name: str,
    title: str,
    filename: str,
) -> Path:
    kappas = sorted({float(row["prior_kappa"]) for row in rows})
    scales = sorted({float(row["body_increment_scale"]) for row in rows})
    matrix = np.full((len(kappas), len(scales)), np.nan)
    for row in rows:
        y_index = kappas.index(float(row["prior_kappa"]))
        x_index = scales.index(float(row["body_increment_scale"]))
        matrix[y_index, x_index] = float(row[metric_name])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=0.5, vmax=1.5)
    ax.set_xticks(range(len(scales)), [f"{scale:g}" for scale in scales])
    ax.set_yticks(range(len(kappas)), [f"{kappa:g}" for kappa in kappas])
    ax.set_xlabel("Body displacement scale")
    ax.set_ylabel("Prior kappa")
    ax.set_title(title)
    for y_index in range(len(kappas)):
        for x_index in range(len(scales)):
            ax.text(x_index, y_index, f"{matrix[y_index, x_index]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="RMSE ratio (<1 favors R1+R2)")
    return save_figure(fig, output_dir, filename)
