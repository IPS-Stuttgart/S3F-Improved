"""Dynamic S3+ x R3 pose benchmark for relaxed S3F prediction."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import (
    SdHalfCondSdHalfGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution

from ..s1r2.plotting import save_figure
from .relaxed_s3f_prototype import (
    SUPPORTED_S3R3_VARIANTS,
    VARIANT_LABELS,
    S3R3CellStatistics,
    S3R3PrototypeConfig,
    _canonical_quaternions,
    _exp_map_identity,
    _quaternion_multiply,
    _rotate_vectors,
    make_s3r3_filter,
    s3r3_cell_statistics,
    s3r3_linear_position_error_stats,
    s3r3_orientation_distance,
    s3r3_orientation_mode,
    s3r3_orientation_point_estimate,
    validate_s3r3_prototype_config,
)

DYNAMIC_POSE_METRIC_FIELDNAMES = [
    "grid_size",
    "variant",
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_point_error_rad",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "cell_radius_rad",
    "cell_sample_count",
    "orientation_increment_norm_rad",
    "orientation_transition_kappa",
    "n_trials",
    "n_steps",
]

DYNAMIC_POSE_CLAIM_FIELDNAMES = [
    "grid_size",
    "comparison",
    "candidate_variant",
    "comparator_variant",
    "candidate_position_rmse",
    "comparator_position_rmse",
    "position_rmse_ratio",
    "position_rmse_gain_pct",
    "candidate_mean_nees",
    "comparator_mean_nees",
    "mean_nees_ratio",
    "candidate_coverage_95",
    "comparator_coverage_95",
    "coverage_delta",
    "candidate_orientation_point_error_rad",
    "comparator_orientation_point_error_rad",
    "orientation_point_error_delta_rad",
    "candidate_runtime_ms_per_step",
    "comparator_runtime_ms_per_step",
    "runtime_ratio",
    "supports_accuracy_claim",
    "supports_consistency_claim",
    "supports_orientation_claim",
    "supports_runtime_claim",
    "supports_overall_claim",
]


@dataclass(frozen=True)
class S3R3DynamicPoseConfig:
    """Configuration for the dynamic S3+ x R3 pose benchmark."""

    prototype: S3R3PrototypeConfig = field(
        default_factory=lambda: S3R3PrototypeConfig(
            grid_sizes=(8, 16, 32),
            n_trials=16,
            n_steps=8,
            seed=47,
            cell_sample_count=27,
        )
    )
    orientation_increment: tuple[float, float, float] = (0.0, 0.18, 0.06)
    orientation_transition_kappa: float = 24.0


@dataclass(frozen=True)
class S3R3DynamicPoseResult:
    """Container for dynamic S3+ x R3 benchmark outputs."""

    metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]


def s3r3_dynamic_pose_config_to_dict(config: S3R3DynamicPoseConfig) -> dict[str, Any]:
    """Return a JSON-serializable dynamic-pose config."""

    return json.loads(json.dumps(asdict(config)))


def run_s3r3_dynamic_pose_benchmark(
    config: S3R3DynamicPoseConfig = S3R3DynamicPoseConfig(),
) -> S3R3DynamicPoseResult:
    """Run dynamic S3+ x R3 relaxed S3F variants and return metrics plus comparison claims."""

    _validate_config(config)
    trials = generate_s3r3_dynamic_pose_trials(config)
    metrics = [
        _run_variant(config, trials, grid_size, variant)
        for grid_size in config.prototype.grid_sizes
        for variant in config.prototype.variants
    ]
    claims = _build_claim_rows(metrics)
    return S3R3DynamicPoseResult(metrics=metrics, claims=claims)


def write_s3r3_dynamic_pose_outputs(
    output_dir: Path,
    config: S3R3DynamicPoseConfig = S3R3DynamicPoseConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the dynamic-pose benchmark and write CSV, metadata, note, and optional plots."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s3r3_dynamic_pose_benchmark(config)

    metrics_path = output_dir / "s3r3_dynamic_pose_metrics.csv"
    _write_csv(metrics_path, result.metrics, DYNAMIC_POSE_METRIC_FIELDNAMES)

    claims_path = output_dir / "s3r3_dynamic_pose_claims.csv"
    _write_csv(claims_path, result.claims, DYNAMIC_POSE_CLAIM_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path}
    plot_paths = _write_plots(output_dir, result.metrics) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_dynamic_pose_note.md"
    _write_note(note_path, result, metrics_path, claims_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, config)
    outputs["metadata"] = metadata_path
    return outputs


def generate_s3r3_dynamic_pose_trials(config: S3R3DynamicPoseConfig) -> list[dict[str, np.ndarray]]:
    """Generate reproducible dynamic-orientation S3+ x R3 tracking trials."""

    _validate_config(config)
    rng = np.random.default_rng(config.prototype.seed)
    modes = _canonical_quaternions(np.asarray(config.prototype.prior_modes, dtype=float))
    weights = np.asarray(config.prototype.prior_weights, dtype=float)
    weights = weights / np.sum(weights)
    body_increment = np.asarray(config.prototype.body_increment, dtype=float)
    delta_quaternion = _orientation_increment_quaternion(config.orientation_increment)

    trials = []
    for _trial_idx in range(config.prototype.n_trials):
        component = int(rng.choice(len(modes), p=weights))
        local_noise = rng.normal(scale=config.prototype.orientation_noise_std, size=3)
        orientation = _quaternion_multiply(modes[component], _exp_map_identity(local_noise)[0])
        orientations = [orientation]
        positions = [np.zeros(3)]
        measurements = []
        for _step in range(config.prototype.n_steps):
            displacement = _rotate_vectors(orientations[-1], body_increment)[0]
            process_noise = rng.normal(scale=config.prototype.process_noise_std, size=3)
            next_position = positions[-1] + displacement + process_noise
            positions.append(next_position)
            measurements.append(next_position + rng.normal(scale=config.prototype.measurement_noise_std, size=3))
            orientations.append(_quaternion_multiply(orientations[-1], delta_quaternion))
        trials.append(
            {
                "orientations": np.asarray(orientations),
                "positions": np.asarray(positions),
                "measurements": np.asarray(measurements),
            }
        )
    return trials


def predict_s3r3_dynamic_pose(
    filter_,
    body_increment: np.ndarray,
    orientation_increment: np.ndarray | tuple[float, ...],
    *,
    variant: str = "r1_r2",
    process_noise_cov: np.ndarray | None = None,
    cell_sample_count: int = 27,
    orientation_transition_kappa: float = 24.0,
) -> S3R3CellStatistics:
    """Predict S3+ x R3 with known quaternion dynamics and relaxed translation statistics."""

    if variant not in SUPPORTED_S3R3_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")
    if orientation_transition_kappa <= 0.0:
        raise ValueError("orientation_transition_kappa must be positive.")

    state = filter_.filter_state
    if state.lin_dim != 3:
        raise ValueError("predict_s3r3_dynamic_pose requires a 3-D linear state.")

    grid = np.asarray(state.gd.get_grid(), dtype=float)
    stats = s3r3_cell_statistics(grid, body_increment, cell_sample_count)
    transition_density = s3r3_orientation_transition_density(
        grid,
        orientation_increment,
        orientation_transition_kappa,
    )

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
        transition_density=transition_density,
        covariance_matrices=covariance_matrices,
        linear_input_vectors=displacements.T,
    )
    return stats


def s3r3_orientation_transition_density(
    grid: np.ndarray,
    orientation_increment: np.ndarray | tuple[float, ...],
    orientation_transition_kappa: float,
) -> SdHalfCondSdHalfGridDistribution:
    """Return a normalized soft grid transition for ``q_next = q_current * delta_q``."""

    if orientation_transition_kappa <= 0.0:
        raise ValueError("orientation_transition_kappa must be positive.")
    grid = np.ascontiguousarray(_canonical_quaternions(np.asarray(grid, dtype=np.float64)))
    delta_quaternion = _orientation_increment_quaternion(orientation_increment)
    return _cached_orientation_transition_density(
        grid.shape,
        grid.tobytes(),
        tuple(float(value) for value in delta_quaternion),
        float(orientation_transition_kappa),
    )


@lru_cache(maxsize=128)
def _cached_orientation_transition_density(
    grid_shape: tuple[int, ...],
    grid_bytes: bytes,
    delta_quaternion_values: tuple[float, float, float, float],
    orientation_transition_kappa: float,
) -> SdHalfCondSdHalfGridDistribution:
    grid = np.frombuffer(grid_bytes, dtype=np.float64).reshape(grid_shape)
    delta_quaternion = np.asarray(delta_quaternion_values, dtype=np.float64)
    targets = _quaternion_multiply(grid, delta_quaternion)
    inner = np.clip(np.abs(grid @ targets.T), 0.0, 1.0)
    scores = np.exp(orientation_transition_kappa * (inner**2 - 1.0))
    column_sums = np.sum(scores, axis=0, keepdims=True)
    manifold_size = _hemisphere_surface(grid.shape[1] - 1)
    density_values = scores / column_sums * (grid.shape[0] / manifold_size)
    return SdHalfCondSdHalfGridDistribution(grid.copy(), density_values, enforce_pdf_nonnegative=True)


def _hemisphere_surface(manifold_dimension: int) -> float:
    return float(2.0 * np.pi ** ((manifold_dimension + 1) / 2.0) / math.gamma((manifold_dimension + 1) / 2.0) / 2.0)


def _orientation_increment_quaternion(orientation_increment: np.ndarray | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(orientation_increment, dtype=float)
    if values.shape == (3,):
        return _exp_map_identity(values)[0]
    if values.shape == (4,):
        return _canonical_quaternions(values)
    raise ValueError("orientation_increment must have shape (3,) tangent or (4,) quaternion.")


def _validate_config(config: S3R3DynamicPoseConfig) -> None:
    validate_s3r3_prototype_config(config.prototype)
    if config.orientation_transition_kappa <= 0.0:
        raise ValueError("orientation_transition_kappa must be positive.")
    _orientation_increment_quaternion(config.orientation_increment)


def _run_variant(
    config: S3R3DynamicPoseConfig,
    trials: list[dict[str, np.ndarray]],
    grid_size: int,
    variant: str,
) -> dict[str, float | int | str]:
    measurement_cov = np.eye(3) * config.prototype.measurement_noise_std**2
    process_noise_cov = np.eye(3) * config.prototype.process_noise_std**2
    body_increment = np.asarray(config.prototype.body_increment, dtype=float)
    orientation_increment = np.asarray(config.orientation_increment, dtype=float)

    sum_position_sq_error = 0.0
    sum_orientation_mode_error = 0.0
    sum_orientation_point_error = 0.0
    sum_nees = 0.0
    coverage_hits = 0
    n_metrics = 0
    runtime = 0.0
    last_cell_radius = 0.0

    for trial in trials:
        filter_ = make_s3r3_filter(config.prototype, grid_size)
        true_orientations = np.asarray(trial["orientations"], dtype=float)
        true_positions = np.asarray(trial["positions"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)

        for step, measurement in enumerate(measurements):
            likelihood = GaussianDistribution(measurement, measurement_cov, check_validity=False)
            start = perf_counter()
            stats = predict_s3r3_dynamic_pose(
                filter_,
                body_increment,
                orientation_increment,
                variant=variant,
                process_noise_cov=process_noise_cov,
                cell_sample_count=config.prototype.cell_sample_count,
                orientation_transition_kappa=config.orientation_transition_kappa,
            )
            filter_.update(likelihoods_linear=[likelihood])
            runtime += perf_counter() - start
            last_cell_radius = stats.cell_radius_rad

            true_position = true_positions[step + 1]
            true_orientation = true_orientations[step + 1]
            error, nees = s3r3_linear_position_error_stats(filter_, true_position)
            sum_position_sq_error += float(error @ error)
            sum_nees += nees
            coverage_hits += int(nees <= 7.814727903251179)
            sum_orientation_mode_error += s3r3_orientation_distance(s3r3_orientation_mode(filter_), true_orientation)
            sum_orientation_point_error += s3r3_orientation_distance(s3r3_orientation_point_estimate(filter_), true_orientation)
            n_metrics += 1

    return {
        "grid_size": grid_size,
        "variant": variant,
        "position_rmse": float(np.sqrt(sum_position_sq_error / n_metrics)),
        "orientation_mode_error_rad": sum_orientation_mode_error / n_metrics,
        "orientation_point_error_rad": sum_orientation_point_error / n_metrics,
        "mean_nees": sum_nees / n_metrics,
        "coverage_95": coverage_hits / n_metrics,
        "runtime_ms_per_step": 1000.0 * runtime / n_metrics,
        "cell_radius_rad": last_cell_radius,
        "cell_sample_count": config.prototype.cell_sample_count,
        "orientation_increment_norm_rad": float(np.linalg.norm(orientation_increment)),
        "orientation_transition_kappa": config.orientation_transition_kappa,
        "n_trials": config.prototype.n_trials,
        "n_steps": config.prototype.n_steps,
    }


def _build_claim_rows(metrics: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str | bool]]:
    rows_by_key = {(int(row["grid_size"]), str(row["variant"])): row for row in metrics}
    claims: list[dict[str, float | int | str | bool]] = []
    grid_sizes = sorted({int(row["grid_size"]) for row in metrics})
    for grid_size in grid_sizes:
        candidate = rows_by_key[(grid_size, "r1_r2")]
        for comparator_variant, comparison in (
            ("baseline", "R1+R2 vs baseline"),
            ("r1", "R1+R2 vs R1"),
        ):
            if (grid_size, comparator_variant) not in rows_by_key:
                continue
            claims.append(_claim_row(candidate, rows_by_key[(grid_size, comparator_variant)], comparison))
    return claims


def _claim_row(
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
    comparison: str,
) -> dict[str, float | int | str | bool]:
    candidate_rmse = float(candidate["position_rmse"])
    comparator_rmse = float(comparator["position_rmse"])
    position_ratio = candidate_rmse / comparator_rmse
    position_gain = 100.0 * (1.0 - position_ratio)
    nees_ratio = float(candidate["mean_nees"]) / float(comparator["mean_nees"])
    coverage_delta = float(candidate["coverage_95"]) - float(comparator["coverage_95"])
    orientation_delta = float(candidate["orientation_point_error_rad"]) - float(comparator["orientation_point_error_rad"])
    runtime_ratio = float(candidate["runtime_ms_per_step"]) / float(comparator["runtime_ms_per_step"])
    supports_accuracy = position_gain > 0.0
    supports_consistency = nees_ratio <= 1.0 and coverage_delta >= -0.02
    supports_orientation = orientation_delta <= 0.05
    supports_runtime = runtime_ratio <= 1.25
    return {
        "grid_size": int(candidate["grid_size"]),
        "comparison": comparison,
        "candidate_variant": str(candidate["variant"]),
        "comparator_variant": str(comparator["variant"]),
        "candidate_position_rmse": candidate_rmse,
        "comparator_position_rmse": comparator_rmse,
        "position_rmse_ratio": position_ratio,
        "position_rmse_gain_pct": position_gain,
        "candidate_mean_nees": float(candidate["mean_nees"]),
        "comparator_mean_nees": float(comparator["mean_nees"]),
        "mean_nees_ratio": nees_ratio,
        "candidate_coverage_95": float(candidate["coverage_95"]),
        "comparator_coverage_95": float(comparator["coverage_95"]),
        "coverage_delta": coverage_delta,
        "candidate_orientation_point_error_rad": float(candidate["orientation_point_error_rad"]),
        "comparator_orientation_point_error_rad": float(comparator["orientation_point_error_rad"]),
        "orientation_point_error_delta_rad": orientation_delta,
        "candidate_runtime_ms_per_step": float(candidate["runtime_ms_per_step"]),
        "comparator_runtime_ms_per_step": float(comparator["runtime_ms_per_step"]),
        "runtime_ratio": runtime_ratio,
        "supports_accuracy_claim": supports_accuracy,
        "supports_consistency_claim": supports_consistency,
        "supports_orientation_claim": supports_orientation,
        "supports_runtime_claim": supports_runtime,
        "supports_overall_claim": supports_accuracy and supports_consistency and supports_orientation and supports_runtime,
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _write_metadata(path: Path, result: S3R3DynamicPoseResult, config: S3R3DynamicPoseConfig) -> None:
    metadata = {
        "experiment": "s3r3_dynamic_pose",
        "config": s3r3_dynamic_pose_config_to_dict(config),
        "metrics_schema": DYNAMIC_POSE_METRIC_FIELDNAMES,
        "claims_schema": DYNAMIC_POSE_CLAIM_FIELDNAMES,
        "metrics_rows": len(result.metrics),
        "claims_rows": len(result.claims),
        "orientation_transition": "soft_grid_density",
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    result: S3R3DynamicPoseResult,
    metrics_path: Path,
    claims_path: Path,
    plot_paths: list[Path],
    config: S3R3DynamicPoseConfig,
) -> None:
    baseline_claims = [row for row in result.claims if row["comparator_variant"] == "baseline"]
    inflation_claims = [row for row in result.claims if row["comparator_variant"] == "r1"]
    best_baseline = max(baseline_claims, key=lambda row: float(row["position_rmse_gain_pct"]))
    lines = [
        "# Dynamic S3+ x R3 Pose Benchmark",
        "",
        "This benchmark uses known orientation dynamics `q_next = q_current * delta_q` and position dynamics `p_next = p_current + R(q_current) u + noise`.",
        "The S3F prediction uses a PyRecEst hyperhemispherical grid transition density for the orientation step and baseline/R1/R1+R2 translation propagation for the linear part.",
        "",
        f"Trials: {config.prototype.n_trials}",
        f"Steps per trial: {config.prototype.n_steps}",
        f"Grid sizes: {list(config.prototype.grid_sizes)}",
        f"Orientation increment tangent vector: {list(config.orientation_increment)}",
        f"Orientation transition kappa: {config.orientation_transition_kappa:g}",
        f"Cell sample count: {config.prototype.cell_sample_count}",
        f"Metrics: `{metrics_path.name}`",
        f"Claims: `{claims_path.name}`",
        "",
        "## Headline",
        "",
        f"`R1+R2` supports the baseline comparison in `{_support_count(baseline_claims)}/{len(baseline_claims)}` grid rows and the inflation comparison in `{_support_count(inflation_claims)}/{len(inflation_claims)}` grid rows.",
        f"Largest baseline RMSE gain is `{float(best_baseline['position_rmse_gain_pct']):.1f}%` at `{best_baseline['grid_size']}` cells.",
        "",
        "## Grid Summary",
        "",
        "| Grid | RMSE gain vs baseline % | NEES ratio | orientation delta [rad] | runtime ratio | overall |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in baseline_claims:
        lines.append(
            f"| {row['grid_size']} | {float(row['position_rmse_gain_pct']):.1f} | {float(row['mean_nees_ratio']):.3f} | {float(row['orientation_point_error_delta_rad']):.3f} | {float(row['runtime_ratio']):.3f} | {row['supports_overall_claim']} |"
        )
    if plot_paths:
        lines.extend(["", "Plots:", *[f"- `{plot_path.name}`" for plot_path in plot_paths]])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _support_count(rows: list[dict[str, float | int | str | bool]]) -> int:
    return sum(bool(row["supports_overall_claim"]) for row in rows)


def _write_plots(output_dir: Path, metrics: list[dict[str, float | int | str]]) -> list[Path]:
    paths = []
    for metric_name, ylabel, filename in (
        ("position_rmse", "Translation RMSE", "s3r3_dynamic_pose_position_rmse.png"),
        ("mean_nees", "Mean Position NEES", "s3r3_dynamic_pose_nees.png"),
        ("orientation_point_error_rad", "Orientation Point Error [rad]", "s3r3_dynamic_pose_orientation_error.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for variant in SUPPORTED_S3R3_VARIANTS:
            rows = [row for row in metrics if row["variant"] == variant]
            rows = sorted(rows, key=lambda row: int(row["grid_size"]))
            ax.plot(
                [int(row["grid_size"]) for row in rows],
                [float(row[metric_name]) for row in rows],
                marker="o",
                label=VARIANT_LABELS[variant],
            )
        ax.set_xlabel("Number of quaternion grid cells")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        paths.append(save_figure(fig, output_dir, filename))
    return paths
