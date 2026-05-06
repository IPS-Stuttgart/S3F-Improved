"""Dynamic high-resolution S3F reference comparison for S3+ x R3."""

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

from ..s1r2.plotting import format_plot_list, save_figure
from .dynamic_pose import (
    S3R3DynamicPoseConfig,
    generate_s3r3_dynamic_pose_trials,
    predict_s3r3_dynamic_pose,
)
from .relaxed_s3f_prototype import (
    SUPPORTED_S3R3_VARIANTS,
    VARIANT_LABELS,
    S3R3PrototypeConfig,
    make_s3r3_filter,
    s3r3_linear_position_error_stats,
    s3r3_linear_position_mean,
    s3r3_orientation_distance,
    s3r3_orientation_mode,
    s3r3_orientation_point_estimate,
    validate_s3r3_prototype_config,
)


DYNAMIC_REFERENCE_VARIANT = "baseline"

S3R3_DYNAMIC_HIGHRES_FIELDNAMES = [
    "grid_size",
    "reference_grid_size",
    "variant",
    "reference_variant",
    "position_rmse_to_reference",
    "orientation_mode_error_to_reference_rad",
    "orientation_point_error_to_reference_rad",
    "position_rmse_to_truth",
    "reference_position_rmse_to_truth",
    "orientation_mode_error_to_truth_rad",
    "orientation_point_error_to_truth_rad",
    "reference_orientation_mode_error_to_truth_rad",
    "reference_orientation_point_error_to_truth_rad",
    "mean_nees_to_truth",
    "coverage_95_to_truth",
    "runtime_ms_per_step",
    "reference_runtime_ms_per_step",
    "runtime_ratio_to_reference",
    "cell_sample_count",
    "orientation_increment_norm_rad",
    "orientation_transition_kappa",
    "n_trials",
    "n_steps",
]

S3R3_DYNAMIC_HIGHRES_CLAIM_FIELDNAMES = [
    "grid_size",
    "reference_grid_size",
    "comparison",
    "candidate_variant",
    "comparator_variant",
    "reference_variant",
    "candidate_position_rmse_to_reference",
    "comparator_position_rmse_to_reference",
    "position_rmse_to_reference_ratio",
    "position_rmse_to_reference_gain_pct",
    "candidate_position_rmse_to_truth",
    "comparator_position_rmse_to_truth",
    "position_rmse_to_truth_ratio",
    "position_rmse_to_truth_gain_pct",
    "candidate_mean_nees_to_truth",
    "comparator_mean_nees_to_truth",
    "mean_nees_ratio",
    "candidate_coverage_95_to_truth",
    "comparator_coverage_95_to_truth",
    "coverage_delta",
    "candidate_orientation_point_error_to_reference_rad",
    "comparator_orientation_point_error_to_reference_rad",
    "orientation_point_error_to_reference_delta_rad",
    "candidate_runtime_ms_per_step",
    "comparator_runtime_ms_per_step",
    "runtime_ratio",
    "supports_reference_claim",
    "supports_truth_accuracy_claim",
    "supports_consistency_claim",
    "supports_orientation_reference_claim",
    "supports_runtime_claim",
    "supports_overall_claim",
]


@dataclass(frozen=True)
class S3R3DynamicHighResReferenceConfig:
    """Configuration for dynamic S3+ x R3 coarse-grid comparison against a denser S3F reference."""

    prototype: S3R3PrototypeConfig = field(
        default_factory=lambda: S3R3PrototypeConfig(
            grid_sizes=(8, 16, 32),
            variants=SUPPORTED_S3R3_VARIANTS,
            n_trials=8,
            n_steps=8,
            seed=59,
            cell_sample_count=27,
        )
    )
    reference_grid_size: int = 64
    orientation_increment: tuple[float, float, float] = (0.0, 0.18, 0.06)
    orientation_transition_kappa: float = 24.0


@dataclass(frozen=True)
class S3R3DynamicHighResReferenceResult:
    """Container for dynamic high-resolution reference outputs."""

    metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]


@dataclass
class _ComparisonTotals:
    position_ref_sq_error: float = 0.0
    orientation_mode_ref_error: float = 0.0
    orientation_point_ref_error: float = 0.0
    position_truth_sq_error: float = 0.0
    orientation_mode_truth_error: float = 0.0
    orientation_point_truth_error: float = 0.0
    nees_sum: float = 0.0
    coverage_hits: int = 0
    runtime_s: float = 0.0


def s3r3_dynamic_highres_reference_config_to_dict(config: S3R3DynamicHighResReferenceConfig) -> dict[str, Any]:
    """Return a JSON-serializable dynamic high-resolution reference config."""

    return json.loads(json.dumps(asdict(config) | {"reference_variant": DYNAMIC_REFERENCE_VARIANT}))


def run_s3r3_dynamic_highres_reference_benchmark(
    config: S3R3DynamicHighResReferenceConfig = S3R3DynamicHighResReferenceConfig(),
) -> S3R3DynamicHighResReferenceResult:
    """Compare dynamic coarse S3+ x R3 variants against a denser dynamic baseline S3F reference."""

    _validate_config(config)
    dynamic_config = S3R3DynamicPoseConfig(
        prototype=config.prototype,
        orientation_increment=config.orientation_increment,
        orientation_transition_kappa=config.orientation_transition_kappa,
    )
    trials = generate_s3r3_dynamic_pose_trials(dynamic_config)
    prototype = config.prototype
    measurement_cov = np.eye(3) * prototype.measurement_noise_std**2
    process_noise_cov = np.eye(3) * prototype.process_noise_std**2
    body_increment = np.asarray(prototype.body_increment, dtype=float)
    orientation_increment = np.asarray(config.orientation_increment, dtype=float)
    keys = [(grid_size, variant) for grid_size in prototype.grid_sizes for variant in prototype.variants]
    totals = {key: _ComparisonTotals() for key in keys}
    reference_totals = _ComparisonTotals()
    metric_count = 0

    for trial in trials:
        reference_filter = make_s3r3_filter(prototype, config.reference_grid_size)
        candidate_filters = {key: make_s3r3_filter(prototype, key[0]) for key in keys}
        true_orientations = np.asarray(trial["orientations"], dtype=float)
        true_positions = np.asarray(trial["positions"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)

        for step, measurement in enumerate(measurements):
            true_position = true_positions[step + 1]
            true_orientation = true_orientations[step + 1]
            reference_runtime = _predict_update(
                reference_filter,
                measurement,
                measurement_cov,
                body_increment,
                orientation_increment,
                DYNAMIC_REFERENCE_VARIANT,
                process_noise_cov,
                prototype.cell_sample_count,
                config.orientation_transition_kappa,
            )
            reference_totals.runtime_s += reference_runtime
            reference_mean = s3r3_linear_position_mean(reference_filter)
            reference_mode = s3r3_orientation_mode(reference_filter)
            reference_point = s3r3_orientation_point_estimate(reference_filter)
            reference_error, _reference_nees = s3r3_linear_position_error_stats(reference_filter, true_position)
            reference_totals.position_truth_sq_error += float(reference_error @ reference_error)
            reference_totals.orientation_mode_truth_error += s3r3_orientation_distance(reference_mode, true_orientation)
            reference_totals.orientation_point_truth_error += s3r3_orientation_distance(reference_point, true_orientation)

            for key, candidate_filter in candidate_filters.items():
                elapsed = _predict_update(
                    candidate_filter,
                    measurement,
                    measurement_cov,
                    body_increment,
                    orientation_increment,
                    key[1],
                    process_noise_cov,
                    prototype.cell_sample_count,
                    config.orientation_transition_kappa,
                )
                _accumulate_candidate(
                    totals[key],
                    candidate_filter,
                    true_position,
                    true_orientation,
                    reference_mean,
                    reference_mode,
                    reference_point,
                    elapsed,
                )
            metric_count += 1

    reference_truth = _reference_truth_summary(reference_totals, metric_count)
    metrics = [
        _row_from_totals(
            grid_size=grid_size,
            variant=variant,
            totals=totals[(grid_size, variant)],
            config=config,
            metric_count=metric_count,
            reference_truth=reference_truth,
        )
        for grid_size, variant in keys
    ]
    return S3R3DynamicHighResReferenceResult(metrics=metrics, claims=_build_claim_rows(metrics))


def write_s3r3_dynamic_highres_reference_outputs(
    output_dir: Path,
    config: S3R3DynamicHighResReferenceConfig = S3R3DynamicHighResReferenceConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the dynamic high-resolution reference benchmark and write CSV, metadata, note, and optional plots."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s3r3_dynamic_highres_reference_benchmark(config)

    metrics_path = output_dir / "s3r3_dynamic_highres_reference_metrics.csv"
    _write_csv(metrics_path, result.metrics, S3R3_DYNAMIC_HIGHRES_FIELDNAMES)

    claims_path = output_dir / "s3r3_dynamic_highres_reference_claims.csv"
    _write_csv(claims_path, result.claims, S3R3_DYNAMIC_HIGHRES_CLAIM_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path}
    plot_paths = _write_plots(output_dir, result.metrics, result.claims) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_dynamic_highres_reference_note.md"
    _write_note(note_path, result, metrics_path, claims_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: S3R3DynamicHighResReferenceConfig) -> None:
    validate_s3r3_prototype_config(
        config.prototype,
        reference_grid_size=config.reference_grid_size,
        required_variants=SUPPORTED_S3R3_VARIANTS,
    )
    if config.orientation_transition_kappa <= 0.0:
        raise ValueError("orientation_transition_kappa must be positive.")
    orientation_increment = np.asarray(config.orientation_increment, dtype=float)
    if orientation_increment.shape not in {(3,), (4,)}:
        raise ValueError("orientation_increment must have shape (3,) tangent or (4,) quaternion.")
    if orientation_increment.shape == (4,) and np.linalg.norm(orientation_increment) <= 0.0:
        raise ValueError("orientation_increment quaternion must be nonzero.")


def _predict_update(
    filter_,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
    body_increment: np.ndarray,
    orientation_increment: np.ndarray,
    variant: str,
    process_noise_cov: np.ndarray,
    cell_sample_count: int,
    orientation_transition_kappa: float,
) -> float:
    likelihood = GaussianDistribution(measurement, measurement_cov, check_validity=False)
    start = perf_counter()
    predict_s3r3_dynamic_pose(
        filter_,
        body_increment,
        orientation_increment,
        variant=variant,
        process_noise_cov=process_noise_cov,
        cell_sample_count=cell_sample_count,
        orientation_transition_kappa=orientation_transition_kappa,
    )
    filter_.update(likelihoods_linear=[likelihood])
    return perf_counter() - start


def _accumulate_candidate(
    totals: _ComparisonTotals,
    candidate_filter,
    true_position: np.ndarray,
    true_orientation: np.ndarray,
    reference_mean: np.ndarray,
    reference_mode: np.ndarray,
    reference_point: np.ndarray,
    elapsed_s: float,
) -> None:
    candidate_mean = s3r3_linear_position_mean(candidate_filter)
    candidate_mode = s3r3_orientation_mode(candidate_filter)
    candidate_point = s3r3_orientation_point_estimate(candidate_filter)
    reference_delta = candidate_mean - reference_mean
    truth_error, nees = s3r3_linear_position_error_stats(candidate_filter, true_position)

    totals.position_ref_sq_error += float(reference_delta @ reference_delta)
    totals.orientation_mode_ref_error += s3r3_orientation_distance(candidate_mode, reference_mode)
    totals.orientation_point_ref_error += s3r3_orientation_distance(candidate_point, reference_point)
    totals.position_truth_sq_error += float(truth_error @ truth_error)
    totals.orientation_mode_truth_error += s3r3_orientation_distance(candidate_mode, true_orientation)
    totals.orientation_point_truth_error += s3r3_orientation_distance(candidate_point, true_orientation)
    totals.nees_sum += nees
    totals.coverage_hits += int(nees <= 7.814727903251179)
    totals.runtime_s += elapsed_s


def _reference_truth_summary(totals: _ComparisonTotals, metric_count: int) -> dict[str, float]:
    return {
        "position_rmse_to_truth": float(np.sqrt(totals.position_truth_sq_error / metric_count)),
        "orientation_mode_error_to_truth_rad": totals.orientation_mode_truth_error / metric_count,
        "orientation_point_error_to_truth_rad": totals.orientation_point_truth_error / metric_count,
        "runtime_ms_per_step": 1000.0 * totals.runtime_s / metric_count,
    }


def _row_from_totals(
    grid_size: int,
    variant: str,
    totals: _ComparisonTotals,
    config: S3R3DynamicHighResReferenceConfig,
    metric_count: int,
    reference_truth: dict[str, float],
) -> dict[str, float | int | str]:
    runtime_ms = 1000.0 * totals.runtime_s / metric_count
    orientation_increment_norm = _orientation_increment_norm(config.orientation_increment)
    reference_runtime_ms = reference_truth["runtime_ms_per_step"]
    return {
        "grid_size": grid_size,
        "reference_grid_size": config.reference_grid_size,
        "variant": variant,
        "reference_variant": DYNAMIC_REFERENCE_VARIANT,
        "position_rmse_to_reference": float(np.sqrt(totals.position_ref_sq_error / metric_count)),
        "orientation_mode_error_to_reference_rad": totals.orientation_mode_ref_error / metric_count,
        "orientation_point_error_to_reference_rad": totals.orientation_point_ref_error / metric_count,
        "position_rmse_to_truth": float(np.sqrt(totals.position_truth_sq_error / metric_count)),
        "reference_position_rmse_to_truth": reference_truth["position_rmse_to_truth"],
        "orientation_mode_error_to_truth_rad": totals.orientation_mode_truth_error / metric_count,
        "orientation_point_error_to_truth_rad": totals.orientation_point_truth_error / metric_count,
        "reference_orientation_mode_error_to_truth_rad": reference_truth["orientation_mode_error_to_truth_rad"],
        "reference_orientation_point_error_to_truth_rad": reference_truth["orientation_point_error_to_truth_rad"],
        "mean_nees_to_truth": totals.nees_sum / metric_count,
        "coverage_95_to_truth": totals.coverage_hits / metric_count,
        "runtime_ms_per_step": runtime_ms,
        "reference_runtime_ms_per_step": reference_runtime_ms,
        "runtime_ratio_to_reference": runtime_ms / reference_runtime_ms,
        "cell_sample_count": config.prototype.cell_sample_count,
        "orientation_increment_norm_rad": orientation_increment_norm,
        "orientation_transition_kappa": config.orientation_transition_kappa,
        "n_trials": config.prototype.n_trials,
        "n_steps": config.prototype.n_steps,
    }


def _build_claim_rows(metrics: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str | bool]]:
    rows_by_key = {(int(row["grid_size"]), str(row["variant"])): row for row in metrics}
    claims: list[dict[str, float | int | str | bool]] = []
    for grid_size in sorted({int(row["grid_size"]) for row in metrics}):
        candidate = rows_by_key[(grid_size, "r1_r2")]
        for comparator_variant, comparison in (
            ("baseline", "R1+R2 vs coarse baseline"),
            ("r1", "R1+R2 vs R1"),
        ):
            if (grid_size, comparator_variant) in rows_by_key:
                claims.append(_claim_row(candidate, rows_by_key[(grid_size, comparator_variant)], comparison))
    return claims


def _orientation_increment_norm(orientation_increment: tuple[float, ...]) -> float:
    values = np.asarray(orientation_increment, dtype=float)
    if values.shape == (3,):
        return float(np.linalg.norm(values))
    quaternion = values / np.linalg.norm(values)
    return float(2.0 * np.arccos(np.clip(abs(quaternion[3]), 0.0, 1.0)))


def _claim_row(
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
    comparison: str,
) -> dict[str, float | int | str | bool]:
    candidate_ref = float(candidate["position_rmse_to_reference"])
    comparator_ref = float(comparator["position_rmse_to_reference"])
    candidate_truth = float(candidate["position_rmse_to_truth"])
    comparator_truth = float(comparator["position_rmse_to_truth"])
    ref_ratio = candidate_ref / comparator_ref
    truth_ratio = candidate_truth / comparator_truth
    nees_ratio = float(candidate["mean_nees_to_truth"]) / float(comparator["mean_nees_to_truth"])
    coverage_delta = float(candidate["coverage_95_to_truth"]) - float(comparator["coverage_95_to_truth"])
    orientation_delta = float(candidate["orientation_point_error_to_reference_rad"]) - float(comparator["orientation_point_error_to_reference_rad"])
    runtime_ratio = float(candidate["runtime_ms_per_step"]) / float(comparator["runtime_ms_per_step"])
    supports_reference = ref_ratio < 1.0
    supports_truth_accuracy = truth_ratio <= 1.02
    supports_consistency = nees_ratio <= 1.0 and coverage_delta >= -0.02
    supports_orientation_reference = orientation_delta <= 0.05
    supports_runtime = runtime_ratio <= 1.25
    return {
        "grid_size": int(candidate["grid_size"]),
        "reference_grid_size": int(candidate["reference_grid_size"]),
        "comparison": comparison,
        "candidate_variant": str(candidate["variant"]),
        "comparator_variant": str(comparator["variant"]),
        "reference_variant": str(candidate["reference_variant"]),
        "candidate_position_rmse_to_reference": candidate_ref,
        "comparator_position_rmse_to_reference": comparator_ref,
        "position_rmse_to_reference_ratio": ref_ratio,
        "position_rmse_to_reference_gain_pct": 100.0 * (1.0 - ref_ratio),
        "candidate_position_rmse_to_truth": candidate_truth,
        "comparator_position_rmse_to_truth": comparator_truth,
        "position_rmse_to_truth_ratio": truth_ratio,
        "position_rmse_to_truth_gain_pct": 100.0 * (1.0 - truth_ratio),
        "candidate_mean_nees_to_truth": float(candidate["mean_nees_to_truth"]),
        "comparator_mean_nees_to_truth": float(comparator["mean_nees_to_truth"]),
        "mean_nees_ratio": nees_ratio,
        "candidate_coverage_95_to_truth": float(candidate["coverage_95_to_truth"]),
        "comparator_coverage_95_to_truth": float(comparator["coverage_95_to_truth"]),
        "coverage_delta": coverage_delta,
        "candidate_orientation_point_error_to_reference_rad": float(candidate["orientation_point_error_to_reference_rad"]),
        "comparator_orientation_point_error_to_reference_rad": float(comparator["orientation_point_error_to_reference_rad"]),
        "orientation_point_error_to_reference_delta_rad": orientation_delta,
        "candidate_runtime_ms_per_step": float(candidate["runtime_ms_per_step"]),
        "comparator_runtime_ms_per_step": float(comparator["runtime_ms_per_step"]),
        "runtime_ratio": runtime_ratio,
        "supports_reference_claim": supports_reference,
        "supports_truth_accuracy_claim": supports_truth_accuracy,
        "supports_consistency_claim": supports_consistency,
        "supports_orientation_reference_claim": supports_orientation_reference,
        "supports_runtime_claim": supports_runtime,
        "supports_overall_claim": supports_reference and supports_truth_accuracy and supports_consistency and supports_runtime,
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _write_metadata(path: Path, result: S3R3DynamicHighResReferenceResult, config: S3R3DynamicHighResReferenceConfig) -> None:
    metadata = {
        "claims_rows": len(result.claims),
        "claims_schema": S3R3_DYNAMIC_HIGHRES_CLAIM_FIELDNAMES,
        "config": s3r3_dynamic_highres_reference_config_to_dict(config),
        "experiment": "s3r3_dynamic_highres_reference",
        "metrics_rows": len(result.metrics),
        "metrics_schema": S3R3_DYNAMIC_HIGHRES_FIELDNAMES,
        "reference_model": "denser_dynamic_baseline_s3f",
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    result: S3R3DynamicHighResReferenceResult,
    metrics_path: Path,
    claims_path: Path,
    plot_paths: list[Path],
    config: S3R3DynamicHighResReferenceConfig,
) -> None:
    baseline_claims = [claim for claim in result.claims if claim["comparator_variant"] == "baseline"]
    best_claim = max(baseline_claims, key=lambda claim: float(claim["position_rmse_to_reference_gain_pct"]))
    support_count = sum(1 for claim in baseline_claims if bool(claim["supports_overall_claim"]))
    reference_count = sum(1 for claim in baseline_claims if bool(claim["supports_reference_claim"]))
    lines = [
        "# Dynamic S3+ x R3 High-Resolution Reference",
        "",
        "This benchmark compares dynamic coarse-grid S3F variants against a denser dynamic baseline S3F reference.",
        "It tests whether relaxed coarse-grid prediction follows the high-resolution representative-cell prediction more closely.",
        "",
        f"Trials: {config.prototype.n_trials}",
        f"Steps per trial: {config.prototype.n_steps}",
        f"Coarse grid sizes: {list(config.prototype.grid_sizes)}",
        f"Reference grid size: {config.reference_grid_size}",
        f"Orientation increment tangent vector: {list(config.orientation_increment)}",
        f"Orientation transition kappa: {config.orientation_transition_kappa:g}",
        f"Cell sample count: {config.prototype.cell_sample_count}",
        f"Metrics: `{metrics_path.name}`",
        f"Claims: `{claims_path.name}`",
        "",
        "## Headline",
        "",
        f"`R1+R2` is closer to the dynamic high-resolution reference than the coarse baseline in `{reference_count}/{len(baseline_claims)}` grid rows.",
        f"The stricter overall claim is supported in `{support_count}/{len(baseline_claims)}` grid rows.",
        f"Largest reference RMSE gain is `{float(best_claim['position_rmse_to_reference_gain_pct']):.1f}%` at `{best_claim['grid_size']}` cells.",
        "",
        "## Baseline Comparison",
        "",
        _format_claim_table(baseline_claims),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "This is still a synthetic reference comparison. The reference is a denser dynamic S3F baseline, not ground truth or a real trajectory filter.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_claim_table(rows: list[dict[str, float | int | str | bool]]) -> str:
    header = "| Cells | ref RMSE gain % | truth RMSE gain % | NEES ratio | coverage delta | runtime ratio | overall |"
    separator = "|---:|---:|---:|---:|---:|---:|---|"
    body = []
    for row in sorted(rows, key=lambda claim: int(claim["grid_size"])):
        body.append(
            "| "
            f"{int(row['grid_size'])} | "
            f"{float(row['position_rmse_to_reference_gain_pct']):.1f} | "
            f"{float(row['position_rmse_to_truth_gain_pct']):.1f} | "
            f"{float(row['mean_nees_ratio']):.3f} | "
            f"{float(row['coverage_delta']):.3f} | "
            f"{float(row['runtime_ratio']):.3f} | "
            f"{row['supports_overall_claim']} |"
        )
    return "\n".join([header, separator, *body])


def _write_plots(
    output_dir: Path,
    metrics: list[dict[str, float | int | str]],
    claims: list[dict[str, float | int | str | bool]],
) -> list[Path]:
    return [
        _write_metric_plot(output_dir, metrics, "position_rmse_to_reference", "Translation RMSE to Dynamic Reference", "s3r3_dynamic_position_rmse_to_reference.png"),
        _write_claim_plot(output_dir, claims, "position_rmse_to_reference_gain_pct", "R1+R2 Reference RMSE Gain [%]", "s3r3_dynamic_reference_rmse_gain.png"),
        _write_metric_plot(output_dir, metrics, "mean_nees_to_truth", "Mean Position NEES to Truth", "s3r3_dynamic_highres_nees.png"),
        _write_metric_plot(output_dir, metrics, "runtime_ratio_to_reference", "Runtime / Reference Runtime", "s3r3_dynamic_runtime_ratio_to_reference.png"),
    ]


def _write_metric_plot(output_dir: Path, rows: list[dict[str, float | int | str]], metric_name: str, y_label: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant in SUPPORTED_S3R3_VARIANTS:
        ordered = sorted([row for row in rows if row["variant"] == variant], key=lambda row: int(row["grid_size"]))
        if ordered:
            ax.plot(
                [int(row["grid_size"]) for row in ordered],
                [float(row[metric_name]) for row in ordered],
                marker="o",
                linewidth=1.8,
                label=VARIANT_LABELS[variant],
            )
    ax.set_xlabel("Number of quaternion grid cells")
    ax.set_ylabel(y_label)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, filename)


def _write_claim_plot(output_dir: Path, claims: list[dict[str, float | int | str | bool]], metric_name: str, y_label: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    baseline_claims = sorted(
        [claim for claim in claims if claim["comparator_variant"] == "baseline"],
        key=lambda claim: int(claim["grid_size"]),
    )
    ax.plot(
        [int(claim["grid_size"]) for claim in baseline_claims],
        [float(claim[metric_name]) for claim in baseline_claims],
        marker="o",
        linewidth=1.8,
        label="R1+R2 vs coarse baseline",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Number of quaternion grid cells")
    ax.set_ylabel(y_label)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, filename)
