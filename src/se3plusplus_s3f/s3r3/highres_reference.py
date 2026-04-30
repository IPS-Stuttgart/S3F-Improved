"""High-resolution S3F reference benchmark for the S3+ x R3 prototype."""

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
from .relaxed_s3f_prototype import (
    SUPPORTED_S3R3_VARIANTS,
    VARIANT_LABELS,
    S3R3PrototypeConfig,
    generate_s3r3_trials,
    make_s3r3_filter,
    predict_s3r3_relaxed,
    s3r3_linear_position_error_stats,
    s3r3_linear_position_mean,
    s3r3_orientation_distance,
    s3r3_orientation_mode,
)


REFERENCE_VARIANT = "baseline"
S3R3_HIGHRES_FIELDNAMES = [
    "grid_size",
    "reference_grid_size",
    "variant",
    "reference_variant",
    "position_rmse_to_reference",
    "orientation_mode_error_to_reference_rad",
    "position_rmse_to_truth",
    "reference_position_rmse_to_truth",
    "orientation_mode_error_to_truth_rad",
    "reference_orientation_mode_error_to_truth_rad",
    "mean_nees_to_truth",
    "coverage_95_to_truth",
    "runtime_ms_per_step",
    "reference_runtime_ms_per_step",
    "runtime_ratio_to_reference",
    "cell_sample_count",
    "n_trials",
    "n_steps",
]


@dataclass(frozen=True)
class S3R3HighResReferenceConfig:
    """Configuration for S3+ x R3 coarse-grid comparison against a denser S3F reference."""

    prototype: S3R3PrototypeConfig = field(
        default_factory=lambda: S3R3PrototypeConfig(
            grid_sizes=(8, 16, 32),
            n_trials=8,
            n_steps=8,
            seed=29,
        )
    )
    reference_grid_size: int = 64


@dataclass
class _ComparisonTotals:
    position_ref_sq_error: float = 0.0
    orientation_ref_error: float = 0.0
    position_truth_sq_error: float = 0.0
    orientation_truth_error: float = 0.0
    nees_sum: float = 0.0
    coverage_hits: int = 0
    runtime_s: float = 0.0


def s3r3_highres_reference_config_to_dict(config: S3R3HighResReferenceConfig) -> dict[str, Any]:
    """Return a JSON-serializable S3R3 high-resolution reference config."""

    return json.loads(json.dumps(asdict(config) | {"reference_variant": REFERENCE_VARIANT}))


def run_s3r3_highres_reference_benchmark(
    config: S3R3HighResReferenceConfig = S3R3HighResReferenceConfig(),
) -> list[dict[str, float | int | str]]:
    """Compare coarse S3+ x R3 relaxed S3F variants against a denser baseline S3F reference."""

    _validate_config(config)
    prototype = config.prototype
    trials = generate_s3r3_trials(prototype)
    measurement_cov = np.eye(3) * prototype.measurement_noise_std**2
    process_noise_cov = np.eye(3) * prototype.process_noise_std**2
    body_increment = np.asarray(prototype.body_increment, dtype=float)
    keys = [(grid_size, variant) for grid_size in prototype.grid_sizes for variant in prototype.variants]
    totals = {key: _ComparisonTotals() for key in keys}
    reference_totals = _ComparisonTotals()
    metric_count = 0

    for trial in trials:
        reference_filter = make_s3r3_filter(prototype, config.reference_grid_size)
        candidate_filters = {key: make_s3r3_filter(prototype, key[0]) for key in keys}
        true_orientation, true_positions, measurements = _trial_arrays(trial)

        for step, measurement in enumerate(measurements):
            true_position = true_positions[step + 1]
            reference_runtime = _predict_update(
                reference_filter,
                measurement,
                measurement_cov,
                body_increment,
                REFERENCE_VARIANT,
                process_noise_cov,
                prototype.cell_sample_count,
            )
            reference_totals.runtime_s += reference_runtime
            reference_mean = s3r3_linear_position_mean(reference_filter)
            reference_mode = s3r3_orientation_mode(reference_filter)
            reference_error, _reference_nees = s3r3_linear_position_error_stats(reference_filter, true_position)
            reference_totals.position_truth_sq_error += float(reference_error @ reference_error)
            reference_totals.orientation_truth_error += s3r3_orientation_distance(reference_mode, true_orientation)

            for key, candidate_filter in candidate_filters.items():
                elapsed = _predict_update(
                    candidate_filter,
                    measurement,
                    measurement_cov,
                    body_increment,
                    key[1],
                    process_noise_cov,
                    prototype.cell_sample_count,
                )
                _accumulate_candidate(
                    totals[key],
                    candidate_filter,
                    true_position,
                    true_orientation,
                    reference_mean,
                    reference_mode,
                    elapsed,
                )

            metric_count += 1

    reference_rmse = float(np.sqrt(reference_totals.position_truth_sq_error / metric_count))
    reference_orientation_error = reference_totals.orientation_truth_error / metric_count
    reference_runtime_ms = 1000.0 * reference_totals.runtime_s / metric_count
    return [
        _row_from_totals(
            grid_size=grid_size,
            variant=variant,
            totals=totals[(grid_size, variant)],
            config=config,
            metric_count=metric_count,
            reference_rmse=reference_rmse,
            reference_orientation_error=reference_orientation_error,
            reference_runtime_ms=reference_runtime_ms,
        )
        for grid_size, variant in keys
    ]


def _trial_arrays(trial: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(trial["orientation"], dtype=float),
        np.asarray(trial["positions"], dtype=float),
        np.asarray(trial["measurements"], dtype=float),
    )


def write_s3r3_highres_reference_outputs(
    output_dir: Path,
    config: S3R3HighResReferenceConfig = S3R3HighResReferenceConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the S3R3 high-resolution reference benchmark and write outputs."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_s3r3_highres_reference_benchmark(config)

    metrics_path = output_dir / "s3r3_highres_reference_metrics.csv"
    _write_csv(metrics_path, rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_highres_reference_note.md"
    _write_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
    return outputs


def _predict_update(
    filter_,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
    body_increment: np.ndarray,
    variant: str,
    process_noise_cov: np.ndarray,
    cell_sample_count: int,
) -> float:
    likelihood = GaussianDistribution(measurement, measurement_cov, check_validity=False)
    start = perf_counter()
    predict_s3r3_relaxed(
        filter_,
        body_increment,
        variant=variant,
        process_noise_cov=process_noise_cov,
        cell_sample_count=cell_sample_count,
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
    elapsed_s: float,
) -> None:
    candidate_mean = s3r3_linear_position_mean(candidate_filter)
    candidate_mode = s3r3_orientation_mode(candidate_filter)
    reference_delta = candidate_mean - reference_mean
    truth_error, nees = s3r3_linear_position_error_stats(candidate_filter, true_position)

    totals.position_ref_sq_error += float(reference_delta @ reference_delta)
    totals.orientation_ref_error += s3r3_orientation_distance(candidate_mode, reference_mode)
    totals.position_truth_sq_error += float(truth_error @ truth_error)
    totals.orientation_truth_error += s3r3_orientation_distance(candidate_mode, true_orientation)
    totals.nees_sum += nees
    totals.coverage_hits += int(nees <= 7.814727903251179)
    totals.runtime_s += elapsed_s


def _validate_config(config: S3R3HighResReferenceConfig) -> None:
    prototype = config.prototype
    if not prototype.grid_sizes:
        raise ValueError("grid_sizes must not be empty.")
    if min(prototype.grid_sizes) <= 0:
        raise ValueError("all grid sizes must be positive.")
    if config.reference_grid_size <= max(prototype.grid_sizes):
        raise ValueError("reference_grid_size must be greater than every coarse grid size.")
    if prototype.n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if prototype.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if prototype.cell_sample_count <= 0:
        raise ValueError("cell_sample_count must be positive.")
    for variant in prototype.variants:
        if variant not in SUPPORTED_S3R3_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}.")


def _row_from_totals(
    grid_size: int,
    variant: str,
    totals: _ComparisonTotals,
    config: S3R3HighResReferenceConfig,
    metric_count: int,
    reference_rmse: float,
    reference_orientation_error: float,
    reference_runtime_ms: float,
) -> dict[str, float | int | str]:
    runtime_ms = 1000.0 * totals.runtime_s / metric_count
    return {
        "grid_size": grid_size,
        "reference_grid_size": config.reference_grid_size,
        "variant": variant,
        "reference_variant": REFERENCE_VARIANT,
        "position_rmse_to_reference": float(np.sqrt(totals.position_ref_sq_error / metric_count)),
        "orientation_mode_error_to_reference_rad": totals.orientation_ref_error / metric_count,
        "position_rmse_to_truth": float(np.sqrt(totals.position_truth_sq_error / metric_count)),
        "reference_position_rmse_to_truth": reference_rmse,
        "orientation_mode_error_to_truth_rad": totals.orientation_truth_error / metric_count,
        "reference_orientation_mode_error_to_truth_rad": reference_orientation_error,
        "mean_nees_to_truth": totals.nees_sum / metric_count,
        "coverage_95_to_truth": totals.coverage_hits / metric_count,
        "runtime_ms_per_step": runtime_ms,
        "reference_runtime_ms_per_step": reference_runtime_ms,
        "runtime_ratio_to_reference": runtime_ms / reference_runtime_ms,
        "n_trials": config.prototype.n_trials,
        "n_steps": config.prototype.n_steps,
        "cell_sample_count": config.prototype.cell_sample_count,
    }


def _metric_fieldnames() -> list[str]:
    return list(S3R3_HIGHRES_FIELDNAMES)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = _metric_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows([row[name] for name in fieldnames] for row in rows)


def _write_metadata(
    path: Path,
    rows: list[dict[str, float | int | str]],
    config: S3R3HighResReferenceConfig,
) -> None:
    metadata_items = (
        ("config", s3r3_highres_reference_config_to_dict(config)),
        ("experiment", "s3r3_highres_reference"),
        ("metrics_rows", len(rows)),
        ("metrics_schema", S3R3_HIGHRES_FIELDNAMES),
        ("reference_model", "denser_baseline_s3f"),
    )
    path.write_text(f"{json.dumps(dict(metadata_items), indent=2, sort_keys=True)}\n", encoding="utf-8")


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    metrics_path: Path,
    plot_paths: list[Path],
    config: S3R3HighResReferenceConfig,
) -> None:
    best_reference_match = min(rows, key=lambda row: float(row["position_rmse_to_reference"]))
    fastest_row = min(rows, key=lambda row: float(row["runtime_ratio_to_reference"]))
    best_label = VARIANT_LABELS[str(best_reference_match["variant"])]
    speed_label = VARIANT_LABELS[str(fastest_row["variant"])]
    content = f"""# S3+ x R3 High-Resolution Reference

This benchmark compares coarse S3+ x R3 S3F variants against a denser baseline S3F reference.
It tests whether relaxed coarse-grid propagation follows the denser S3F reference more closely than representative-cell propagation.

- trials: {config.prototype.n_trials}
- steps per trial: {config.prototype.n_steps}
- coarse grid sizes: {list(config.prototype.grid_sizes)}
- reference grid size: {config.reference_grid_size}
- cell sample count: {config.prototype.cell_sample_count}
- metrics file: `{metrics_path.name}`

## Best Reference Match

`{best_label}` at `{best_reference_match["grid_size"]}` cells has RMSE-to-reference `{float(best_reference_match["position_rmse_to_reference"]):.4f}` and orientation mode error-to-reference `{float(best_reference_match["orientation_mode_error_to_reference_rad"]):.3f}` rad.

## Fastest Row

`{speed_label}` at `{fastest_row["grid_size"]}` cells ran at `{float(fastest_row["runtime_ratio_to_reference"]):.3f}` of the reference runtime.

## Metrics

{_format_metrics_table(rows)}

Plots:
{format_plot_list(plot_paths)}
"""
    path.write_text(content, encoding="utf-8")


def _format_metrics_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Variant | Cells | RMSE to ref | Mode error to ref rad | RMSE to truth | NEES | Coverage | Runtime/reference |"
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in sorted(rows, key=lambda item: (int(item["grid_size"]), str(item["variant"]))):
        cells = (
            VARIANT_LABELS[str(row["variant"])],
            str(int(row["grid_size"])),
            f"{float(row['position_rmse_to_reference']):.4f}",
            f"{float(row['orientation_mode_error_to_reference_rad']):.3f}",
            f"{float(row['position_rmse_to_truth']):.4f}",
            f"{float(row['mean_nees_to_truth']):.3f}",
            f"{float(row['coverage_95_to_truth']):.3f}",
            f"{float(row['runtime_ratio_to_reference']):.3f}",
        )
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    plot_specs = [
        ("position_rmse_to_reference", "Translation RMSE to Reference", "s3r3_position_rmse_to_reference.png"),
        ("orientation_mode_error_to_reference_rad", "Mode Error to Reference [rad]", "s3r3_orientation_mode_error_to_reference.png"),
        ("runtime_ratio_to_reference", "Runtime / Reference Runtime", "s3r3_runtime_ratio_to_reference.png"),
    ]
    paths = []
    for metric_name, y_label, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for variant, xs, ys in _plot_series(rows, metric_name):
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=VARIANT_LABELS[variant])
        ax.set_xlabel("Number of quaternion grid cells")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        paths.append(save_figure(fig, output_dir, filename))
    return paths


def _plot_series(
    rows: list[dict[str, float | int | str]],
    metric_name: str,
) -> list[tuple[str, list[int], list[float]]]:
    series = []
    for variant in SUPPORTED_S3R3_VARIANTS:
        ordered = sorted((row for row in rows if row["variant"] == variant), key=lambda row: int(row["grid_size"]))
        if ordered:
            series.append(
                (
                    variant,
                    [int(row["grid_size"]) for row in ordered],
                    [float(row[metric_name]) for row in ordered],
                )
            )
    return series
