"""High-resolution S3F reference benchmark for the S1 x R2 model problem."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pyrecest.filters.relaxed_s3f_circular import SUPPORTED_RELAXED_S3F_VARIANTS, circular_error

from .plotting import format_plot_list, write_metric_line_plots
from .relaxed_s3f_pilot import (
    VARIANT_LABELS,
    PilotConfig,
    generate_pilot_trials,
    make_initial_filter,
    pilot_config_to_dict,
)
from .s3f_common import (
    linear_position_error_stats,
    linear_position_mean,
    make_linear_likelihood,
    orientation_mode_and_mean,
    predict_update_linear_position,
)


HIGHRES_REFERENCE_FIELDNAMES = [
    "grid_size",
    "reference_grid_size",
    "variant",
    "reference_variant",
    "position_rmse_to_reference",
    "orientation_mode_error_to_reference_rad",
    "orientation_mean_error_to_reference_rad",
    "position_rmse_to_truth",
    "reference_position_rmse_to_truth",
    "mean_nees_to_truth",
    "coverage_95_to_truth",
    "runtime_ms_per_step",
    "reference_runtime_ms_per_step",
    "runtime_ratio_to_reference",
    "n_trials",
    "n_steps",
]

REFERENCE_VARIANT = "baseline"


@dataclass(frozen=True)
class HighResReferenceConfig:
    """Configuration for coarse-grid S3F comparison against a fine reference."""

    pilot: PilotConfig = field(
        default_factory=lambda: PilotConfig(
            n_trials=16,
            n_steps=16,
            seed=17,
        )
    )
    reference_grid_size: int = 256


@dataclass
class _CandidateAccumulator:
    position_ref_sq_error: float = 0.0
    orientation_mode_ref_error: float = 0.0
    orientation_mean_ref_error: float = 0.0
    position_truth_sq_error: float = 0.0
    nees_sum: float = 0.0
    coverage_hits: int = 0
    runtime_s: float = 0.0


def highres_reference_config_to_dict(config: HighResReferenceConfig) -> dict[str, Any]:
    """Return a JSON-serializable high-resolution reference config."""

    return {
        "pilot": pilot_config_to_dict(config.pilot),
        "reference_grid_size": config.reference_grid_size,
        "reference_variant": REFERENCE_VARIANT,
    }


def run_highres_reference_benchmark(
    config: HighResReferenceConfig = HighResReferenceConfig(),
) -> list[dict[str, float | int | str]]:
    """Compare coarse relaxed S3F variants against a high-resolution baseline."""

    _validate_config(config)
    pilot_config = config.pilot
    trials = generate_pilot_trials(pilot_config)
    measurement_cov = np.eye(2) * pilot_config.measurement_noise_std**2
    process_noise_cov = np.eye(2) * pilot_config.process_noise_std**2
    body_increment = np.asarray(pilot_config.body_increment, dtype=float)

    keys = [(grid_size, variant) for grid_size in pilot_config.grid_sizes for variant in pilot_config.variants]
    accumulators = {key: _CandidateAccumulator() for key in keys}
    reference_truth_sq_error = 0.0
    reference_runtime_s = 0.0
    n_metrics = 0

    for trial in trials:
        reference_filter = make_initial_filter(pilot_config, config.reference_grid_size)
        candidate_filters = {
            key: make_initial_filter(pilot_config, key[0])
            for key in keys
        }
        true_positions = np.asarray(trial["positions"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)

        for step, measurement in enumerate(measurements):
            next_position = true_positions[step + 1]
            reference_likelihood = make_linear_likelihood(measurement, measurement_cov)
            reference_runtime_s += predict_update_linear_position(
                reference_filter,
                body_increment,
                REFERENCE_VARIANT,
                process_noise_cov,
                reference_likelihood,
            )
            reference_mean = linear_position_mean(reference_filter)
            reference_mode, reference_circular_mean = orientation_mode_and_mean(reference_filter)
            reference_error, _ = linear_position_error_stats(reference_filter, next_position)
            reference_truth_sq_error += float(reference_error @ reference_error)

            for key, candidate_filter in candidate_filters.items():
                _, variant = key
                likelihood = make_linear_likelihood(measurement, measurement_cov)
                accumulator = accumulators[key]
                accumulator.runtime_s += predict_update_linear_position(
                    candidate_filter,
                    body_increment,
                    variant,
                    process_noise_cov,
                    likelihood,
                )
                candidate_mean = linear_position_mean(candidate_filter)
                candidate_mode, candidate_circular_mean = orientation_mode_and_mean(candidate_filter)
                reference_delta = candidate_mean - reference_mean
                truth_error, nees = linear_position_error_stats(candidate_filter, next_position)

                accumulator.position_ref_sq_error += float(reference_delta @ reference_delta)
                accumulator.orientation_mode_ref_error += circular_error(candidate_mode, reference_mode)
                accumulator.orientation_mean_ref_error += circular_error(
                    candidate_circular_mean,
                    reference_circular_mean,
                )
                accumulator.position_truth_sq_error += float(truth_error @ truth_error)
                accumulator.nees_sum += nees
                accumulator.coverage_hits += int(nees <= 5.991464547107979)

            n_metrics += 1

    reference_position_rmse = float(np.sqrt(reference_truth_sq_error / n_metrics))
    reference_runtime_ms = 1000.0 * reference_runtime_s / n_metrics
    return [
        _row_from_accumulator(
            grid_size=grid_size,
            variant=variant,
            accumulator=accumulators[(grid_size, variant)],
            config=config,
            n_metrics=n_metrics,
            reference_position_rmse=reference_position_rmse,
            reference_runtime_ms=reference_runtime_ms,
        )
        for grid_size, variant in keys
    ]


def write_highres_reference_outputs(
    output_dir: Path,
    config: HighResReferenceConfig = HighResReferenceConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the high-resolution reference benchmark and write outputs."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_highres_reference_benchmark(config)

    metrics_path = output_dir / "highres_reference_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HIGHRES_REFERENCE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "highres_reference_note.md"
    _write_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: HighResReferenceConfig) -> None:
    pilot = config.pilot
    if not pilot.grid_sizes:
        raise ValueError("grid_sizes must not be empty.")
    if min(pilot.grid_sizes) <= 0:
        raise ValueError("all grid sizes must be positive.")
    if config.reference_grid_size <= max(pilot.grid_sizes):
        raise ValueError("reference_grid_size must be greater than every coarse grid size.")
    if pilot.n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if pilot.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if not pilot.variants:
        raise ValueError("variants must not be empty.")
    for variant in pilot.variants:
        if variant not in SUPPORTED_RELAXED_S3F_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}.")


def _row_from_accumulator(
    grid_size: int,
    variant: str,
    accumulator: _CandidateAccumulator,
    config: HighResReferenceConfig,
    n_metrics: int,
    reference_position_rmse: float,
    reference_runtime_ms: float,
) -> dict[str, float | int | str]:
    runtime_ms = 1000.0 * accumulator.runtime_s / n_metrics
    return {
        "grid_size": grid_size,
        "reference_grid_size": config.reference_grid_size,
        "variant": variant,
        "reference_variant": REFERENCE_VARIANT,
        "position_rmse_to_reference": float(np.sqrt(accumulator.position_ref_sq_error / n_metrics)),
        "orientation_mode_error_to_reference_rad": accumulator.orientation_mode_ref_error / n_metrics,
        "orientation_mean_error_to_reference_rad": accumulator.orientation_mean_ref_error / n_metrics,
        "position_rmse_to_truth": float(np.sqrt(accumulator.position_truth_sq_error / n_metrics)),
        "reference_position_rmse_to_truth": reference_position_rmse,
        "mean_nees_to_truth": accumulator.nees_sum / n_metrics,
        "coverage_95_to_truth": accumulator.coverage_hits / n_metrics,
        "runtime_ms_per_step": runtime_ms,
        "reference_runtime_ms_per_step": reference_runtime_ms,
        "runtime_ratio_to_reference": runtime_ms / reference_runtime_ms,
        "n_trials": config.pilot.n_trials,
        "n_steps": config.pilot.n_steps,
    }


def _write_metadata(path: Path, rows: list[dict[str, float | int | str]], config: HighResReferenceConfig) -> None:
    content = {
        "experiment": "highres_reference",
        "config": highres_reference_config_to_dict(config),
        "metrics_schema": HIGHRES_REFERENCE_FIELDNAMES,
        "metrics_rows": len(rows),
    }
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    plot_specs = [
        (
            "position_rmse_to_reference",
            "Translation RMSE to Reference",
            "translation_rmse_to_reference_vs_grid.png",
        ),
        (
            "orientation_mean_error_to_reference_rad",
            "Mean Orientation Error to Reference [rad]",
            "orientation_mean_error_to_reference_vs_grid.png",
        ),
        (
            "runtime_ratio_to_reference",
            "Runtime / Reference Runtime",
            "runtime_ratio_to_reference_vs_grid.png",
        ),
    ]
    plotter = write_metric_line_plots
    return plotter(output_dir, rows, plot_specs, SUPPORTED_RELAXED_S3F_VARIANTS, VARIANT_LABELS)


def _write_note(path: Path, rows, metrics_path: Path, plot_paths: list[Path], config: HighResReferenceConfig) -> None:
    best_reference_match = min(rows, key=lambda row: float(row["position_rmse_to_reference"]))
    best_speed = min(rows, key=lambda row: float(row["runtime_ratio_to_reference"]))

    content = f"""# High-Resolution S3F Reference Note

## What Was Run

This benchmark compares coarse-grid S3F variants against a high-resolution
baseline S3F reference on the same synthetic `S1 x R2` trials. The reference
uses `{config.reference_grid_size}` cells and variant `{REFERENCE_VARIANT}`.

- trials: {config.pilot.n_trials}
- steps per trial: {config.pilot.n_steps}
- coarse grid sizes: {list(config.pilot.grid_sizes)}
- metrics: `{metrics_path.name}`

## First Result

Closest translation match to the high-resolution reference:
`{best_reference_match["variant"]}` at grid size `{best_reference_match["grid_size"]}`
with RMSE `{float(best_reference_match["position_rmse_to_reference"]):.4f}`.

Fastest coarse approximation:
`{best_speed["variant"]}` at grid size `{best_speed["grid_size"]}` with runtime ratio
`{float(best_speed["runtime_ratio_to_reference"]):.3f}` relative to the reference.

## Plots

{format_plot_list(plot_paths)}

## Interpretation

This benchmark supports the narrower approximation question: whether relaxed
coarse-grid S3F follows an expensive high-resolution S3F baseline more closely
than representative-cell coarse S3F. It is not a comparison against EKF, UKF,
particle filters, VIO, or full SE(3)+ filtering.
"""
    path.write_text(content, encoding="utf-8")
