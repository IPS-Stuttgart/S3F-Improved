"""Runtime profiling for S1 x R2 relaxed S3F phases."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.filters.relaxed_s3f_circular import (
    SUPPORTED_RELAXED_S3F_VARIANTS,
    uniform_circular_cell_statistics,
)

from .plotting import format_plot_list, save_figure
from .relaxed_s3f_pilot import PilotConfig, generate_pilot_trials, make_initial_filter, pilot_config_to_dict
from .s3f_common import linear_position_error_stats, make_linear_likelihood, orientation_mode_and_mean


RUNTIME_PROFILE_FIELDNAMES = [
    "grid_size",
    "variant",
    "setup_ms_per_trial",
    "likelihood_ms_per_step",
    "cell_stats_ms_per_step",
    "covariance_prep_ms_per_step",
    "predict_linear_ms_per_step",
    "update_ms_per_step",
    "metrics_ms_per_step",
    "unclassified_ms_per_step",
    "total_wall_ms_per_step",
    "position_rmse",
    "mean_nees",
    "n_trials",
    "n_steps",
]

TIMED_PHASES = [
    "likelihood_ms_per_step",
    "cell_stats_ms_per_step",
    "covariance_prep_ms_per_step",
    "predict_linear_ms_per_step",
    "update_ms_per_step",
    "metrics_ms_per_step",
    "unclassified_ms_per_step",
]

PHASE_LABELS = {
    "likelihood_ms_per_step": "likelihood",
    "cell_stats_ms_per_step": "cell stats",
    "covariance_prep_ms_per_step": "cov prep",
    "predict_linear_ms_per_step": "predict_linear",
    "update_ms_per_step": "update",
    "metrics_ms_per_step": "metrics",
    "unclassified_ms_per_step": "other",
}


@dataclass(frozen=True)
class RuntimeProfileConfig:
    """Configuration for profiling relaxed S3F runtime phases."""

    pilot: PilotConfig = field(
        default_factory=lambda: PilotConfig(
            n_trials=16,
            n_steps=16,
        )
    )


@dataclass
class _RuntimeAccumulator:
    setup_s: float = 0.0
    likelihood_s: float = 0.0
    cell_stats_s: float = 0.0
    covariance_prep_s: float = 0.0
    predict_linear_s: float = 0.0
    update_s: float = 0.0
    metrics_s: float = 0.0
    total_wall_s: float = 0.0
    position_sq_error: float = 0.0
    nees: float = 0.0
    n_metrics: int = 0


def runtime_profile_config_to_dict(config: RuntimeProfileConfig) -> dict[str, Any]:
    """Return a JSON-serializable runtime profile config."""

    return {"pilot": pilot_config_to_dict(config.pilot)}


def run_s3f_runtime_profile(config: RuntimeProfileConfig = RuntimeProfileConfig()) -> list[dict[str, float | int | str]]:
    """Profile S3F runtime phases on deterministic synthetic trials."""

    trials = generate_pilot_trials(config.pilot)
    rows = []
    for n_cells in config.pilot.grid_sizes:
        for variant in config.pilot.variants:
            if variant not in SUPPORTED_RELAXED_S3F_VARIANTS:
                raise ValueError(f"Unknown variant {variant!r}.")
            rows.append(_profile_variant(config, trials, n_cells, variant))
    return rows


def write_s3f_runtime_profile_outputs(
    output_dir: Path,
    config: RuntimeProfileConfig = RuntimeProfileConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the runtime profile and write CSV, optional plots, metadata, and note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_s3f_runtime_profile(config)

    metrics_path = output_dir / "s3f_runtime_profile.csv"
    _write_csv(metrics_path, rows)

    outputs = {"metrics": metrics_path}
    plot_paths = _write_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3f_runtime_profile_note.md"
    _write_note(note_path, rows, metrics_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, config)
    outputs["metadata"] = metadata_path
    return outputs


def _profile_variant(
    config: RuntimeProfileConfig,
    trials: list[dict[str, np.ndarray | float]],
    n_cells: int,
    variant: str,
) -> dict[str, float | int | str]:
    pilot = config.pilot
    measurement_cov = np.eye(2) * pilot.measurement_noise_std**2
    process_noise_cov = np.eye(2) * pilot.process_noise_std**2
    body_increment = np.asarray(pilot.body_increment, dtype=float)
    accumulator = _RuntimeAccumulator()

    for trial in trials:
        setup_start = perf_counter()
        filter_ = make_initial_filter(pilot, n_cells)
        accumulator.setup_s += perf_counter() - setup_start

        true_positions = np.asarray(trial["positions"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)

        for step, measurement in enumerate(measurements):
            step_start = perf_counter()

            likelihood_start = perf_counter()
            likelihood = make_linear_likelihood(measurement, measurement_cov)
            accumulator.likelihood_s += perf_counter() - likelihood_start

            predict_timings = _profile_predict_circular_relaxed(filter_, body_increment, variant, process_noise_cov)
            accumulator.cell_stats_s += predict_timings["cell_stats_s"]
            accumulator.covariance_prep_s += predict_timings["covariance_prep_s"]
            accumulator.predict_linear_s += predict_timings["predict_linear_s"]

            update_start = perf_counter()
            filter_.update(likelihoods_linear=[likelihood])
            accumulator.update_s += perf_counter() - update_start

            metrics_start = perf_counter()
            error, nees = linear_position_error_stats(filter_, true_positions[step + 1])
            orientation_mode_and_mean(filter_)
            accumulator.metrics_s += perf_counter() - metrics_start

            accumulator.position_sq_error += float(error @ error)
            accumulator.nees += nees
            accumulator.n_metrics += 1
            accumulator.total_wall_s += perf_counter() - step_start

    return _row_from_accumulator(n_cells, variant, accumulator, config)


def _profile_predict_circular_relaxed(filter_, body_increment: np.ndarray, variant: str, process_noise_cov: np.ndarray) -> dict[str, float]:
    state = filter_.filter_state
    n_cells = len(state.linear_distributions)
    if state.lin_dim != 2:
        raise ValueError("runtime profiling requires a 2-D linear state.")

    stats_start = perf_counter()
    grid = np.asarray(state.gd.get_grid(), dtype=float).reshape(-1)
    stats = uniform_circular_cell_statistics(n_cells, body_increment, grid=grid)
    cell_stats_s = perf_counter() - stats_start

    prep_start = perf_counter()
    if variant == "baseline":
        displacements = stats.representative_displacements
        covariance_inflations = np.zeros_like(stats.covariance_inflations)
    elif variant == "r1":
        displacements = stats.mean_displacements
        covariance_inflations = np.zeros_like(stats.covariance_inflations)
    elif variant == "r1_r2":
        displacements = stats.mean_displacements
        covariance_inflations = stats.covariance_inflations
    else:
        raise ValueError(f"Unknown variant {variant!r}.")

    covariance_matrices = np.stack(
        [process_noise_cov + covariance_inflations[idx] for idx in range(n_cells)],
        axis=2,
    )
    covariance_prep_s = perf_counter() - prep_start

    predict_start = perf_counter()
    filter_.predict_linear(
        covariance_matrices=np.asarray(covariance_matrices),
        linear_input_vectors=np.asarray(displacements.T),
    )
    predict_linear_s = perf_counter() - predict_start
    return {
        "cell_stats_s": cell_stats_s,
        "covariance_prep_s": covariance_prep_s,
        "predict_linear_s": predict_linear_s,
    }


def _row_from_accumulator(
    n_cells: int,
    variant: str,
    accumulator: _RuntimeAccumulator,
    config: RuntimeProfileConfig,
) -> dict[str, float | int | str]:
    n_steps = accumulator.n_metrics
    timed_s = (
        accumulator.likelihood_s
        + accumulator.cell_stats_s
        + accumulator.covariance_prep_s
        + accumulator.predict_linear_s
        + accumulator.update_s
        + accumulator.metrics_s
    )
    unclassified_s = max(accumulator.total_wall_s - timed_s, 0.0)
    row: dict[str, float | int | str] = {
        "grid_size": n_cells,
        "variant": variant,
        "setup_ms_per_trial": 1000.0 * accumulator.setup_s / config.pilot.n_trials,
        "likelihood_ms_per_step": 1000.0 * accumulator.likelihood_s / n_steps,
        "cell_stats_ms_per_step": 1000.0 * accumulator.cell_stats_s / n_steps,
        "covariance_prep_ms_per_step": 1000.0 * accumulator.covariance_prep_s / n_steps,
        "predict_linear_ms_per_step": 1000.0 * accumulator.predict_linear_s / n_steps,
        "update_ms_per_step": 1000.0 * accumulator.update_s / n_steps,
        "metrics_ms_per_step": 1000.0 * accumulator.metrics_s / n_steps,
        "unclassified_ms_per_step": 1000.0 * unclassified_s / n_steps,
        "total_wall_ms_per_step": 1000.0 * accumulator.total_wall_s / n_steps,
        "position_rmse": float(np.sqrt(accumulator.position_sq_error / n_steps)),
        "mean_nees": accumulator.nees / n_steps,
    }
    row["n_steps"] = config.pilot.n_steps
    row["n_trials"] = config.pilot.n_trials
    return row


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(RUNTIME_PROFILE_FIELDNAMES)
        for row in rows:
            writer.writerow([row[field] for field in RUNTIME_PROFILE_FIELDNAMES])


def _write_metadata(path: Path, rows: list[dict[str, float | int | str]], config: RuntimeProfileConfig) -> None:
    metadata: dict[str, Any] = {}
    metadata["experiment"] = "s3f_runtime_profile"
    metadata["config"] = runtime_profile_config_to_dict(config)
    metadata["metrics_rows"] = len(rows)
    metadata["metrics_schema"] = list(RUNTIME_PROFILE_FIELDNAMES)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    paths = [
        _write_phase_stack_plot(output_dir, rows),
        _write_total_runtime_plot(output_dir, rows),
    ]
    return paths


def _write_phase_stack_plot(output_dir: Path, rows: list[dict[str, float | int | str]]) -> Path:
    labels = [f"{row['variant']} {row['grid_size']}" for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bottoms = np.zeros(len(rows), dtype=float)
    for phase in TIMED_PHASES:
        values = np.asarray([float(row[phase]) for row in rows])
        ax.bar(labels, values, bottom=bottoms, label=PHASE_LABELS[phase])
        bottoms += values
    ax.set_ylabel("Runtime [ms/step]")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    return save_figure(fig, output_dir, "s3f_runtime_phase_stack.png")


def _write_total_runtime_plot(output_dir: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    variants = tuple(dict.fromkeys(str(row["variant"]) for row in rows))
    for variant in variants:
        variant_rows = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda row: int(row["grid_size"]),
        )
        ax.plot(
            [int(row["grid_size"]) for row in variant_rows],
            [float(row["total_wall_ms_per_step"]) for row in variant_rows],
            marker="o",
            linewidth=1.8,
            label=variant,
        )
    ax.set_xlabel("Number of circular cells")
    ax.set_ylabel("Total runtime [ms/step]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, "s3f_total_runtime_vs_grid.png")


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    metrics_path: Path,
    plot_paths: list[Path],
    config: RuntimeProfileConfig,
) -> None:
    slowest = max(rows, key=lambda row: float(row["total_wall_ms_per_step"]))
    dominant_phase = max(TIMED_PHASES, key=lambda phase: float(slowest[phase]))
    dominant_share = float(slowest[dominant_phase]) / float(slowest["total_wall_ms_per_step"])
    lines = [
        "# S3F Runtime Profile Note",
        "",
        "The S1 x R2 relaxed S3F loop was profiled by timing likelihood creation,",
        "circular cell statistics, covariance/input preparation, PyRecEst `predict_linear`,",
        "PyRecEst update, metric bookkeeping, and unclassified loop overhead.",
        "",
        f"Trials: {config.pilot.n_trials}",
        f"Steps per trial: {config.pilot.n_steps}",
        f"Grid sizes: {list(config.pilot.grid_sizes)}",
        f"Variants: {list(config.pilot.variants)}",
        f"Metrics file: `{metrics_path.name}`",
        "",
        (
            "Slowest row: "
            f"`{slowest['variant']}` at `{slowest['grid_size']}` cells with "
            f"`{float(slowest['total_wall_ms_per_step']):.3f}` ms/step."
        ),
        (
            "Dominant phase there: "
            f"`{PHASE_LABELS[dominant_phase]}` at `{float(slowest[dominant_phase]):.3f}` ms/step "
            f"({dominant_share:.1%})."
        ),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "Use this profile to decide whether optimization belongs in this experiment",
        "layer, the relaxed circular helper, or PyRecEst's S3F prediction/update internals.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
