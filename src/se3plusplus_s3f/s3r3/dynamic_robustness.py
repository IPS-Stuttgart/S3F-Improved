"""Robustness sweep for dynamic S3+ x R3 relaxed S3F prediction."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..s1r2.plotting import format_plot_list, save_figure
from .dynamic_pose import (
    DYNAMIC_POSE_CLAIM_FIELDNAMES,
    DYNAMIC_POSE_METRIC_FIELDNAMES,
    S3R3DynamicPoseConfig,
    run_s3r3_dynamic_pose_benchmark,
)
from .relaxed_s3f_prototype import S3R3PrototypeConfig, SUPPORTED_S3R3_VARIANTS, validate_s3r3_prototype_config


DYNAMIC_ROBUSTNESS_METRIC_FIELDNAMES = [
    "scenario_id",
    "seed",
    "orientation_increment_scale",
    *DYNAMIC_POSE_METRIC_FIELDNAMES,
]

DYNAMIC_ROBUSTNESS_CLAIM_FIELDNAMES = [
    "scenario_id",
    "seed",
    "orientation_increment_scale",
    "orientation_increment_norm_rad",
    *DYNAMIC_POSE_CLAIM_FIELDNAMES,
]

DYNAMIC_ROBUSTNESS_AGGREGATE_FIELDNAMES = [
    "orientation_increment_scale",
    "orientation_increment_norm_rad",
    "grid_size",
    "comparison",
    "candidate_variant",
    "comparator_variant",
    "scenario_count",
    "win_count",
    "win_rate",
    "overall_support_count",
    "overall_support_rate",
    "mean_position_rmse_gain_pct",
    "median_position_rmse_gain_pct",
    "min_position_rmse_gain_pct",
    "mean_position_rmse_ratio",
    "mean_nees_ratio",
    "mean_coverage_delta",
    "mean_orientation_point_error_delta_rad",
    "mean_runtime_ratio",
    "max_runtime_ratio",
]

BASELINE_COMPARISON = "R1+R2 vs baseline"
INFLATION_COMPARISON = "R1+R2 vs R1"


@dataclass(frozen=True)
class S3R3DynamicRobustnessConfig:
    """Configuration for sweeping dynamic S3+ x R3 prediction across seeds and turn rates."""

    prototype: S3R3PrototypeConfig = field(
        default_factory=lambda: S3R3PrototypeConfig(
            grid_sizes=(8, 16, 32, 64),
            variants=SUPPORTED_S3R3_VARIANTS,
            n_trials=32,
            n_steps=12,
            seed=47,
            cell_sample_count=27,
        )
    )
    seeds: tuple[int, ...] = (47, 48, 49, 50, 51)
    orientation_increment: tuple[float, float, float] = (0.0, 0.18, 0.06)
    orientation_increment_scales: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    orientation_transition_kappa: float = 24.0


@dataclass(frozen=True)
class S3R3DynamicRobustnessResult:
    """Container for dynamic robustness sweep outputs."""

    metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]
    aggregates: list[dict[str, float | int | str]]


def s3r3_dynamic_robustness_config_to_dict(config: S3R3DynamicRobustnessConfig) -> dict[str, Any]:
    """Return a JSON-serializable dynamic-robustness config."""

    return json.loads(json.dumps(asdict(config)))


def run_s3r3_dynamic_robustness_sweep(
    config: S3R3DynamicRobustnessConfig = S3R3DynamicRobustnessConfig(),
) -> S3R3DynamicRobustnessResult:
    """Run dynamic S3+ x R3 prediction across seeds and orientation-increment scales."""

    _validate_config(config)
    metrics: list[dict[str, float | int | str]] = []
    claims: list[dict[str, float | int | str | bool]] = []
    base_increment = np.asarray(config.orientation_increment, dtype=float)

    for seed in config.seeds:
        for scale in config.orientation_increment_scales:
            scenario_id = _scenario_id(seed, scale)
            scaled_increment = base_increment * scale
            scenario_config = S3R3DynamicPoseConfig(
                prototype=replace(config.prototype, seed=seed),
                orientation_increment=tuple(float(value) for value in scaled_increment),
                orientation_transition_kappa=config.orientation_transition_kappa,
            )
            result = run_s3r3_dynamic_pose_benchmark(scenario_config)
            metrics.extend(_scenario_metric_row(row, scenario_id, seed, scale) for row in result.metrics)
            increment_norm = float(np.linalg.norm(scaled_increment))
            claims.extend(_scenario_claim_row(row, scenario_id, seed, scale, increment_norm) for row in result.claims)

    return S3R3DynamicRobustnessResult(metrics=metrics, claims=claims, aggregates=_build_aggregate_rows(claims))


def write_s3r3_dynamic_robustness_outputs(
    output_dir: Path,
    config: S3R3DynamicRobustnessConfig = S3R3DynamicRobustnessConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the dynamic robustness sweep and write CSVs, metadata, note, and optional plots."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s3r3_dynamic_robustness_sweep(config)

    metrics_path = output_dir / "s3r3_dynamic_robustness_metrics.csv"
    _write_csv(metrics_path, result.metrics, DYNAMIC_ROBUSTNESS_METRIC_FIELDNAMES)

    claims_path = output_dir / "s3r3_dynamic_robustness_claims.csv"
    _write_csv(claims_path, result.claims, DYNAMIC_ROBUSTNESS_CLAIM_FIELDNAMES)

    aggregates_path = output_dir / "s3r3_dynamic_robustness_aggregates.csv"
    _write_csv(aggregates_path, result.aggregates, DYNAMIC_ROBUSTNESS_AGGREGATE_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path, "aggregates": aggregates_path}
    plot_paths = _write_plots(output_dir, result.aggregates) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_dynamic_robustness_note.md"
    _write_note(note_path, result, metrics_path, claims_path, aggregates_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: S3R3DynamicRobustnessConfig) -> None:
    validate_s3r3_prototype_config(config.prototype, required_variants=SUPPORTED_S3R3_VARIANTS)
    if not config.seeds:
        raise ValueError("seeds must not be empty.")
    if not config.orientation_increment_scales:
        raise ValueError("orientation_increment_scales must not be empty.")
    if min(config.orientation_increment_scales) <= 0.0:
        raise ValueError("all orientation_increment_scales must be positive.")
    if config.orientation_transition_kappa <= 0.0:
        raise ValueError("orientation_transition_kappa must be positive.")
    if np.asarray(config.orientation_increment, dtype=float).shape != (3,):
        raise ValueError("orientation_increment must contain exactly three tangent-vector entries.")


def _scenario_metric_row(
    row: dict[str, float | int | str],
    scenario_id: str,
    seed: int,
    scale: float,
) -> dict[str, float | int | str]:
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "orientation_increment_scale": scale,
        **row,
    }


def _scenario_claim_row(
    row: dict[str, float | int | str | bool],
    scenario_id: str,
    seed: int,
    scale: float,
    increment_norm: float,
) -> dict[str, float | int | str | bool]:
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "orientation_increment_scale": scale,
        "orientation_increment_norm_rad": increment_norm,
        **row,
    }


def _build_aggregate_rows(
    claims: list[dict[str, float | int | str | bool]],
) -> list[dict[str, float | int | str]]:
    rows = []
    keys = sorted(
        {
            (float(row["orientation_increment_scale"]), int(row["grid_size"]), str(row["comparison"]))
            for row in claims
        }
    )
    for scale, grid_size, comparison in keys:
        group = [
            row
            for row in claims
            if float(row["orientation_increment_scale"]) == scale and int(row["grid_size"]) == grid_size and str(row["comparison"]) == comparison
        ]
        rows.append(_aggregate_row(scale, grid_size, comparison, group))
    return rows


def _aggregate_row(
    scale: float,
    grid_size: int,
    comparison: str,
    rows: list[dict[str, float | int | str | bool]],
) -> dict[str, float | int | str]:
    first = rows[0]
    gains = [float(row["position_rmse_gain_pct"]) for row in rows]
    runtime_ratios = [float(row["runtime_ratio"]) for row in rows]
    scenario_count = len(rows)
    win_count = sum(1 for gain in gains if gain > 0.0)
    support_count = sum(1 for row in rows if bool(row["supports_overall_claim"]))
    return {
        "orientation_increment_scale": scale,
        "orientation_increment_norm_rad": float(first["orientation_increment_norm_rad"]),
        "grid_size": grid_size,
        "comparison": comparison,
        "candidate_variant": str(first["candidate_variant"]),
        "comparator_variant": str(first["comparator_variant"]),
        "scenario_count": scenario_count,
        "win_count": win_count,
        "win_rate": win_count / scenario_count,
        "overall_support_count": support_count,
        "overall_support_rate": support_count / scenario_count,
        "mean_position_rmse_gain_pct": _mean(rows, "position_rmse_gain_pct"),
        "median_position_rmse_gain_pct": _median(gains),
        "min_position_rmse_gain_pct": min(gains),
        "mean_position_rmse_ratio": _mean(rows, "position_rmse_ratio"),
        "mean_nees_ratio": _mean(rows, "mean_nees_ratio"),
        "mean_coverage_delta": _mean(rows, "coverage_delta"),
        "mean_orientation_point_error_delta_rad": _mean(rows, "orientation_point_error_delta_rad"),
        "mean_runtime_ratio": _mean(rows, "runtime_ratio"),
        "max_runtime_ratio": max(runtime_ratios),
    }


def _mean(rows: list[dict[str, float | int | str | bool]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _write_metadata(path: Path, result: S3R3DynamicRobustnessResult, config: S3R3DynamicRobustnessConfig) -> None:
    metadata = {
        "aggregates_rows": len(result.aggregates),
        "aggregates_schema": DYNAMIC_ROBUSTNESS_AGGREGATE_FIELDNAMES,
        "claims_rows": len(result.claims),
        "claims_schema": DYNAMIC_ROBUSTNESS_CLAIM_FIELDNAMES,
        "config": s3r3_dynamic_robustness_config_to_dict(config),
        "experiment": "s3r3_dynamic_robustness",
        "metrics_rows": len(result.metrics),
        "metrics_schema": DYNAMIC_ROBUSTNESS_METRIC_FIELDNAMES,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    result: S3R3DynamicRobustnessResult,
    metrics_path: Path,
    claims_path: Path,
    aggregates_path: Path,
    plot_paths: list[Path],
    config: S3R3DynamicRobustnessConfig,
) -> None:
    baseline_claims = [claim for claim in result.claims if claim["comparison"] == BASELINE_COMPARISON]
    baseline_aggregates = [row for row in result.aggregates if row["comparison"] == BASELINE_COMPARISON]
    support_count = sum(1 for claim in baseline_claims if bool(claim["supports_overall_claim"]))
    win_count = sum(1 for claim in baseline_claims if float(claim["position_rmse_gain_pct"]) > 0.0)
    worst_aggregate = min(baseline_aggregates, key=lambda row: float(row["mean_position_rmse_gain_pct"]))
    best_aggregate = max(baseline_aggregates, key=lambda row: float(row["mean_position_rmse_gain_pct"]))

    lines = [
        "# Dynamic S3+ x R3 Robustness Sweep",
        "",
        "This report repeats the dynamic S3+ x R3 pose benchmark across random seeds and orientation-increment scales.",
        "It checks whether the R1+R2 benefit survives changes in trial randomness and turn-rate magnitude.",
        "",
        f"Seeds: {list(config.seeds)}",
        f"Orientation-increment base tangent vector: {list(config.orientation_increment)}",
        f"Orientation-increment scales: {list(config.orientation_increment_scales)}",
        f"Orientation transition kappa: {config.orientation_transition_kappa:g}",
        f"Trials per scenario: {config.prototype.n_trials}",
        f"Steps per trial: {config.prototype.n_steps}",
        f"Grid sizes: {list(config.prototype.grid_sizes)}",
        f"Cell sample count: {config.prototype.cell_sample_count}",
        f"Metrics: `{metrics_path.name}`",
        f"Claims: `{claims_path.name}`",
        f"Aggregates: `{aggregates_path.name}`",
        "",
        "## Headline",
        "",
        f"`R1+R2` beats baseline RMSE in `{win_count}/{len(baseline_claims)}` seed/scale/grid rows and supports the overall baseline claim in `{support_count}/{len(baseline_claims)}` rows.",
        (
            f"Best mean RMSE gain is `{float(best_aggregate['mean_position_rmse_gain_pct']):.1f}%` "
            f"at scale `{best_aggregate['orientation_increment_scale']}`, `{best_aggregate['grid_size']}` cells."
        ),
        (
            f"Worst mean RMSE gain is `{float(worst_aggregate['mean_position_rmse_gain_pct']):.1f}%` "
            f"at scale `{worst_aggregate['orientation_increment_scale']}`, `{worst_aggregate['grid_size']}` cells."
        ),
        "",
        "## Baseline Aggregate Table",
        "",
        _format_baseline_table(baseline_aggregates),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "This remains synthetic evidence. It strengthens the dynamic claim by testing repeatability across seeds and turn rates, not by validating against real pose data.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_baseline_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| scale | grid | win rate | support rate | mean RMSE gain % | mean NEES ratio | mean coverage delta | mean runtime ratio |"
    separator = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in sorted(rows, key=lambda item: (float(item["orientation_increment_scale"]), int(item["grid_size"]))):
        body.append(
            "| "
            f"{float(row['orientation_increment_scale']):.3g} | "
            f"{int(row['grid_size'])} | "
            f"{float(row['win_rate']):.2f} | "
            f"{float(row['overall_support_rate']):.2f} | "
            f"{float(row['mean_position_rmse_gain_pct']):.1f} | "
            f"{float(row['mean_nees_ratio']):.3f} | "
            f"{float(row['mean_coverage_delta']):.3f} | "
            f"{float(row['mean_runtime_ratio']):.3f} |"
        )
    return "\n".join([header, separator, *body])


def _write_plots(output_dir: Path, aggregates: list[dict[str, float | int | str]]) -> list[Path]:
    baseline_rows = [row for row in aggregates if row["comparison"] == BASELINE_COMPARISON]
    return [
        _write_gain_by_scale_plot(output_dir, baseline_rows),
        _write_heatmap(output_dir, baseline_rows, "mean_position_rmse_gain_pct", "Mean RMSE gain [%]", "s3r3_dynamic_robustness_rmse_gain_heatmap.png"),
        _write_heatmap(output_dir, baseline_rows, "win_rate", "RMSE win rate", "s3r3_dynamic_robustness_win_rate_heatmap.png"),
        _write_heatmap(output_dir, baseline_rows, "mean_nees_ratio", "Mean NEES ratio", "s3r3_dynamic_robustness_nees_ratio_heatmap.png"),
    ]


def _write_gain_by_scale_plot(output_dir: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for grid_size in sorted({int(row["grid_size"]) for row in rows}):
        series = sorted([row for row in rows if int(row["grid_size"]) == grid_size], key=lambda row: float(row["orientation_increment_scale"]))
        ax.plot(
            [float(row["orientation_increment_scale"]) for row in series],
            [float(row["mean_position_rmse_gain_pct"]) for row in series],
            marker="o",
            linewidth=1.6,
            label=f"{grid_size} cells",
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Orientation-increment scale")
    ax.set_ylabel("Mean R1+R2 RMSE gain over baseline [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, "s3r3_dynamic_robustness_rmse_gain_by_scale.png")


def _write_heatmap(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    metric_name: str,
    label: str,
    filename: str,
) -> Path:
    scales = sorted({float(row["orientation_increment_scale"]) for row in rows})
    grids = sorted({int(row["grid_size"]) for row in rows})
    matrix = [[_aggregate_value(rows, scale, grid_size, metric_name) for grid_size in grids] for scale in scales]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    image = ax.imshow(matrix, aspect="auto", origin="lower")
    ax.set_xticks(range(len(grids)), [str(value) for value in grids])
    ax.set_yticks(range(len(scales)), [f"{value:.3g}" for value in scales])
    ax.set_xlabel("Number of quaternion grid cells")
    ax.set_ylabel("Orientation-increment scale")
    ax.set_title(label)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", color="white" if value < 1.0 else "black")
    fig.colorbar(image, ax=ax, label=label)
    return save_figure(fig, output_dir, filename)


def _aggregate_value(rows: list[dict[str, float | int | str]], scale: float, grid_size: int, metric_name: str) -> float:
    matches = [
        row
        for row in rows
        if float(row["orientation_increment_scale"]) == scale and int(row["grid_size"]) == grid_size
    ]
    if not matches:
        return float("nan")
    return float(matches[0][metric_name])


def _scenario_id(seed: int, scale: float) -> str:
    return f"seed{seed}_s{_number_token(scale)}"


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")
