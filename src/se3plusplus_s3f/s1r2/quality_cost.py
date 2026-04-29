"""Quality-vs-cost report for S1 x R2 relaxed S3F grids."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .baseline_comparison import ParticleSensitivityConfig, particle_sensitivity_config_to_dict, run_particle_sensitivity
from .highres_reference import HighResReferenceConfig, highres_reference_config_to_dict, run_highres_reference_benchmark
from .plotting import format_plot_list, save_figure
from .relaxed_s3f_pilot import VARIANT_LABELS, PilotConfig


QUALITY_COST_FIELDNAMES = [
    "variant",
    "grid_size",
    "position_rmse",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "position_rmse_to_reference",
    "orientation_mean_error_to_reference_rad",
    "runtime_ratio_to_reference",
    "reference_grid_size",
    "reference_position_rmse",
    "reference_runtime_ms_per_step",
    "n_trials",
    "n_steps",
]

QUALITY_COST_CLAIM_FIELDNAMES = [
    "claim_id",
    "comparison",
    "candidate_variant",
    "candidate_grid_size",
    "comparator_variant",
    "comparator_grid_size",
    "candidate_position_rmse",
    "comparator_position_rmse",
    "position_rmse_ratio",
    "candidate_mean_nees",
    "comparator_mean_nees",
    "mean_nees_ratio",
    "candidate_runtime_ms_per_step",
    "comparator_runtime_ms_per_step",
    "runtime_ratio",
    "candidate_position_rmse_to_reference",
    "comparator_position_rmse_to_reference",
    "reference_rmse_ratio",
    "supports_accuracy_claim",
    "supports_consistency_claim",
    "supports_runtime_claim",
]

QUALITY_COST_PARETO_FIELDNAMES = [
    "filter",
    "label",
    "variant",
    "grid_size",
    "particle_count",
    "resource_count",
    "position_rmse",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "position_rmse_to_reference",
    "runtime_ratio_to_reference",
    "n_trials",
    "n_steps",
]

QUALITY_COST_REPEAT_PARETO_FIELDNAMES = [
    "repeat_index",
    "seed",
    "particle_seed",
    *QUALITY_COST_PARETO_FIELDNAMES,
]

QUALITY_COST_SUMMARY_FIELDNAMES = [
    "filter",
    "label",
    "variant",
    "grid_size",
    "particle_count",
    "resource_count",
    "position_rmse_mean",
    "position_rmse_std",
    "position_rmse_ci95_low",
    "position_rmse_ci95_high",
    "mean_nees_mean",
    "mean_nees_std",
    "coverage_95_mean",
    "coverage_95_std",
    "runtime_ms_per_step_mean",
    "runtime_ms_per_step_std",
    "runtime_ms_per_step_ci95_low",
    "runtime_ms_per_step_ci95_high",
    "n_repeats",
    "n_trials",
    "n_steps",
]

QUALITY_COST_PAIRWISE_FIELDNAMES = [
    "pair_id",
    "candidate_label",
    "comparator_label",
    "candidate_filter",
    "candidate_variant",
    "candidate_resource_count",
    "comparator_filter",
    "comparator_variant",
    "comparator_resource_count",
    "position_rmse_delta_mean",
    "position_rmse_delta_std",
    "position_rmse_delta_ci95_low",
    "position_rmse_delta_ci95_high",
    "candidate_rmse_win_count",
    "comparator_rmse_win_count",
    "runtime_ms_per_step_delta_mean",
    "runtime_ms_per_step_delta_std",
    "runtime_ms_per_step_delta_ci95_low",
    "runtime_ms_per_step_delta_ci95_high",
    "runtime_ratio_mean",
    "candidate_runtime_win_count",
    "comparator_runtime_win_count",
    "candidate_dominance_count",
    "comparator_dominance_count",
    "n_repeats",
    "n_trials",
    "n_steps",
]

QUALITY_COST_VARIANTS = ("baseline", "r1", "r1_r2")
QUALITY_COST_PARTICLE_COUNTS = (128, 512, 2048, 8192)

QUALITY_COST_PAIRWISE_SPECS = [
    {
        "pair_id": "r1_r2_64_vs_particle_2048",
        "candidate": {"filter": "s3f", "variant": "r1_r2", "resource_count": 64},
        "comparator": {"filter": "particle_filter", "variant": "bootstrap", "resource_count": 2048},
    },
    {
        "pair_id": "r1_r2_64_vs_particle_8192",
        "candidate": {"filter": "s3f", "variant": "r1_r2", "resource_count": 64},
        "comparator": {"filter": "particle_filter", "variant": "bootstrap", "resource_count": 8192},
    },
    {
        "pair_id": "r1_r2_32_vs_particle_512",
        "candidate": {"filter": "s3f", "variant": "r1_r2", "resource_count": 32},
        "comparator": {"filter": "particle_filter", "variant": "bootstrap", "resource_count": 512},
    },
    {
        "pair_id": "r1_r2_8_vs_baseline_16",
        "candidate": {"filter": "s3f", "variant": "r1_r2", "resource_count": 8},
        "comparator": {"filter": "s3f", "variant": "baseline", "resource_count": 16},
    },
]


@dataclass(frozen=True)
class QualityCostResult:
    """Container for quality-cost report tables."""

    metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]
    pareto: list[dict[str, float | int | str]]
    repeat_pareto: list[dict[str, float | int | str]] = field(default_factory=list)
    summary: list[dict[str, float | int | str]] = field(default_factory=list)
    pairwise: list[dict[str, float | int | str]] = field(default_factory=list)


@dataclass(frozen=True)
class QualityCostConfig:
    """Configuration for the relaxed S3F quality-vs-cost report."""

    reference: HighResReferenceConfig = field(
        default_factory=lambda: HighResReferenceConfig(
            pilot=PilotConfig(
                variants=QUALITY_COST_VARIANTS,
                n_trials=16,
                n_steps=16,
                seed=17,
            ),
            reference_grid_size=256,
        )
    )
    particle_counts: tuple[int, ...] = QUALITY_COST_PARTICLE_COUNTS
    particle_seed: int = 101
    particle_resample_threshold: float = 0.5
    repeats: int = 1
    repeat_seed_stride: int = 1000


def quality_cost_config_to_dict(config: QualityCostConfig) -> dict[str, Any]:
    """Return a JSON-serializable quality-cost config."""

    return {
        "reference": highres_reference_config_to_dict(config.reference),
        "particle_sensitivity": particle_sensitivity_config_to_dict(_particle_config(config)),
        "repeats": config.repeats,
        "repeat_seed_stride": config.repeat_seed_stride,
    }


def run_quality_cost_report(config: QualityCostConfig = QualityCostConfig()) -> QualityCostResult:
    """Run the quality-vs-cost report from a high-resolution S3F reference sweep."""

    if config.repeats <= 0:
        raise ValueError("repeats must be positive.")

    repeat_results = [_run_single_quality_cost_report(_repeat_config(config, repeat_index)) for repeat_index in range(config.repeats)]
    result = repeat_results[0]
    if config.repeats == 1:
        return result

    repeat_pareto = _repeat_pareto_rows(config, repeat_results)
    summary = _summarize_repeat_pareto(repeat_pareto)
    pairwise = _build_pairwise_rows(repeat_pareto)
    return QualityCostResult(
        metrics=result.metrics,
        claims=result.claims,
        pareto=result.pareto,
        repeat_pareto=repeat_pareto,
        summary=summary,
        pairwise=pairwise,
    )


def _run_single_quality_cost_report(config: QualityCostConfig) -> QualityCostResult:
    reference_rows = run_highres_reference_benchmark(config.reference)
    rows = [_quality_row_from_reference(row) for row in reference_rows]
    claims = _build_claim_rows(rows)
    pareto = _build_pareto_rows(rows, run_particle_sensitivity(_particle_config(config)))
    return QualityCostResult(metrics=rows, claims=claims, pareto=pareto)


def write_quality_cost_outputs(
    output_dir: Path,
    config: QualityCostConfig = QualityCostConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the report and write metrics, claims, optional plots, metadata, and note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_quality_cost_report(config)

    metrics_path = output_dir / "quality_cost_metrics.csv"
    _write_csv(metrics_path, result.metrics, QUALITY_COST_FIELDNAMES)

    claims_path = output_dir / "quality_cost_claims.csv"
    _write_csv(claims_path, result.claims, QUALITY_COST_CLAIM_FIELDNAMES)

    pareto_path = output_dir / "quality_cost_pareto.csv"
    _write_csv(pareto_path, result.pareto, QUALITY_COST_PARETO_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path, "pareto": pareto_path}
    repeat_pareto_path = output_dir / "quality_cost_repeat_pareto.csv"
    summary_path = output_dir / "quality_cost_summary.csv"
    if result.repeat_pareto:
        _write_csv(repeat_pareto_path, result.repeat_pareto, QUALITY_COST_REPEAT_PARETO_FIELDNAMES)
        _write_csv(summary_path, result.summary, QUALITY_COST_SUMMARY_FIELDNAMES)
        outputs["repeat_pareto"] = repeat_pareto_path
        outputs["summary"] = summary_path
    pairwise_path = output_dir / "quality_cost_pairwise.csv"
    if result.pairwise:
        _write_csv(pairwise_path, result.pairwise, QUALITY_COST_PAIRWISE_FIELDNAMES)
        outputs["pairwise"] = pairwise_path

    plot_paths = _write_plots(output_dir, result.metrics, result.pareto) if write_plots else []
    if write_plots and result.pairwise:
        plot_paths.append(_write_pairwise_delta_plot(output_dir, result.pairwise))
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "quality_cost_note.md"
    _write_note(note_path, result, metrics_path, claims_path, pareto_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, config)
    outputs["metadata"] = metadata_path
    return outputs


def _particle_config(config: QualityCostConfig) -> ParticleSensitivityConfig:
    return ParticleSensitivityConfig(
        pilot=config.reference.pilot,
        particle_counts=config.particle_counts,
        particle_seed=config.particle_seed,
        particle_resample_threshold=config.particle_resample_threshold,
    )


def _repeat_config(config: QualityCostConfig, repeat_index: int) -> QualityCostConfig:
    seed_offset = repeat_index * config.repeat_seed_stride
    repeat_pilot = replace(
        config.reference.pilot,
        seed=config.reference.pilot.seed + seed_offset,
    )
    return replace(
        config,
        reference=replace(config.reference, pilot=repeat_pilot),
        particle_seed=config.particle_seed + seed_offset,
        repeats=1,
    )


def _repeat_pareto_rows(
    config: QualityCostConfig,
    repeat_results: list[QualityCostResult],
) -> list[dict[str, float | int | str]]:
    rows = []
    for repeat_index, result in enumerate(repeat_results):
        repeat_config = _repeat_config(config, repeat_index)
        for row in result.pareto:
            rows.append(
                {
                    "repeat_index": repeat_index,
                    "seed": repeat_config.reference.pilot.seed,
                    "particle_seed": repeat_config.particle_seed,
                    **row,
                }
            )
    return rows


def _summarize_repeat_pareto(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (
            str(row["filter"]),
            str(row["label"]),
            str(row["variant"]),
            str(row["grid_size"]),
            str(row["particle_count"]),
            str(row["resource_count"]),
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, group_rows in grouped.items():
        filter_name, label, variant, grid_size, particle_count, resource_count = key
        summary.append(
            {
                "filter": filter_name,
                "label": label,
                "variant": variant,
                "grid_size": grid_size,
                "particle_count": particle_count,
                "resource_count": resource_count,
                **_summary_metric_fields(group_rows, "position_rmse"),
                **_summary_metric_fields(group_rows, "mean_nees", include_ci=False),
                **_summary_metric_fields(group_rows, "coverage_95", include_ci=False),
                **_summary_metric_fields(group_rows, "runtime_ms_per_step"),
                "n_repeats": len(group_rows),
                "n_trials": group_rows[0]["n_trials"],
                "n_steps": group_rows[0]["n_steps"],
            }
        )
    return sorted(summary, key=lambda row: (float(row["runtime_ms_per_step_mean"]), str(row["filter"])))


def _summary_metric_fields(
    rows: list[dict[str, float | int | str]],
    metric: str,
    include_ci: bool = True,
) -> dict[str, float]:
    values = [float(row[metric]) for row in rows]
    mean_value = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        std_value = math.sqrt(variance)
    else:
        std_value = 0.0

    fields = {
        f"{metric}_mean": mean_value,
        f"{metric}_std": std_value,
    }
    if include_ci:
        half_width = 1.96 * std_value / math.sqrt(len(values))
        fields[f"{metric}_ci95_low"] = mean_value - half_width
        fields[f"{metric}_ci95_high"] = mean_value + half_width
    return fields


def _build_pairwise_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    rows_by_repeat: dict[int, list[dict[str, float | int | str]]] = {}
    for row in rows:
        rows_by_repeat.setdefault(int(row["repeat_index"]), []).append(row)

    pairwise_rows = []
    for spec in QUALITY_COST_PAIRWISE_SPECS:
        paired_rows = []
        for repeat_index in sorted(rows_by_repeat):
            repeat_rows = rows_by_repeat[repeat_index]
            candidate = _find_pairwise_match(repeat_rows, spec["candidate"])
            comparator = _find_pairwise_match(repeat_rows, spec["comparator"])
            if candidate is not None and comparator is not None:
                paired_rows.append((candidate, comparator))
        if paired_rows:
            pairwise_rows.append(_pairwise_row(str(spec["pair_id"]), paired_rows))
    return pairwise_rows


def _find_pairwise_match(
    rows: list[dict[str, float | int | str]],
    spec: dict[str, str | int],
) -> dict[str, float | int | str] | None:
    for row in rows:
        if (
            row["filter"] == spec["filter"]
            and row["variant"] == spec["variant"]
            and int(row["resource_count"]) == int(spec["resource_count"])
        ):
            return row
    return None


def _pairwise_row(
    pair_id: str,
    paired_rows: list[tuple[dict[str, float | int | str], dict[str, float | int | str]]],
) -> dict[str, float | int | str]:
    first_candidate, first_comparator = paired_rows[0]
    rmse_deltas = [
        float(candidate["position_rmse"]) - float(comparator["position_rmse"])
        for candidate, comparator in paired_rows
    ]
    runtime_deltas = [
        float(candidate["runtime_ms_per_step"]) - float(comparator["runtime_ms_per_step"])
        for candidate, comparator in paired_rows
    ]
    runtime_ratios = [
        _ratio(candidate["runtime_ms_per_step"], comparator["runtime_ms_per_step"])
        for candidate, comparator in paired_rows
    ]
    return {
        "pair_id": pair_id,
        "candidate_label": first_candidate["label"],
        "comparator_label": first_comparator["label"],
        "candidate_filter": first_candidate["filter"],
        "candidate_variant": first_candidate["variant"],
        "candidate_resource_count": first_candidate["resource_count"],
        "comparator_filter": first_comparator["filter"],
        "comparator_variant": first_comparator["variant"],
        "comparator_resource_count": first_comparator["resource_count"],
        **_paired_value_fields(rmse_deltas, "position_rmse_delta"),
        "candidate_rmse_win_count": sum(delta < 0.0 for delta in rmse_deltas),
        "comparator_rmse_win_count": sum(delta > 0.0 for delta in rmse_deltas),
        **_paired_value_fields(runtime_deltas, "runtime_ms_per_step_delta"),
        "runtime_ratio_mean": sum(runtime_ratios) / len(runtime_ratios),
        "candidate_runtime_win_count": sum(delta < 0.0 for delta in runtime_deltas),
        "comparator_runtime_win_count": sum(delta > 0.0 for delta in runtime_deltas),
        "candidate_dominance_count": sum(_dominates(candidate, comparator) for candidate, comparator in paired_rows),
        "comparator_dominance_count": sum(_dominates(comparator, candidate) for candidate, comparator in paired_rows),
        "n_repeats": len(paired_rows),
        "n_trials": first_candidate["n_trials"],
        "n_steps": first_candidate["n_steps"],
    }


def _paired_value_fields(values: list[float], prefix: str) -> dict[str, float]:
    mean_value = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        std_value = math.sqrt(variance)
    else:
        std_value = 0.0
    half_width = 1.96 * std_value / math.sqrt(len(values))
    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_std": std_value,
        f"{prefix}_ci95_low": mean_value - half_width,
        f"{prefix}_ci95_high": mean_value + half_width,
    }


def _quality_row_from_reference(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    return {
        "variant": row["variant"],
        "grid_size": row["grid_size"],
        "position_rmse": row["position_rmse_to_truth"],
        "mean_nees": row["mean_nees_to_truth"],
        "coverage_95": row["coverage_95_to_truth"],
        "runtime_ms_per_step": row["runtime_ms_per_step"],
        "position_rmse_to_reference": row["position_rmse_to_reference"],
        "orientation_mean_error_to_reference_rad": row["orientation_mean_error_to_reference_rad"],
        "runtime_ratio_to_reference": row["runtime_ratio_to_reference"],
        "reference_grid_size": row["reference_grid_size"],
        "reference_position_rmse": row["reference_position_rmse_to_truth"],
        "reference_runtime_ms_per_step": row["reference_runtime_ms_per_step"],
        "n_trials": row["n_trials"],
        "n_steps": row["n_steps"],
    }


def _build_claim_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str | bool]]:
    indexed = {(str(row["variant"]), int(row["grid_size"])): row for row in rows}
    grid_sizes = sorted({int(row["grid_size"]) for row in rows})
    claims: list[dict[str, float | int | str | bool]] = []

    for grid_size in grid_sizes:
        candidate = indexed.get(("r1_r2", grid_size))
        comparator = indexed.get(("baseline", grid_size))
        if candidate is not None and comparator is not None:
            claims.append(_claim_row(f"same_grid_{grid_size}", "R1+R2 vs baseline at the same grid size", candidate, comparator))

    for current_grid, next_grid in zip(grid_sizes, grid_sizes[1:], strict=False):
        candidate = indexed.get(("r1_r2", current_grid))
        comparator = indexed.get(("baseline", next_grid))
        if candidate is not None and comparator is not None:
            claims.append(_claim_row(f"coarser_r1_r2_{current_grid}_vs_baseline_{next_grid}", "R1+R2 at a coarser grid vs baseline at the next grid size", candidate, comparator))

    return claims


def _claim_row(
    claim_id: str,
    comparison: str,
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
) -> dict[str, float | int | str | bool]:
    position_ratio = _ratio(candidate["position_rmse"], comparator["position_rmse"])
    nees_ratio = _ratio(candidate["mean_nees"], comparator["mean_nees"])
    runtime_ratio = _ratio(candidate["runtime_ms_per_step"], comparator["runtime_ms_per_step"])
    reference_ratio = _ratio(candidate["position_rmse_to_reference"], comparator["position_rmse_to_reference"])
    return {
        "claim_id": claim_id,
        "comparison": comparison,
        "candidate_variant": candidate["variant"],
        "candidate_grid_size": candidate["grid_size"],
        "comparator_variant": comparator["variant"],
        "comparator_grid_size": comparator["grid_size"],
        "candidate_position_rmse": candidate["position_rmse"],
        "comparator_position_rmse": comparator["position_rmse"],
        "position_rmse_ratio": position_ratio,
        "candidate_mean_nees": candidate["mean_nees"],
        "comparator_mean_nees": comparator["mean_nees"],
        "mean_nees_ratio": nees_ratio,
        "candidate_runtime_ms_per_step": candidate["runtime_ms_per_step"],
        "comparator_runtime_ms_per_step": comparator["runtime_ms_per_step"],
        "runtime_ratio": runtime_ratio,
        "candidate_position_rmse_to_reference": candidate["position_rmse_to_reference"],
        "comparator_position_rmse_to_reference": comparator["position_rmse_to_reference"],
        "reference_rmse_ratio": reference_ratio,
        "supports_accuracy_claim": position_ratio <= 1.0,
        "supports_consistency_claim": _distance_to_consistent_nees(candidate["mean_nees"]) <= _distance_to_consistent_nees(comparator["mean_nees"]),
        "supports_runtime_claim": runtime_ratio <= 1.1,
    }


def _ratio(numerator: float | int | str, denominator: float | int | str) -> float:
    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return float("inf")
    return float(numerator) / denominator_float


def _distance_to_consistent_nees(value: float | int | str) -> float:
    return abs(float(value) - 2.0)


def _build_pareto_rows(
    s3f_rows: list[dict[str, float | int | str]],
    particle_sensitivity_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    pareto_rows = [_pareto_row_from_s3f(row) for row in s3f_rows]
    pareto_rows.extend(
        _pareto_row_from_particle(row)
        for row in particle_sensitivity_rows
        if row["filter"] == "particle_filter"
    )
    return pareto_rows


def _pareto_row_from_s3f(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    return {
        "filter": "s3f",
        "label": _row_label(row),
        "variant": row["variant"],
        "grid_size": row["grid_size"],
        "particle_count": "",
        "resource_count": row["grid_size"],
        "position_rmse": row["position_rmse"],
        "mean_nees": row["mean_nees"],
        "coverage_95": row["coverage_95"],
        "runtime_ms_per_step": row["runtime_ms_per_step"],
        "position_rmse_to_reference": row["position_rmse_to_reference"],
        "runtime_ratio_to_reference": row["runtime_ratio_to_reference"],
        "n_trials": row["n_trials"],
        "n_steps": row["n_steps"],
    }


def _pareto_row_from_particle(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    particle_count = row["particle_count"]
    return {
        "filter": "particle_filter",
        "label": f"Particle filter ({particle_count})",
        "variant": row["variant"],
        "grid_size": "",
        "particle_count": particle_count,
        "resource_count": particle_count,
        "position_rmse": row["position_rmse"],
        "mean_nees": row["mean_nees"],
        "coverage_95": row["coverage_95"],
        "runtime_ms_per_step": row["runtime_ms_per_step"],
        "position_rmse_to_reference": "",
        "runtime_ratio_to_reference": "",
        "n_trials": row["n_trials"],
        "n_steps": row["n_steps"],
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(
    path: Path,
    result: QualityCostResult,
    config: QualityCostConfig,
) -> None:
    metadata = {
        "experiment": "quality_cost",
        "config": quality_cost_config_to_dict(config),
        "metrics_schema": QUALITY_COST_FIELDNAMES,
        "metrics_rows": len(result.metrics),
        "claims_schema": QUALITY_COST_CLAIM_FIELDNAMES,
        "claims_rows": len(result.claims),
        "pareto_schema": QUALITY_COST_PARETO_FIELDNAMES,
        "pareto_rows": len(result.pareto),
        "repeat_pareto_schema": QUALITY_COST_REPEAT_PARETO_FIELDNAMES,
        "repeat_pareto_rows": len(result.repeat_pareto),
        "summary_schema": QUALITY_COST_SUMMARY_FIELDNAMES,
        "summary_rows": len(result.summary),
        "pairwise_schema": QUALITY_COST_PAIRWISE_FIELDNAMES,
        "pairwise_rows": len(result.pairwise),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]], pareto_rows: list[dict[str, float | int | str]]) -> list[Path]:
    specs = [
        ("position_rmse", "Translation RMSE", "quality_cost_rmse_runtime.png"),
        ("mean_nees", "Mean Position NEES", "quality_cost_nees_runtime.png"),
        ("position_rmse_to_reference", "Translation RMSE to Reference", "quality_cost_reference_rmse_runtime.png"),
    ]
    plots = [_write_runtime_tradeoff_plot(output_dir, rows, metric, ylabel, filename) for metric, ylabel, filename in specs]
    plots.append(_write_pareto_plot(output_dir, pareto_rows, "position_rmse", "Translation RMSE", "quality_cost_pareto_rmse_runtime.png"))
    plots.append(_write_pareto_plot(output_dir, pareto_rows, "mean_nees", "Mean Position NEES", "quality_cost_pareto_nees_runtime.png"))
    return plots


def _write_runtime_tradeoff_plot(output_dir: Path, rows: list[dict[str, float | int | str]], metric: str, ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for variant in QUALITY_COST_VARIANTS:
        variant_rows = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda row: int(row["grid_size"]),
        )
        runtimes = [float(row["runtime_ms_per_step"]) for row in variant_rows]
        values = [float(row[metric]) for row in variant_rows]
        ax.plot(runtimes, values, marker="o", linewidth=1.8, label=VARIANT_LABELS[variant])
        for runtime, value, row in zip(runtimes, values, variant_rows, strict=True):
            ax.annotate(str(row["grid_size"]), (runtime, value), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Runtime [ms/step]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, filename)


def _write_pareto_plot(output_dir: Path, rows: list[dict[str, float | int | str]], metric: str, ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for variant in QUALITY_COST_VARIANTS:
        variant_rows = sorted(
            [row for row in rows if row["filter"] == "s3f" and row["variant"] == variant],
            key=lambda row: int(row["grid_size"]),
        )
        ax.plot(
            [float(row["runtime_ms_per_step"]) for row in variant_rows],
            [float(row[metric]) for row in variant_rows],
            marker="o",
            linewidth=1.8,
            label=VARIANT_LABELS[variant],
        )

    particle_rows = sorted(
        [row for row in rows if row["filter"] == "particle_filter"],
        key=lambda row: int(row["particle_count"]),
    )
    if particle_rows:
        ax.plot(
            [float(row["runtime_ms_per_step"]) for row in particle_rows],
            [float(row[metric]) for row in particle_rows],
            marker="s",
            linewidth=1.8,
            label="Particle filter",
        )

    ax.set_xlabel("Runtime [ms/step]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, filename)


def _write_pairwise_delta_plot(output_dir: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fig_height = max(3.2, 1.0 + 0.55 * len(rows))
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    sorted_rows = list(reversed(rows))
    labels = [
        f"{row['candidate_label']} vs {row['comparator_label']}"
        for row in sorted_rows
    ]
    means = [float(row["position_rmse_delta_mean"]) for row in sorted_rows]
    lows = [float(row["position_rmse_delta_ci95_low"]) for row in sorted_rows]
    highs = [float(row["position_rmse_delta_ci95_high"]) for row in sorted_rows]
    left_errors = [mean - low for mean, low in zip(means, lows, strict=True)]
    right_errors = [high - mean for mean, high in zip(means, highs, strict=True)]
    y_positions = list(range(len(sorted_rows)))
    colors = ["#4C78A8" if mean <= 0.0 else "#F58518" for mean in means]

    ax.barh(y_positions, means, color=colors, alpha=0.85)
    ax.errorbar(means, y_positions, xerr=[left_errors, right_errors], fmt="none", ecolor="black", capsize=3, linewidth=1.0)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Paired RMSE delta, candidate minus comparator")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir, "quality_cost_pairwise_rmse_delta.png")


def _write_note(
    path: Path,
    result: QualityCostResult,
    metrics_path: Path,
    claims_path: Path,
    pareto_path: Path,
    plot_paths: list[Path],
    config: QualityCostConfig,
) -> None:
    rows = result.metrics
    best_accuracy = min(result.pareto, key=lambda row: float(row["position_rmse"]))
    best_reference_match = min(rows, key=lambda row: float(row["position_rmse_to_reference"]))
    best_particle = _best_filter_row(result.pareto, "particle_filter")
    best_r1_r2 = min(
        [row for row in result.pareto if row["filter"] == "s3f" and row["variant"] == "r1_r2"],
        key=lambda row: float(row["position_rmse"]),
    )
    supported_grid_saving = [
        claim
        for claim in result.claims
        if str(claim["claim_id"]).startswith("coarser_r1_r2")
        and bool(claim["supports_accuracy_claim"])
        and bool(claim["supports_runtime_claim"])
    ]
    lines = [
        "# Quality-Cost Report",
        "",
        "This report combines coarse-grid S3F quality, consistency, runtime, and",
        "distance to a high-resolution S3F reference in one table.",
        "",
        f"Trials: {config.reference.pilot.n_trials}",
        f"Steps per trial: {config.reference.pilot.n_steps}",
        f"Grid sizes: {list(config.reference.pilot.grid_sizes)}",
        f"Reference grid size: {config.reference.reference_grid_size}",
        f"Particle counts: {list(config.particle_counts)}",
        f"Repeats: {config.repeats}",
        f"Metrics file: `{metrics_path.name}`",
        f"Claims file: `{claims_path.name}`",
        f"Pareto file: `{pareto_path.name}`",
        *_repeat_file_scope_lines(config),
        "",
        "## Best Rows",
        "",
        (
            "Best truth RMSE: "
            f"`{best_accuracy['label']}` with RMSE `{float(best_accuracy['position_rmse']):.4f}`, "
            f"NEES `{float(best_accuracy['mean_nees']):.3f}`, and runtime "
            f"`{float(best_accuracy['runtime_ms_per_step']):.3f}` ms/step."
        ),
        (
            "Closest high-resolution match: "
            f"`{_row_label(best_reference_match)}` with reference RMSE "
            f"`{float(best_reference_match['position_rmse_to_reference']):.4f}`."
        ),
        "",
        "## Quality-Cost Table",
        "",
        _format_metrics_table(rows),
        "",
        "## Grid-Saving Checks",
        "",
        _format_claims_table(result.claims),
        "",
        "## S3F-vs-Particle Pareto Table",
        "",
        _format_pareto_table(result.pareto),
        "",
        "## RMSE-Runtime Frontier",
        "",
        _format_pareto_table(_rmse_runtime_frontier(result.pareto)),
        *_repeat_summary_lines(result),
        *_pairwise_comparison_lines(result),
        "",
        "## Interpretation",
        "",
        _interpret_supported_claims(supported_grid_saving),
        _interpret_particle_comparison(best_r1_r2, best_particle),
        _interpret_nearest_particle_comparison(best_r1_r2, result.pareto),
        _interpret_repeat_summary(result.summary),
        _interpret_pairwise_comparisons(result.pairwise),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "The S3F-to-reference columns remain S3F-internal. The Pareto table is the",
        "shared truth-metric comparison against the bootstrap particle filter.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_label(row: dict[str, float | int | str]) -> str:
    return f"{VARIANT_LABELS[str(row['variant'])]} ({row['grid_size']} cells)"


def _repeat_file_scope_lines(config: QualityCostConfig) -> list[str]:
    if config.repeats <= 1:
        return []
    return [
        "Repeat summary files aggregate all repeats; paired comparison files are written when selected pairs are available. Metrics, claims, and Pareto files keep the first repeat schema for compatibility."
    ]


def _format_metrics_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Variant | Cells | RMSE | NEES | Runtime ms/step | RMSE to ref | Runtime/ref |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in sorted(rows, key=lambda item: (int(item["grid_size"]), str(item["variant"]))):
        body.append(
            "| "
            f"{VARIANT_LABELS[str(row['variant'])]} | "
            f"{int(row['grid_size'])} | "
            f"{float(row['position_rmse']):.4f} | "
            f"{float(row['mean_nees']):.3f} | "
            f"{float(row['runtime_ms_per_step']):.3f} | "
            f"{float(row['position_rmse_to_reference']):.4f} | "
            f"{float(row['runtime_ratio_to_reference']):.3f} |"
        )
    return "\n".join([header, separator, *body])


def _format_claims_table(claims: list[dict[str, float | int | str | bool]]) -> str:
    header = "| Candidate | Comparator | RMSE ratio | NEES ratio | Runtime ratio | Supports? |"
    separator = "|---|---|---:|---:|---:|---|"
    body = []
    for claim in claims:
        supports = [
            name
            for name, key in (
                ("accuracy", "supports_accuracy_claim"),
                ("consistency", "supports_consistency_claim"),
                ("runtime", "supports_runtime_claim"),
            )
            if bool(claim[key])
        ]
        body.append(
            "| "
            f"{claim['candidate_variant']} {claim['candidate_grid_size']} | "
            f"{claim['comparator_variant']} {claim['comparator_grid_size']} | "
            f"{float(claim['position_rmse_ratio']):.3f} | "
            f"{float(claim['mean_nees_ratio']):.3f} | "
            f"{float(claim['runtime_ratio']):.3f} | "
            f"{', '.join(supports) if supports else 'none'} |"
        )
    return "\n".join([header, separator, *body])


def _format_pareto_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Method | Resource | RMSE | NEES | Coverage | Runtime ms/step |"
    separator = "|---|---:|---:|---:|---:|---:|"
    body = []
    for row in sorted(rows, key=lambda item: (float(item["runtime_ms_per_step"]), str(item["filter"]))):
        body.append(
            "| "
            f"{row['label']} | "
            f"{int(row['resource_count'])} | "
            f"{float(row['position_rmse']):.4f} | "
            f"{float(row['mean_nees']):.3f} | "
            f"{float(row['coverage_95']):.3f} | "
            f"{float(row['runtime_ms_per_step']):.3f} |"
        )
    return "\n".join([header, separator, *body])


def _repeat_summary_lines(result: QualityCostResult) -> list[str]:
    if not result.summary:
        return []
    return [
        "",
        "## Repeat Summary",
        "",
        _format_summary_table(result.summary),
        "",
        "Repeat summary file: `quality_cost_summary.csv`",
        "Per-repeat Pareto file: `quality_cost_repeat_pareto.csv`",
    ]


def _format_summary_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Method | Resource | RMSE mean | RMSE 95% CI | Runtime mean | Runtime 95% CI | Repeats |"
    separator = "|---|---:|---:|---|---:|---|---:|"
    body = []
    for row in sorted(rows, key=lambda item: (float(item["runtime_ms_per_step_mean"]), str(item["filter"]))):
        body.append(
            "| "
            f"{row['label']} | "
            f"{int(row['resource_count'])} | "
            f"{float(row['position_rmse_mean']):.4f} | "
            f"[{float(row['position_rmse_ci95_low']):.4f}, {float(row['position_rmse_ci95_high']):.4f}] | "
            f"{float(row['runtime_ms_per_step_mean']):.3f} | "
            f"[{float(row['runtime_ms_per_step_ci95_low']):.3f}, {float(row['runtime_ms_per_step_ci95_high']):.3f}] | "
            f"{int(row['n_repeats'])} |"
        )
    return "\n".join([header, separator, *body])


def _pairwise_comparison_lines(result: QualityCostResult) -> list[str]:
    if not result.pairwise:
        return []
    return [
        "",
        "## Paired Comparisons",
        "",
        "Deltas are candidate minus comparator; negative RMSE and runtime deltas favor the candidate.",
        "",
        _format_pairwise_table(result.pairwise),
        "",
        "Paired comparison file: `quality_cost_pairwise.csv`",
    ]


def _format_pairwise_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Pair | RMSE delta mean | RMSE delta 95% CI | RMSE wins | Runtime delta mean | Runtime ratio | Dominance |"
    separator = "|---|---:|---|---:|---:|---:|---:|"
    body = []
    for row in rows:
        n_repeats = int(row["n_repeats"])
        body.append(
            "| "
            f"{row['candidate_label']} vs {row['comparator_label']} | "
            f"{float(row['position_rmse_delta_mean']):.4f} | "
            f"[{float(row['position_rmse_delta_ci95_low']):.4f}, {float(row['position_rmse_delta_ci95_high']):.4f}] | "
            f"{int(row['candidate_rmse_win_count'])}/{n_repeats} vs {int(row['comparator_rmse_win_count'])}/{n_repeats} | "
            f"{float(row['runtime_ms_per_step_delta_mean']):.3f} | "
            f"{float(row['runtime_ratio_mean']):.3f} | "
            f"{int(row['candidate_dominance_count'])}/{n_repeats} vs {int(row['comparator_dominance_count'])}/{n_repeats} |"
        )
    return "\n".join([header, separator, *body])


def _rmse_runtime_frontier(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Return rows not dominated on truth RMSE and runtime."""

    frontier = []
    for candidate in rows:
        if not any(_dominates(other, candidate) for other in rows if other is not candidate):
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (float(row["runtime_ms_per_step"]), float(row["position_rmse"])))


def _dominates(
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
) -> bool:
    candidate_rmse = float(candidate["position_rmse"])
    comparator_rmse = float(comparator["position_rmse"])
    candidate_runtime = float(candidate["runtime_ms_per_step"])
    comparator_runtime = float(comparator["runtime_ms_per_step"])
    return (
        candidate_rmse <= comparator_rmse
        and candidate_runtime <= comparator_runtime
        and (candidate_rmse < comparator_rmse or candidate_runtime < comparator_runtime)
    )


def _interpret_supported_claims(claims: list[dict[str, float | int | str | bool]]) -> str:
    if not claims:
        return "No coarser-grid R1+R2 row met both the accuracy and runtime checks against the next baseline grid size."
    claim_text = []
    for claim in claims:
        claim_text.append(
            f"`R1+R2` at `{claim['candidate_grid_size']}` cells matched or beat baseline at "
            f"`{claim['comparator_grid_size']}` cells on RMSE without being more than 10% slower."
        )
    return " ".join(claim_text)


def _best_filter_row(rows: list[dict[str, float | int | str]], filter_name: str) -> dict[str, float | int | str] | None:
    matching_rows = [row for row in rows if row["filter"] == filter_name]
    if not matching_rows:
        return None
    return min(matching_rows, key=lambda row: float(row["position_rmse"]))


def _interpret_particle_comparison(
    best_r1_r2: dict[str, float | int | str],
    best_particle: dict[str, float | int | str] | None,
) -> str:
    if best_particle is None:
        return "No particle-filter rows were included in this report."

    rmse_ratio = _ratio(best_r1_r2["position_rmse"], best_particle["position_rmse"])
    runtime_ratio = _ratio(best_r1_r2["runtime_ms_per_step"], best_particle["runtime_ms_per_step"])
    return (
        f"Best `R1+R2` uses `{best_r1_r2['resource_count']}` cells with RMSE "
        f"`{float(best_r1_r2['position_rmse']):.4f}` at `{float(best_r1_r2['runtime_ms_per_step']):.3f}` ms/step; "
        f"best particle row uses `{best_particle['particle_count']}` particles with RMSE "
        f"`{float(best_particle['position_rmse']):.4f}` at `{float(best_particle['runtime_ms_per_step']):.3f}` ms/step. "
        f"The R1+R2/best-particle ratios are `{rmse_ratio:.3f}` for RMSE and `{runtime_ratio:.3f}` for runtime."
    )


def _interpret_nearest_particle_comparison(
    best_r1_r2: dict[str, float | int | str],
    rows: list[dict[str, float | int | str]],
) -> str:
    particle_rows = [row for row in rows if row["filter"] == "particle_filter"]
    if not particle_rows:
        return "No particle-filter row is available for same-runtime comparison."

    nearest_runtime_particle = min(
        particle_rows,
        key=lambda row: abs(float(row["runtime_ms_per_step"]) - float(best_r1_r2["runtime_ms_per_step"])),
    )
    not_slower_particles = [
        row
        for row in particle_rows
        if float(row["runtime_ms_per_step"]) <= float(best_r1_r2["runtime_ms_per_step"])
    ]
    best_not_slower_particle = (
        min(not_slower_particles, key=lambda row: float(row["position_rmse"]))
        if not_slower_particles
        else None
    )

    nearest_text = (
        f"Nearest-runtime particle row: `{nearest_runtime_particle['label']}` has RMSE "
        f"`{float(nearest_runtime_particle['position_rmse']):.4f}` at "
        f"`{float(nearest_runtime_particle['runtime_ms_per_step']):.3f}` ms/step."
    )
    if best_not_slower_particle is None:
        return nearest_text + " No particle row is at least as fast as the best R1+R2 row."

    comparison_text = (
        f"Best particle row no slower than best R1+R2: `{best_not_slower_particle['label']}` "
        f"with RMSE `{float(best_not_slower_particle['position_rmse']):.4f}` at "
        f"`{float(best_not_slower_particle['runtime_ms_per_step']):.3f}` ms/step."
    )
    if _dominates(best_not_slower_particle, best_r1_r2):
        comparison_text += " In this run it dominates the best R1+R2 row on RMSE and runtime."
    else:
        comparison_text += (
            " In this run it is faster, but does not dominate the best R1+R2 row on RMSE."
        )
    return nearest_text + " " + comparison_text


def _interpret_repeat_summary(summary: list[dict[str, float | int | str]]) -> str:
    if not summary:
        return ""

    r1_r2_rows = [row for row in summary if row["filter"] == "s3f" and row["variant"] == "r1_r2"]
    if not r1_r2_rows:
        return "Repeat summary did not include an R1+R2 row."

    particle_rows = [row for row in summary if row["filter"] == "particle_filter"]
    if not particle_rows:
        return "Repeat summary did not include particle-filter rows."

    best_r1_r2 = min(r1_r2_rows, key=lambda row: float(row["position_rmse_mean"]))
    nearest_runtime_particle = min(
        particle_rows,
        key=lambda row: abs(float(row["runtime_ms_per_step_mean"]) - float(best_r1_r2["runtime_ms_per_step_mean"])),
    )
    not_slower_particles = [
        row
        for row in particle_rows
        if float(row["runtime_ms_per_step_mean"]) <= float(best_r1_r2["runtime_ms_per_step_mean"])
    ]
    best_not_slower_particle = (
        min(not_slower_particles, key=lambda row: float(row["position_rmse_mean"]))
        if not_slower_particles
        else None
    )
    n_repeats = int(best_r1_r2["n_repeats"])

    nearest_text = (
        f"Across `{n_repeats}` repeats, nearest-runtime particle row by mean runtime is "
        f"`{nearest_runtime_particle['label']}` with mean RMSE "
        f"`{float(nearest_runtime_particle['position_rmse_mean']):.4f}` at "
        f"`{float(nearest_runtime_particle['runtime_ms_per_step_mean']):.3f}` ms/step."
    )
    if best_not_slower_particle is None:
        return nearest_text + " No particle row is at least as fast as the best R1+R2 row in repeat means."

    comparison_text = (
        f"Best particle row no slower than best R1+R2 in repeat means is "
        f"`{best_not_slower_particle['label']}` with mean RMSE "
        f"`{float(best_not_slower_particle['position_rmse_mean']):.4f}` at "
        f"`{float(best_not_slower_particle['runtime_ms_per_step_mean']):.3f}` ms/step."
    )
    if _summary_dominates(best_not_slower_particle, best_r1_r2):
        if _summary_ci_non_overlapping(best_not_slower_particle, best_r1_r2):
            comparison_text += (
                " It mean-dominates the best R1+R2 row with non-overlapping approximate 95% CIs."
            )
        else:
            comparison_text += (
                " It mean-dominates the best R1+R2 row, but the approximate 95% CIs overlap, so this should be treated as a trend rather than a robust separation."
            )
    else:
        comparison_text += " No no-slower particle row dominates the best R1+R2 row in repeat means."
    return nearest_text + " " + comparison_text


def _interpret_pairwise_comparisons(pairwise_rows: list[dict[str, float | int | str]]) -> str:
    if not pairwise_rows:
        return ""

    interpretations = []
    for row in pairwise_rows:
        rmse_low = float(row["position_rmse_delta_ci95_low"])
        rmse_high = float(row["position_rmse_delta_ci95_high"])
        runtime_ratio = float(row["runtime_ratio_mean"])
        n_repeats = int(row["n_repeats"])
        candidate_wins = int(row["candidate_rmse_win_count"])
        comparator_wins = int(row["comparator_rmse_win_count"])
        if rmse_high < 0.0:
            rmse_text = "has lower paired RMSE with the approximate 95% CI below zero"
        elif rmse_low > 0.0:
            rmse_text = "has higher paired RMSE with the approximate 95% CI above zero"
        else:
            rmse_text = "has a paired RMSE delta CI crossing zero"
        runtime_text = "faster" if runtime_ratio < 1.0 else "slower"
        interpretations.append(
            f"Paired `{row['candidate_label']}` vs `{row['comparator_label']}`: candidate {rmse_text}; "
            f"RMSE wins are `{candidate_wins}/{n_repeats}` vs `{comparator_wins}/{n_repeats}`, "
            f"and candidate is `{runtime_ratio:.3f}x` the comparator runtime ({runtime_text})."
        )
    return " ".join(interpretations)


def _summary_dominates(
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
) -> bool:
    candidate_rmse = float(candidate["position_rmse_mean"])
    comparator_rmse = float(comparator["position_rmse_mean"])
    candidate_runtime = float(candidate["runtime_ms_per_step_mean"])
    comparator_runtime = float(comparator["runtime_ms_per_step_mean"])
    return (
        candidate_rmse <= comparator_rmse
        and candidate_runtime <= comparator_runtime
        and (candidate_rmse < comparator_rmse or candidate_runtime < comparator_runtime)
    )


def _summary_ci_non_overlapping(
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
) -> bool:
    return (
        float(candidate["position_rmse_ci95_high"]) < float(comparator["position_rmse_ci95_low"])
        and float(candidate["runtime_ms_per_step_ci95_high"]) < float(comparator["runtime_ms_per_step_ci95_low"])
    )
