"""Quality-vs-cost report for S1 x R2 relaxed S3F grids."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

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

QUALITY_COST_VARIANTS = ("baseline", "r1", "r1_r2")


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


def quality_cost_config_to_dict(config: QualityCostConfig) -> dict[str, Any]:
    """Return a JSON-serializable quality-cost config."""

    return {"reference": highres_reference_config_to_dict(config.reference)}


def run_quality_cost_report(config: QualityCostConfig = QualityCostConfig()) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str | bool]]]:
    """Run the quality-vs-cost report from a high-resolution S3F reference sweep."""

    reference_rows = run_highres_reference_benchmark(config.reference)
    rows = [_quality_row_from_reference(row) for row in reference_rows]
    claims = _build_claim_rows(rows)
    return rows, claims


def write_quality_cost_outputs(
    output_dir: Path,
    config: QualityCostConfig = QualityCostConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the report and write metrics, claims, optional plots, metadata, and note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, claims = run_quality_cost_report(config)

    metrics_path = output_dir / "quality_cost_metrics.csv"
    _write_csv(metrics_path, rows, QUALITY_COST_FIELDNAMES)

    claims_path = output_dir / "quality_cost_claims.csv"
    _write_csv(claims_path, claims, QUALITY_COST_CLAIM_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path}
    plot_paths = _write_plots(output_dir, rows) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "quality_cost_note.md"
    _write_note(note_path, rows, claims, metrics_path, claims_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, claims, config)
    outputs["metadata"] = metadata_path
    return outputs


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


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(
    path: Path,
    rows: list[dict[str, float | int | str]],
    claims: list[dict[str, float | int | str | bool]],
    config: QualityCostConfig,
) -> None:
    metadata = {
        "experiment": "quality_cost",
        "config": quality_cost_config_to_dict(config),
        "metrics_schema": QUALITY_COST_FIELDNAMES,
        "metrics_rows": len(rows),
        "claims_schema": QUALITY_COST_CLAIM_FIELDNAMES,
        "claims_rows": len(claims),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    specs = [
        ("position_rmse", "Translation RMSE", "quality_cost_rmse_runtime.png"),
        ("mean_nees", "Mean Position NEES", "quality_cost_nees_runtime.png"),
        ("position_rmse_to_reference", "Translation RMSE to Reference", "quality_cost_reference_rmse_runtime.png"),
    ]
    return [_write_runtime_tradeoff_plot(output_dir, rows, metric, ylabel, filename) for metric, ylabel, filename in specs]


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


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    claims: list[dict[str, float | int | str | bool]],
    metrics_path: Path,
    claims_path: Path,
    plot_paths: list[Path],
    config: QualityCostConfig,
) -> None:
    best_accuracy = min(rows, key=lambda row: float(row["position_rmse"]))
    best_reference_match = min(rows, key=lambda row: float(row["position_rmse_to_reference"]))
    supported_grid_saving = [
        claim
        for claim in claims
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
        f"Metrics file: `{metrics_path.name}`",
        f"Claims file: `{claims_path.name}`",
        "",
        "## Best Rows",
        "",
        (
            "Best truth RMSE: "
            f"`{_row_label(best_accuracy)}` with RMSE `{float(best_accuracy['position_rmse']):.4f}`, "
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
        _format_claims_table(claims),
        "",
        "## Interpretation",
        "",
        _interpret_supported_claims(supported_grid_saving),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "This report is still an S3F-internal comparison. It answers whether relaxed",
        "coarse grids approach the high-resolution S3F behavior sooner; it does not",
        "replace the particle-filter sensitivity benchmark.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_label(row: dict[str, float | int | str]) -> str:
    return f"{VARIANT_LABELS[str(row['variant'])]} ({row['grid_size']} cells)"


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
