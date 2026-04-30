"""Stress sweep for S3+ x R3 relaxed S3F behavior."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..s1r2.plotting import format_plot_list, save_figure
from .relaxed_s3f_prototype import (
    S3R3PrototypeConfig,
    SUPPORTED_S3R3_VARIANTS,
    run_s3r3_relaxed_prototype,
    validate_s3r3_prototype_config,
)


S3R3_STRESS_FIELDNAMES = [
    "scenario_id",
    "prior_kappa",
    "body_increment_scale",
    "body_increment_norm",
    "grid_size",
    "variant",
    "position_rmse",
    "orientation_mode_error_rad",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "cell_radius_rad",
    "cell_sample_count",
    "n_trials",
    "n_steps",
]

S3R3_STRESS_CLAIM_FIELDNAMES = [
    "scenario_id",
    "prior_kappa",
    "body_increment_scale",
    "body_increment_norm",
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
    "candidate_runtime_ms_per_step",
    "comparator_runtime_ms_per_step",
    "runtime_ratio",
    "supports_accuracy_claim",
    "supports_consistency_claim",
    "supports_runtime_claim",
    "supports_overall_claim",
]

S3R3_STRESS_SUMMARY_FIELDNAMES = [
    "scenario_id",
    "prior_kappa",
    "body_increment_scale",
    "body_increment_norm",
    "baseline_comparisons",
    "baseline_overall_supports",
    "baseline_mean_rmse_gain_pct",
    "baseline_mean_nees_ratio",
    "baseline_mean_runtime_ratio",
    "inflation_comparisons",
    "inflation_overall_supports",
    "inflation_mean_rmse_gain_pct",
    "inflation_mean_nees_ratio",
    "inflation_mean_runtime_ratio",
]

STRESS_SWEEP_VARIANTS = SUPPORTED_S3R3_VARIANTS
BASELINE_COMPARISON = "R1+R2 vs baseline"
INFLATION_COMPARISON = "R1+R2 vs R1"


@dataclass(frozen=True)
class S3R3StressSweepResult:
    """Container for the S3R3 stress-sweep tables."""

    metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]
    summary: list[dict[str, float | int | str]]


@dataclass(frozen=True)
class S3R3StressSweepConfig:
    """Configuration for sweeping S3R3 relaxed-S3F stress regimes."""

    prototype: S3R3PrototypeConfig = S3R3PrototypeConfig(
        grid_sizes=(8, 16, 32),
        variants=STRESS_SWEEP_VARIANTS,
        n_trials=4,
        n_steps=5,
        seed=31,
    )
    prior_kappas: tuple[float, ...] = (1.5, 3.0, 8.0)
    body_increment_scales: tuple[float, ...] = (0.5, 1.0, 1.5)


def s3r3_stress_sweep_config_to_dict(config: S3R3StressSweepConfig) -> dict[str, Any]:
    """Return a JSON-serializable stress-sweep config."""

    return json.loads(json.dumps(asdict(config)))


def run_s3r3_stress_sweep(
    config: S3R3StressSweepConfig = S3R3StressSweepConfig(),
) -> S3R3StressSweepResult:
    """Run S3R3 relaxed S3F across orientation-width and displacement-size stress regimes."""

    _validate_config(config)
    metrics: list[dict[str, float | int | str]] = []
    for prior_kappa in config.prior_kappas:
        for body_increment_scale in config.body_increment_scales:
            scenario_config = _scenario_config(config.prototype, prior_kappa, body_increment_scale)
            scenario_rows = run_s3r3_relaxed_prototype(scenario_config)
            scenario_id = _scenario_id(prior_kappa, body_increment_scale)
            for row in scenario_rows:
                metrics.append(_stress_metric_row(row, scenario_config, scenario_id, prior_kappa, body_increment_scale))

    claims = _build_claim_rows(metrics)
    return S3R3StressSweepResult(metrics=metrics, claims=claims, summary=_build_summary_rows(claims))


def write_s3r3_stress_sweep_outputs(
    output_dir: Path,
    config: S3R3StressSweepConfig = S3R3StressSweepConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the stress sweep and write metrics, claims, summary, optional plots, metadata, and note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s3r3_stress_sweep(config)

    metrics_path = output_dir / "s3r3_stress_sweep_metrics.csv"
    _write_csv(metrics_path, result.metrics, S3R3_STRESS_FIELDNAMES)

    claims_path = output_dir / "s3r3_stress_sweep_claims.csv"
    _write_csv(claims_path, result.claims, S3R3_STRESS_CLAIM_FIELDNAMES)

    summary_path = output_dir / "s3r3_stress_sweep_summary.csv"
    _write_csv(summary_path, result.summary, S3R3_STRESS_SUMMARY_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path, "summary": summary_path}
    plot_paths = _write_plots(output_dir, result.claims, result.summary) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_stress_sweep_note.md"
    _write_note(note_path, result, metrics_path, claims_path, summary_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: S3R3StressSweepConfig) -> None:
    validate_s3r3_prototype_config(config.prototype, required_variants=STRESS_SWEEP_VARIANTS)
    if not config.prior_kappas:
        raise ValueError("prior_kappas must not be empty.")
    if min(config.prior_kappas) <= 0.0:
        raise ValueError("all prior_kappas must be positive.")
    if not config.body_increment_scales:
        raise ValueError("body_increment_scales must not be empty.")
    if min(config.body_increment_scales) <= 0.0:
        raise ValueError("all body_increment_scales must be positive.")


def _scenario_config(
    prototype: S3R3PrototypeConfig,
    prior_kappa: float,
    body_increment_scale: float,
) -> S3R3PrototypeConfig:
    return S3R3PrototypeConfig(
        grid_sizes=prototype.grid_sizes,
        variants=prototype.variants,
        n_trials=prototype.n_trials,
        n_steps=prototype.n_steps,
        seed=prototype.seed + _scenario_seed_offset(prior_kappa, body_increment_scale),
        body_increment=tuple(body_increment_scale * value for value in prototype.body_increment),
        measurement_noise_std=prototype.measurement_noise_std,
        process_noise_std=prototype.process_noise_std,
        initial_position_std=prototype.initial_position_std,
        prior_modes=prototype.prior_modes,
        prior_weights=prototype.prior_weights,
        prior_kappa=prior_kappa,
        orientation_noise_std=prototype.orientation_noise_std,
        cell_sample_count=prototype.cell_sample_count,
    )


def _scenario_seed_offset(prior_kappa: float, body_increment_scale: float) -> int:
    return int(round(1000.0 * prior_kappa + 100.0 * body_increment_scale))


def _stress_metric_row(
    row: dict[str, float | int | str],
    scenario_config: S3R3PrototypeConfig,
    scenario_id: str,
    prior_kappa: float,
    body_increment_scale: float,
) -> dict[str, float | int | str]:
    body_increment = tuple(float(value) for value in scenario_config.body_increment)
    return {
        "scenario_id": scenario_id,
        "prior_kappa": prior_kappa,
        "body_increment_scale": body_increment_scale,
        "body_increment_norm": sum(value * value for value in body_increment) ** 0.5,
        **row,
    }


def _build_claim_rows(
    metrics: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str | bool]]:
    claims: list[dict[str, float | int | str | bool]] = []
    indexed = _index_metrics(metrics)
    for scenario_id, grid_size in sorted({(str(row["scenario_id"]), int(row["grid_size"])) for row in metrics}):
        r1_r2 = indexed[(scenario_id, grid_size, "r1_r2")]
        baseline = indexed[(scenario_id, grid_size, "baseline")]
        r1 = indexed[(scenario_id, grid_size, "r1")]
        claims.append(_claim_row(BASELINE_COMPARISON, r1_r2, baseline))
        claims.append(_claim_row(INFLATION_COMPARISON, r1_r2, r1))
    return claims


def _index_metrics(
    metrics: list[dict[str, float | int | str]],
) -> dict[tuple[str, int, str], dict[str, float | int | str]]:
    return {(str(row["scenario_id"]), int(row["grid_size"]), str(row["variant"])): row for row in metrics}


def _claim_row(
    comparison: str,
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
) -> dict[str, float | int | str | bool]:
    candidate_rmse = float(candidate["position_rmse"])
    comparator_rmse = float(comparator["position_rmse"])
    candidate_nees = float(candidate["mean_nees"])
    comparator_nees = float(comparator["mean_nees"])
    candidate_coverage = float(candidate["coverage_95"])
    comparator_coverage = float(comparator["coverage_95"])
    candidate_runtime = float(candidate["runtime_ms_per_step"])
    comparator_runtime = float(comparator["runtime_ms_per_step"])
    rmse_ratio = _ratio(candidate_rmse, comparator_rmse)
    nees_ratio = _ratio(candidate_nees, comparator_nees)
    runtime_ratio = _ratio(candidate_runtime, comparator_runtime)
    coverage_delta = candidate_coverage - comparator_coverage
    supports_accuracy = rmse_ratio < 1.0
    supports_consistency = nees_ratio < 1.0 and coverage_delta >= 0.0
    supports_runtime = runtime_ratio <= 1.10
    return {
        "scenario_id": candidate["scenario_id"],
        "prior_kappa": candidate["prior_kappa"],
        "body_increment_scale": candidate["body_increment_scale"],
        "body_increment_norm": candidate["body_increment_norm"],
        "grid_size": int(candidate["grid_size"]),
        "comparison": comparison,
        "candidate_variant": candidate["variant"],
        "comparator_variant": comparator["variant"],
        "candidate_position_rmse": candidate_rmse,
        "comparator_position_rmse": comparator_rmse,
        "position_rmse_ratio": rmse_ratio,
        "position_rmse_gain_pct": 100.0 * (1.0 - rmse_ratio),
        "candidate_mean_nees": candidate_nees,
        "comparator_mean_nees": comparator_nees,
        "mean_nees_ratio": nees_ratio,
        "candidate_coverage_95": candidate_coverage,
        "comparator_coverage_95": comparator_coverage,
        "coverage_delta": coverage_delta,
        "candidate_runtime_ms_per_step": candidate_runtime,
        "comparator_runtime_ms_per_step": comparator_runtime,
        "runtime_ratio": runtime_ratio,
        "supports_accuracy_claim": supports_accuracy,
        "supports_consistency_claim": supports_consistency,
        "supports_runtime_claim": supports_runtime,
        "supports_overall_claim": supports_accuracy and supports_consistency and supports_runtime,
    }


def _build_summary_rows(
    claims: list[dict[str, float | int | str | bool]],
) -> list[dict[str, float | int | str]]:
    rows = []
    scenario_ids = sorted({str(claim["scenario_id"]) for claim in claims})
    for scenario_id in scenario_ids:
        scenario_claims = [claim for claim in claims if claim["scenario_id"] == scenario_id]
        first = scenario_claims[0]
        baseline_claims = [claim for claim in scenario_claims if claim["comparison"] == BASELINE_COMPARISON]
        inflation_claims = [claim for claim in scenario_claims if claim["comparison"] == INFLATION_COMPARISON]
        rows.append(
            {
                "scenario_id": scenario_id,
                "prior_kappa": first["prior_kappa"],
                "body_increment_scale": first["body_increment_scale"],
                "body_increment_norm": first["body_increment_norm"],
                **_comparison_summary("baseline", baseline_claims),
                **_comparison_summary("inflation", inflation_claims),
            }
        )
    return rows


def _comparison_summary(
    prefix: str,
    claims: list[dict[str, float | int | str | bool]],
) -> dict[str, float | int]:
    return {
        f"{prefix}_comparisons": len(claims),
        f"{prefix}_overall_supports": sum(1 for claim in claims if bool(claim["supports_overall_claim"])),
        f"{prefix}_mean_rmse_gain_pct": _mean(claims, "position_rmse_gain_pct"),
        f"{prefix}_mean_nees_ratio": _mean(claims, "mean_nees_ratio"),
        f"{prefix}_mean_runtime_ratio": _mean(claims, "runtime_ratio"),
    }


def _mean(rows: list[dict[str, float | int | str | bool]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else float("nan")


def _ratio(candidate: float, comparator: float) -> float:
    return candidate / comparator if comparator else float("inf")


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: Path, result: S3R3StressSweepResult, config: S3R3StressSweepConfig) -> None:
    metadata = {
        "claims_rows": len(result.claims),
        "claims_schema": S3R3_STRESS_CLAIM_FIELDNAMES,
        "config": s3r3_stress_sweep_config_to_dict(config),
        "experiment": "s3r3_stress_sweep",
        "metrics_rows": len(result.metrics),
        "metrics_schema": S3R3_STRESS_FIELDNAMES,
        "summary_rows": len(result.summary),
        "summary_schema": S3R3_STRESS_SUMMARY_FIELDNAMES,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    result: S3R3StressSweepResult,
    metrics_path: Path,
    claims_path: Path,
    summary_path: Path,
    plot_paths: list[Path],
    config: S3R3StressSweepConfig,
) -> None:
    baseline_claims = [claim for claim in result.claims if claim["comparison"] == BASELINE_COMPARISON]
    inflation_claims = [claim for claim in result.claims if claim["comparison"] == INFLATION_COMPARISON]
    best_claim = max(baseline_claims, key=lambda claim: float(claim["position_rmse_gain_pct"]))
    support_count = sum(1 for claim in baseline_claims if bool(claim["supports_overall_claim"]))
    inflation_support_count = sum(1 for claim in inflation_claims if bool(claim["supports_overall_claim"]))
    lines = [
        "# S3+ x R3 Stress Sweep",
        "",
        "This report checks where relaxed S3F helps as orientation prior width and body-frame displacement size change.",
        "",
        f"Trials per scenario: {config.prototype.n_trials}",
        f"Steps per trial: {config.prototype.n_steps}",
        f"Grid sizes: {list(config.prototype.grid_sizes)}",
        f"Prior kappas: {list(config.prior_kappas)} (lower means broader orientation prior)",
        f"Body-increment scales: {list(config.body_increment_scales)}",
        f"Metrics: `{metrics_path.name}`",
        f"Claims: `{claims_path.name}`",
        f"Summary: `{summary_path.name}`",
        "",
        "## Headline",
        "",
        (
            f"`R1+R2` supports the baseline comparison in `{support_count}/{len(baseline_claims)}` grid/scenario rows "
            f"and the inflation comparison in `{inflation_support_count}/{len(inflation_claims)}` rows."
        ),
        (
            f"Largest baseline RMSE gain is `{float(best_claim['position_rmse_gain_pct']):.1f}%` at "
            f"`{best_claim['scenario_id']}`, `{best_claim['grid_size']}` cells."
        ),
        "",
        "## Scenario Summary",
        "",
        _format_summary_table(result.summary),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "The sweep is still synthetic. It is intended to map expected operating regimes, not to replace real-data validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_summary_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Scenario | kappa | disp scale | R1+R2/base support | mean RMSE gain % | mean NEES ratio | mean runtime ratio |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in rows:
        support = f"{int(row['baseline_overall_supports'])}/{int(row['baseline_comparisons'])}"
        body.append(
            "| "
            f"{row['scenario_id']} | "
            f"{float(row['prior_kappa']):.3g} | "
            f"{float(row['body_increment_scale']):.3g} | "
            f"{support} | "
            f"{float(row['baseline_mean_rmse_gain_pct']):.1f} | "
            f"{float(row['baseline_mean_nees_ratio']):.3f} | "
            f"{float(row['baseline_mean_runtime_ratio']):.3f} |"
        )
    return "\n".join([header, separator, *body])


def _write_plots(
    output_dir: Path,
    claims: list[dict[str, float | int | str | bool]],
    summary: list[dict[str, float | int | str]],
) -> list[Path]:
    return [
        _write_gain_by_grid_plot(output_dir, claims),
        _write_summary_heatmap(output_dir, summary, "baseline_mean_rmse_gain_pct", "Mean RMSE Gain [%]", "s3r3_stress_mean_rmse_gain.png"),
        _write_summary_heatmap(output_dir, summary, "baseline_mean_nees_ratio", "Mean NEES Ratio", "s3r3_stress_mean_nees_ratio.png"),
    ]


def _write_gain_by_grid_plot(output_dir: Path, claims: list[dict[str, float | int | str | bool]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    baseline_claims = [claim for claim in claims if claim["comparison"] == BASELINE_COMPARISON]
    for scenario_id in sorted({str(claim["scenario_id"]) for claim in baseline_claims}):
        rows = sorted((claim for claim in baseline_claims if claim["scenario_id"] == scenario_id), key=lambda claim: int(claim["grid_size"]))
        ax.plot(
            [int(row["grid_size"]) for row in rows],
            [float(row["position_rmse_gain_pct"]) for row in rows],
            marker="o",
            linewidth=1.3,
            alpha=0.78,
            label=scenario_id,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Number of quaternion grid cells")
    ax.set_ylabel("R1+R2 RMSE gain over baseline [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return save_figure(fig, output_dir, "s3r3_stress_rmse_gain_by_grid.png")


def _write_summary_heatmap(output_dir: Path, rows: list[dict[str, float | int | str]], metric_name: str, label: str, filename: str) -> Path:
    kappas = sorted({float(row["prior_kappa"]) for row in rows})
    scales = sorted({float(row["body_increment_scale"]) for row in rows})
    matrix = [[_summary_value(rows, kappa, scale, metric_name) for scale in scales] for kappa in kappas]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    image = ax.imshow(matrix, aspect="auto", origin="lower")
    ax.set_xticks(range(len(scales)), [f"{value:.3g}" for value in scales])
    ax.set_yticks(range(len(kappas)), [f"{value:.3g}" for value in kappas])
    ax.set_xlabel("Body-increment scale")
    ax.set_ylabel("Prior kappa")
    ax.set_title(label)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", color="white" if value < 1.0 else "black")
    fig.colorbar(image, ax=ax, label=label)
    return save_figure(fig, output_dir, filename)


def _summary_value(rows: list[dict[str, float | int | str]], prior_kappa: float, body_increment_scale: float, metric_name: str) -> float:
    matches = [
        row
        for row in rows
        if float(row["prior_kappa"]) == prior_kappa and float(row["body_increment_scale"]) == body_increment_scale
    ]
    if not matches:
        return float("nan")
    return float(matches[0][metric_name])


def _scenario_id(prior_kappa: float, body_increment_scale: float) -> str:
    return f"k{_number_token(prior_kappa)}_d{_number_token(body_increment_scale)}"


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")
