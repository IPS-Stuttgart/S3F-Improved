"""Covariance-only diagnostic ablation for the S1 x R2 relaxed S3F benchmark.

This module intentionally keeps the diagnostic separate from the main
quality-cost pipeline.  The proposed predictors remain the representative-cell
baseline, R1 mean correction, and R1+R2 moment correction.  The additional
``cov_only`` row is not a moment-matched predictor: it keeps the representative
prediction mean but adds the unresolved within-cell displacement covariance.
It is useful for testing whether coarse-grid failures are dominated by missing
covariance rather than mean bias.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from pyrecest.filters.relaxed_s3f_circular import (
    circular_error,
    predict_circular_relaxed,
    uniform_circular_cell_statistics,
)

from .relaxed_s3f_pilot import PilotConfig, generate_pilot_trials, make_initial_filter, pilot_config_to_dict
from .s3f_common import linear_position_error_stats, make_linear_likelihood, orientation_mode_and_mean

COVARIANCE_ONLY_VARIANT = "cov_only"
COVARIANCE_DIAGNOSTIC_VARIANTS = ("baseline", "r1", COVARIANCE_ONLY_VARIANT, "r1_r2")

VARIANT_LABELS = {
    "baseline": "Baseline S3F",
    "r1": "S3F + R1",
    COVARIANCE_ONLY_VARIANT: "S3F + covariance-only diagnostic",
    "r1_r2": "S3F + R1 + R2",
}

METRIC_FIELDNAMES = [
    "grid_size",
    "variant",
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_mean_error_rad",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "n_trials",
    "n_steps",
]

PAIRWISE_FIELDNAMES = [
    "grid_size",
    "candidate_variant",
    "comparator_variant",
    "candidate_position_rmse",
    "comparator_position_rmse",
    "position_rmse_delta",
    "candidate_mean_nees",
    "comparator_mean_nees",
    "mean_nees_delta",
    "candidate_coverage_95",
    "comparator_coverage_95",
    "coverage_delta",
]


@dataclass(frozen=True)
class CovarianceDiagnosticConfig:
    """Configuration for the covariance-only diagnostic ablation."""

    pilot: PilotConfig = field(
        default_factory=lambda: PilotConfig(
            variants=COVARIANCE_DIAGNOSTIC_VARIANTS,
            n_trials=16,
            n_steps=16,
            seed=17,
        )
    )


def covariance_diagnostic_config_to_dict(config: CovarianceDiagnosticConfig) -> dict[str, Any]:
    """Return a JSON-serializable diagnostic configuration."""

    return {"pilot": pilot_config_to_dict(config.pilot)}


def run_covariance_diagnostic(
    config: CovarianceDiagnosticConfig = CovarianceDiagnosticConfig(),
) -> list[dict[str, float | int | str]]:
    """Run the covariance-only diagnostic ablation."""

    _validate_config(config)
    trials = generate_pilot_trials(config.pilot)
    rows: list[dict[str, float | int | str]] = []
    for n_cells in config.pilot.grid_sizes:
        for variant in config.pilot.variants:
            rows.append(_run_variant(config.pilot, trials, n_cells, variant))
    return rows


def write_covariance_diagnostic_outputs(
    output_dir: Path,
    config: CovarianceDiagnosticConfig = CovarianceDiagnosticConfig(),
) -> dict[str, Path]:
    """Run the diagnostic and write metrics, pairwise rows, note, and metadata."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_covariance_diagnostic(config)
    pairwise = _build_pairwise_rows(rows)

    metrics_path = output_dir / "covariance_diagnostic_metrics.csv"
    _write_csv(metrics_path, rows, METRIC_FIELDNAMES)

    pairwise_path = output_dir / "covariance_diagnostic_pairwise.csv"
    _write_csv(pairwise_path, pairwise, PAIRWISE_FIELDNAMES)

    note_path = output_dir / "covariance_diagnostic_note.md"
    _write_note(note_path, rows, pairwise, metrics_path, pairwise_path, config)

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, rows, pairwise, config)

    return {
        "metrics": metrics_path,
        "pairwise": pairwise_path,
        "note": note_path,
        "metadata": metadata_path,
    }


def _validate_config(config: CovarianceDiagnosticConfig) -> None:
    if not config.pilot.grid_sizes:
        raise ValueError("grid_sizes must not be empty.")
    if min(config.pilot.grid_sizes) <= 0:
        raise ValueError("all grid sizes must be positive.")
    if config.pilot.n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if config.pilot.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    for variant in config.pilot.variants:
        if variant not in COVARIANCE_DIAGNOSTIC_VARIANTS:
            raise ValueError(f"Unknown diagnostic variant {variant!r}.")


def _run_variant(
    config: PilotConfig,
    trials: list[dict[str, np.ndarray | float]],
    n_cells: int,
    variant: str,
) -> dict[str, float | int | str]:
    measurement_cov = np.eye(2) * config.measurement_noise_std**2
    process_noise_cov = np.eye(2) * config.process_noise_std**2
    body_increment = np.asarray(config.body_increment, dtype=float)

    sum_position_sq_error = 0.0
    sum_orientation_mode_error = 0.0
    sum_orientation_mean_error = 0.0
    sum_nees = 0.0
    coverage_hits = 0
    n_metrics = 0
    runtime = 0.0

    for trial in trials:
        filter_ = make_initial_filter(config, n_cells)
        true_angle = float(trial["angle"])
        true_positions = np.asarray(trial["positions"], dtype=float)
        measurements = np.asarray(trial["measurements"], dtype=float)

        for step, measurement in enumerate(measurements):
            likelihood = make_linear_likelihood(measurement, measurement_cov)
            runtime += _predict_update_variant(filter_, body_increment, variant, process_noise_cov, likelihood)

            error, nees = linear_position_error_stats(filter_, true_positions[step + 1])
            sum_position_sq_error += float(error @ error)
            sum_nees += nees
            coverage_hits += int(nees <= 5.991464547107979)

            mode_angle, mean_angle = orientation_mode_and_mean(filter_)
            sum_orientation_mode_error += circular_error(mode_angle, true_angle)
            sum_orientation_mean_error += circular_error(mean_angle, true_angle)
            n_metrics += 1

    return {
        "grid_size": n_cells,
        "variant": variant,
        "position_rmse": float(np.sqrt(sum_position_sq_error / n_metrics)),
        "orientation_mode_error_rad": sum_orientation_mode_error / n_metrics,
        "orientation_mean_error_rad": sum_orientation_mean_error / n_metrics,
        "mean_nees": sum_nees / n_metrics,
        "coverage_95": coverage_hits / n_metrics,
        "runtime_ms_per_step": 1000.0 * runtime / n_metrics,
        "n_trials": config.n_trials,
        "n_steps": config.n_steps,
    }


def _predict_update_variant(
    filter_,
    body_increment: np.ndarray,
    variant: str,
    process_noise_cov: np.ndarray,
    likelihood: GaussianDistribution,
) -> float:
    start = perf_counter()
    if variant == COVARIANCE_ONLY_VARIANT:
        _predict_covariance_only(filter_, body_increment, process_noise_cov)
    else:
        predict_circular_relaxed(
            filter_,
            body_increment,
            variant=variant,
            process_noise_cov=process_noise_cov,
        )
    filter_.update(likelihoods_linear=[likelihood])
    return perf_counter() - start


def _predict_covariance_only(filter_, body_increment: np.ndarray, process_noise_cov: np.ndarray) -> None:
    """Apply representative-cell mean plus unresolved within-cell covariance."""

    state = filter_.filter_state
    n_cells = len(state.linear_distributions)
    if state.lin_dim != 2:
        raise ValueError("covariance-only diagnostic requires a 2-D linear state.")

    grid = np.asarray(state.gd.get_grid(), dtype=float).reshape(-1)
    stats = uniform_circular_cell_statistics(n_cells, body_increment, grid=grid)
    covariance_inflations = np.asarray(stats.covariance_inflations, dtype=float)
    covariance_matrices = np.stack([process_noise_cov + covariance_inflations[idx] for idx in range(n_cells)], axis=2)
    filter_.predict_linear(
        covariance_matrices=covariance_matrices,
        linear_input_vectors=np.asarray(stats.representative_displacements, dtype=float).T,
    )


def _build_pairwise_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    comparisons = [
        (COVARIANCE_ONLY_VARIANT, "baseline"),
        ("r1", "baseline"),
        ("r1_r2", COVARIANCE_ONLY_VARIANT),
        ("r1_r2", "baseline"),
    ]
    by_key = {(int(row["grid_size"]), str(row["variant"])): row for row in rows}
    pairwise = []
    for grid_size in sorted({int(row["grid_size"]) for row in rows}):
        for candidate_variant, comparator_variant in comparisons:
            candidate = by_key.get((grid_size, candidate_variant))
            comparator = by_key.get((grid_size, comparator_variant))
            if candidate is None or comparator is None:
                continue
            pairwise.append(_pairwise_row(grid_size, candidate, comparator))
    return pairwise


def _pairwise_row(
    grid_size: int,
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    return {
        "grid_size": grid_size,
        "candidate_variant": candidate["variant"],
        "comparator_variant": comparator["variant"],
        "candidate_position_rmse": candidate["position_rmse"],
        "comparator_position_rmse": comparator["position_rmse"],
        "position_rmse_delta": float(candidate["position_rmse"]) - float(comparator["position_rmse"]),
        "candidate_mean_nees": candidate["mean_nees"],
        "comparator_mean_nees": comparator["mean_nees"],
        "mean_nees_delta": float(candidate["mean_nees"]) - float(comparator["mean_nees"]),
        "candidate_coverage_95": candidate["coverage_95"],
        "comparator_coverage_95": comparator["coverage_95"],
        "coverage_delta": float(candidate["coverage_95"]) - float(comparator["coverage_95"]),
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(
    path: Path,
    rows: list[dict[str, float | int | str]],
    pairwise: list[dict[str, float | int | str]],
    config: CovarianceDiagnosticConfig,
) -> None:
    metadata = {
        "experiment": "s1r2_covariance_diagnostic",
        "config": covariance_diagnostic_config_to_dict(config),
        "metrics_schema": METRIC_FIELDNAMES,
        "metrics_rows": len(rows),
        "pairwise_schema": PAIRWISE_FIELDNAMES,
        "pairwise_rows": len(pairwise),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    rows: list[dict[str, float | int | str]],
    pairwise: list[dict[str, float | int | str]],
    metrics_path: Path,
    pairwise_path: Path,
    config: CovarianceDiagnosticConfig,
) -> None:
    grid_sizes = sorted({int(row["grid_size"]) for row in rows})
    coarse_grid = grid_sizes[0]
    coarse_pairs = [row for row in pairwise if int(row["grid_size"]) == coarse_grid]
    cov_vs_base = _find_pair(coarse_pairs, COVARIANCE_ONLY_VARIANT, "baseline")
    full_vs_cov = _find_pair(coarse_pairs, "r1_r2", COVARIANCE_ONLY_VARIANT)

    lines = [
        "# S1/R2 Covariance-Only Diagnostic",
        "",
        "This diagnostic isolates the R2 covariance term by keeping the representative-cell prediction mean",
        "while adding the unresolved within-cell displacement covariance. It is not a proposed predictor,",
        "because the covariance is centered at the within-cell mean whereas the propagated mean remains representative.",
        "",
        f"- trials: {config.pilot.n_trials}",
        f"- steps per trial: {config.pilot.n_steps}",
        f"- grid sizes: {list(config.pilot.grid_sizes)}",
        f"- metrics: `{metrics_path.name}`",
        f"- pairwise rows: `{pairwise_path.name}`",
        "",
        "## Coarsest-grid diagnostic",
        "",
    ]
    if cov_vs_base is not None:
        lines.append(
            f"At {coarse_grid} cells, covariance-only vs baseline has RMSE delta "
            f"`{float(cov_vs_base['position_rmse_delta']):.4f}` and NEES delta "
            f"`{float(cov_vs_base['mean_nees_delta']):.4f}`."
        )
    if full_vs_cov is not None:
        lines.append(
            f"At {coarse_grid} cells, R1+R2 vs covariance-only has RMSE delta "
            f"`{float(full_vs_cov['position_rmse_delta']):.4f}` and NEES delta "
            f"`{float(full_vs_cov['mean_nees_delta']):.4f}`."
        )
    lines.extend(
        [
            "",
            "Use this output to decide whether the manuscript can claim that covariance underestimation",
            "is the dominant coarse-grid mechanism, or whether the R1 and R2 terms interact materially.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_pair(
    pairwise: list[dict[str, float | int | str]],
    candidate_variant: str,
    comparator_variant: str,
) -> dict[str, float | int | str] | None:
    for row in pairwise:
        if row["candidate_variant"] == candidate_variant and row["comparator_variant"] == comparator_variant:
            return row
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "s1r2_covariance_diagnostic")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = CovarianceDiagnosticConfig(
        pilot=PilotConfig(
            grid_sizes=tuple(args.grid_sizes),
            variants=COVARIANCE_DIAGNOSTIC_VARIANTS,
            n_trials=args.trials,
            n_steps=args.steps,
            seed=args.seed,
        )
    )
    outputs = write_covariance_diagnostic_outputs(args.output_dir, config)
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
