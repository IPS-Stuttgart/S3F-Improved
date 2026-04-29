import csv
import json

import numpy as np

from se3plusplus_s3f.s1r2.highres_reference import HighResReferenceConfig
from se3plusplus_s3f.s1r2.quality_cost import (
    QualityCostConfig,
    _interpret_nearest_particle_comparison,
    _interpret_repeat_summary,
    _rmse_runtime_frontier,
    run_quality_cost_report,
    write_quality_cost_outputs,
)
from se3plusplus_s3f.s1r2.relaxed_s3f_pilot import PilotConfig


def test_quality_cost_report_smoke_outputs_metrics_and_claims(tmp_path):
    config = QualityCostConfig(
        reference=HighResReferenceConfig(
            pilot=PilotConfig(
                grid_sizes=(8, 16),
                n_trials=1,
                n_steps=2,
            ),
            reference_grid_size=32,
        ),
        particle_counts=(16, 32),
    )

    result = run_quality_cost_report(config)

    assert len(result.metrics) == 6
    assert {row["variant"] for row in result.metrics} == {"baseline", "r1", "r1_r2"}
    assert {int(row["grid_size"]) for row in result.metrics} == {8, 16}
    assert all(np.isfinite(float(row["position_rmse"])) for row in result.metrics)
    assert all(np.isfinite(float(row["position_rmse_to_reference"])) for row in result.metrics)
    assert all(float(row["runtime_ms_per_step"]) > 0.0 for row in result.metrics)

    assert len(result.claims) == 3
    assert {claim["claim_id"] for claim in result.claims} == {
        "coarser_r1_r2_8_vs_baseline_16",
        "same_grid_8",
        "same_grid_16",
    }
    assert all(float(claim["runtime_ratio"]) > 0.0 for claim in result.claims)

    assert len(result.pareto) == 8
    assert {row["filter"] for row in result.pareto} == {"s3f", "particle_filter"}
    assert {
        int(row["particle_count"])
        for row in result.pareto
        if row["filter"] == "particle_filter"
    } == {16, 32}
    assert all(np.isfinite(float(row["position_rmse"])) for row in result.pareto)
    assert all(float(row["runtime_ms_per_step"]) > 0.0 for row in result.pareto)

    outputs = write_quality_cost_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "claims", "pareto", "note", "metadata"):
        assert outputs[output_name].is_file()

    with outputs["metrics"].open(newline="", encoding="utf-8") as metrics_file:
        assert len(list(csv.DictReader(metrics_file))) == 6
    with outputs["claims"].open(newline="", encoding="utf-8") as claims_file:
        assert len(list(csv.DictReader(claims_file))) == 3
    with outputs["pareto"].open(newline="", encoding="utf-8") as pareto_file:
        assert len(list(csv.DictReader(pareto_file))) == 8

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "quality_cost"
    assert metadata["metrics_rows"] == 6
    assert metadata["claims_rows"] == 3
    assert metadata["pareto_rows"] == 8

    note_text = outputs["note"].read_text(encoding="utf-8")
    assert "## RMSE-Runtime Frontier" in note_text
    assert "Nearest-runtime particle row" in note_text


def test_rmse_runtime_frontier_excludes_dominated_rows():
    rows = [
        _pareto_test_row("fast coarse", "s3f", 0.40, 0.10),
        _pareto_test_row("middle relaxed", "s3f", 0.20, 0.30),
        _pareto_test_row("dominated relaxed", "s3f", 0.25, 0.35),
        _pareto_test_row("accurate particle", "particle_filter", 0.15, 0.60),
    ]

    frontier = _rmse_runtime_frontier(rows)

    assert [row["label"] for row in frontier] == [
        "fast coarse",
        "middle relaxed",
        "accurate particle",
    ]


def test_nearest_particle_interpretation_calls_out_dominance():
    best_r1_r2 = _pareto_test_row("S3F + R1 + R2 (64 cells)", "s3f", 0.17, 0.55)
    rows = [
        best_r1_r2,
        _pareto_test_row("Particle filter (512)", "particle_filter", 0.20, 0.22),
        _pareto_test_row("Particle filter (2048)", "particle_filter", 0.16, 0.50),
        _pareto_test_row("Particle filter (8192)", "particle_filter", 0.15, 1.60),
    ]

    interpretation = _interpret_nearest_particle_comparison(best_r1_r2, rows)

    assert "Nearest-runtime particle row: `Particle filter (2048)`" in interpretation
    assert "dominates the best R1+R2 row" in interpretation


def test_quality_cost_repeats_write_summary_outputs(tmp_path):
    config = QualityCostConfig(
        reference=HighResReferenceConfig(
            pilot=PilotConfig(
                grid_sizes=(8, 16),
                n_trials=1,
                n_steps=1,
                seed=3,
            ),
            reference_grid_size=32,
        ),
        particle_counts=(512,),
        particle_seed=5,
        repeats=2,
        repeat_seed_stride=11,
    )

    result = run_quality_cost_report(config)

    assert len(result.metrics) == 6
    assert len(result.pareto) == 7
    assert len(result.repeat_pareto) == 14
    assert len(result.summary) == 7
    assert len(result.pairwise) == 1
    assert result.pairwise[0]["pair_id"] == "r1_r2_8_vs_baseline_16"
    assert np.isfinite(float(result.pairwise[0]["position_rmse_delta_mean"]))
    assert {int(row["seed"]) for row in result.repeat_pareto} == {3, 14}
    assert {int(row["particle_seed"]) for row in result.repeat_pareto} == {5, 16}
    assert all(int(row["n_repeats"]) == 2 for row in result.summary)

    outputs = write_quality_cost_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("repeat_pareto", "summary", "pairwise"):
        assert outputs[output_name].is_file()

    with outputs["repeat_pareto"].open(newline="", encoding="utf-8") as repeat_file:
        assert len(list(csv.DictReader(repeat_file))) == 14
    with outputs["summary"].open(newline="", encoding="utf-8") as summary_file:
        assert len(list(csv.DictReader(summary_file))) == 7
    with outputs["pairwise"].open(newline="", encoding="utf-8") as pairwise_file:
        assert len(list(csv.DictReader(pairwise_file))) == 1

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["repeat_pareto_rows"] == 14
    assert metadata["summary_rows"] == 7
    assert metadata["pairwise_rows"] == 1

    note_text = outputs["note"].read_text(encoding="utf-8")
    assert "## Repeat Summary" in note_text
    assert "## Paired Comparisons" in note_text
    assert "Across `2` repeats" in note_text


def test_repeat_summary_interpretation_calls_out_nonoverlapping_ci():
    best_r1_r2 = _summary_test_row(
        "S3F + R1 + R2 (64 cells)",
        "s3f",
        "r1_r2",
        rmse_mean=0.17,
        rmse_ci=(0.168, 0.172),
        runtime_mean=0.55,
        runtime_ci=(0.54, 0.56),
    )
    particle = _summary_test_row(
        "Particle filter (2048)",
        "particle_filter",
        "bootstrap",
        rmse_mean=0.16,
        rmse_ci=(0.158, 0.162),
        runtime_mean=0.50,
        runtime_ci=(0.49, 0.51),
    )

    interpretation = _interpret_repeat_summary([best_r1_r2, particle])

    assert "Across `5` repeats" in interpretation
    assert "non-overlapping approximate 95% CIs" in interpretation


def _pareto_test_row(
    label: str,
    filter_name: str,
    rmse: float,
    runtime: float,
) -> dict[str, float | int | str]:
    return {
        "filter": filter_name,
        "label": label,
        "variant": "r1_r2" if filter_name == "s3f" else "bootstrap",
        "grid_size": 64 if filter_name == "s3f" else "",
        "particle_count": "" if filter_name == "s3f" else 2048,
        "resource_count": 64 if filter_name == "s3f" else 2048,
        "position_rmse": rmse,
        "mean_nees": 2.0,
        "coverage_95": 0.95,
        "runtime_ms_per_step": runtime,
        "position_rmse_to_reference": "",
        "runtime_ratio_to_reference": "",
        "n_trials": 1,
        "n_steps": 1,
    }


def _summary_test_row(
    label: str,
    filter_name: str,
    variant: str,
    rmse_mean: float,
    rmse_ci: tuple[float, float],
    runtime_mean: float,
    runtime_ci: tuple[float, float],
) -> dict[str, float | int | str]:
    return {
        "filter": filter_name,
        "label": label,
        "variant": variant,
        "grid_size": 64 if filter_name == "s3f" else "",
        "particle_count": "" if filter_name == "s3f" else 2048,
        "resource_count": 64 if filter_name == "s3f" else 2048,
        "position_rmse_mean": rmse_mean,
        "position_rmse_std": 0.01,
        "position_rmse_ci95_low": rmse_ci[0],
        "position_rmse_ci95_high": rmse_ci[1],
        "mean_nees_mean": 2.0,
        "mean_nees_std": 0.1,
        "coverage_95_mean": 0.95,
        "coverage_95_std": 0.01,
        "runtime_ms_per_step_mean": runtime_mean,
        "runtime_ms_per_step_std": 0.01,
        "runtime_ms_per_step_ci95_low": runtime_ci[0],
        "runtime_ms_per_step_ci95_high": runtime_ci[1],
        "n_repeats": 5,
        "n_trials": 32,
        "n_steps": 20,
    }
