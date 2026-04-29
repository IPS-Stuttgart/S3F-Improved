import csv
import json

import numpy as np

from se3plusplus_s3f.s1r2.highres_reference import HighResReferenceConfig
from se3plusplus_s3f.s1r2.quality_cost import (
    QualityCostConfig,
    _interpret_nearest_particle_comparison,
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
