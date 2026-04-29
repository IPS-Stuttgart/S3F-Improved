import csv
import json

import numpy as np

from se3plusplus_s3f.s1r2.highres_reference import HighResReferenceConfig
from se3plusplus_s3f.s1r2.quality_cost import (
    QualityCostConfig,
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
