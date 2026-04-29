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
        )
    )

    rows, claims = run_quality_cost_report(config)

    assert len(rows) == 6
    assert {row["variant"] for row in rows} == {"baseline", "r1", "r1_r2"}
    assert {int(row["grid_size"]) for row in rows} == {8, 16}
    assert all(np.isfinite(float(row["position_rmse"])) for row in rows)
    assert all(np.isfinite(float(row["position_rmse_to_reference"])) for row in rows)
    assert all(float(row["runtime_ms_per_step"]) > 0.0 for row in rows)

    assert len(claims) == 3
    assert {claim["claim_id"] for claim in claims} == {
        "coarser_r1_r2_8_vs_baseline_16",
        "same_grid_8",
        "same_grid_16",
    }
    assert all(float(claim["runtime_ratio"]) > 0.0 for claim in claims)

    outputs = write_quality_cost_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "claims", "note", "metadata"):
        assert outputs[output_name].is_file()

    with outputs["metrics"].open(newline="", encoding="utf-8") as metrics_file:
        assert len(list(csv.DictReader(metrics_file))) == 6
    with outputs["claims"].open(newline="", encoding="utf-8") as claims_file:
        assert len(list(csv.DictReader(claims_file))) == 3

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "quality_cost"
    assert metadata["metrics_rows"] == 6
    assert metadata["claims_rows"] == 3
