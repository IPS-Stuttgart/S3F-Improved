import csv
import json

import numpy as np
import pytest

from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig
from se3plusplus_s3f.s3r3.stress_sweep import (
    BASELINE_COMPARISON,
    INFLATION_COMPARISON,
    S3R3StressSweepConfig,
    run_s3r3_stress_sweep,
    write_s3r3_stress_sweep_outputs,
)


def test_s3r3_stress_sweep_smoke_outputs_claims(tmp_path):
    config = S3R3StressSweepConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
        ),
        prior_kappas=(2.0,),
        body_increment_scales=(1.25,),
    )

    result = run_s3r3_stress_sweep(config)

    assert len(result.metrics) == 3
    assert len(result.claims) == 2
    assert len(result.summary) == 1
    assert {row["variant"] for row in result.metrics} == {"baseline", "r1", "r1_r2"}
    assert {claim["comparison"] for claim in result.claims} == {BASELINE_COMPARISON, INFLATION_COMPARISON}
    assert all(np.isfinite(float(claim["position_rmse_ratio"])) for claim in result.claims)
    assert all(np.isfinite(float(claim["mean_nees_ratio"])) for claim in result.claims)

    outputs = write_s3r3_stress_sweep_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "claims", "summary", "note", "metadata"):
        assert outputs[output_name].is_file()

    assert _csv_row_count(outputs["metrics"]) == len(result.metrics)
    assert _csv_row_count(outputs["claims"]) == len(result.claims)

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_stress_sweep"
    assert metadata["metrics_rows"] == 3
    assert metadata["claims_rows"] == 2
    assert metadata["summary_rows"] == 1

    note_text = outputs["note"].read_text(encoding="utf-8")
    assert "## Headline" in note_text
    assert "Scenario Summary" in note_text


def test_s3r3_stress_sweep_rejects_nonpositive_stress_values():
    config = S3R3StressSweepConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8,)),
        prior_kappas=(0.0,),
        body_increment_scales=(1.0,),
    )

    with pytest.raises(ValueError, match="prior_kappas"):
        run_s3r3_stress_sweep(config)


def _csv_row_count(path):
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _row in csv.DictReader(file))
