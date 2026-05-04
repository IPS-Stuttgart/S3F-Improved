import csv
import json

import numpy as np
import pytest

from se3plusplus_s3f.s3r3.dynamic_robustness import (
    BASELINE_COMPARISON,
    INFLATION_COMPARISON,
    S3R3DynamicRobustnessConfig,
    run_s3r3_dynamic_robustness_sweep,
    write_s3r3_dynamic_robustness_outputs,
)
from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig


def test_s3r3_dynamic_robustness_smoke_outputs_aggregates(tmp_path):
    config = S3R3DynamicRobustnessConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
            cell_sample_count=8,
        ),
        seeds=(47, 48),
        orientation_increment_scales=(0.5, 1.0),
    )

    result = run_s3r3_dynamic_robustness_sweep(config)

    assert len(result.metrics) == 12
    assert len(result.claims) == 8
    assert len(result.aggregates) == 4
    assert {row["variant"] for row in result.metrics} == {"baseline", "r1", "r1_r2"}
    assert {row["comparison"] for row in result.claims} == {BASELINE_COMPARISON, INFLATION_COMPARISON}
    assert all(np.isfinite(float(row["mean_position_rmse_gain_pct"])) for row in result.aggregates)
    assert all(0.0 <= float(row["win_rate"]) <= 1.0 for row in result.aggregates)

    outputs = write_s3r3_dynamic_robustness_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "claims", "aggregates", "note", "metadata"):
        assert outputs[output_name].is_file()

    assert _csv_row_count(outputs["metrics"]) == len(result.metrics)
    assert _csv_row_count(outputs["claims"]) == len(result.claims)
    assert _csv_row_count(outputs["aggregates"]) == len(result.aggregates)

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_dynamic_robustness"
    assert metadata["metrics_rows"] == 12
    assert metadata["claims_rows"] == 8
    assert metadata["aggregates_rows"] == 4

    note_text = outputs["note"].read_text(encoding="utf-8")
    assert "## Headline" in note_text
    assert "Baseline Aggregate Table" in note_text


def test_s3r3_dynamic_robustness_rejects_empty_scales():
    config = S3R3DynamicRobustnessConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8,)),
        orientation_increment_scales=(),
    )

    with pytest.raises(ValueError, match="orientation_increment_scales"):
        run_s3r3_dynamic_robustness_sweep(config)


def _csv_row_count(path):
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _row in csv.DictReader(file))
