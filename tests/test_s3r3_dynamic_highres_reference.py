import csv
import json

import numpy as np
import pytest

from se3plusplus_s3f.s3r3.dynamic_highres_reference import (
    S3R3DynamicHighResReferenceConfig,
    run_s3r3_dynamic_highres_reference_benchmark,
    write_s3r3_dynamic_highres_reference_outputs,
)
from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig


def test_s3r3_dynamic_highres_reference_smoke_outputs_claims(tmp_path):
    config = S3R3DynamicHighResReferenceConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
            cell_sample_count=8,
        ),
        reference_grid_size=16,
    )

    result = run_s3r3_dynamic_highres_reference_benchmark(config)

    assert len(result.metrics) == 3
    assert len(result.claims) == 2
    assert {row["variant"] for row in result.metrics} == {"baseline", "r1", "r1_r2"}
    assert {row["comparator_variant"] for row in result.claims} == {"baseline", "r1"}
    assert all(np.isfinite(float(row["position_rmse_to_reference"])) for row in result.metrics)
    assert all(np.isfinite(float(row["position_rmse_to_reference_ratio"])) for row in result.claims)

    outputs = write_s3r3_dynamic_highres_reference_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "claims", "note", "metadata"):
        assert outputs[output_name].is_file()

    assert _csv_row_count(outputs["metrics"]) == len(result.metrics)
    assert _csv_row_count(outputs["claims"]) == len(result.claims)

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_dynamic_highres_reference"
    assert metadata["metrics_rows"] == 3
    assert metadata["claims_rows"] == 2
    assert metadata["config"]["reference_grid_size"] == 16

    note_text = outputs["note"].read_text(encoding="utf-8")
    assert "## Headline" in note_text
    assert "Baseline Comparison" in note_text


def test_s3r3_dynamic_highres_reference_rejects_low_reference_grid_size():
    config = S3R3DynamicHighResReferenceConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8,)),
        reference_grid_size=8,
    )

    with pytest.raises(ValueError, match="reference_grid_size"):
        run_s3r3_dynamic_highres_reference_benchmark(config)


def _csv_row_count(path):
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _row in csv.DictReader(file))
