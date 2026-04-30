import csv

import numpy as np
import pytest

from se3plusplus_s3f.s3r3.highres_reference import (
    S3R3HighResReferenceConfig,
    run_s3r3_highres_reference_benchmark,
    write_s3r3_highres_reference_outputs,
)
from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig


def test_s3r3_highres_reference_smoke_outputs_metrics(tmp_path):
    config = S3R3HighResReferenceConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
        ),
        reference_grid_size=16,
    )

    rows = run_s3r3_highres_reference_benchmark(config)

    assert len(rows) == 3
    assert {row["variant"] for row in rows} == {"baseline", "r1", "r1_r2"}
    assert all(int(row["grid_size"]) == 8 for row in rows)
    assert all(int(row["reference_grid_size"]) == 16 for row in rows)
    assert all(float(row["position_rmse_to_reference"]) >= 0.0 for row in rows)
    assert all(np.isfinite(float(row["orientation_mode_error_to_reference_rad"])) for row in rows)
    assert all(float(row["runtime_ms_per_step"]) > 0.0 for row in rows)

    output_paths = write_s3r3_highres_reference_outputs(tmp_path / "out", config, write_plots=False)
    assert {"metrics", "note", "metadata"} <= set(output_paths)
    assert all(path.is_file() for path in output_paths.values())

    written_rows = list(csv.DictReader(output_paths["metrics"].open(newline="", encoding="utf-8")))
    assert len(written_rows) == len(rows)


def test_s3r3_highres_reference_requires_finer_reference():
    config = S3R3HighResReferenceConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8, 16)),
        reference_grid_size=16,
    )

    with pytest.raises(ValueError, match="reference_grid_size"):
        run_s3r3_highres_reference_benchmark(config)
