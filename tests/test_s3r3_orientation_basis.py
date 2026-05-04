import csv
import json

import numpy as np

from se3plusplus_s3f.s3r3.orientation_basis import (
    S3R3OrientationBasisConfig,
    run_s3r3_orientation_basis_diagnostic,
    write_s3r3_orientation_basis_outputs,
)
from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig


def test_s3r3_orientation_basis_smoke_outputs_metrics(tmp_path):
    config = S3R3OrientationBasisConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
            cell_sample_count=8,
        )
    )

    rows = run_s3r3_orientation_basis_diagnostic(config)

    assert len(rows) == 1
    row = rows[0]
    assert int(row["grid_size"]) == 8
    assert int(row["basis_grid_matches_s3f"]) == 1
    assert float(row["basis_prior_max_abs_diff"]) < 1e-12
    assert np.isfinite(float(row["mean_mode_error_rad"]))
    assert np.isfinite(float(row["mean_point_error_rad"]))
    assert float(row["mean_effective_cells"]) > 0.0

    outputs = write_s3r3_orientation_basis_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "note", "metadata"):
        assert outputs[output_name].is_file()

    with outputs["metrics"].open(newline="", encoding="utf-8") as metrics_file:
        assert len(list(csv.DictReader(metrics_file))) == 1

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_orientation_basis"
    assert metadata["config"]["prototype"]["grid_sizes"] == [8]
    assert metadata["metrics_rows"] == 1


def test_s3r3_orientation_basis_rejects_unknown_variant():
    config = S3R3OrientationBasisConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8,)),
        variant="unknown",
    )

    try:
        run_s3r3_orientation_basis_diagnostic(config)
    except ValueError as exc:
        assert "Unknown variant" in str(exc)
    else:
        raise AssertionError("expected unknown variant to be rejected")
