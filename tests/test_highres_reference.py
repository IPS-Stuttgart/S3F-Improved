import csv

import numpy as np
import pytest

from se3plusplus_s3f.s1r2.highres_reference import (
    HighResReferenceConfig,
    run_highres_reference_benchmark,
    write_highres_reference_outputs,
)
from se3plusplus_s3f.s1r2.relaxed_s3f_pilot import PilotConfig


def test_highres_reference_benchmark_smoke(tmp_path):
    config = HighResReferenceConfig(
        pilot=PilotConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
        ),
        reference_grid_size=16,
    )

    rows = run_highres_reference_benchmark(config)

    assert {row["variant"] for row in rows} == {"baseline", "r1", "r1_r2"}
    assert all(int(row["grid_size"]) == 8 for row in rows)
    assert all(int(row["reference_grid_size"]) == 16 for row in rows)
    assert all(float(row["position_rmse_to_reference"]) >= 0.0 for row in rows)
    assert all(np.isfinite(float(row["orientation_mean_error_to_reference_rad"])) for row in rows)
    assert all(float(row["runtime_ms_per_step"]) > 0.0 for row in rows)

    outputs = write_highres_reference_outputs(tmp_path / "out", config, write_plots=False)
    assert outputs["metrics"].exists()
    assert outputs["note"].exists()
    assert outputs["metadata"].exists()
    with outputs["metrics"].open(newline="", encoding="utf-8") as file:
        written_rows = list(csv.DictReader(file))
    assert len(written_rows) == 3


def test_highres_reference_requires_finer_reference():
    config = HighResReferenceConfig(
        pilot=PilotConfig(grid_sizes=(8, 16)),
        reference_grid_size=16,
    )

    with pytest.raises(ValueError, match="reference_grid_size"):
        run_highres_reference_benchmark(config)
