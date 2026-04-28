import csv
import json
from pathlib import Path

import numpy.testing as npt

from se3plusplus_s3f.wp1.relaxed_s3f_pilot import (
    PilotConfig,
    load_pilot_config,
    run_relaxed_s3f_pilot,
    write_relaxed_s3f_pilot_outputs,
)


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
REFERENCE_METRICS = [
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_mean_error_rad",
    "mean_nees",
    "coverage_95",
]


def test_pilot_runner_smoke(tmp_path):
    config = PilotConfig(grid_sizes=(8,), n_trials=1, n_steps=2)
    rows = run_relaxed_s3f_pilot(config)
    assert len(rows) == 3

    outputs = write_relaxed_s3f_pilot_outputs(tmp_path, config, write_plots=False)
    assert outputs["metrics"].exists()
    assert outputs["note"].exists()
    assert outputs["metadata"].exists()

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "wp1_s1_r2_relaxed_s3f"
    assert metadata["config"]["grid_sizes"] == [8]
    assert metadata["metrics_rows"] == 3


def test_wp1_reference_metrics_match_committed_reference():
    config = load_pilot_config(REFERENCE_DIR / "wp1_relaxed_s3f_regression_config.json")
    with (REFERENCE_DIR / "wp1_relaxed_s3f_regression_metrics.csv").open(newline="", encoding="utf-8") as file:
        expected_rows = list(csv.DictReader(file))
    actual_rows = run_relaxed_s3f_pilot(config)

    actual_by_key = {
        (int(row["grid_size"]), str(row["variant"])): row
        for row in actual_rows
    }
    expected_by_key = {
        (int(row["grid_size"]), str(row["variant"])): row
        for row in expected_rows
    }
    assert set(actual_by_key) == set(expected_by_key)

    for key, expected_row in expected_by_key.items():
        actual_row = actual_by_key[key]
        for metric in REFERENCE_METRICS:
            npt.assert_allclose(
                float(actual_row[metric]),
                float(expected_row[metric]),
                rtol=1e-10,
                atol=1e-10,
            )
