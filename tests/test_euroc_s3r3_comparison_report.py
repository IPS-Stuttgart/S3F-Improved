import csv
import json

import numpy as np
import pytest

from se3plusplus_s3f.s3r3.euroc_comparison_report import (
    EuRoCS3R3ComparisonReportConfig,
    run_euroc_s3r3_comparison_report,
    write_euroc_s3r3_comparison_report_outputs,
)


def test_euroc_s3r3_comparison_report_smoke_outputs_metrics_and_claims(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=7)
    config = EuRoCS3R3ComparisonReportConfig(
        grid_sizes=(8,),
        reference_grid_size=16,
        stride=1,
        max_steps=3,
        cell_sample_count=8,
        orientation_transition_kappa=24.0,
        particle_counts=(32,),
    )

    result = run_euroc_s3r3_comparison_report(groundtruth_path, config)

    assert len(result.metrics) == 6
    assert len(result.claims) == 4
    assert {row["filter"] for row in result.metrics} == {"s3f_reference", "s3f", "manifold_ukf", "particle_filter"}
    assert {row["variant"] for row in result.metrics if row["filter"] == "s3f"} == {"baseline", "r1", "r1_r2"}
    assert all(int(row["n_steps"]) == 3 for row in result.metrics)
    assert all(np.isfinite(float(row["position_rmse_to_truth"])) for row in result.metrics)
    assert all(np.isfinite(float(row["position_rmse_to_reference"])) for row in result.metrics)
    assert all("supports_reference_claim" in row for row in result.claims)

    outputs = write_euroc_s3r3_comparison_report_outputs(groundtruth_path, tmp_path / "out", config, write_plots=True)
    for output_name in ("metrics", "claims", "note", "metadata", "euroc_s3r3_reference_rmse", "euroc_s3r3_reference_gain"):
        assert outputs[output_name].exists()

    with outputs["metrics"].open(newline="", encoding="utf-8") as file:
        written_metrics = list(csv.DictReader(file))
    with outputs["claims"].open(newline="", encoding="utf-8") as file:
        written_claims = list(csv.DictReader(file))
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))

    assert len(written_metrics) == 6
    assert len(written_claims) == 4
    assert metadata["experiment"] == "euroc_s3r3_comparison_report"
    assert metadata["config"]["reference_grid_size"] == 16


def test_euroc_s3r3_comparison_report_can_skip_external_baselines(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=5)
    config = EuRoCS3R3ComparisonReportConfig(
        grid_sizes=(8,),
        reference_grid_size=16,
        stride=1,
        max_steps=2,
        cell_sample_count=8,
        include_manifold_ukf=False,
        particle_counts=(),
    )

    result = run_euroc_s3r3_comparison_report(groundtruth_path, config)

    assert len(result.metrics) == 4
    assert len(result.claims) == 2
    assert {row["filter"] for row in result.metrics} == {"s3f_reference", "s3f"}


def test_euroc_s3r3_comparison_report_rejects_bad_reference_grid(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=5)
    config = EuRoCS3R3ComparisonReportConfig(grid_sizes=(8, 16), reference_grid_size=16, max_steps=1)

    with pytest.raises(ValueError, match="reference_grid_size"):
        run_euroc_s3r3_comparison_report(groundtruth_path, config)


def test_euroc_s3r3_comparison_report_rejects_invalid_particle_count(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=5)
    config = EuRoCS3R3ComparisonReportConfig(grid_sizes=(8,), reference_grid_size=16, particle_counts=(0,), max_steps=1)

    with pytest.raises(ValueError, match="particle_counts"):
        run_euroc_s3r3_comparison_report(groundtruth_path, config)


def _write_groundtruth(path, n_poses: int) -> None:
    lines = [
        "#timestamp [ns] p_RS_R_x [m] p_RS_R_y [m] p_RS_R_z [m] q_RS_w [] q_RS_x [] q_RS_y [] q_RS_z []"
    ]
    for idx in range(n_poses):
        timestamp = 1_403_636_580_863_555_584 + idx * 50_000_000
        roll = 0.04 * idx
        pitch = 0.02 * idx
        yaw = 0.2 + 0.04 * idx
        quaternion = _rpy_quaternion(roll, pitch, yaw)
        x = 0.12 * idx
        y = 0.03 * idx * idx
        z = 0.02 * np.sin(0.4 * idx)
        lines.append(f"{timestamp:.10f} {x:.10f} {y:.10f} {z:.10f} {quaternion[3]:.10f} {quaternion[0]:.10f} {quaternion[1]:.10f} {quaternion[2]:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rpy_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )
