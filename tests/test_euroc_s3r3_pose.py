import csv
import json

import numpy as np
import numpy.testing as npt
import pytest

from se3plusplus_s3f.s3r3.euroc_pose import (
    EuRoCS3R3PoseConfig,
    load_euroc_pose_groundtruth,
    run_euroc_s3r3_pose,
    write_euroc_s3r3_pose_outputs,
)


def test_load_euroc_pose_groundtruth_from_whitespace_file(tmp_path):
    path = tmp_path / "groundtruth.txt"
    _write_groundtruth(path, n_poses=6)

    trajectory = load_euroc_pose_groundtruth(path)

    assert trajectory.positions.shape == (6, 3)
    assert trajectory.quaternions.shape == (6, 4)
    npt.assert_allclose(trajectory.timestamps_s[:3], [0.0, 0.05, 0.10], atol=1e-6)
    npt.assert_allclose(np.linalg.norm(trajectory.quaternions, axis=1), np.ones(6), atol=1e-9)
    npt.assert_allclose(trajectory.quaternions[0], [0.0, 0.0, np.sin(0.1), np.cos(0.1)], atol=1e-9)


def test_euroc_s3r3_pose_smoke_outputs_metrics_and_claims(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=7)
    config = EuRoCS3R3PoseConfig(grid_size=8, stride=1, max_steps=3, cell_sample_count=8, orientation_transition_kappa=24.0)

    result = run_euroc_s3r3_pose(groundtruth_path, config)

    assert {row["variant"] for row in result.metrics} == {"baseline", "r1", "r1_r2"}
    assert len(result.metrics) == 3
    assert len(result.claims) == 2
    assert all(int(row["n_steps"]) == 3 for row in result.metrics)
    assert all(float(row["path_length_m"]) > 0.0 for row in result.metrics)
    assert all(np.isfinite(float(row["position_rmse"])) for row in result.metrics)
    assert all(np.isfinite(float(row["orientation_point_error_rad"])) for row in result.metrics)
    assert all("supports_overall_claim" in row for row in result.claims)

    outputs = write_euroc_s3r3_pose_outputs(groundtruth_path, tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "claims", "note", "metadata"):
        assert outputs[output_name].exists()

    with outputs["metrics"].open(newline="", encoding="utf-8") as file:
        written_metrics = list(csv.DictReader(file))
    with outputs["claims"].open(newline="", encoding="utf-8") as file:
        written_claims = list(csv.DictReader(file))
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))

    assert len(written_metrics) == 3
    assert len(written_claims) == 2
    assert metadata["experiment"] == "euroc_s3r3_pose"
    assert metadata["config"]["grid_size"] == 8


def test_euroc_s3r3_pose_rejects_bad_slice(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=3)
    config = EuRoCS3R3PoseConfig(grid_size=8, stride=2, max_steps=3)

    with pytest.raises(ValueError, match="exceeds trajectory length"):
        run_euroc_s3r3_pose(groundtruth_path, config)


def test_euroc_s3r3_pose_rejects_invalid_grid(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=4)
    config = EuRoCS3R3PoseConfig(grid_size=0, max_steps=1)

    with pytest.raises(ValueError, match="grid_size"):
        run_euroc_s3r3_pose(groundtruth_path, config)


def _write_groundtruth(path, n_poses: int) -> None:
    lines = [
        "#timestamp [ns] p_RS_R_x [m] p_RS_R_y [m] p_RS_R_z [m] q_RS_w [] q_RS_x [] q_RS_y [] q_RS_z []"
    ]
    for idx in range(n_poses):
        timestamp = 1_403_636_580_863_555_584 + idx * 50_000_000
        yaw = 0.2 + 0.04 * idx
        qw = np.cos(0.5 * yaw)
        qz = np.sin(0.5 * yaw)
        x = 0.12 * idx
        y = 0.03 * idx * idx
        z = 0.02 * np.sin(0.4 * idx)
        lines.append(f"{timestamp:.10f} {x:.10f} {y:.10f} {z:.10f} {qw:.10f} 0.0 0.0 {qz:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
