import csv

import numpy as np
import numpy.testing as npt

from se3plusplus_s3f.s1r2.euroc_planar import (
    EuRoCPlanarConfig,
    load_euroc_planar_groundtruth,
    run_euroc_planar_relaxed_s3f,
    write_euroc_planar_outputs,
)


def test_load_euroc_planar_groundtruth_from_whitespace_file(tmp_path):
    path = tmp_path / "groundtruth.txt"
    _write_groundtruth(path, n_poses=6)

    trajectory = load_euroc_planar_groundtruth(path)

    assert trajectory.positions_xy.shape == (6, 2)
    npt.assert_allclose(trajectory.timestamps_s[:3], [0.0, 0.05, 0.10], atol=1e-6)
    npt.assert_allclose(trajectory.yaw[:3], [0.2, 0.22, 0.24])


def test_euroc_planar_smoke_outputs_metrics(tmp_path):
    groundtruth_path = tmp_path / "groundtruth.txt"
    _write_groundtruth(groundtruth_path, n_poses=12)
    config = EuRoCPlanarConfig(grid_size=8, stride=2, max_steps=4)

    rows = run_euroc_planar_relaxed_s3f(groundtruth_path, config)
    assert {row["variant"] for row in rows} == {"baseline", "r1", "r1_r2"}
    assert all(int(row["n_steps"]) == 4 for row in rows)
    assert all(float(row["path_length_m"]) > 0.0 for row in rows)
    assert all(np.isfinite(float(row["position_rmse"])) for row in rows)

    outputs = write_euroc_planar_outputs(groundtruth_path, tmp_path / "out", config)
    assert outputs["metrics"].exists()
    assert outputs["note"].exists()
    with outputs["metrics"].open(newline="", encoding="utf-8") as file:
        written_rows = list(csv.DictReader(file))
    assert len(written_rows) == 3


def _write_groundtruth(path, n_poses: int) -> None:
    lines = [
        "#timestamp [ns] p_RS_R_x [m] p_RS_R_y [m] p_RS_R_z [m] q_RS_w [] q_RS_x [] q_RS_y [] q_RS_z []"
    ]
    for idx in range(n_poses):
        timestamp = 1_403_636_580_863_555_584 + idx * 50_000_000
        yaw = 0.2 + 0.02 * idx
        qw = np.cos(0.5 * yaw)
        qz = np.sin(0.5 * yaw)
        x = 0.12 * idx
        y = 0.03 * idx * idx
        lines.append(f"{timestamp:.10f} {x:.10f} {y:.10f} 0.0 {qw:.10f} 0.0 0.0 {qz:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
