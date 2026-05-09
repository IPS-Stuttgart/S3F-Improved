import csv
import json

import numpy as np
import pyrecest.filters.relaxed_s3f_so3 as pyrecest_relaxed_s3f_so3
from pyrecest.filters.hyperhemispherical_grid_filter import HyperhemisphericalGridFilter

from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import (
    S3R3PrototypeConfig,
    make_s3r3_filter,
    make_s3r3_orientation_filter,
    predict_s3r3_relaxed,
    run_s3r3_relaxed_prototype,
    s3r3_cell_statistics,
    s3r3_orientation_filter_from_s3f,
    s3r3_orientation_point_estimate,
    write_s3r3_relaxed_outputs,
)


def test_s3r3_relaxed_helpers_are_reexported_from_pyrecest():
    assert s3r3_cell_statistics is pyrecest_relaxed_s3f_so3.s3r3_cell_statistics
    assert predict_s3r3_relaxed is pyrecest_relaxed_s3f_so3.predict_s3r3_relaxed


def test_s3r3_cell_statistics_covariance_is_psd():
    grid = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sin(0.5), np.cos(0.5)],
        ]
    )
    stats = s3r3_cell_statistics(grid, np.array([0.4, 0.1, 0.2]), cell_sample_count=27)

    assert stats.representative_displacements.shape == (2, 3)
    assert stats.mean_displacements.shape == (2, 3)
    assert stats.covariance_inflations.shape == (2, 3, 3)
    for covariance in stats.covariance_inflations:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12


def test_s3r3_filter_uses_pyrecest_hyperhemispherical_orientation_basis():
    config = S3R3PrototypeConfig(grid_sizes=(8,), n_trials=1, n_steps=1)

    orientation_filter = make_s3r3_orientation_filter(config, 8)
    s3f_filter = make_s3r3_filter(config, 8)
    extracted_orientation_filter = s3r3_orientation_filter_from_s3f(s3f_filter)

    assert isinstance(orientation_filter, HyperhemisphericalGridFilter)
    assert isinstance(extracted_orientation_filter, HyperhemisphericalGridFilter)
    np.testing.assert_allclose(orientation_filter.filter_state.get_grid(), s3f_filter.filter_state.gd.get_grid(), atol=1e-12)
    np.testing.assert_allclose(orientation_filter.filter_state.get_grid(), extracted_orientation_filter.filter_state.get_grid(), atol=1e-12)

    expected_weights = np.asarray(orientation_filter.filter_state.grid_values, dtype=float)
    actual_weights = np.asarray(extracted_orientation_filter.filter_state.grid_values, dtype=float)
    np.testing.assert_allclose(actual_weights / np.sum(actual_weights), expected_weights / np.sum(expected_weights), atol=1e-12)

    point_estimate = s3r3_orientation_point_estimate(s3f_filter)
    assert point_estimate.shape == (4,)
    assert point_estimate[-1] >= 0.0
    np.testing.assert_allclose(np.linalg.norm(point_estimate), 1.0, atol=1e-12)


def test_s3r3_predict_preserves_grid_masses_and_inflates_covariance():
    config = S3R3PrototypeConfig(grid_sizes=(8,), n_trials=1, n_steps=1)
    filter_ = make_s3r3_filter(config, 8)
    weights_before = np.asarray(filter_.filter_state.gd.grid_values, dtype=float)
    weights_before = weights_before / np.sum(weights_before)

    stats = predict_s3r3_relaxed(
        filter_,
        np.asarray(config.body_increment),
        variant="r1_r2",
        process_noise_cov=0.01 * np.eye(3),
        cell_sample_count=config.cell_sample_count,
    )

    weights_after = np.asarray(filter_.filter_state.gd.grid_values, dtype=float)
    weights_after = weights_after / np.sum(weights_after)
    np.testing.assert_allclose(weights_after, weights_before, atol=1e-12)
    assert any(np.linalg.norm(covariance) > 0.0 for covariance in stats.covariance_inflations)


def test_s3r3_prototype_smoke_outputs_metrics(tmp_path):
    config = S3R3PrototypeConfig(grid_sizes=(8,), n_trials=1, n_steps=2)

    rows = run_s3r3_relaxed_prototype(config)

    assert len(rows) == 3
    assert {row["variant"] for row in rows} == {"baseline", "r1", "r1_r2"}
    assert all(int(row["grid_size"]) == 8 for row in rows)
    assert all(np.isfinite(float(row["position_rmse"])) for row in rows)
    assert all(float(row["runtime_ms_per_step"]) > 0.0 for row in rows)

    outputs = write_s3r3_relaxed_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "note", "metadata"):
        assert outputs[output_name].is_file()

    with outputs["metrics"].open(newline="", encoding="utf-8") as metrics_file:
        assert len(list(csv.DictReader(metrics_file))) == 3

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_relaxed_prototype"
    assert metadata["config"]["grid_sizes"] == [8]
    assert metadata["metrics_rows"] == 3
    assert metadata["cell_model"] == "pyrecest.relaxed_s3f_so3.local_tangent_samples"
