import csv
import json

import numpy as np
from pyrecest.filters import so3_right_multiplication_grid_transition

from se3plusplus_s3f.s3r3.dynamic_pose import (
    S3R3DynamicPoseConfig,
    generate_s3r3_dynamic_pose_trials,
    predict_s3r3_dynamic_pose,
    run_s3r3_dynamic_pose_benchmark,
    s3r3_orientation_transition_density,
    write_s3r3_dynamic_pose_outputs,
)
from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import (
    S3R3PrototypeConfig,
    make_s3r3_filter,
)


def test_s3r3_orientation_transition_density_is_column_normalized():
    config = S3R3PrototypeConfig(grid_sizes=(8,), n_trials=1, n_steps=1)
    filter_ = make_s3r3_filter(config, 8)
    grid = np.asarray(filter_.filter_state.gd.get_grid(), dtype=float)

    transition = s3r3_orientation_transition_density(grid, (0.0, 0.18, 0.06), 24.0)
    expected_transition = so3_right_multiplication_grid_transition(
        grid,
        (0.0, 0.18, 0.06),
        24.0,
    )

    s3_hemisphere_surface = np.pi**2
    column_integrals = np.mean(transition.grid_values, axis=0) * s3_hemisphere_surface
    np.testing.assert_allclose(column_integrals, np.ones(grid.shape[0]), atol=1e-12)
    assert transition.grid_values.shape == (8, 8)
    np.testing.assert_allclose(transition.grid_values, expected_transition.grid_values, atol=1e-12)


def test_s3r3_dynamic_predict_preserves_mass_and_changes_orientation_weights():
    config = S3R3PrototypeConfig(grid_sizes=(8,), n_trials=1, n_steps=1)
    filter_ = make_s3r3_filter(config, 8)
    weights_before = np.asarray(filter_.filter_state.gd.grid_values, dtype=float)
    weights_before = weights_before / np.sum(weights_before)

    stats = predict_s3r3_dynamic_pose(
        filter_,
        np.asarray(config.body_increment),
        (0.0, 0.18, 0.06),
        variant="r1_r2",
        process_noise_cov=0.01 * np.eye(3),
        cell_sample_count=8,
        orientation_transition_kappa=24.0,
    )

    weights_after = np.asarray(filter_.filter_state.gd.grid_values, dtype=float)
    weights_after = weights_after / np.sum(weights_after)
    np.testing.assert_allclose(np.sum(weights_after), 1.0, atol=1e-12)
    assert np.linalg.norm(weights_after - weights_before) > 0.0
    assert stats.covariance_inflations.shape == (8, 3, 3)


def test_s3r3_dynamic_pose_trials_have_orientation_sequence():
    config = S3R3DynamicPoseConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8,), n_trials=2, n_steps=3)
    )

    trials = generate_s3r3_dynamic_pose_trials(config)

    assert len(trials) == 2
    assert trials[0]["orientations"].shape == (4, 4)
    assert trials[0]["positions"].shape == (4, 3)
    assert trials[0]["measurements"].shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(trials[0]["orientations"], axis=1), np.ones(4), atol=1e-12)


def test_s3r3_dynamic_pose_smoke_outputs_metrics_and_claims(tmp_path):
    config = S3R3DynamicPoseConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
            cell_sample_count=8,
        )
    )

    result = run_s3r3_dynamic_pose_benchmark(config)

    assert len(result.metrics) == 3
    assert len(result.claims) == 2
    assert {row["variant"] for row in result.metrics} == {"baseline", "r1", "r1_r2"}
    assert all(np.isfinite(float(row["position_rmse"])) for row in result.metrics)
    assert all("supports_overall_claim" in row for row in result.claims)

    outputs = write_s3r3_dynamic_pose_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "claims", "note", "metadata"):
        assert outputs[output_name].is_file()

    with outputs["metrics"].open(newline="", encoding="utf-8") as metrics_file:
        assert len(list(csv.DictReader(metrics_file))) == 3
    with outputs["claims"].open(newline="", encoding="utf-8") as claims_file:
        assert len(list(csv.DictReader(claims_file))) == 2

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_dynamic_pose"
    assert metadata["config"]["prototype"]["grid_sizes"] == [8]
