import numpy as np
import numpy.testing as npt
import pytest

from se3plusplus_s3f.s3r3.manifold_ukf import (
    SO3R3ManifoldUKFConfig,
    SO3R3PoseState,
    make_so3r3_manifold_ukf,
    predict_so3r3_manifold_ukf,
    so3r3_inverse_retract,
    so3r3_manifold_ukf_orientation,
    so3r3_manifold_ukf_position_error_stats,
    so3r3_retract,
    update_so3r3_manifold_ukf,
)


def test_so3r3_retraction_round_trip_is_local_inverse():
    state = SO3R3PoseState(
        orientation=np.array([0.0, 0.0, np.sin(0.1), np.cos(0.1)]),
        position=np.array([1.0, -0.5, 0.2]),
    )
    tangent = np.array([0.03, -0.02, 0.04, 0.1, -0.2, 0.05])

    moved = so3r3_retract(state, tangent)
    recovered = so3r3_inverse_retract(state, moved)

    npt.assert_allclose(recovered, tangent, atol=1e-12)
    npt.assert_allclose(np.linalg.norm(moved.orientation), 1.0, atol=1e-12)


def test_so3r3_manifold_ukf_predict_update_smoke():
    config = SO3R3ManifoldUKFConfig(
        measurement_noise_std=0.05,
        process_noise_std=0.01,
        initial_position_std=0.08,
        initial_orientation_std=0.30,
        orientation_process_std=0.05,
        alpha=0.5,
    )
    filter_ = make_so3r3_manifold_ukf(
        initial_position=np.zeros(3),
        initial_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        config=config,
    )

    predict_so3r3_manifold_ukf(
        filter_,
        body_increment=np.array([0.2, 0.0, 0.0]),
        orientation_increment=np.array([0.0, 0.0, np.sin(0.05), np.cos(0.05)]),
    )
    update_so3r3_manifold_ukf(filter_, np.array([0.21, -0.01, 0.0]))

    state, covariance = filter_.filter_state
    error, nees = so3r3_manifold_ukf_position_error_stats(filter_, np.array([0.2, 0.0, 0.0]))

    assert state.position.shape == (3,)
    assert error.shape == (3,)
    assert np.isfinite(nees)
    npt.assert_allclose(np.linalg.norm(so3r3_manifold_ukf_orientation(filter_)), 1.0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(np.asarray(covariance, dtype=float)) > -1e-9)


def test_so3r3_manifold_ukf_rejects_invalid_config():
    with pytest.raises(ValueError, match="alpha"):
        make_so3r3_manifold_ukf(
            initial_position=np.zeros(3),
            initial_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            config=SO3R3ManifoldUKFConfig(alpha=0.0),
        )
