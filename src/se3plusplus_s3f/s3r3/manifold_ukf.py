"""SO3 x R3 manifold UKF baseline for controlled pose tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyrecest.filters import UKFOnManifolds

from .relaxed_s3f_prototype import (
    _canonical_quaternions,
    _exp_map_identity,
    _quaternion_multiply,
    _rotate_vectors,
)


@dataclass(frozen=True)
class SO3R3PoseState:
    """Pose state with scalar-last quaternion orientation and R3 position."""

    orientation: np.ndarray
    position: np.ndarray


@dataclass(frozen=True)
class SO3R3PoseControl:
    """Known per-step body-frame translation and quaternion increment."""

    body_increment: np.ndarray
    orientation_increment: np.ndarray


@dataclass(frozen=True)
class SO3R3ManifoldUKFConfig:
    """Configuration for the SO3 x R3 UKF baseline."""

    measurement_noise_std: float = 0.05
    process_noise_std: float = 0.01
    initial_position_std: float = 0.08
    initial_orientation_std: float = 0.40
    orientation_process_std: float = 0.10
    alpha: float = 0.5


def make_so3r3_manifold_ukf(
    initial_position: np.ndarray,
    initial_orientation: np.ndarray,
    config: SO3R3ManifoldUKFConfig = SO3R3ManifoldUKFConfig(),
) -> UKFOnManifolds:
    """Create a PyRecEst UKF-M baseline for controlled SO3 x R3 pose tracking."""

    validate_so3r3_manifold_ukf_config(config)
    state0 = SO3R3PoseState(
        orientation=_canonical_quaternions(initial_orientation).reshape(4),
        position=np.asarray(initial_position, dtype=float).reshape(3),
    )
    initial_covariance = np.diag(
        [
            config.initial_orientation_std**2,
            config.initial_orientation_std**2,
            config.initial_orientation_std**2,
            config.initial_position_std**2,
            config.initial_position_std**2,
            config.initial_position_std**2,
        ]
    )
    process_noise_covariance = np.diag(
        [
            config.orientation_process_std**2,
            config.orientation_process_std**2,
            config.orientation_process_std**2,
            config.process_noise_std**2,
            config.process_noise_std**2,
            config.process_noise_std**2,
        ]
    )
    measurement_covariance = np.eye(3) * config.measurement_noise_std**2
    return UKFOnManifolds(
        f=so3r3_pose_dynamics,
        h=so3r3_position_measurement,
        phi=so3r3_retract,
        phi_inv=so3r3_inverse_retract,
        Q=process_noise_covariance,
        R=measurement_covariance,
        alpha=config.alpha,
        state0=state0,
        P0=initial_covariance,
    )


def validate_so3r3_manifold_ukf_config(config: SO3R3ManifoldUKFConfig) -> None:
    """Validate the UKF baseline noise and sigma-point parameters."""

    for name, value in (
        ("measurement_noise_std", config.measurement_noise_std),
        ("process_noise_std", config.process_noise_std),
        ("initial_position_std", config.initial_position_std),
        ("initial_orientation_std", config.initial_orientation_std),
        ("orientation_process_std", config.orientation_process_std),
        ("alpha", config.alpha),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")


def predict_so3r3_manifold_ukf(
    filter_: UKFOnManifolds,
    body_increment: np.ndarray,
    orientation_increment: np.ndarray,
) -> None:
    """Predict the SO3 x R3 UKF one controlled pose step."""

    control = SO3R3PoseControl(
        body_increment=np.asarray(body_increment, dtype=float).reshape(3),
        orientation_increment=_canonical_quaternions(orientation_increment).reshape(4),
    )
    filter_.predict(omega=control, dt=1.0)


def update_so3r3_manifold_ukf(filter_: UKFOnManifolds, measurement: np.ndarray) -> None:
    """Update the SO3 x R3 UKF with an R3 position measurement."""

    filter_.update(np.asarray(measurement, dtype=float).reshape(3))


def so3r3_pose_dynamics(
    state: SO3R3PoseState,
    control: SO3R3PoseControl,
    noise: np.ndarray,
    _dt: float,
) -> SO3R3PoseState:
    """Controlled pose dynamics used by the UKF baseline."""

    noise = np.asarray(noise, dtype=float).reshape(6)
    displacement = _rotate_vectors(state.orientation, control.body_increment)[0]
    next_orientation = _quaternion_multiply(
        _quaternion_multiply(state.orientation, control.orientation_increment),
        _exp_map_identity(noise[:3])[0],
    )
    next_position = np.asarray(state.position, dtype=float).reshape(3) + displacement + noise[3:]
    return SO3R3PoseState(
        orientation=_canonical_quaternions(next_orientation).reshape(4),
        position=next_position,
    )


def so3r3_position_measurement(state: SO3R3PoseState) -> np.ndarray:
    """Return the R3 position measurement predicted from a pose state."""

    return np.asarray(state.position, dtype=float).reshape(3)


def so3r3_retract(state: SO3R3PoseState, tangent: np.ndarray) -> SO3R3PoseState:
    """Apply a local SO3 x R3 tangent perturbation to a pose state."""

    tangent = np.asarray(tangent, dtype=float).reshape(6)
    return SO3R3PoseState(
        orientation=_quaternion_multiply(state.orientation, _exp_map_identity(tangent[:3])[0]),
        position=np.asarray(state.position, dtype=float).reshape(3) + tangent[3:],
    )


def so3r3_inverse_retract(reference: SO3R3PoseState, state: SO3R3PoseState) -> np.ndarray:
    """Return local SO3 x R3 tangent coordinates of ``state`` around ``reference``."""

    orientation_delta = _quaternion_multiply(_quaternion_inverse(reference.orientation), state.orientation)
    return np.concatenate(
        (
            _log_map_identity(orientation_delta),
            np.asarray(state.position, dtype=float).reshape(3) - np.asarray(reference.position, dtype=float).reshape(3),
        )
    )


def so3r3_manifold_ukf_position_error_stats(filter_: UKFOnManifolds, true_position: np.ndarray) -> tuple[np.ndarray, float]:
    """Return R3 position error and marginal position NEES for a UKF state."""

    state, covariance = filter_.filter_state
    error = np.asarray(state.position, dtype=float).reshape(3) - np.asarray(true_position, dtype=float).reshape(3)
    position_covariance = np.asarray(covariance, dtype=float)[3:6, 3:6] + 1e-10 * np.eye(3)
    return error, float(error @ np.linalg.solve(position_covariance, error))


def so3r3_manifold_ukf_orientation(filter_: UKFOnManifolds) -> np.ndarray:
    """Return the current UKF orientation point estimate."""

    state, _covariance = filter_.filter_state
    return _canonical_quaternions(state.orientation).reshape(4)


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    xyzw = _canonical_quaternions(quaternion).reshape(4)
    return _canonical_quaternions(np.array([-xyzw[0], -xyzw[1], -xyzw[2], xyzw[3]], dtype=float))


def _log_map_identity(quaternion: np.ndarray) -> np.ndarray:
    xyzw = _canonical_quaternions(quaternion).reshape(4)
    vector = xyzw[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, float(xyzw[3]))
    return angle * vector / vector_norm
