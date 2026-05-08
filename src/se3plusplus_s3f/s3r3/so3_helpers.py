"""NumPy-compatible wrappers around PyRecEst SO(3) helpers."""

from __future__ import annotations

import numpy as np
from pyrecest.distributions import so3_helpers as _so3_helpers


def canonical_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    return np.asarray(_so3_helpers.canonicalize_quaternions(quaternions), dtype=float).reshape(quaternions.shape)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    output_shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1]) + (4,)
    return np.asarray(_so3_helpers.quaternion_multiply(left, right), dtype=float).reshape(output_shape)


def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    return np.asarray(_so3_helpers.quaternion_conjugate(quaternion), dtype=float).reshape(quaternion.shape)


def exp_map_identity(tangent_vectors: np.ndarray) -> np.ndarray:
    return np.asarray(_so3_helpers.exp_map_identity(np.asarray(tangent_vectors, dtype=float)), dtype=float)


def log_map_identity(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    result = np.asarray(_so3_helpers.log_map_identity(quaternions), dtype=float)
    return result.reshape((3,) if quaternions.ndim == 1 else quaternions.shape[:-1] + (3,))


def quaternion_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray(_so3_helpers.quaternions_to_rotation_matrices(np.asarray(quaternions, dtype=float)), dtype=float)


def rotate_vectors(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.asarray(_so3_helpers.rotate_vectors(np.asarray(quaternions, dtype=float), np.asarray(vector, dtype=float)), dtype=float)


def quaternion_distance_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = canonical_quaternions(quaternions)
    inner = np.clip(np.abs(quaternions @ quaternions.T), 0.0, 1.0)
    return 2.0 * np.arccos(inner)


def geodesic_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.asarray(_so3_helpers.geodesic_distance(left, right), dtype=float).reshape(-1)[0])
