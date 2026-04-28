"""EuRoC ground-truth adapter for the S1 x R2 relaxed-S3F smoke test."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyrecest.filters.relaxed_s3f_circular import (
    SUPPORTED_RELAXED_S3F_VARIANTS,
    circular_error,
    rotation_matrix,
)
from scipy.special import i0

from .s3f_common import (
    linear_position_error_stats,
    make_linear_likelihood,
    make_s3f_filter,
    orientation_mode_and_mean,
    predict_update_linear_position,
)


EUROC_PLANAR_METRIC_FIELDNAMES = [
    "variant",
    "grid_size",
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_mean_error_rad",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "n_steps",
    "duration_s",
    "path_length_m",
    "mean_body_increment_m",
]


@dataclass(frozen=True)
class EuRoCPlanarTrajectory:
    """Planar projection of a EuRoC ground-truth trajectory."""

    timestamps_s: np.ndarray
    positions_xy: np.ndarray
    yaw: np.ndarray


@dataclass(frozen=True)
class EuRoCPlanarConfig:
    """Configuration for a short EuRoC trajectory smoke run."""

    grid_size: int = 16
    variants: tuple[str, ...] = SUPPORTED_RELAXED_S3F_VARIANTS
    start_index: int = 0
    stride: int = 20
    max_steps: int = 60
    seed: int = 13
    measurement_noise_std: float = 0.05
    process_noise_std: float = 0.01
    initial_position_std: float = 0.08
    orientation_prior_kappa: float = 6.0


def load_euroc_planar_groundtruth(path: Path) -> EuRoCPlanarTrajectory:
    """Load a EuRoC/TUM-style ground-truth pose file as x, y, yaw."""

    table = _load_numeric_table(path)
    if table.shape[1] < 8:
        raise ValueError(f"Expected at least 8 columns in {path}.")

    timestamps = table[:, 0]
    if float(np.nanmax(timestamps)) > 1e12:
        timestamps_s = (timestamps - timestamps[0]) / 1e9
    else:
        timestamps_s = timestamps - timestamps[0]

    positions_xy = table[:, 1:3].astype(float)
    quaternions_wxyz = table[:, 4:8].astype(float)
    yaw = _yaw_from_quaternions_wxyz(quaternions_wxyz)

    if not np.all(np.isfinite(positions_xy)):
        raise ValueError(f"Non-finite positions in {path}.")
    if not np.all(np.isfinite(yaw)):
        raise ValueError(f"Non-finite yaw values in {path}.")

    return EuRoCPlanarTrajectory(
        timestamps_s=timestamps_s,
        positions_xy=positions_xy,
        yaw=yaw,
    )


def run_euroc_planar_relaxed_s3f(
    groundtruth_path: Path,
    config: EuRoCPlanarConfig = EuRoCPlanarConfig(),
) -> list[dict[str, float | int | str]]:
    """Run relaxed S3F variants on a planar projection of one EuRoC trajectory."""

    trajectory = _slice_trajectory(load_euroc_planar_groundtruth(groundtruth_path), config)
    if config.grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    rows = []
    for variant in config.variants:
        if variant not in SUPPORTED_RELAXED_S3F_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}.")
        rows.append(_run_euroc_variant(trajectory, config, variant))
    return rows


def write_euroc_planar_outputs(
    groundtruth_path: Path,
    output_dir: Path,
    config: EuRoCPlanarConfig = EuRoCPlanarConfig(),
) -> dict[str, Path]:
    """Run the EuRoC planar smoke test and write metrics plus a short note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_euroc_planar_relaxed_s3f(groundtruth_path, config)

    metrics_path = output_dir / "euroc_planar_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EUROC_PLANAR_METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    note_path = output_dir / "euroc_planar_note.md"
    _write_note(note_path, groundtruth_path, rows, config)
    return {"metrics": metrics_path, "note": note_path}


def _run_euroc_variant(
    trajectory: EuRoCPlanarTrajectory,
    config: EuRoCPlanarConfig,
    variant: str,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(config.seed)
    filter_ = _make_initial_filter(
        initial_position=trajectory.positions_xy[0],
        initial_yaw=float(trajectory.yaw[0]),
        config=config,
    )
    measurement_cov = np.eye(2) * config.measurement_noise_std**2
    process_noise_cov = np.eye(2) * config.process_noise_std**2

    position_sq_error = 0.0
    orientation_mode_error = 0.0
    orientation_mean_error = 0.0
    nees_sum = 0.0
    coverage_hits = 0
    runtime = 0.0
    body_increment_norms = []

    for step in range(trajectory.positions_xy.shape[0] - 1):
        current_position = trajectory.positions_xy[step]
        next_position = trajectory.positions_xy[step + 1]
        current_yaw = float(trajectory.yaw[step])
        next_yaw = float(trajectory.yaw[step + 1])
        body_increment = rotation_matrix(-current_yaw) @ (next_position - current_position)
        body_increment_norms.append(float(np.linalg.norm(body_increment)))

        measurement = next_position + rng.normal(0.0, config.measurement_noise_std, size=2)
        likelihood = make_linear_likelihood(measurement, measurement_cov)
        runtime += predict_update_linear_position(
            filter_,
            body_increment,
            variant,
            process_noise_cov,
            likelihood,
        )

        error, nees = linear_position_error_stats(filter_, next_position)
        position_sq_error += float(error @ error)
        nees_sum += nees
        coverage_hits += int(nees <= 5.991464547107979)

        mode_yaw, mean_yaw = orientation_mode_and_mean(filter_)
        orientation_mode_error += circular_error(mode_yaw, next_yaw)
        orientation_mean_error += circular_error(mean_yaw, next_yaw)

    n_steps = trajectory.positions_xy.shape[0] - 1
    body_increment_norms_array = np.asarray(body_increment_norms, dtype=float)
    return {
        "variant": variant,
        "grid_size": config.grid_size,
        "position_rmse": float(np.sqrt(position_sq_error / n_steps)),
        "orientation_mode_error_rad": orientation_mode_error / n_steps,
        "orientation_mean_error_rad": orientation_mean_error / n_steps,
        "mean_nees": nees_sum / n_steps,
        "coverage_95": coverage_hits / n_steps,
        "runtime_ms_per_step": 1000.0 * runtime / n_steps,
        "n_steps": n_steps,
        "duration_s": float(trajectory.timestamps_s[-1] - trajectory.timestamps_s[0]),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(trajectory.positions_xy, axis=0), axis=1))),
        "mean_body_increment_m": float(np.mean(body_increment_norms_array)),
    }


def _make_initial_filter(
    initial_position: np.ndarray,
    initial_yaw: float,
    config: EuRoCPlanarConfig,
):
    grid = np.linspace(0.0, 2.0 * np.pi, config.grid_size, endpoint=False)
    prior_values = np.exp(config.orientation_prior_kappa * np.cos(grid - initial_yaw)) / (
        2.0 * np.pi * i0(config.orientation_prior_kappa)
    )
    prior_values = prior_values / (np.sum(prior_values) * 2.0 * np.pi / config.grid_size)
    initial_cov = np.eye(2) * config.initial_position_std**2
    return make_s3f_filter(grid, prior_values, initial_position, initial_cov)


def _slice_trajectory(
    trajectory: EuRoCPlanarTrajectory,
    config: EuRoCPlanarConfig,
) -> EuRoCPlanarTrajectory:
    if config.start_index < 0:
        raise ValueError("start_index must be nonnegative.")
    if config.stride <= 0:
        raise ValueError("stride must be positive.")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    stop = config.start_index + config.stride * (config.max_steps + 1)
    indices = np.arange(config.start_index, stop, config.stride, dtype=int)
    if indices[-1] >= trajectory.positions_xy.shape[0]:
        raise ValueError(
            "Requested EuRoC slice exceeds trajectory length: "
            f"last index {indices[-1]}, length {trajectory.positions_xy.shape[0]}."
        )
    return EuRoCPlanarTrajectory(
        timestamps_s=trajectory.timestamps_s[indices],
        positions_xy=trajectory.positions_xy[indices],
        yaw=trajectory.yaw[indices],
    )


def _load_numeric_table(path: Path) -> np.ndarray:
    first_data_line = ""
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_data_line = stripped
                break
    if not first_data_line:
        raise ValueError(f"No numeric rows found in {path}.")

    delimiter = "," if "," in first_data_line else None
    table = np.loadtxt(path, comments="#", delimiter=delimiter, dtype=float)
    table = np.atleast_2d(table)
    if table.shape[0] < 2:
        raise ValueError(f"Expected at least two poses in {path}.")
    return table


def _yaw_from_quaternions_wxyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions_wxyz, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Quaternions must have positive norm.")
    normalized = quaternions_wxyz / norms[:, None]
    w = normalized[:, 0]
    x = normalized[:, 1]
    y = normalized[:, 2]
    z = normalized[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.mod(yaw, 2.0 * np.pi)


def _write_note(
    path: Path,
    groundtruth_path: Path,
    rows: list[dict[str, float | int | str]],
    config: EuRoCPlanarConfig,
) -> None:
    best_rmse = min(rows, key=lambda row: float(row["position_rmse"]))
    content = f"""# EuRoC Planar Relaxed S3F Smoke Note

## What Was Run

This run projects EuRoC ground-truth poses to `S1 x R2` using planar position
and yaw. It uses ground-truth relative body-frame displacements as controls and
noisy planar position measurements. This is a trajectory-geometry smoke test,
not a visual-inertial frontend.

- ground truth: `{groundtruth_path}`
- grid size: {config.grid_size}
- steps: {config.max_steps}
- stride: {config.stride}

## Result

Lowest translation RMSE in this run:
`{best_rmse["variant"]}` with RMSE `{float(best_rmse["position_rmse"]):.4f}`.
"""
    path.write_text(content, encoding="utf-8")
