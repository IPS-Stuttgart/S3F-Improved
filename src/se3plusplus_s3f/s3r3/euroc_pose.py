"""EuRoC 3D ground-truth adapter for dynamic S3+ x R3 relaxed S3F."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter

from ..s1r2.plotting import format_plot_list, save_figure
from .dynamic_pose import predict_s3r3_dynamic_pose
from .manifold_ukf import (
    SO3R3ManifoldUKFConfig,
    make_so3r3_manifold_ukf,
    predict_so3r3_manifold_ukf,
    so3r3_manifold_ukf_orientation,
    so3r3_manifold_ukf_position_error_stats,
    update_so3r3_manifold_ukf,
)
from .relaxed_s3f_prototype import (
    SUPPORTED_S3R3_VARIANTS,
    VARIANT_LABELS,
    S3R3PrototypeConfig,
    _canonical_quaternions,
    _quaternion_multiply,
    _rotate_vectors,
    make_s3r3_orientation_filter,
    s3r3_linear_position_error_stats,
    s3r3_orientation_distance,
    s3r3_orientation_mode,
    s3r3_orientation_point_estimate,
)


EUROC_S3R3_VARIANT_LABELS = {
    **VARIANT_LABELS,
    "manifold_ukf": "Manifold UKF",
}

EUROC_S3R3_METRIC_FIELDNAMES = [
    "variant",
    "grid_size",
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_point_error_rad",
    "mean_nees",
    "coverage_95",
    "runtime_ms_per_step",
    "n_steps",
    "duration_s",
    "path_length_m",
    "mean_body_increment_m",
    "mean_orientation_increment_rad",
    "orientation_transition_kappa",
    "cell_sample_count",
]

EUROC_S3R3_CLAIM_FIELDNAMES = [
    "comparison",
    "candidate_variant",
    "comparator_variant",
    "candidate_position_rmse",
    "comparator_position_rmse",
    "position_rmse_ratio",
    "position_rmse_gain_pct",
    "candidate_mean_nees",
    "comparator_mean_nees",
    "mean_nees_ratio",
    "candidate_coverage_95",
    "comparator_coverage_95",
    "coverage_delta",
    "candidate_orientation_point_error_rad",
    "comparator_orientation_point_error_rad",
    "orientation_point_error_delta_rad",
    "candidate_runtime_ms_per_step",
    "comparator_runtime_ms_per_step",
    "runtime_ratio",
    "supports_accuracy_claim",
    "supports_consistency_claim",
    "supports_orientation_claim",
    "supports_runtime_claim",
    "supports_overall_claim",
]


@dataclass(frozen=True)
class EuRoCPoseTrajectory:
    """EuRoC ground-truth pose trajectory in internal quaternion order x, y, z, w."""

    timestamps_s: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray


@dataclass(frozen=True)
class EuRoCS3R3PoseConfig:
    """Configuration for a short EuRoC 3D pose smoke run."""

    grid_size: int = 16
    variants: tuple[str, ...] = SUPPORTED_S3R3_VARIANTS
    start_index: int = 0
    stride: int = 20
    max_steps: int = 50
    seed: int = 19
    measurement_noise_std: float = 0.05
    process_noise_std: float = 0.01
    initial_position_std: float = 0.08
    orientation_prior_kappa: float = 12.0
    orientation_transition_kappa: float = 48.0
    cell_sample_count: int = 27
    prior_yaw_offsets_rad: tuple[float, ...] = (0.0,)
    prior_weights: tuple[float, ...] = (1.0,)
    include_manifold_ukf: bool = True
    ukf_alpha: float = 0.5
    ukf_orientation_process_std: float = 0.10


@dataclass(frozen=True)
class EuRoCS3R3PoseResult:
    """Container for EuRoC S3R3 pose metrics and claim rows."""

    metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]


def euroc_s3r3_pose_config_to_dict(config: EuRoCS3R3PoseConfig) -> dict[str, Any]:
    """Return a JSON-serializable EuRoC S3R3 pose config."""

    return json.loads(json.dumps(asdict(config)))


def load_euroc_pose_groundtruth(path: Path) -> EuRoCPoseTrajectory:
    """Load a EuRoC/TUM-style ground-truth pose file as 3D positions and quaternions."""

    table = _load_numeric_table(path)
    if table.shape[1] < 8:
        raise ValueError(f"Expected at least 8 columns in {path}.")

    timestamps = table[:, 0]
    if float(np.nanmax(timestamps)) > 1e12:
        timestamps_s = (timestamps - timestamps[0]) / 1e9
    else:
        timestamps_s = timestamps - timestamps[0]

    positions = table[:, 1:4].astype(float)
    quaternions_wxyz = table[:, 4:8].astype(float)
    quaternions_xyzw = _canonical_quaternions(
        np.column_stack(
            (
                quaternions_wxyz[:, 1],
                quaternions_wxyz[:, 2],
                quaternions_wxyz[:, 3],
                quaternions_wxyz[:, 0],
            )
        )
    )

    if not np.all(np.isfinite(positions)):
        raise ValueError(f"Non-finite positions in {path}.")
    if not np.all(np.isfinite(quaternions_xyzw)):
        raise ValueError(f"Non-finite quaternions in {path}.")

    return EuRoCPoseTrajectory(
        timestamps_s=timestamps_s,
        positions=positions,
        quaternions=quaternions_xyzw,
    )


def run_euroc_s3r3_pose(
    groundtruth_path: Path,
    config: EuRoCS3R3PoseConfig = EuRoCS3R3PoseConfig(),
) -> EuRoCS3R3PoseResult:
    """Run dynamic S3R3 variants on one EuRoC ground-truth pose segment."""

    _validate_config(config)
    trajectory = _slice_trajectory(load_euroc_pose_groundtruth(groundtruth_path), config)
    controls = _trajectory_controls(trajectory)
    rng = np.random.default_rng(config.seed)
    measurements = trajectory.positions[1:] + rng.normal(0.0, config.measurement_noise_std, size=(controls["body_increments"].shape[0], 3))
    rows = [
        _run_variant(trajectory, controls, measurements, config, variant)
        for variant in config.variants
    ]
    if config.include_manifold_ukf:
        rows.append(_run_manifold_ukf_variant(trajectory, controls, measurements, config))
    return EuRoCS3R3PoseResult(metrics=rows, claims=_build_claim_rows(rows))


def write_euroc_s3r3_pose_outputs(
    groundtruth_path: Path,
    output_dir: Path,
    config: EuRoCS3R3PoseConfig = EuRoCS3R3PoseConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the EuRoC S3R3 pose benchmark and write metrics, claims, metadata, note, and optional plots."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_euroc_s3r3_pose(groundtruth_path, config)

    metrics_path = output_dir / "euroc_s3r3_pose_metrics.csv"
    _write_csv(metrics_path, result.metrics, EUROC_S3R3_METRIC_FIELDNAMES)

    claims_path = output_dir / "euroc_s3r3_pose_claims.csv"
    _write_csv(claims_path, result.claims, EUROC_S3R3_CLAIM_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path}
    plot_paths = _write_plots(output_dir, result.metrics) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "euroc_s3r3_pose_note.md"
    _write_note(note_path, groundtruth_path, result, metrics_path, claims_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, groundtruth_path, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: EuRoCS3R3PoseConfig) -> None:
    if config.grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    unknown_variants = [variant for variant in config.variants if variant not in SUPPORTED_S3R3_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown variant {unknown_variants[0]!r}.")
    if config.start_index < 0:
        raise ValueError("start_index must be nonnegative.")
    if config.stride <= 0:
        raise ValueError("stride must be positive.")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if config.cell_sample_count <= 0:
        raise ValueError("cell_sample_count must be positive.")
    if config.orientation_transition_kappa <= 0.0:
        raise ValueError("orientation_transition_kappa must be positive.")
    if config.ukf_alpha <= 0.0:
        raise ValueError("ukf_alpha must be positive.")
    if len(config.prior_yaw_offsets_rad) == 0:
        raise ValueError("prior_yaw_offsets_rad must not be empty.")
    if len(config.prior_yaw_offsets_rad) != len(config.prior_weights):
        raise ValueError("prior_yaw_offsets_rad and prior_weights must have the same length.")
    prior_weights = np.asarray(config.prior_weights, dtype=float)
    if np.any(prior_weights < 0.0) or float(np.sum(prior_weights)) <= 0.0:
        raise ValueError("prior_weights must be nonnegative with positive sum.")
    for name, value in (
        ("measurement_noise_std", config.measurement_noise_std),
        ("process_noise_std", config.process_noise_std),
        ("initial_position_std", config.initial_position_std),
        ("orientation_prior_kappa", config.orientation_prior_kappa),
        ("ukf_orientation_process_std", config.ukf_orientation_process_std),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")


def _slice_trajectory(
    trajectory: EuRoCPoseTrajectory,
    config: EuRoCS3R3PoseConfig,
) -> EuRoCPoseTrajectory:
    stop = config.start_index + config.stride * (config.max_steps + 1)
    indices = np.arange(config.start_index, stop, config.stride, dtype=int)
    if indices[-1] >= trajectory.positions.shape[0]:
        raise ValueError(
            "Requested EuRoC slice exceeds trajectory length: "
            f"last index {indices[-1]}, length {trajectory.positions.shape[0]}."
        )
    return EuRoCPoseTrajectory(
        timestamps_s=trajectory.timestamps_s[indices],
        positions=trajectory.positions[indices],
        quaternions=trajectory.quaternions[indices],
    )


def _trajectory_controls(trajectory: EuRoCPoseTrajectory) -> dict[str, np.ndarray]:
    body_increments = []
    orientation_increments = []
    for current_position, next_position, current_orientation, next_orientation in zip(
        trajectory.positions[:-1],
        trajectory.positions[1:],
        trajectory.quaternions[:-1],
        trajectory.quaternions[1:],
        strict=True,
    ):
        current_inverse = _quaternion_inverse(current_orientation)
        body_increment = _rotate_vectors(current_inverse, next_position - current_position)[0]
        orientation_increment = _quaternion_multiply(current_inverse, next_orientation)
        body_increments.append(body_increment)
        orientation_increments.append(orientation_increment)
    return {
        "body_increments": np.asarray(body_increments),
        "orientation_increments": np.asarray(orientation_increments),
    }


def _run_variant(
    trajectory: EuRoCPoseTrajectory,
    controls: dict[str, np.ndarray],
    measurements: np.ndarray,
    config: EuRoCS3R3PoseConfig,
    variant: str,
) -> dict[str, float | int | str]:
    filter_ = _make_initial_filter(trajectory.positions[0], trajectory.quaternions[0], config)
    measurement_cov = np.eye(3) * config.measurement_noise_std**2
    process_noise_cov = np.eye(3) * config.process_noise_std**2

    position_sq_error = 0.0
    orientation_mode_error = 0.0
    orientation_point_error = 0.0
    nees_sum = 0.0
    coverage_hits = 0
    runtime = 0.0

    for step, measurement in enumerate(measurements):
        likelihood = GaussianDistribution(measurement, measurement_cov, check_validity=False)
        start = perf_counter()
        predict_s3r3_dynamic_pose(
            filter_,
            controls["body_increments"][step],
            controls["orientation_increments"][step],
            variant=variant,
            process_noise_cov=process_noise_cov,
            cell_sample_count=config.cell_sample_count,
            orientation_transition_kappa=config.orientation_transition_kappa,
        )
        filter_.update(likelihoods_linear=[likelihood])
        runtime += perf_counter() - start

        true_position = trajectory.positions[step + 1]
        true_orientation = trajectory.quaternions[step + 1]
        error, nees = s3r3_linear_position_error_stats(filter_, true_position)
        position_sq_error += float(error @ error)
        nees_sum += nees
        coverage_hits += int(nees <= 7.814727903251179)
        orientation_mode_error += s3r3_orientation_distance(s3r3_orientation_mode(filter_), true_orientation)
        orientation_point_error += s3r3_orientation_distance(s3r3_orientation_point_estimate(filter_), true_orientation)

    n_steps = measurements.shape[0]
    body_norms = np.linalg.norm(controls["body_increments"], axis=1)
    orientation_norms = np.asarray([_quaternion_angle(quaternion) for quaternion in controls["orientation_increments"]])
    return {
        "variant": variant,
        "grid_size": config.grid_size,
        "position_rmse": float(np.sqrt(position_sq_error / n_steps)),
        "orientation_mode_error_rad": orientation_mode_error / n_steps,
        "orientation_point_error_rad": orientation_point_error / n_steps,
        "mean_nees": nees_sum / n_steps,
        "coverage_95": coverage_hits / n_steps,
        "runtime_ms_per_step": 1000.0 * runtime / n_steps,
        "n_steps": n_steps,
        "duration_s": float(trajectory.timestamps_s[-1] - trajectory.timestamps_s[0]),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1))),
        "mean_body_increment_m": float(np.mean(body_norms)),
        "mean_orientation_increment_rad": float(np.mean(orientation_norms)),
        "orientation_transition_kappa": config.orientation_transition_kappa,
        "cell_sample_count": config.cell_sample_count,
    }


def _run_manifold_ukf_variant(
    trajectory: EuRoCPoseTrajectory,
    controls: dict[str, np.ndarray],
    measurements: np.ndarray,
    config: EuRoCS3R3PoseConfig,
) -> dict[str, float | int | str]:
    filter_ = make_so3r3_manifold_ukf(
        trajectory.positions[0],
        _ukf_initial_orientation(trajectory.quaternions[0], config),
        SO3R3ManifoldUKFConfig(
            measurement_noise_std=config.measurement_noise_std,
            process_noise_std=config.process_noise_std,
            initial_position_std=config.initial_position_std,
            initial_orientation_std=_ukf_initial_orientation_std(config),
            orientation_process_std=config.ukf_orientation_process_std,
            alpha=config.ukf_alpha,
        ),
    )

    position_sq_error = 0.0
    orientation_error = 0.0
    nees_sum = 0.0
    coverage_hits = 0
    runtime = 0.0

    for step, measurement in enumerate(measurements):
        start = perf_counter()
        predict_so3r3_manifold_ukf(
            filter_,
            controls["body_increments"][step],
            controls["orientation_increments"][step],
        )
        update_so3r3_manifold_ukf(filter_, measurement)
        runtime += perf_counter() - start

        true_position = trajectory.positions[step + 1]
        true_orientation = trajectory.quaternions[step + 1]
        error, nees = so3r3_manifold_ukf_position_error_stats(filter_, true_position)
        position_sq_error += float(error @ error)
        nees_sum += nees
        coverage_hits += int(nees <= 7.814727903251179)
        orientation_error += s3r3_orientation_distance(so3r3_manifold_ukf_orientation(filter_), true_orientation)

    n_steps = measurements.shape[0]
    body_norms = np.linalg.norm(controls["body_increments"], axis=1)
    orientation_norms = np.asarray([_quaternion_angle(quaternion) for quaternion in controls["orientation_increments"]])
    return {
        "variant": "manifold_ukf",
        "grid_size": config.grid_size,
        "position_rmse": float(np.sqrt(position_sq_error / n_steps)),
        "orientation_mode_error_rad": orientation_error / n_steps,
        "orientation_point_error_rad": orientation_error / n_steps,
        "mean_nees": nees_sum / n_steps,
        "coverage_95": coverage_hits / n_steps,
        "runtime_ms_per_step": 1000.0 * runtime / n_steps,
        "n_steps": n_steps,
        "duration_s": float(trajectory.timestamps_s[-1] - trajectory.timestamps_s[0]),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1))),
        "mean_body_increment_m": float(np.mean(body_norms)),
        "mean_orientation_increment_rad": float(np.mean(orientation_norms)),
        "orientation_transition_kappa": config.orientation_transition_kappa,
        "cell_sample_count": config.cell_sample_count,
    }


def _initial_prior_modes(
    initial_orientation: np.ndarray,
    config: EuRoCS3R3PoseConfig,
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(float(value) for value in _yaw_offset_orientation(initial_orientation, yaw_offset))
        for yaw_offset in config.prior_yaw_offsets_rad
    )


def _normalized_prior_weights(config: EuRoCS3R3PoseConfig) -> tuple[float, ...]:
    weights = np.asarray(config.prior_weights, dtype=float)
    weights = weights / np.sum(weights)
    return tuple(float(value) for value in weights)


def _ukf_initial_orientation(
    initial_orientation: np.ndarray,
    config: EuRoCS3R3PoseConfig,
) -> np.ndarray:
    mean_offset = _weighted_yaw_offset_mean(config)
    return _yaw_offset_orientation(initial_orientation, mean_offset)


def _ukf_initial_orientation_std(config: EuRoCS3R3PoseConfig) -> float:
    local_std = float(np.sqrt(2.0 / config.orientation_prior_kappa))
    mode_std = _weighted_yaw_offset_std(config, _weighted_yaw_offset_mean(config))
    return max(local_std, mode_std, 1e-6)


def _weighted_yaw_offset_mean(config: EuRoCS3R3PoseConfig) -> float:
    offsets = np.asarray(config.prior_yaw_offsets_rad, dtype=float)
    weights = np.asarray(_normalized_prior_weights(config), dtype=float)
    sin_sum = float(np.sum(weights * np.sin(offsets)))
    cos_sum = float(np.sum(weights * np.cos(offsets)))
    if np.hypot(sin_sum, cos_sum) > 1e-12:
        return float(np.arctan2(sin_sum, cos_sum))
    return float(np.average(np.unwrap(offsets), weights=weights))


def _weighted_yaw_offset_std(config: EuRoCS3R3PoseConfig, mean_offset: float) -> float:
    offsets = np.asarray(config.prior_yaw_offsets_rad, dtype=float)
    weights = np.asarray(_normalized_prior_weights(config), dtype=float)
    residuals = np.angle(np.exp(1j * (offsets - mean_offset)))
    return float(np.sqrt(np.sum(weights * residuals**2)))


def _yaw_offset_orientation(initial_orientation: np.ndarray, yaw_offset: float) -> np.ndarray:
    return _quaternion_multiply(_canonical_quaternions(initial_orientation), _yaw_quaternion(yaw_offset))


def _yaw_quaternion(yaw_offset: float) -> np.ndarray:
    return _canonical_quaternions(
        np.array([0.0, 0.0, np.sin(0.5 * yaw_offset), np.cos(0.5 * yaw_offset)], dtype=float)
    )


def _make_initial_filter(
    initial_position: np.ndarray,
    initial_orientation: np.ndarray,
    config: EuRoCS3R3PoseConfig,
) -> StateSpaceSubdivisionFilter:
    prototype = S3R3PrototypeConfig(
        grid_sizes=(config.grid_size,),
        variants=config.variants,
        n_trials=1,
        n_steps=config.max_steps,
        seed=config.seed,
        measurement_noise_std=config.measurement_noise_std,
        process_noise_std=config.process_noise_std,
        initial_position_std=config.initial_position_std,
        prior_modes=_initial_prior_modes(initial_orientation, config),
        prior_weights=_normalized_prior_weights(config),
        prior_kappa=config.orientation_prior_kappa,
        cell_sample_count=config.cell_sample_count,
    )
    orientation_filter = make_s3r3_orientation_filter(prototype, config.grid_size)
    gd = orientation_filter.filter_state
    initial_covariance = np.eye(3) * config.initial_position_std**2
    gaussians = [
        GaussianDistribution(np.asarray(initial_position, dtype=float), initial_covariance.copy(), check_validity=False)
        for _ in range(gd.get_grid().shape[0])
    ]
    return StateSpaceSubdivisionFilter(StateSpaceSubdivisionGaussianDistribution(gd, gaussians))


def _build_claim_rows(metrics: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str | bool]]:
    rows_by_variant = {str(row["variant"]): row for row in metrics}
    claims = []
    candidate = rows_by_variant.get("r1_r2")
    if candidate is None:
        return claims
    for comparator_variant, comparison in (
        ("baseline", "R1+R2 vs baseline"),
        ("r1", "R1+R2 vs R1"),
        ("manifold_ukf", "R1+R2 vs manifold UKF"),
    ):
        if comparator_variant in rows_by_variant:
            claims.append(_claim_row(candidate, rows_by_variant[comparator_variant], comparison))
    return claims


def _claim_row(
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
    comparison: str,
) -> dict[str, float | int | str | bool]:
    candidate_rmse = float(candidate["position_rmse"])
    comparator_rmse = float(comparator["position_rmse"])
    position_ratio = candidate_rmse / comparator_rmse
    nees_ratio = float(candidate["mean_nees"]) / float(comparator["mean_nees"])
    coverage_delta = float(candidate["coverage_95"]) - float(comparator["coverage_95"])
    orientation_delta = float(candidate["orientation_point_error_rad"]) - float(comparator["orientation_point_error_rad"])
    runtime_ratio = float(candidate["runtime_ms_per_step"]) / float(comparator["runtime_ms_per_step"])
    supports_accuracy = position_ratio < 1.0
    supports_consistency = nees_ratio <= 1.0 and coverage_delta >= -0.02
    supports_orientation = orientation_delta <= 0.05
    supports_runtime = runtime_ratio <= 1.25
    return {
        "comparison": comparison,
        "candidate_variant": str(candidate["variant"]),
        "comparator_variant": str(comparator["variant"]),
        "candidate_position_rmse": candidate_rmse,
        "comparator_position_rmse": comparator_rmse,
        "position_rmse_ratio": position_ratio,
        "position_rmse_gain_pct": 100.0 * (1.0 - position_ratio),
        "candidate_mean_nees": float(candidate["mean_nees"]),
        "comparator_mean_nees": float(comparator["mean_nees"]),
        "mean_nees_ratio": nees_ratio,
        "candidate_coverage_95": float(candidate["coverage_95"]),
        "comparator_coverage_95": float(comparator["coverage_95"]),
        "coverage_delta": coverage_delta,
        "candidate_orientation_point_error_rad": float(candidate["orientation_point_error_rad"]),
        "comparator_orientation_point_error_rad": float(comparator["orientation_point_error_rad"]),
        "orientation_point_error_delta_rad": orientation_delta,
        "candidate_runtime_ms_per_step": float(candidate["runtime_ms_per_step"]),
        "comparator_runtime_ms_per_step": float(comparator["runtime_ms_per_step"]),
        "runtime_ratio": runtime_ratio,
        "supports_accuracy_claim": supports_accuracy,
        "supports_consistency_claim": supports_consistency,
        "supports_orientation_claim": supports_orientation,
        "supports_runtime_claim": supports_runtime,
        "supports_overall_claim": supports_accuracy and supports_consistency and supports_orientation and supports_runtime,
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _write_metadata(
    path: Path,
    result: EuRoCS3R3PoseResult,
    groundtruth_path: Path,
    config: EuRoCS3R3PoseConfig,
) -> None:
    metadata = {
        "claims_rows": len(result.claims),
        "claims_schema": EUROC_S3R3_CLAIM_FIELDNAMES,
        "config": euroc_s3r3_pose_config_to_dict(config),
        "experiment": "euroc_s3r3_pose",
        "groundtruth_path": str(groundtruth_path),
        "metrics_rows": len(result.metrics),
        "metrics_schema": EUROC_S3R3_METRIC_FIELDNAMES,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    groundtruth_path: Path,
    result: EuRoCS3R3PoseResult,
    metrics_path: Path,
    claims_path: Path,
    plot_paths: list[Path],
    config: EuRoCS3R3PoseConfig,
) -> None:
    best_rmse = min(result.metrics, key=lambda row: float(row["position_rmse"]))
    baseline_claims = [row for row in result.claims if row["comparator_variant"] == "baseline"]
    lines = [
        "# EuRoC S3+ x R3 Pose Smoke Note",
        "",
        "This run uses EuRoC ground-truth 3D poses to derive per-step body-frame translation controls and quaternion increments.",
        "No visual-inertial frontend is used; position measurements are simulated by adding Gaussian noise to ground-truth positions.",
        "The `manifold_ukf` row is a PyRecEst UKF-M baseline on SO3 x R3 with a 6-D tangent covariance.",
        "",
        f"Ground truth: `{groundtruth_path}`",
        f"Grid size: {config.grid_size}",
        f"Steps: {config.max_steps}",
        f"Stride: {config.stride}",
        f"Cell sample count: {config.cell_sample_count}",
        f"Prior yaw offsets rad: {list(config.prior_yaw_offsets_rad)}",
        f"Prior weights: {list(_normalized_prior_weights(config))}",
        f"Manifold UKF included: {config.include_manifold_ukf}",
        f"Metrics: `{metrics_path.name}`",
        f"Claims: `{claims_path.name}`",
        "",
        "## Result",
        "",
        f"Lowest translation RMSE: `{best_rmse['variant']}` with RMSE `{float(best_rmse['position_rmse']):.4f}`.",
    ]
    if baseline_claims:
        claim = baseline_claims[0]
        lines.extend(
            [
                f"`R1+R2` vs baseline translation RMSE gain: `{float(claim['position_rmse_gain_pct']):.1f}%`.",
                f"`R1+R2` vs baseline NEES ratio: `{float(claim['mean_nees_ratio']):.3f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            _format_metrics_table(result.metrics),
            "",
            "Plots:",
            format_plot_list(plot_paths),
            "",
            "This is a real-trajectory geometry smoke test. It validates the S3/R3 prediction path on measured pose increments, not full visual-inertial estimation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metrics_table(rows: list[dict[str, float | int | str]]) -> str:
    header = "| Variant | RMSE | point err rad | NEES | coverage | runtime ms |"
    separator = "|---|---:|---:|---:|---:|---:|"
    body = []
    for row in rows:
        body.append(
            "| "
            f"{EUROC_S3R3_VARIANT_LABELS[str(row['variant'])]} | "
            f"{float(row['position_rmse']):.4f} | "
            f"{float(row['orientation_point_error_rad']):.3f} | "
            f"{float(row['mean_nees']):.3f} | "
            f"{float(row['coverage_95']):.3f} | "
            f"{float(row['runtime_ms_per_step']):.3f} |"
        )
    return "\n".join([header, separator, *body])


def _write_plots(output_dir: Path, rows: list[dict[str, float | int | str]]) -> list[Path]:
    return [
        _write_bar_plot(output_dir, rows, "position_rmse", "Translation RMSE [m]", "euroc_s3r3_pose_rmse.png"),
        _write_bar_plot(output_dir, rows, "mean_nees", "Mean Position NEES", "euroc_s3r3_pose_nees.png"),
        _write_bar_plot(output_dir, rows, "runtime_ms_per_step", "Runtime [ms/step]", "euroc_s3r3_pose_runtime.png"),
    ]


def _write_bar_plot(output_dir: Path, rows: list[dict[str, float | int | str]], metric_name: str, y_label: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [EUROC_S3R3_VARIANT_LABELS[str(row["variant"])] for row in rows]
    values = [float(row[metric_name]) for row in rows]
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=15)
    return save_figure(fig, output_dir, filename)


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


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    xyzw = _canonical_quaternions(quaternion).reshape(4)
    return _canonical_quaternions(np.array([-xyzw[0], -xyzw[1], -xyzw[2], xyzw[3]], dtype=float))


def _quaternion_angle(quaternion: np.ndarray) -> float:
    xyzw = _canonical_quaternions(quaternion).reshape(4)
    return float(2.0 * np.arccos(np.clip(abs(xyzw[3]), 0.0, 1.0)))
