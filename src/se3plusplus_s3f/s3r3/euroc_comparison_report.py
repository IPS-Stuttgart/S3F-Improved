"""EuRoC S3+ x R3 comparison report against high-resolution S3F, UKF, and PF baselines."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution

from ..s1r2.plotting import format_plot_list, save_figure
from .dynamic_pose import predict_s3r3_dynamic_pose
from .euroc_pose import (
    EuRoCPoseTrajectory,
    EuRoCS3R3PoseConfig,
    _initial_prior_modes,
    _make_initial_filter,
    _normalized_prior_weights,
    _quaternion_angle,
    _slice_trajectory,
    _trajectory_controls,
    _ukf_initial_orientation,
    _ukf_initial_orientation_std,
    load_euroc_pose_groundtruth,
)
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
    _canonical_quaternions,
    _exp_map_identity,
    _quaternion_multiply,
    _rotate_vectors,
    _symmetrize,
    s3r3_linear_position_error_stats,
    s3r3_linear_position_mean,
    s3r3_orientation_distance,
    s3r3_orientation_mode,
    s3r3_orientation_point_estimate,
)

plt.switch_backend("Agg")

REFERENCE_FILTER = "s3f_reference"
S3F_FILTER = "s3f"
UKF_FILTER = "manifold_ukf"
PARTICLE_FILTER = "particle_filter"
REFERENCE_VARIANT = "baseline"
PARTICLE_VARIANT = "bootstrap"
R1_R2_VARIANT = "r1_r2"

EUROC_S3R3_COMPARISON_LABELS = {
    **VARIANT_LABELS,
    REFERENCE_FILTER: "High-res baseline S3F",
    UKF_FILTER: "Manifold UKF",
    PARTICLE_VARIANT: "Bootstrap PF",
}

EUROC_S3R3_COMPARISON_METRIC_FIELDNAMES = [
    "filter",
    "variant",
    "grid_size",
    "reference_grid_size",
    "particle_count",
    "resource_count",
    "position_rmse_to_truth",
    "position_rmse_to_reference",
    "orientation_mode_error_to_truth_rad",
    "orientation_mode_error_to_reference_rad",
    "orientation_point_error_to_truth_rad",
    "orientation_point_error_to_reference_rad",
    "mean_nees_to_truth",
    "coverage_95_to_truth",
    "runtime_ms_per_step",
    "n_steps",
    "duration_s",
    "path_length_m",
    "mean_body_increment_m",
    "mean_orientation_increment_rad",
    "orientation_transition_kappa",
    "cell_sample_count",
]

EUROC_S3R3_COMPARISON_CLAIM_FIELDNAMES = [
    "comparison",
    "candidate_filter",
    "candidate_variant",
    "candidate_grid_size",
    "candidate_particle_count",
    "comparator_filter",
    "comparator_variant",
    "comparator_grid_size",
    "comparator_particle_count",
    "candidate_position_rmse_to_reference",
    "comparator_position_rmse_to_reference",
    "position_rmse_to_reference_ratio",
    "position_rmse_to_reference_gain_pct",
    "candidate_position_rmse_to_truth",
    "comparator_position_rmse_to_truth",
    "position_rmse_to_truth_ratio",
    "position_rmse_to_truth_gain_pct",
    "candidate_mean_nees_to_truth",
    "comparator_mean_nees_to_truth",
    "mean_nees_ratio",
    "candidate_coverage_95_to_truth",
    "comparator_coverage_95_to_truth",
    "coverage_delta",
    "candidate_orientation_point_error_to_reference_rad",
    "comparator_orientation_point_error_to_reference_rad",
    "orientation_point_error_to_reference_delta_rad",
    "candidate_runtime_ms_per_step",
    "comparator_runtime_ms_per_step",
    "runtime_ratio",
    "supports_reference_claim",
    "supports_truth_accuracy_claim",
    "supports_consistency_claim",
    "supports_orientation_reference_claim",
    "supports_runtime_claim",
    "supports_overall_claim",
]


@dataclass(frozen=True)
class EuRoCS3R3ComparisonReportConfig:
    """Configuration for a EuRoC S3/R3 report with high-resolution and external baselines."""

    grid_sizes: tuple[int, ...] = (8, 16, 32)
    variants: tuple[str, ...] = SUPPORTED_S3R3_VARIANTS
    reference_grid_size: int = 64
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
    particle_counts: tuple[int, ...] = (128,)
    particle_resample_threshold: float = 0.5


@dataclass(frozen=True)
class EuRoCS3R3ComparisonReportResult:
    """Container for EuRoC S3/R3 comparison report rows."""

    metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]


@dataclass
class _MetricTotals:
    position_truth_sq_error: float = 0.0
    position_reference_sq_error: float = 0.0
    orientation_mode_truth_error: float = 0.0
    orientation_mode_reference_error: float = 0.0
    orientation_point_truth_error: float = 0.0
    orientation_point_reference_error: float = 0.0
    nees_sum: float = 0.0
    coverage_hits: int = 0
    runtime_s: float = 0.0


@dataclass
class _ParticleState:
    orientations: np.ndarray
    positions: np.ndarray
    weights: np.ndarray


def euroc_s3r3_comparison_report_config_to_dict(config: EuRoCS3R3ComparisonReportConfig) -> dict[str, Any]:
    """Return a JSON-serializable EuRoC S3/R3 comparison config."""

    return json.loads(json.dumps(asdict(config)))


def run_euroc_s3r3_comparison_report(
    groundtruth_path: Path,
    config: EuRoCS3R3ComparisonReportConfig = EuRoCS3R3ComparisonReportConfig(),
) -> EuRoCS3R3ComparisonReportResult:
    """Run EuRoC S3/R3 variants against a high-resolution S3F reference, UKF, and PF."""

    _validate_config(config)
    trajectory = _slice_trajectory(load_euroc_pose_groundtruth(groundtruth_path), _pose_config(config, config.grid_sizes[0]))
    controls = _trajectory_controls(trajectory)
    measurements = _noisy_position_measurements(trajectory, config)
    shared_context = _trajectory_context(trajectory, controls, config)
    process_noise_cov = np.eye(3) * config.process_noise_std**2
    measurement_cov = np.eye(3) * config.measurement_noise_std**2

    reference_filter = _make_initial_filter(trajectory.positions[0], trajectory.quaternions[0], _pose_config(config, config.reference_grid_size))
    candidate_filters = {
        (grid_size, variant): _make_initial_filter(trajectory.positions[0], trajectory.quaternions[0], _pose_config(config, grid_size))
        for grid_size in config.grid_sizes
        for variant in config.variants
    }
    ukf_filter = _make_ukf_filter(trajectory, config) if config.include_manifold_ukf else None
    rng = np.random.default_rng(config.seed + 10_000)
    particle_states = {particle_count: _initial_particles(trajectory, config, particle_count, rng) for particle_count in config.particle_counts}

    reference_totals = _MetricTotals()
    candidate_totals = {key: _MetricTotals() for key in candidate_filters}
    ukf_totals = _MetricTotals()
    particle_totals = {particle_count: _MetricTotals() for particle_count in config.particle_counts}

    for step, measurement in enumerate(measurements):
        true_position = trajectory.positions[step + 1]
        true_orientation = trajectory.quaternions[step + 1]
        body_increment = controls["body_increments"][step]
        orientation_increment = controls["orientation_increments"][step]

        reference_snapshot = _run_reference_step(
            reference_filter,
            measurement,
            measurement_cov,
            body_increment,
            orientation_increment,
            process_noise_cov,
            true_position,
            true_orientation,
            config,
            reference_totals,
        )

        for key, filter_ in candidate_filters.items():
            elapsed = _predict_update_s3f(filter_, measurement, measurement_cov, body_increment, orientation_increment, process_noise_cov, key[1], config)
            _accumulate_s3f(candidate_totals[key], filter_, true_position, true_orientation, reference_snapshot, elapsed)

        if ukf_filter is not None:
            elapsed = _predict_update_ukf(ukf_filter, measurement, body_increment, orientation_increment)
            _accumulate_ukf(ukf_totals, ukf_filter, true_position, true_orientation, reference_snapshot, elapsed)

        for particle_count, particle_state in particle_states.items():
            elapsed = _predict_update_particles(particle_state, measurement, measurement_cov, body_increment, orientation_increment, process_noise_cov, config, rng)
            _accumulate_particles(particle_totals[particle_count], particle_state, true_position, true_orientation, reference_snapshot, elapsed)
            _resample_particle_state(particle_state, rng, config.particle_resample_threshold)

    n_steps = measurements.shape[0]
    metrics = [
        _metric_row(
            filter_name=REFERENCE_FILTER,
            variant=REFERENCE_VARIANT,
            grid_size=config.reference_grid_size,
            particle_count="",
            resource_count=config.reference_grid_size,
            totals=reference_totals,
            n_steps=n_steps,
            config=config,
            context=shared_context,
        )
    ]
    metrics.extend(
        _metric_row(
            filter_name=S3F_FILTER,
            variant=variant,
            grid_size=grid_size,
            particle_count="",
            resource_count=grid_size,
            totals=candidate_totals[(grid_size, variant)],
            n_steps=n_steps,
            config=config,
            context=shared_context,
        )
        for grid_size in config.grid_sizes
        for variant in config.variants
    )
    if ukf_filter is not None:
        metrics.append(
            _metric_row(
                filter_name=UKF_FILTER,
                variant=UKF_FILTER,
                grid_size="",
                particle_count="",
                resource_count="",
                totals=ukf_totals,
                n_steps=n_steps,
                config=config,
                context=shared_context,
            )
        )
    metrics.extend(
        _metric_row(
            filter_name=PARTICLE_FILTER,
            variant=PARTICLE_VARIANT,
            grid_size="",
            particle_count=particle_count,
            resource_count=particle_count,
            totals=particle_totals[particle_count],
            n_steps=n_steps,
            config=config,
            context=shared_context,
        )
        for particle_count in config.particle_counts
    )
    return EuRoCS3R3ComparisonReportResult(metrics=metrics, claims=_build_claim_rows(metrics))


def write_euroc_s3r3_comparison_report_outputs(
    groundtruth_path: Path,
    output_dir: Path,
    config: EuRoCS3R3ComparisonReportConfig = EuRoCS3R3ComparisonReportConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the EuRoC S3/R3 comparison report and write CSVs, plots, metadata, and a note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_euroc_s3r3_comparison_report(groundtruth_path, config)

    metrics_path = output_dir / "euroc_s3r3_comparison_metrics.csv"
    _write_csv(metrics_path, result.metrics, EUROC_S3R3_COMPARISON_METRIC_FIELDNAMES)

    claims_path = output_dir / "euroc_s3r3_comparison_claims.csv"
    _write_csv(claims_path, result.claims, EUROC_S3R3_COMPARISON_CLAIM_FIELDNAMES)

    outputs = {"metrics": metrics_path, "claims": claims_path}
    plot_paths = _write_plots(output_dir, result.metrics, result.claims) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "euroc_s3r3_comparison_note.md"
    _write_note(note_path, groundtruth_path, result, metrics_path, claims_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, groundtruth_path, config)
    outputs["metadata"] = metadata_path
    return outputs


def _validate_config(config: EuRoCS3R3ComparisonReportConfig) -> None:
    if not config.grid_sizes:
        raise ValueError("grid_sizes must not be empty.")
    if min(config.grid_sizes) <= 0:
        raise ValueError("all grid_sizes must be positive.")
    if config.reference_grid_size <= max(config.grid_sizes):
        raise ValueError("reference_grid_size must be greater than every coarse grid size.")
    unknown_variants = [variant for variant in config.variants if variant not in SUPPORTED_S3R3_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown variant {unknown_variants[0]!r}.")
    if R1_R2_VARIANT not in config.variants or REFERENCE_VARIANT not in config.variants:
        raise ValueError("variants must include baseline and r1_r2.")
    for name, value in (
        ("stride", config.stride),
        ("max_steps", config.max_steps),
        ("cell_sample_count", config.cell_sample_count),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.start_index < 0:
        raise ValueError("start_index must be nonnegative.")
    for name, value in (
        ("measurement_noise_std", config.measurement_noise_std),
        ("process_noise_std", config.process_noise_std),
        ("initial_position_std", config.initial_position_std),
        ("orientation_prior_kappa", config.orientation_prior_kappa),
        ("orientation_transition_kappa", config.orientation_transition_kappa),
        ("ukf_alpha", config.ukf_alpha),
        ("ukf_orientation_process_std", config.ukf_orientation_process_std),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if min(config.particle_counts, default=1) <= 0:
        raise ValueError("particle_counts must be positive.")
    if not 0.0 < config.particle_resample_threshold <= 1.0:
        raise ValueError("particle_resample_threshold must be in (0, 1].")
    _pose_config(config, config.grid_sizes[0])


def _pose_config(config: EuRoCS3R3ComparisonReportConfig, grid_size: int) -> EuRoCS3R3PoseConfig:
    return EuRoCS3R3PoseConfig(
        grid_size=grid_size,
        variants=config.variants,
        start_index=config.start_index,
        stride=config.stride,
        max_steps=config.max_steps,
        seed=config.seed,
        measurement_noise_std=config.measurement_noise_std,
        process_noise_std=config.process_noise_std,
        initial_position_std=config.initial_position_std,
        orientation_prior_kappa=config.orientation_prior_kappa,
        orientation_transition_kappa=config.orientation_transition_kappa,
        cell_sample_count=config.cell_sample_count,
        prior_yaw_offsets_rad=config.prior_yaw_offsets_rad,
        prior_weights=config.prior_weights,
        include_manifold_ukf=config.include_manifold_ukf,
        ukf_alpha=config.ukf_alpha,
        ukf_orientation_process_std=config.ukf_orientation_process_std,
    )


def _noisy_position_measurements(trajectory: EuRoCPoseTrajectory, config: EuRoCS3R3ComparisonReportConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    return trajectory.positions[1:] + rng.normal(0.0, config.measurement_noise_std, size=(trajectory.positions.shape[0] - 1, 3))


def _trajectory_context(
    trajectory: EuRoCPoseTrajectory,
    controls: dict[str, np.ndarray],
    config: EuRoCS3R3ComparisonReportConfig,
) -> dict[str, float | int]:
    body_norms = np.linalg.norm(controls["body_increments"], axis=1)
    orientation_norms = np.asarray([_quaternion_angle(quaternion) for quaternion in controls["orientation_increments"]])
    return {
        "duration_s": float(trajectory.timestamps_s[-1] - trajectory.timestamps_s[0]),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1))),
        "mean_body_increment_m": float(np.mean(body_norms)),
        "mean_orientation_increment_rad": float(np.mean(orientation_norms)),
        "orientation_transition_kappa": float(config.orientation_transition_kappa),
        "cell_sample_count": int(config.cell_sample_count),
    }


def _make_ukf_filter(trajectory: EuRoCPoseTrajectory, config: EuRoCS3R3ComparisonReportConfig):
    pose_config = _pose_config(config, config.grid_sizes[0])
    return make_so3r3_manifold_ukf(
        trajectory.positions[0],
        _ukf_initial_orientation(trajectory.quaternions[0], pose_config),
        SO3R3ManifoldUKFConfig(
            measurement_noise_std=config.measurement_noise_std,
            process_noise_std=config.process_noise_std,
            initial_position_std=config.initial_position_std,
            initial_orientation_std=_ukf_initial_orientation_std(pose_config),
            orientation_process_std=config.ukf_orientation_process_std,
            alpha=config.ukf_alpha,
        ),
    )


def _run_reference_step(
    reference_filter,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
    body_increment: np.ndarray,
    orientation_increment: np.ndarray,
    process_noise_cov: np.ndarray,
    true_position: np.ndarray,
    true_orientation: np.ndarray,
    config: EuRoCS3R3ComparisonReportConfig,
    totals: _MetricTotals,
) -> dict[str, np.ndarray]:
    elapsed = _predict_update_s3f(
        reference_filter,
        measurement,
        measurement_cov,
        body_increment,
        orientation_increment,
        process_noise_cov,
        REFERENCE_VARIANT,
        config,
    )
    position_mean = s3r3_linear_position_mean(reference_filter)
    mode = s3r3_orientation_mode(reference_filter)
    point = s3r3_orientation_point_estimate(reference_filter)
    error, nees = s3r3_linear_position_error_stats(reference_filter, true_position)
    totals.position_truth_sq_error += float(error @ error)
    totals.orientation_mode_truth_error += s3r3_orientation_distance(mode, true_orientation)
    totals.orientation_point_truth_error += s3r3_orientation_distance(point, true_orientation)
    totals.nees_sum += nees
    totals.coverage_hits += int(nees <= 7.814727903251179)
    totals.runtime_s += elapsed
    return {"position": position_mean, "mode": mode, "point": point}


def _predict_update_s3f(
    filter_,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
    body_increment: np.ndarray,
    orientation_increment: np.ndarray,
    process_noise_cov: np.ndarray,
    variant: str,
    config: EuRoCS3R3ComparisonReportConfig,
) -> float:
    likelihood = GaussianDistribution(measurement, measurement_cov, check_validity=False)
    start = perf_counter()
    predict_s3r3_dynamic_pose(
        filter_,
        body_increment,
        orientation_increment,
        variant=variant,
        process_noise_cov=process_noise_cov,
        cell_sample_count=config.cell_sample_count,
        orientation_transition_kappa=config.orientation_transition_kappa,
    )
    filter_.update(likelihoods_linear=[likelihood])
    return perf_counter() - start


def _predict_update_ukf(
    filter_,
    measurement: np.ndarray,
    body_increment: np.ndarray,
    orientation_increment: np.ndarray,
) -> float:
    start = perf_counter()
    predict_so3r3_manifold_ukf(filter_, body_increment, orientation_increment)
    update_so3r3_manifold_ukf(filter_, measurement)
    return perf_counter() - start


def _accumulate_s3f(
    totals: _MetricTotals,
    filter_,
    true_position: np.ndarray,
    true_orientation: np.ndarray,
    reference_snapshot: dict[str, np.ndarray],
    elapsed_s: float,
) -> None:
    position_mean = s3r3_linear_position_mean(filter_)
    mode = s3r3_orientation_mode(filter_)
    point = s3r3_orientation_point_estimate(filter_)
    error, nees = s3r3_linear_position_error_stats(filter_, true_position)
    _accumulate_common(totals, position_mean, mode, point, error, nees, true_orientation, reference_snapshot, elapsed_s)


def _accumulate_ukf(
    totals: _MetricTotals,
    filter_,
    true_position: np.ndarray,
    true_orientation: np.ndarray,
    reference_snapshot: dict[str, np.ndarray],
    elapsed_s: float,
) -> None:
    state, _covariance = filter_.filter_state
    position_mean = np.asarray(state.position, dtype=float).reshape(3)
    point = so3r3_manifold_ukf_orientation(filter_)
    error, nees = so3r3_manifold_ukf_position_error_stats(filter_, true_position)
    _accumulate_common(totals, position_mean, point, point, error, nees, true_orientation, reference_snapshot, elapsed_s)


def _accumulate_particles(
    totals: _MetricTotals,
    particle_state: _ParticleState,
    true_position: np.ndarray,
    true_orientation: np.ndarray,
    reference_snapshot: dict[str, np.ndarray],
    elapsed_s: float,
) -> None:
    position_mean, position_covariance = _weighted_position_stats(particle_state.positions, particle_state.weights)
    point = _weighted_quaternion_mean(particle_state.orientations, particle_state.weights)
    error = position_mean - np.asarray(true_position, dtype=float)
    covariance_reg = position_covariance + 1e-10 * np.eye(3)
    nees = float(error @ np.linalg.solve(covariance_reg, error))
    _accumulate_common(totals, position_mean, point, point, error, nees, true_orientation, reference_snapshot, elapsed_s)


def _accumulate_common(
    totals: _MetricTotals,
    position_mean: np.ndarray,
    mode: np.ndarray,
    point: np.ndarray,
    position_error: np.ndarray,
    nees: float,
    true_orientation: np.ndarray,
    reference_snapshot: dict[str, np.ndarray],
    elapsed_s: float,
) -> None:
    reference_delta = position_mean - reference_snapshot["position"]
    totals.position_truth_sq_error += float(position_error @ position_error)
    totals.position_reference_sq_error += float(reference_delta @ reference_delta)
    totals.orientation_mode_truth_error += s3r3_orientation_distance(mode, true_orientation)
    totals.orientation_mode_reference_error += s3r3_orientation_distance(mode, reference_snapshot["mode"])
    totals.orientation_point_truth_error += s3r3_orientation_distance(point, true_orientation)
    totals.orientation_point_reference_error += s3r3_orientation_distance(point, reference_snapshot["point"])
    totals.nees_sum += nees
    totals.coverage_hits += int(nees <= 7.814727903251179)
    totals.runtime_s += elapsed_s


def _initial_particles(
    trajectory: EuRoCPoseTrajectory,
    config: EuRoCS3R3ComparisonReportConfig,
    particle_count: int,
    rng: np.random.Generator,
) -> _ParticleState:
    pose_config = _pose_config(config, config.grid_sizes[0])
    modes = _canonical_quaternions(np.asarray(_initial_prior_modes(trajectory.quaternions[0], pose_config), dtype=float))
    weights = np.asarray(_normalized_prior_weights(pose_config), dtype=float)
    components = rng.choice(len(modes), size=particle_count, p=weights)
    tangent_std = float(np.sqrt(2.0 / config.orientation_prior_kappa))
    local_quaternions = _exp_map_identity(rng.normal(scale=tangent_std, size=(particle_count, 3)))
    orientations = _quaternion_multiply(modes[components], local_quaternions)
    positions = trajectory.positions[0] + rng.normal(0.0, config.initial_position_std, size=(particle_count, 3))
    return _ParticleState(orientations=orientations, positions=positions, weights=np.full(particle_count, 1.0 / particle_count))


def _predict_update_particles(
    particle_state: _ParticleState,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
    body_increment: np.ndarray,
    orientation_increment: np.ndarray,
    process_noise_cov: np.ndarray,
    config: EuRoCS3R3ComparisonReportConfig,
    rng: np.random.Generator,
) -> float:
    start = perf_counter()
    particle_count = particle_state.positions.shape[0]
    position_noise = rng.multivariate_normal(np.zeros(3), process_noise_cov, size=particle_count)
    displacement = _rotate_vectors(particle_state.orientations, body_increment)
    particle_state.positions = particle_state.positions + displacement + position_noise
    orientation_noise = _exp_map_identity(rng.normal(scale=_orientation_process_std(config), size=(particle_count, 3)))
    particle_state.orientations = _quaternion_multiply(_quaternion_multiply(particle_state.orientations, orientation_increment), orientation_noise)
    particle_state.weights = _update_particle_weights(particle_state.positions, particle_state.weights, measurement, measurement_cov)
    return perf_counter() - start


def _orientation_process_std(config: EuRoCS3R3ComparisonReportConfig) -> float:
    return float(1.0 / np.sqrt(max(config.orientation_transition_kappa, 1e-9)))


# jscpd:ignore-start
def _update_particle_weights(
    positions: np.ndarray,
    weights: np.ndarray,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
) -> np.ndarray:
    innovation = positions - np.asarray(measurement, dtype=float)
    precision = np.linalg.inv(measurement_cov)
    log_likelihood = -0.5 * np.einsum("ni,ij,nj->n", innovation, precision, innovation)
    log_weights = np.log(weights + np.finfo(float).tiny) + log_likelihood
    log_weights -= np.max(log_weights)
    updated = np.exp(log_weights)
    weight_sum = float(np.sum(updated))
    if weight_sum <= 0.0:
        return np.full_like(weights, 1.0 / weights.shape[0])
    return updated / weight_sum


def _resample_particle_state(particle_state: _ParticleState, rng: np.random.Generator, threshold: float) -> None:
    effective_sample_size = 1.0 / float(np.sum(particle_state.weights**2))
    if effective_sample_size >= threshold * particle_state.weights.shape[0]:
        return
    indices = rng.choice(particle_state.weights.shape[0], size=particle_state.weights.shape[0], replace=True, p=particle_state.weights)
    particle_state.orientations = particle_state.orientations[indices].copy()
    particle_state.positions = particle_state.positions[indices].copy()
    particle_state.weights = np.full_like(particle_state.weights, 1.0 / particle_state.weights.shape[0])
# jscpd:ignore-end


def _weighted_position_stats(positions: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = weights @ positions
    centered = positions - mean
    covariance = centered.T @ (centered * weights[:, None])
    return mean, _symmetrize(covariance)


def _weighted_quaternion_mean(orientations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    canonical = _canonical_quaternions(orientations).reshape(-1, 4)
    scatter = canonical.T @ (canonical * weights[:, None])
    _eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    return _canonical_quaternions(eigenvectors[:, int(np.argmax(_eigenvalues))])


def _metric_row(
    *,
    filter_name: str,
    variant: str,
    grid_size: int | str,
    particle_count: int | str,
    resource_count: int | str,
    totals: _MetricTotals,
    n_steps: int,
    config: EuRoCS3R3ComparisonReportConfig,
    context: dict[str, float | int],
) -> dict[str, float | int | str]:
    return {
        "filter": filter_name,
        "variant": variant,
        "grid_size": grid_size,
        "reference_grid_size": config.reference_grid_size,
        "particle_count": particle_count,
        "resource_count": resource_count,
        "position_rmse_to_truth": float(np.sqrt(totals.position_truth_sq_error / n_steps)),
        "position_rmse_to_reference": float(np.sqrt(totals.position_reference_sq_error / n_steps)),
        "orientation_mode_error_to_truth_rad": totals.orientation_mode_truth_error / n_steps,
        "orientation_mode_error_to_reference_rad": totals.orientation_mode_reference_error / n_steps,
        "orientation_point_error_to_truth_rad": totals.orientation_point_truth_error / n_steps,
        "orientation_point_error_to_reference_rad": totals.orientation_point_reference_error / n_steps,
        "mean_nees_to_truth": totals.nees_sum / n_steps,
        "coverage_95_to_truth": totals.coverage_hits / n_steps,
        "runtime_ms_per_step": 1000.0 * totals.runtime_s / n_steps,
        "n_steps": n_steps,
        **context,
    }


def _build_claim_rows(metrics: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str | bool]]:
    claims: list[dict[str, float | int | str | bool]] = []
    rows_by_s3f = {(int(row["grid_size"]), str(row["variant"])): row for row in metrics if row["filter"] == S3F_FILTER}
    for grid_size in sorted({int(row["grid_size"]) for row in metrics if row["filter"] == S3F_FILTER}):
        candidate = rows_by_s3f[(grid_size, R1_R2_VARIANT)]
        for comparator_variant, comparison in (
            (REFERENCE_VARIANT, "R1+R2 vs coarse baseline"),
            ("r1", "R1+R2 vs R1"),
        ):
            if (grid_size, comparator_variant) in rows_by_s3f:
                claims.append(_claim_row(candidate, rows_by_s3f[(grid_size, comparator_variant)], comparison))

    candidates = [row for row in metrics if row["filter"] == S3F_FILTER and row["variant"] == R1_R2_VARIANT]
    if not candidates:
        return claims
    best_reference_candidate = min(candidates, key=lambda row: float(row["position_rmse_to_reference"]))
    for comparator in metrics:
        if comparator["filter"] == UKF_FILTER:
            claims.append(_claim_row(best_reference_candidate, comparator, "best R1+R2 vs manifold UKF"))
        elif comparator["filter"] == PARTICLE_FILTER:
            claims.append(_claim_row(best_reference_candidate, comparator, f"best R1+R2 vs bootstrap PF {comparator['particle_count']}"))
    return claims


# jscpd:ignore-start
def _claim_row(
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
    comparison: str,
) -> dict[str, float | int | str | bool]:
    ref_ratio = _ratio(float(candidate["position_rmse_to_reference"]), float(comparator["position_rmse_to_reference"]))
    truth_ratio = _ratio(float(candidate["position_rmse_to_truth"]), float(comparator["position_rmse_to_truth"]))
    nees_ratio = _ratio(float(candidate["mean_nees_to_truth"]), float(comparator["mean_nees_to_truth"]))
    coverage_delta = float(candidate["coverage_95_to_truth"]) - float(comparator["coverage_95_to_truth"])
    orientation_delta = float(candidate["orientation_point_error_to_reference_rad"]) - float(comparator["orientation_point_error_to_reference_rad"])
    runtime_ratio = _ratio(float(candidate["runtime_ms_per_step"]), float(comparator["runtime_ms_per_step"]))
    supports_reference = ref_ratio < 1.0
    supports_truth_accuracy = truth_ratio <= 1.02
    supports_consistency = nees_ratio <= 1.1 and coverage_delta >= -0.05
    supports_orientation = orientation_delta <= 0.05
    supports_runtime = runtime_ratio <= 1.25
    return {
        "comparison": comparison,
        "candidate_filter": str(candidate["filter"]),
        "candidate_variant": str(candidate["variant"]),
        "candidate_grid_size": candidate["grid_size"],
        "candidate_particle_count": candidate["particle_count"],
        "comparator_filter": str(comparator["filter"]),
        "comparator_variant": str(comparator["variant"]),
        "comparator_grid_size": comparator["grid_size"],
        "comparator_particle_count": comparator["particle_count"],
        "candidate_position_rmse_to_reference": float(candidate["position_rmse_to_reference"]),
        "comparator_position_rmse_to_reference": float(comparator["position_rmse_to_reference"]),
        "position_rmse_to_reference_ratio": ref_ratio,
        "position_rmse_to_reference_gain_pct": 100.0 * (1.0 - ref_ratio),
        "candidate_position_rmse_to_truth": float(candidate["position_rmse_to_truth"]),
        "comparator_position_rmse_to_truth": float(comparator["position_rmse_to_truth"]),
        "position_rmse_to_truth_ratio": truth_ratio,
        "position_rmse_to_truth_gain_pct": 100.0 * (1.0 - truth_ratio),
        "candidate_mean_nees_to_truth": float(candidate["mean_nees_to_truth"]),
        "comparator_mean_nees_to_truth": float(comparator["mean_nees_to_truth"]),
        "mean_nees_ratio": nees_ratio,
        "candidate_coverage_95_to_truth": float(candidate["coverage_95_to_truth"]),
        "comparator_coverage_95_to_truth": float(comparator["coverage_95_to_truth"]),
        "coverage_delta": coverage_delta,
        "candidate_orientation_point_error_to_reference_rad": float(candidate["orientation_point_error_to_reference_rad"]),
        "comparator_orientation_point_error_to_reference_rad": float(comparator["orientation_point_error_to_reference_rad"]),
        "orientation_point_error_to_reference_delta_rad": orientation_delta,
        "candidate_runtime_ms_per_step": float(candidate["runtime_ms_per_step"]),
        "comparator_runtime_ms_per_step": float(comparator["runtime_ms_per_step"]),
        "runtime_ratio": runtime_ratio,
        "supports_reference_claim": supports_reference,
        "supports_truth_accuracy_claim": supports_truth_accuracy,
        "supports_consistency_claim": supports_consistency,
        "supports_orientation_reference_claim": supports_orientation,
        "supports_runtime_claim": supports_runtime,
        "supports_overall_claim": supports_reference and supports_truth_accuracy and supports_consistency and supports_orientation,
    }


def _ratio(candidate: float, comparator: float) -> float:
    return float(candidate / comparator) if comparator > 0.0 else float("inf")


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
# jscpd:ignore-end


def _write_metadata(
    path: Path,
    result: EuRoCS3R3ComparisonReportResult,
    groundtruth_path: Path,
    config: EuRoCS3R3ComparisonReportConfig,
) -> None:
    metadata = {
        "claims_rows": len(result.claims),
        "claims_schema": EUROC_S3R3_COMPARISON_CLAIM_FIELDNAMES,
        "config": euroc_s3r3_comparison_report_config_to_dict(config),
        "experiment": "euroc_s3r3_comparison_report",
        "groundtruth_path": str(groundtruth_path),
        "metrics_rows": len(result.metrics),
        "metrics_schema": EUROC_S3R3_COMPARISON_METRIC_FIELDNAMES,
        "reference_model": "denser_dynamic_baseline_s3f_on_same_euroc_controls",
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    groundtruth_path: Path,
    result: EuRoCS3R3ComparisonReportResult,
    metrics_path: Path,
    claims_path: Path,
    plot_paths: list[Path],
    config: EuRoCS3R3ComparisonReportConfig,
) -> None:
    baseline_claims = [claim for claim in result.claims if claim["comparison"] == "R1+R2 vs coarse baseline"]
    external_claims = [claim for claim in result.claims if claim["comparison"].startswith("best R1+R2 vs")]
    best_reference_row = min(
        [row for row in result.metrics if row["filter"] == S3F_FILTER and row["variant"] == R1_R2_VARIANT],
        key=lambda row: float(row["position_rmse_to_reference"]),
    )
    lines = [
        "# EuRoC S3+ x R3 Comparison Report",
        "",
        "This report uses EuRoC ground-truth 3D poses to derive body-frame translation controls and quaternion increments.",
        "Position measurements are simulated from ground truth, so this is a controlled trajectory-filter comparison rather than a visual-inertial frontend benchmark.",
        "The reference row is a denser dynamic baseline S3F run on the same controls and measurements.",
        "",
        f"Ground truth: `{groundtruth_path}`",
        f"Coarse grid sizes: {list(config.grid_sizes)}",
        f"Reference grid size: {config.reference_grid_size}",
        f"Steps: {config.max_steps}",
        f"Stride: {config.stride}",
        f"Cell sample count: {config.cell_sample_count}",
        f"Particle counts: {list(config.particle_counts)}",
        f"Metrics: `{metrics_path.name}`",
        f"Claims: `{claims_path.name}`",
        "",
        "## Headline",
        "",
        f"Best R1+R2 reference row: `{best_reference_row['grid_size']}` cells with reference RMSE `{float(best_reference_row['position_rmse_to_reference']):.4f}`.",
        f"R1+R2 is closer to the high-resolution reference than the coarse baseline in `{_support_count(baseline_claims, 'supports_reference_claim')}/{len(baseline_claims)}` grid rows.",
        f"Best R1+R2 beats external baselines on reference closeness in `{_support_count(external_claims, 'supports_reference_claim')}/{len(external_claims)}` comparison rows.",
        "",
        "## Baseline Rows",
        "",
        _format_claim_table(baseline_claims),
        "",
        "## External Rows",
        "",
        _format_claim_table(external_claims),
        "",
        "Plots:",
        format_plot_list(plot_paths),
        "",
        "This result is most useful as evidence about controlled 3D pose propagation and consistency. It does not claim full VIO performance.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _support_count(rows: list[dict[str, float | int | str | bool]], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def _format_claim_table(rows: list[dict[str, float | int | str | bool]]) -> str:
    if not rows:
        return "_No comparison rows._"
    header = "| Comparison | ref gain % | truth gain % | NEES ratio | coverage delta | runtime ratio | ref claim | overall |"
    separator = "|---|---:|---:|---:|---:|---:|---|---|"
    body = []
    for row in rows:
        body.append(
            "| "
            f"{row['comparison']} | "
            f"{float(row['position_rmse_to_reference_gain_pct']):.1f} | "
            f"{float(row['position_rmse_to_truth_gain_pct']):.1f} | "
            f"{float(row['mean_nees_ratio']):.3f} | "
            f"{float(row['coverage_delta']):.3f} | "
            f"{float(row['runtime_ratio']):.3f} | "
            f"{row['supports_reference_claim']} | "
            f"{row['supports_overall_claim']} |"
        )
    return "\n".join([header, separator, *body])


def _write_plots(
    output_dir: Path,
    metrics: list[dict[str, float | int | str]],
    claims: list[dict[str, float | int | str | bool]],
) -> list[Path]:
    return [
        _write_s3f_grid_plot(output_dir, metrics, "position_rmse_to_reference", "Position RMSE to High-Res Reference", "euroc_s3r3_reference_rmse.png"),
        _write_s3f_grid_plot(output_dir, metrics, "position_rmse_to_truth", "Position RMSE to Truth", "euroc_s3r3_truth_rmse.png"),
        _write_s3f_grid_plot(output_dir, metrics, "mean_nees_to_truth", "Mean Position NEES", "euroc_s3r3_nees.png"),
        _write_runtime_plot(output_dir, metrics),
        _write_claim_gain_plot(output_dir, claims),
    ]


def _write_s3f_grid_plot(output_dir: Path, rows: list[dict[str, float | int | str]], metric_name: str, y_label: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant in SUPPORTED_S3R3_VARIANTS:
        ordered = sorted([row for row in rows if row["filter"] == S3F_FILTER and row["variant"] == variant], key=lambda row: int(row["grid_size"]))
        if ordered:
            ax.plot([int(row["grid_size"]) for row in ordered], [float(row[metric_name]) for row in ordered], marker="o", linewidth=1.8, label=VARIANT_LABELS[variant])
    for row in rows:
        if row["filter"] in {REFERENCE_FILTER, UKF_FILTER, PARTICLE_FILTER}:
            label = _row_label(row)
            ax.axhline(float(row[metric_name]), linestyle="--", linewidth=1.0, alpha=0.7, label=label)
    ax.set_xlabel("Number of quaternion grid cells")
    ax.set_ylabel(y_label)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return save_figure(fig, output_dir, filename)


def _write_runtime_plot(output_dir: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant in SUPPORTED_S3R3_VARIANTS:
        ordered = sorted([row for row in rows if row["filter"] == S3F_FILTER and row["variant"] == variant], key=lambda row: int(row["grid_size"]))
        if ordered:
            ax.plot([int(row["resource_count"]) for row in ordered], [float(row["runtime_ms_per_step"]) for row in ordered], marker="o", linewidth=1.8, label=VARIANT_LABELS[variant])
    external = [row for row in rows if row["filter"] in {REFERENCE_FILTER, UKF_FILTER, PARTICLE_FILTER}]
    if external:
        ax.scatter(
            [float(row["resource_count"]) if str(row["resource_count"]) else 1.0 for row in external],
            [float(row["runtime_ms_per_step"]) for row in external],
            marker="s",
            label="reference/UKF/PF",
        )
    ax.set_xlabel("Grid cells or particles")
    ax.set_ylabel("Runtime [ms/step]")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return save_figure(fig, output_dir, "euroc_s3r3_runtime.png")


def _write_claim_gain_plot(output_dir: Path, claims: list[dict[str, float | int | str | bool]]) -> Path:
    baseline_claims = [claim for claim in claims if claim["comparison"] == "R1+R2 vs coarse baseline"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(
        [str(claim["candidate_grid_size"]) for claim in baseline_claims],
        [float(claim["position_rmse_to_reference_gain_pct"]) for claim in baseline_claims],
        color="#54A24B",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("R1+R2 grid cells")
    ax.set_ylabel("Reference RMSE gain vs baseline [%]")
    ax.grid(True, axis="y", alpha=0.3)
    return save_figure(fig, output_dir, "euroc_s3r3_reference_gain.png")


def _row_label(row: dict[str, float | int | str]) -> str:
    if row["filter"] == REFERENCE_FILTER:
        return EUROC_S3R3_COMPARISON_LABELS[REFERENCE_FILTER]
    if row["filter"] == UKF_FILTER:
        return EUROC_S3R3_COMPARISON_LABELS[UKF_FILTER]
    if row["filter"] == PARTICLE_FILTER:
        return f"{EUROC_S3R3_COMPARISON_LABELS[PARTICLE_VARIANT]} {row['particle_count']}"
    return VARIANT_LABELS[str(row["variant"])]
