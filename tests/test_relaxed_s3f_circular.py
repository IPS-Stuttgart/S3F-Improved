import csv
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter

from se3plusplus_s3f.wp1.relaxed_s3f_circular import (
    predict_circular_relaxed,
    rotate_body_increment,
    uniform_circular_cell_statistics,
)
from se3plusplus_s3f.wp1.relaxed_s3f_pilot import (
    PilotConfig,
    load_pilot_config,
    run_relaxed_s3f_pilot,
    write_relaxed_s3f_pilot_outputs,
)


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
REFERENCE_METRICS = [
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_mean_error_rad",
    "mean_nees",
    "coverage_95",
]


def test_covariance_inflation_is_positive_semidefinite():
    stats = uniform_circular_cell_statistics(12, np.array([0.7, -0.2]))

    for cov in stats.covariance_inflations:
        eigvals = np.linalg.eigvalsh(cov)
        assert float(eigvals[0]) >= -1e-12


def test_closed_form_statistics_match_deterministic_quadrature():
    n_cells = 9
    cell_idx = 3
    body_increment = np.array([0.6, 0.15])
    stats = uniform_circular_cell_statistics(n_cells, body_increment)
    center = stats.grid[cell_idx]
    half_width = 0.5 * stats.cell_width
    samples = np.linspace(center - half_width, center + half_width, 20001)
    rotated = rotate_body_increment(samples, body_increment)

    mean_quad = rotated.mean(axis=0)
    cov_quad = np.cov(rotated.T, bias=True)

    npt.assert_allclose(stats.mean_displacements[cell_idx], mean_quad, atol=1e-5)
    npt.assert_allclose(stats.covariance_inflations[cell_idx], cov_quad, atol=1e-5)


def test_prediction_and_update_conserve_grid_mass():
    filter_ = _make_filter(10)

    predict_circular_relaxed(
        filter_,
        np.array([0.4, 0.1]),
        variant="r1_r2",
        process_noise_cov=np.eye(2) * 0.01,
    )
    npt.assert_allclose(float(filter_.filter_state.gd.integrate()), 1.0, atol=1e-12)

    filter_.update(
        likelihoods_linear=[
            GaussianDistribution(
                np.array([0.2, 0.1]),
                np.eye(2) * 0.05,
                check_validity=False,
            )
        ]
    )
    npt.assert_allclose(float(filter_.filter_state.gd.integrate()), 1.0, atol=1e-12)


def test_relaxations_vanish_as_grid_resolution_increases():
    body_increment = np.array([0.8, -0.1])
    coarse = uniform_circular_cell_statistics(8, body_increment)
    fine = uniform_circular_cell_statistics(64, body_increment)

    coarse_mean_gap = np.linalg.norm(
        coarse.mean_displacements - coarse.representative_displacements,
        axis=1,
    ).max()
    fine_mean_gap = np.linalg.norm(
        fine.mean_displacements - fine.representative_displacements,
        axis=1,
    ).max()
    coarse_cov_norm = np.linalg.norm(coarse.covariance_inflations, axis=(1, 2)).max()
    fine_cov_norm = np.linalg.norm(fine.covariance_inflations, axis=(1, 2)).max()

    assert fine_mean_gap < coarse_mean_gap / 20.0
    assert fine_cov_norm < coarse_cov_norm / 20.0


def test_pilot_runner_smoke(tmp_path):
    config = PilotConfig(grid_sizes=(8,), n_trials=1, n_steps=2)
    rows = run_relaxed_s3f_pilot(config)
    assert len(rows) == 3

    outputs = write_relaxed_s3f_pilot_outputs(tmp_path, config, write_plots=False)
    assert outputs["metrics"].exists()
    assert outputs["note"].exists()
    assert outputs["metadata"].exists()

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "wp1_s1_r2_relaxed_s3f"
    assert metadata["config"]["grid_sizes"] == [8]
    assert metadata["metrics_rows"] == 3


def test_wp1_reference_metrics_match_committed_reference():
    config = load_pilot_config(REFERENCE_DIR / "wp1_relaxed_s3f_regression_config.json")
    with (REFERENCE_DIR / "wp1_relaxed_s3f_regression_metrics.csv").open(newline="", encoding="utf-8") as file:
        expected_rows = list(csv.DictReader(file))
    actual_rows = run_relaxed_s3f_pilot(config)

    actual_by_key = {
        (int(row["grid_size"]), str(row["variant"])): row
        for row in actual_rows
    }
    expected_by_key = {
        (int(row["grid_size"]), str(row["variant"])): row
        for row in expected_rows
    }
    assert set(actual_by_key) == set(expected_by_key)

    for key, expected_row in expected_by_key.items():
        actual_row = actual_by_key[key]
        for metric in REFERENCE_METRICS:
            npt.assert_allclose(
                float(actual_row[metric]),
                float(expected_row[metric]),
                rtol=1e-10,
                atol=1e-10,
            )


def _make_filter(n_cells: int) -> StateSpaceSubdivisionFilter:
    grid = np.linspace(0.0, 2.0 * np.pi, n_cells, endpoint=False).reshape(-1, 1)
    gd = HypertoroidalGridDistribution(
        np.ones(n_cells) / (2.0 * np.pi),
        grid_type="custom",
        grid=grid,
    )
    gd.normalize_in_place(warn_unnorm=False)
    gaussians = [
        GaussianDistribution(np.zeros(2), np.eye(2) * 0.1, check_validity=False)
        for _ in range(n_cells)
    ]
    state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
    return StateSpaceSubdivisionFilter(state)
