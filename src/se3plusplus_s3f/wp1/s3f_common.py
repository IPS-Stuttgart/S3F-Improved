"""Shared WP1 S3F utilities used by the synthetic and EuRoC smoke runs."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from pyrecest.filters.relaxed_s3f_circular import (
    circular_weighted_mean,
    grid_probability_masses,
    predict_circular_relaxed,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter


def make_s3f_filter(
    grid: np.ndarray,
    grid_values: np.ndarray,
    linear_mean: np.ndarray,
    linear_covariance: np.ndarray,
) -> StateSpaceSubdivisionFilter:
    """Construct an S3F state with one Gaussian linear component per grid cell."""

    grid = np.asarray(grid, dtype=float).reshape(-1)
    grid_values = np.asarray(grid_values, dtype=float).reshape(-1)
    mean = np.asarray(linear_mean, dtype=float).reshape(-1)
    covariance = np.asarray(linear_covariance, dtype=float)

    if grid.shape != grid_values.shape:
        raise ValueError("grid and grid_values must have matching shapes.")

    gd = HypertoroidalGridDistribution(
        grid_values,
        grid_type="custom",
        grid=grid.reshape(-1, 1),
        enforce_pdf_nonnegative=True,
    )
    gd.normalize_in_place(warn_unnorm=False)

    gaussians = [
        GaussianDistribution(mean.copy(), covariance.copy(), check_validity=False)
        for _ in range(grid.shape[0])
    ]
    state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
    return StateSpaceSubdivisionFilter(state)


def make_linear_likelihood(
    mean: np.ndarray,
    covariance: np.ndarray,
) -> GaussianDistribution:
    """Create a Gaussian likelihood for a linear position measurement."""

    return GaussianDistribution(
        np.asarray(mean, dtype=float),
        np.asarray(covariance, dtype=float),
        check_validity=False,
    )


def predict_update_linear_position(
    filter_: StateSpaceSubdivisionFilter,
    body_increment: np.ndarray,
    variant: str,
    process_noise_cov: np.ndarray,
    likelihood: GaussianDistribution,
) -> float:
    """Run one relaxed circular prediction and linear likelihood update."""

    start = perf_counter()
    predict_circular_relaxed(
        filter_,
        body_increment,
        variant=variant,
        process_noise_cov=process_noise_cov,
    )
    filter_.update(likelihoods_linear=[likelihood])
    return perf_counter() - start


def linear_position_error_stats(
    filter_: StateSpaceSubdivisionFilter,
    true_position: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return linear position error and NEES for the current filter state."""

    state = filter_.filter_state
    mean = np.asarray(state.linear_mean(), dtype=float)
    covariance = np.asarray(state.linear_covariance(), dtype=float)
    error = mean - np.asarray(true_position, dtype=float)
    covariance_reg = covariance + 1e-10 * np.eye(error.shape[0])
    nees = float(error @ np.linalg.solve(covariance_reg, error))
    return error, nees


def linear_position_mean(filter_: StateSpaceSubdivisionFilter) -> np.ndarray:
    """Return the current linear position mean."""

    return np.asarray(filter_.filter_state.linear_mean(), dtype=float)


def orientation_mode_and_mean(filter_: StateSpaceSubdivisionFilter) -> tuple[float, float]:
    """Return the modal grid angle and circular mean angle."""

    state = filter_.filter_state
    weights = grid_probability_masses(state.gd.grid_values)
    grid = np.asarray(state.gd.get_grid(), dtype=float).reshape(-1)
    mode = float(grid[int(np.argmax(weights))])
    mean = circular_weighted_mean(grid, weights)
    return mode, mean
