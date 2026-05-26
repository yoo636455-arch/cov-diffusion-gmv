"""
test_covariance_combination.py
--------------------------------
Tests for combine_covariances (spec §35.8).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate import combine_covariances


def _random_spd(n=10, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + np.eye(n) * 1e-3


@pytest.fixture
def sample_cov():
    return _random_spd(seed=42)


@pytest.fixture
def scenarios():
    return [_random_spd(seed=i) for i in range(5)]


def test_alpha_zero_returns_mean_of_scenarios(sample_cov, scenarios):
    combined = combine_covariances(sample_cov, scenarios, alpha=0.0)
    expected = np.mean(np.stack(scenarios, axis=0), axis=0)
    assert np.allclose(combined, expected, atol=1e-8), (
        f"alpha=0 should return mean of scenarios"
    )


def test_alpha_one_returns_sample_cov(sample_cov, scenarios):
    combined = combine_covariances(sample_cov, scenarios, alpha=1.0)
    assert np.allclose(combined, sample_cov, atol=1e-8), (
        f"alpha=1 should return sample_cov"
    )


def test_alpha_half_is_equal_blend(sample_cov, scenarios):
    combined = combine_covariances(sample_cov, scenarios, alpha=0.5)
    diff_mean = np.mean(np.stack(scenarios, axis=0), axis=0)
    expected = 0.5 * sample_cov + 0.5 * diff_mean
    assert np.allclose(combined, expected, atol=1e-8)


def test_output_is_symmetric(sample_cov, scenarios):
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        combined = combine_covariances(sample_cov, scenarios, alpha=alpha)
        assert np.allclose(combined, combined.T, atol=1e-10), (
            f"Output not symmetric for alpha={alpha}"
        )


def test_invalid_alpha_raises(sample_cov, scenarios):
    with pytest.raises(ValueError):
        combine_covariances(sample_cov, scenarios, alpha=1.5)
    with pytest.raises(ValueError):
        combine_covariances(sample_cov, scenarios, alpha=-0.1)


def test_averaging_in_covariance_not_log_space(sample_cov, scenarios):
    """
    Verify that we average in covariance space, not log-covariance space.
    (Spec §20.1 strict rule.)
    """
    combined = combine_covariances(sample_cov, scenarios, alpha=0.0)
    direct_mean = np.mean(np.stack(scenarios, axis=0), axis=0)
    assert np.allclose(combined, direct_mean, atol=1e-8)
