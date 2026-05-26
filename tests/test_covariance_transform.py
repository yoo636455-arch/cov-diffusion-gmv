"""
test_covariance_transform.py
-----------------------------
Tests for SPD covariance round-trip (spec §35.5).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transforms import covariance_to_log_vech, log_vech_to_covariance


def _random_spd(n=10, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + np.eye(n) * 1e-4


def test_round_trip_reconstruction():
    """vech -> ivech -> expm should recover the ridge-stabilized covariance."""
    for seed in range(5):
        S = _random_spd(seed=seed)
        eps = 1e-8
        vec = covariance_to_log_vech(S, ridge_epsilon=eps)
        S_rec = log_vech_to_covariance(vec)

        S_expected = S + eps * np.eye(10)
        assert np.allclose(S_expected, S_rec, atol=1e-6), (
            f"Round-trip mismatch at seed={seed}: max error {np.abs(S_expected - S_rec).max():.2e}"
        )


def test_output_vector_dimension():
    S = _random_spd()
    vec = covariance_to_log_vech(S)
    assert vec.shape == (55,), f"Expected (55,), got {vec.shape}"


def test_reconstruction_is_spd():
    """Reconstructed matrix must be symmetric positive definite."""
    for seed in range(10):
        S = _random_spd(seed=seed)
        vec = covariance_to_log_vech(S)
        S_rec = log_vech_to_covariance(vec)

        eigvals = np.linalg.eigvalsh(S_rec)
        assert eigvals.min() > 0, f"Reconstructed matrix is not PD: min eig = {eigvals.min():.2e}"
        assert np.allclose(S_rec, S_rec.T, atol=1e-10), "Reconstructed matrix is not symmetric"


def test_non_pd_raises():
    """A matrix with non-positive eigenvalue should fail after ridge jitter is too small."""
    S_bad = np.zeros((10, 10))  # singular
    with pytest.raises(Exception):
        # ridge_epsilon=0 should fail for singular matrix
        covariance_to_log_vech(S_bad, ridge_epsilon=0.0)
