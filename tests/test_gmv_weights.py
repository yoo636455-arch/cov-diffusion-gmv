"""
test_gmv_weights.py
--------------------
Tests for long-only GMV optimization (spec §35.9).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gmv import solve_long_only_gmv


def _random_spd(n=10, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + np.eye(n) * 1e-2


def test_weights_sum_to_one():
    for seed in range(5):
        cov = _random_spd(seed=seed)
        w = solve_long_only_gmv(cov)
        assert abs(w.sum() - 1.0) < 1e-5, f"Weights sum {w.sum()} != 1 at seed={seed}"


def test_weights_non_negative():
    for seed in range(5):
        cov = _random_spd(seed=seed)
        w = solve_long_only_gmv(cov)
        assert w.min() >= -1e-7, f"Negative weight {w.min()} at seed={seed}"


def test_output_shape():
    n = 10
    cov = _random_spd(n=n)
    w = solve_long_only_gmv(cov)
    assert w.shape == (n,)


def test_equal_covariance_gives_equal_weight():
    """Identity covariance -> uniform weight (all assets identical risk)."""
    cov = np.eye(10)
    w = solve_long_only_gmv(cov)
    expected = np.full(10, 0.1)
    assert np.allclose(w, expected, atol=1e-4), f"Weights {w} != 0.1 for identity covariance"


def test_nearly_singular_covariance_handled():
    """Near-singular covariance should not crash (jitter is applied)."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(10, 1))
    cov = v @ v.T  # rank-1 matrix (singular)
    # Should not raise
    w = solve_long_only_gmv(cov)
    assert abs(w.sum() - 1.0) < 1e-5
    assert w.min() >= -1e-7
