"""
test_beta_schedules.py
-----------------------
Tests for beta schedule factory (spec §35.6).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beta_schedules import compute_alpha_bar, make_beta_schedule

BETA_MIN = 1e-4
BETA_MAX = 0.02


@pytest.mark.parametrize("schedule_type", ["linear", "quadratic", "logarithmic"])
@pytest.mark.parametrize("T", [25, 50, 100])
class TestBetaSchedules:
    def test_correct_length(self, schedule_type, T):
        betas = make_beta_schedule(schedule_type, T)
        assert betas.shape == (T,), f"Expected ({T},), got {betas.shape}"

    def test_values_strictly_in_01(self, schedule_type, T):
        betas = make_beta_schedule(schedule_type, T)
        assert np.all(betas > 0), "Some betas <= 0"
        assert np.all(betas < 1), "Some betas >= 1"

    def test_first_value_near_beta_min(self, schedule_type, T):
        betas = make_beta_schedule(schedule_type, T)
        assert abs(betas[0] - BETA_MIN) < 1e-7, f"First beta {betas[0]} != {BETA_MIN}"

    def test_last_value_near_beta_max(self, schedule_type, T):
        betas = make_beta_schedule(schedule_type, T)
        assert abs(betas[-1] - BETA_MAX) < 1e-7, f"Last beta {betas[-1]} != {BETA_MAX}"


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        make_beta_schedule("cosine", 50)


def test_T_less_than_2_raises():
    with pytest.raises(ValueError, match="T must be at least 2"):
        make_beta_schedule("linear", 1)


def test_alpha_bar_is_decreasing():
    """alpha_bar should be monotonically decreasing (more noise over time)."""
    betas = make_beta_schedule("linear", 100)
    ab = compute_alpha_bar(betas)
    assert ab.shape == (100,)
    assert np.all(ab[:-1] >= ab[1:]), "alpha_bar is not monotonically decreasing"


def test_alpha_bar_first_and_last():
    betas = make_beta_schedule("linear", 50)
    ab = compute_alpha_bar(betas)
    # First: 1 - beta_1
    expected_first = 1.0 - betas[0]
    assert abs(ab[0] - expected_first) < 1e-12
    # Last: product of all alphas
    expected_last = np.prod(1.0 - betas)
    assert abs(ab[-1] - expected_last) < 1e-12
