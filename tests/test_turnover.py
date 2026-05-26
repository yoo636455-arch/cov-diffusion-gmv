"""
test_turnover.py
-----------------
Tests for turnover computation (spec §35.10 / §23).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.turnover import (
    apply_transaction_costs,
    compute_drifted_weights,
    compute_turnover,
)


def test_zero_turnover_same_weights():
    w = {1: 0.5, 2: 0.5}
    to = compute_turnover(w, w)
    assert abs(to) < 1e-12


def test_full_turnover_disjoint_portfolios():
    """Selling everything and buying everything new = 2× turnover."""
    old = {1: 0.6, 2: 0.4}
    new = {3: 0.7, 4: 0.3}
    to = compute_turnover(new, old)
    # Old weights set to 0 in new, new weights set to 0 in old
    expected = 0.6 + 0.4 + 0.7 + 0.3  # = 2.0
    assert abs(to - expected) < 1e-10


def test_partial_turnover():
    old = {1: 0.5, 2: 0.5}
    new = {1: 0.3, 2: 0.7}
    to = compute_turnover(new, old)
    # |0.3-0.5| + |0.7-0.5| = 0.2 + 0.2 = 0.4
    assert abs(to - 0.4) < 1e-10


def test_compute_drifted_weights():
    """Equal weights with equal returns -> weights stay equal."""
    weights = {1: 0.5, 2: 0.5}
    # Returns: same for both assets
    returns = np.array([[0.01, 0.01], [0.02, 0.02]])  # (2, 2)
    permnos = [1, 2]
    drifted = compute_drifted_weights(weights, returns, permnos)
    assert abs(drifted[1] - 0.5) < 1e-6
    assert abs(drifted[2] - 0.5) < 1e-6


def test_compute_drifted_weights_asymmetric():
    """Asset 1 doubles in value, asset 2 stays flat."""
    weights = {1: 0.5, 2: 0.5}
    returns = np.array([[1.0, 0.0]])  # (1 day, 2 assets)
    permnos = [1, 2]
    drifted = compute_drifted_weights(weights, returns, permnos)
    # Asset 1: 0.5 * 2.0 / portfolio_value = 1.0 / 1.5
    # Asset 2: 0.5 * 1.0 / 1.5
    assert abs(drifted[1] - (1.0 / 1.5)) < 1e-8
    assert abs(drifted[2] - (0.5 / 1.5)) < 1e-8


def test_apply_transaction_costs():
    returns = [0.01, 0.02, 0.03, 0.04]
    # Rebalance at index 0 and index 2
    net = apply_transaction_costs(returns, [0, 2], [0.5, 0.3], cost_bps=10.0)
    # Cost at idx 0: 10/10000 * 0.5 = 0.0005
    assert abs(net[0] - (0.01 - 0.0005)) < 1e-10
    # Cost at idx 2: 10/10000 * 0.3 = 0.0003
    assert abs(net[2] - (0.03 - 0.0003)) < 1e-10
    # Unchanged days
    assert abs(net[1] - 0.02) < 1e-10
    assert abs(net[3] - 0.04) < 1e-10
