"""
turnover.py
-----------
Portfolio turnover computation and transaction-cost adjustment.

Spec §23.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_drifted_weights(
    weights: Dict[int, float],
    returns: np.ndarray,
    permnos: List[int],
) -> Dict[int, float]:
    """
    Compute portfolio weights after a holding period of *returns* (T days).

    Uses the standard buy-and-hold weight drift:
      w_i^post = w_i * (1 + R_i) / (1 + R_p)
    where R_i is the cumulative return of asset i and R_p is the portfolio return.

    Parameters
    ----------
    weights : {permno: weight} at formation date
    returns : (T, N) daily return matrix, columns ordered by *permnos*
    permnos : list of N permno identifiers matching columns of *returns*

    Returns
    -------
    {permno: drifted_weight} at the end of the holding period
    """
    n = len(permnos)
    perm_to_idx = {p: i for i, p in enumerate(permnos)}

    w = np.array([weights.get(p, 0.0) for p in permnos], dtype=float)

    # Cumulative returns
    cum_ret = np.prod(1.0 + returns, axis=0) - 1.0  # (N,)
    portfolio_cum = (w * (1.0 + cum_ret)).sum() - 1.0  # scalar

    if abs(portfolio_cum + 1.0) < 1e-12:
        # Portfolio went to zero – unlikely but guard
        return {p: 1.0 / n for p in permnos}

    drifted = w * (1.0 + cum_ret) / (1.0 + portfolio_cum)
    return {p: float(drifted[i]) for i, p in enumerate(permnos)}


def compute_turnover(
    target_weights: Dict[int, float],
    pretrade_weights: Dict[int, float],
) -> float:
    """
    One-way portfolio turnover as the sum of absolute weight changes
    over the union of holdings.

    Spec §23.1.

    Parameters
    ----------
    target_weights : {permno: new_weight}
    pretrade_weights : {permno: pre-trade drifted weight}

    Returns
    -------
    float – one-way turnover in [0, 2]
    """
    all_permnos = set(target_weights.keys()) | set(pretrade_weights.keys())
    turnover = sum(
        abs(target_weights.get(p, 0.0) - pretrade_weights.get(p, 0.0))
        for p in all_permnos
    )
    return float(turnover)


def apply_transaction_costs(
    daily_returns: List[float],
    rebalance_indices: List[int],
    turnovers: List[float],
    cost_bps: float = 10.0,
) -> List[float]:
    """
    Subtract transaction cost from the first daily return after each rebalance.

    cost_t = cost_bps / 10000 * turnover_t

    Spec §23.2.

    Parameters
    ----------
    daily_returns : list of gross daily returns
    rebalance_indices : index in daily_returns corresponding to the first day
                        of each holding period
    turnovers : list of turnover values (one per rebalance)
    cost_bps : basis points per unit of one-way turnover

    Returns
    -------
    List of net daily returns
    """
    net = list(daily_returns)
    cost_rate = cost_bps / 10_000.0

    for idx, to in zip(rebalance_indices, turnovers):
        if idx < len(net):
            net[idx] = net[idx] - cost_rate * to

    return net
