"""
metrics.py
----------
Evaluation metrics for portfolio performance.

Spec §28.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def annualized_volatility(daily_returns: np.ndarray | List[float]) -> float:
    """Annualized realized volatility: sqrt(252) * std(r)."""
    r = np.asarray(daily_returns, dtype=float)
    return float(np.sqrt(252) * np.std(r, ddof=1))


def annualized_return(daily_returns: np.ndarray | List[float]) -> float:
    """Annualized compounded return: (prod(1+r_d))^(252/D) - 1."""
    r = np.asarray(daily_returns, dtype=float)
    D = len(r)
    if D == 0:
        return np.nan
    cumulative = np.prod(1.0 + r)
    if cumulative <= 0:
        return -1.0
    return float(cumulative ** (252.0 / D) - 1.0)


def sharpe_ratio(daily_returns: np.ndarray | List[float]) -> float:
    """Sharpe ratio assuming zero risk-free rate: sqrt(252) * mean(r) / std(r)."""
    r = np.asarray(daily_returns, dtype=float)
    s = np.std(r, ddof=1)
    if s < 1e-12:
        return np.nan
    return float(np.sqrt(252) * np.mean(r) / s)


def cvar_95(daily_returns: np.ndarray | List[float]) -> float:
    """
    Historical daily CVaR at 95% confidence.

    CVaR_95 = E[loss | loss >= VaR_95]
    loss_d = -r_d
    """
    r = np.asarray(daily_returns, dtype=float)
    losses = -r
    var_95 = np.percentile(losses, 95)
    tail = losses[losses >= var_95]
    if len(tail) == 0:
        return float(var_95)
    return float(np.mean(tail))


def maximum_drawdown(daily_returns: np.ndarray | List[float]) -> float:
    """
    Maximum drawdown: min_d (V_d / max_{tau<=d} V_tau - 1).

    Returns a negative number (or 0).
    """
    r = np.asarray(daily_returns, dtype=float)
    wealth = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min())


def herfindahl_hirschman_index(weights: np.ndarray) -> float:
    """Aggregate weight HHI = sum(w_i^2)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w ** 2))


def max_weight(weights: np.ndarray) -> float:
    """Maximum individual security weight."""
    return float(np.max(np.abs(weights)))


# ---------------------------------------------------------------------------
# Compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    daily_returns: np.ndarray | List[float],
    weights_per_rebalance: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute the full set of test reporting metrics.

    Parameters
    ----------
    daily_returns : gross daily return series
    weights_per_rebalance : list of aggregate weight vectors (one per rebalance date)

    Returns
    -------
    dict of metric name -> float value
    """
    r = np.asarray(daily_returns, dtype=float)

    metrics = {
        "annualized_volatility_gross": annualized_volatility(r),
        "annualized_return_gross": annualized_return(r),
        "sharpe_gross": sharpe_ratio(r),
        "cvar_95_daily_gross": cvar_95(r),
        "maximum_drawdown_gross": maximum_drawdown(r),
    }

    if weights_per_rebalance is not None and len(weights_per_rebalance) > 0:
        avg_max_wt = np.mean([max_weight(w) for w in weights_per_rebalance])
        avg_hhi = np.mean([herfindahl_hirschman_index(w) for w in weights_per_rebalance])
        metrics["average_max_stock_weight"] = float(avg_max_wt)
        metrics["average_weight_hhi"] = float(avg_hhi)
    else:
        metrics["average_max_stock_weight"] = np.nan
        metrics["average_weight_hhi"] = np.nan

    return metrics


def volatility_reduction_vs_sample_gmv(
    proposed_vol: float,
    sample_gmv_vol: float,
) -> float:
    """
    Volatility reduction: 1 - sigma_proposed / sigma_sample_gmv.
    Positive means proposed method has lower volatility.
    """
    if sample_gmv_vol < 1e-12:
        return np.nan
    return float(1.0 - proposed_vol / sample_gmv_vol)
