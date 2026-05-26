"""
backtest.py
-----------
Core backtest engine: aggregate sleeve-level GMV into an equal-weighted
portfolio and compute realized returns.

Used for both validation and test.

Spec §22, §33, §34.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .covariance import get_future_returns
from .turnover import compute_drifted_weights, compute_turnover

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single rebalance step
# ---------------------------------------------------------------------------

def run_one_rebalance(
    sleeve_weights: Dict[int, np.ndarray],
    sleeve_permnos: Dict[int, List[int]],
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    rebalance_date: pd.Timestamp,
    horizon_days: int = 21,
    previous_drifted_weights: Optional[Dict[int, float]] = None,
    repair_log: Optional[list] = None,
) -> Tuple[
    List[float],          # gross daily returns
    Dict[int, float],     # aggregate weights (permno -> weight)
    Dict[int, float],     # end-of-period drifted weights
    float,                # turnover
]:
    """
    Run one rebalance step.

    Parameters
    ----------
    sleeve_weights : {sleeve_id: (N,) weight array}
    sleeve_permnos : {sleeve_id: [permno, ...]}  ordered by position
    crsp_df : cleaned CRSP panel
    trading_dates : full trading calendar
    rebalance_date : formation date t
    horizon_days : 21
    previous_drifted_weights : {permno: weight} from previous period (or None)

    Returns
    -------
    (daily_returns, agg_weights, drifted_weights, turnover)
    """
    n_sleeves = len(sleeve_weights)
    if n_sleeves == 0:
        return [], {}, {}, 0.0

    sleeve_capital = 1.0 / n_sleeves

    # Build aggregate weights
    agg_weights: Dict[int, float] = {}
    for sid, w_arr in sleeve_weights.items():
        permnos = sleeve_permnos[sid]
        for p, w in zip(permnos, w_arr):
            agg_weights[p] = agg_weights.get(p, 0.0) + sleeve_capital * float(w)

    # Normalize (should already sum to 1)
    total_w = sum(agg_weights.values())
    if abs(total_w - 1.0) > 1e-6:
        logger.warning("Aggregate weights sum to %.6f; renormalizing.", total_w)
        agg_weights = {p: w / total_w for p, w in agg_weights.items()}

    # Compute turnover
    if previous_drifted_weights is not None:
        turnover = compute_turnover(agg_weights, previous_drifted_weights)
    else:
        # First rebalance: turnover = sum of absolute new weights (full buy)
        turnover = sum(abs(w) for w in agg_weights.values())

    # Future returns for each permno
    all_permnos = list(agg_weights.keys())
    from .covariance import compute_return_matrix
    from .calendar import get_horizon_dates
    try:
        fwd_dates = get_horizon_dates(trading_dates, rebalance_date, horizon_days)
    except ValueError as exc:
        logger.warning("Horizon dates error at %s: %s", rebalance_date.date(), exc)
        return [], agg_weights, {}, turnover

    panel = crsp_df[
        crsp_df["date"].isin(fwd_dates) & crsp_df["permno"].isin(all_permnos)
    ][["date", "permno", "ret_total"]].copy()

    pivot = panel.pivot(index="date", columns="permno", values="ret_total")
    pivot = pivot.reindex(index=fwd_dates, columns=all_permnos)

    # Any missing returns -> forward-fill with 0 (log the event)
    missing = pivot.isna().sum().sum()
    if missing > 0:
        logger.debug(
            "Forward-filling %d missing returns at %s", missing, rebalance_date.date()
        )
        pivot = pivot.fillna(0.0)

    returns_matrix = pivot.values  # (horizon_days, len(all_permnos))
    w_arr = np.array([agg_weights[p] for p in all_permnos])

    # Daily portfolio returns
    daily_portfolio_returns = (returns_matrix @ w_arr).tolist()

    # Drifted weights at end of period
    drifted = compute_drifted_weights(
        weights=agg_weights,
        returns=returns_matrix,
        permnos=all_permnos,
    )

    return daily_portfolio_returns, agg_weights, drifted, turnover


# ---------------------------------------------------------------------------
# Full backtest loop
# ---------------------------------------------------------------------------

def run_full_backtest(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    rebalance_dates: List[pd.Timestamp],
    sleeve_weight_fn,
    horizon_days: int = 21,
    transaction_cost_bps: float = 10.0,
) -> Dict:
    """
    Run a complete backtest given a callable *sleeve_weight_fn*.

    Parameters
    ----------
    crsp_df : cleaned CRSP panel
    trading_dates : full trading calendar
    rebalance_dates : sorted list of rebalance dates for the split
    sleeve_weight_fn : callable(rebalance_date) -> (sleeve_weights, sleeve_permnos)
        Returns ({sleeve_id: weights_array}, {sleeve_id: [permno, ...]})
    horizon_days : 21
    transaction_cost_bps : float

    Returns
    -------
    Dict with keys:
        'gross_returns'     : list of daily gross returns
        'net_returns'       : list of daily net returns (10 bps default)
        'weights'           : list of (rebalance_date, agg_weights_dict)
        'turnovers'         : list of (rebalance_date, turnover)
        'rebalance_indices' : list of indices into gross_returns where each
                              new holding period starts
    """
    all_gross_returns: List[float] = []
    all_weights: List[Tuple] = []
    all_turnovers: List[Tuple] = []
    rebalance_indices: List[int] = []
    repair_log: List[dict] = []

    previous_drifted = None

    for t in rebalance_dates:
        try:
            sleeve_wts, sleeve_perms = sleeve_weight_fn(t)
        except Exception as exc:
            logger.warning(
                "sleeve_weight_fn failed at %s: %s. Skipping.", t.date(), exc
            )
            continue

        if not sleeve_wts:
            logger.warning("No sleeves at %s; skipping.", t.date())
            continue

        daily_rets, agg_wts, drifted, to = run_one_rebalance(
            sleeve_weights=sleeve_wts,
            sleeve_permnos=sleeve_perms,
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            rebalance_date=t,
            horizon_days=horizon_days,
            previous_drifted_weights=previous_drifted,
            repair_log=repair_log,
        )

        if not daily_rets:
            continue

        rebalance_indices.append(len(all_gross_returns))
        all_gross_returns.extend(daily_rets)
        all_weights.append((t, agg_wts))
        all_turnovers.append((t, to))
        previous_drifted = drifted

    # Apply transaction costs
    from .turnover import apply_transaction_costs
    all_net_returns = apply_transaction_costs(
        daily_returns=all_gross_returns,
        rebalance_indices=rebalance_indices,
        turnovers=[to for _, to in all_turnovers],
        cost_bps=transaction_cost_bps,
    )

    return {
        "gross_returns": all_gross_returns,
        "net_returns": all_net_returns,
        "weights": all_weights,
        "turnovers": all_turnovers,
        "rebalance_indices": rebalance_indices,
        "repair_log": repair_log,
    }
