"""
covariance.py
-------------
Compute historical (conditioning) and future (target) sample covariance matrices
for each sleeve / group at each rebalance date.

Spec §12.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .calendar import get_lookback_dates, get_horizon_dates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core covariance computation
# ---------------------------------------------------------------------------

def compute_return_matrix(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    permnos: list[int],
    rebalance_date: pd.Timestamp,
    window_dates: pd.DatetimeIndex,
) -> Optional[np.ndarray]:
    """
    Build a (T × N) return matrix for the given permnos over the specified dates.

    Returns None if any return is missing for any permno on any date.

    Accepts either a plain DataFrame (filtered via boolean mask) or a DataFrame
    pre-indexed on (date, permno) for fast O(log n) lookups.
    """
    if isinstance(crsp_df.index, pd.MultiIndex):
        # Fast path: pre-indexed on (date, permno)
        try:
            rows = crsp_df.loc[(list(window_dates), permnos), "ret_total"]
        except KeyError:
            return None
        pivot = rows.unstack(level="permno")
        pivot = pivot.reindex(index=window_dates, columns=permnos)
    else:
        # Slow path: full scan (kept for backward compat)
        panel = crsp_df[
            crsp_df["date"].isin(window_dates) & crsp_df["permno"].isin(permnos)
        ][["date", "permno", "ret_total"]].copy()
        pivot = panel.pivot(index="date", columns="permno", values="ret_total")
        pivot = pivot.reindex(index=window_dates, columns=permnos)

    if pivot.isna().any().any():
        logger.debug(
            "Missing returns at %s: %d cells",
            rebalance_date.date(), pivot.isna().sum().sum(),
        )
        return None

    return pivot.values.astype(np.float64)


def compute_sample_covariance(
    return_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the sample covariance matrix from a (T × N) return matrix.

    Uses np.cov which divides by T-1 (unbiased).

    Returns
    -------
    np.ndarray of shape (N, N)
    """
    # np.cov expects (N, T), so transpose
    return np.cov(return_matrix.T)


def build_covariance_pair(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    permnos: list[int],
    rebalance_date: pd.Timestamp,
    lookback_days: int = 126,
    horizon_days: int = 21,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build the (conditioning covariance, target covariance) pair for one sleeve.

    Parameters
    ----------
    crsp_df : cleaned CRSP panel
    trading_dates : full trading calendar
    permnos : ordered list of N PERMNOs for the sleeve
    rebalance_date : the formation date t
    lookback_days : lookback window T_hist (default 126)
    horizon_days : holding-period window T_fwd (default 21)

    Returns
    -------
    Tuple (S_hist, S_fwd, R_hist, R_fwd) or None on failure.
      S_hist : (N, N) historical 126-day sample covariance
      S_fwd  : (N, N) future 21-day realized covariance proxy
      R_hist : (lookback_days, N) historical return matrix
      R_fwd  : (horizon_days,  N) future return matrix
    """
    # ---- lookback returns ---------------------------------------------------
    try:
        lb_dates = get_lookback_dates(trading_dates, rebalance_date, lookback_days)
    except ValueError as exc:
        logger.debug("Lookback failed at %s: %s", rebalance_date.date(), exc)
        return None

    R_hist = compute_return_matrix(
        crsp_df, trading_dates, permnos, rebalance_date, lb_dates
    )
    if R_hist is None:
        logger.debug(
            "Incomplete historical returns for sleeve at %s", rebalance_date.date()
        )
        return None

    # ---- future returns ----------------------------------------------------
    try:
        fwd_dates = get_horizon_dates(trading_dates, rebalance_date, horizon_days)
    except ValueError as exc:
        logger.debug("Horizon failed at %s: %s", rebalance_date.date(), exc)
        return None

    R_fwd = compute_return_matrix(
        crsp_df, trading_dates, permnos, rebalance_date, fwd_dates
    )
    if R_fwd is None:
        logger.debug(
            "Incomplete future returns for sleeve at %s", rebalance_date.date()
        )
        return None

    S_hist = compute_sample_covariance(R_hist)
    S_fwd = compute_sample_covariance(R_fwd)

    return S_hist, S_fwd, R_hist, R_fwd


# ---------------------------------------------------------------------------
# Daily-sliding window for training data augmentation
# ---------------------------------------------------------------------------

def build_daily_sliding_pairs_for_group(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    permnos: list[int],
    anchor_indices: list[int],
    lookback_days: int = 126,
    horizon_days: int = 21,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    For a FIXED 10-stock group, compute (S_hist, S_fwd) pairs at every
    anchor date in *anchor_indices* using a SINGLE return-matrix pivot
    (vectorised — not one pandas call per date).

    Parameters
    ----------
    crsp_df         : cleaned CRSP panel
    trading_dates   : full sorted trading calendar (DatetimeIndex)
    permnos         : the group's 10 permnos in descending-ME order
    anchor_indices  : integer positions in *trading_dates* to use as
                      anchor dates (pre-filtered by caller to valid range)
    lookback_days   : 126
    horizon_days    : 21

    Returns
    -------
    List of (S_hist, S_fwd) ndarray pairs — one per valid anchor date.
    """
    if not permnos or not anchor_indices:
        return []

    # Determine the date range we actually need
    min_idx = min(anchor_indices) - lookback_days
    max_idx = max(anchor_indices) + horizon_days  # inclusive last fwd day

    if min_idx < 0 or max_idx >= len(trading_dates):
        return []

    date_slice = trading_dates[min_idx : max_idx + 1]

    # ONE pivot for all required dates — fast
    sub = crsp_df[
        crsp_df["permno"].isin(permnos) & crsp_df["date"].isin(date_slice)
    ][["date", "permno", "ret_total"]].copy()

    pivot = sub.pivot(index="date", columns="permno", values="ret_total")
    pivot = pivot.reindex(index=date_slice, columns=permnos)
    ret_matrix = pivot.values.astype(np.float64)  # shape (date_slice_len, N)

    # Map trading-calendar index → row index within ret_matrix
    # row_of[i] = i - min_idx
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for anc_idx in anchor_indices:
        row = anc_idx - min_idx            # row for the anchor date itself

        lb_start = row - lookback_days     # lookback: [lb_start, row)
        lb_end   = row
        fwd_start = row + 1                # forward:  [fwd_start, fwd_end)
        fwd_end   = row + 1 + horizon_days

        if lb_start < 0 or fwd_end > len(ret_matrix):
            continue

        R_hist = ret_matrix[lb_start:lb_end]    # (lookback_days, N)
        R_fwd  = ret_matrix[fwd_start:fwd_end]  # (horizon_days,  N)

        if (np.isnan(R_hist).any() or np.isnan(R_fwd).any()
                or R_hist.shape[0] != lookback_days
                or R_fwd.shape[0] != horizon_days):
            continue

        S_hist = np.cov(R_hist.T)
        S_fwd  = np.cov(R_fwd.T)
        pairs.append((S_hist, S_fwd))

    return pairs


def get_future_returns(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    permnos: list[int],
    rebalance_date: pd.Timestamp,
    horizon_days: int = 21,
) -> Optional[np.ndarray]:
    """
    Return the (horizon_days × N) future return matrix for a sleeve.
    Returns None if any return is missing.
    """
    try:
        fwd_dates = get_horizon_dates(trading_dates, rebalance_date, horizon_days)
    except ValueError:
        return None

    return compute_return_matrix(
        crsp_df, trading_dates, permnos, rebalance_date, fwd_dates
    )
