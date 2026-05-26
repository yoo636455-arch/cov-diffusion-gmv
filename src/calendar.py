"""
calendar.py
-----------
Build CRSP-derived trading calendars and non-overlapping rebalance-date schedules.

Key functions
-------------
build_trading_calendar        - extract sorted unique trading dates from cleaned CRSP data
build_non_overlapping_rebalance_dates - spec §7.2 rebalance schedule
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------

def build_trading_calendar(crsp_df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Extract a sorted unique list of actual trading dates from the cleaned CRSP panel.

    Parameters
    ----------
    crsp_df : pd.DataFrame
        Cleaned panel with a 'date' column.

    Returns
    -------
    pd.DatetimeIndex of trading dates in ascending order.
    """
    dates = pd.to_datetime(crsp_df["date"]).drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


# ---------------------------------------------------------------------------
# Rebalance date schedule
# ---------------------------------------------------------------------------

def build_non_overlapping_rebalance_dates(
    trading_dates: pd.DatetimeIndex,
    start_target_date: str | pd.Timestamp,
    end_target_date: str | pd.Timestamp,
    lookback_days: int = 126,
    horizon_days: int = 21,
) -> List[pd.Timestamp]:
    """
    Return rebalance dates whose complete input lookback window exists and
    whose complete 21-day future target/holding window lies entirely within
    [start_target_date, end_target_date].

    Spec §6.2 – observation assignment rule:
    * An observation is assigned to a split based on the dates of its FUTURE
      21-day target window, not the input lookback window.
    * The input lookback window may extend into the prior split.

    Parameters
    ----------
    trading_dates : pd.DatetimeIndex
        Complete sorted list of actual trading dates from CRSP.
    start_target_date : str or pd.Timestamp
        First date of the split's target window (inclusive).
    end_target_date : str or pd.Timestamp
        Last date of the split's target window (inclusive).
    lookback_days : int
        Number of trading days in the input window (default 126).
    horizon_days : int
        Number of trading days in the target/holding window (default 21).

    Returns
    -------
    List[pd.Timestamp]
        Sorted list of valid rebalance dates.
    """
    start_target_date = pd.Timestamp(start_target_date)
    end_target_date = pd.Timestamp(end_target_date)

    dates_arr = trading_dates.sort_values()
    n = len(dates_arr)

    rebalance_dates: List[pd.Timestamp] = []

    # The minimum index to start: we need lookback_days before rebalance date
    # and horizon_days after.  The index of the rebalance date must be at least
    # lookback_days - 1 (0-indexed), so dates_arr[lookback_days - 1] is the
    # earliest possible rebalance date with a full 126-day lookback.

    # We iterate through all candidate rebalance dates and check:
    #   1. Full lookback exists (index >= lookback_days - 1)
    #   2. Full horizon exists  (index + horizon_days <= n - 1)
    #   3. Target window start  >= start_target_date
    #   4. Target window end    <= end_target_date

    # For non-overlapping 21-day holding periods, we advance by horizon_days
    # from one rebalance date to the next.

    # Find the first trading date on or after start_target_date
    # (since t+1 must be >= start_target_date, t must be the trading day
    # just before start_target_date or the split boundary itself).
    # We look for the first t such that dates_arr[t_idx + 1] >= start_target_date
    # and dates_arr[t_idx + horizon_days] <= end_target_date.

    # Approach: iterate by fixed horizon_days steps
    # Start from the first date where t+1 >= start_target_date

    for t_idx in range(lookback_days - 1, n - horizon_days):
        target_start = dates_arr[t_idx + 1]
        target_end = dates_arr[t_idx + horizon_days]

        # Check split boundaries
        if target_start < start_target_date:
            continue
        if target_end > end_target_date:
            break  # Further dates will only have later target ends

        rebalance_date = dates_arr[t_idx]
        rebalance_dates.append(rebalance_date)

    # Now enforce non-overlapping: step forward by horizon_days at a time.
    # The loop above found ALL valid rebalance dates; we select the
    # non-overlapping subset by taking every horizon_days-th step.
    if not rebalance_dates:
        logger.warning(
            "No rebalance dates found for window [%s, %s]",
            start_target_date.date(), end_target_date.date(),
        )
        return []

    # Build non-overlapping schedule by stepping exactly horizon_days
    # trading days forward from the first eligible date.
    selected: List[pd.Timestamp] = []
    all_valid = pd.DatetimeIndex(rebalance_dates)

    # First valid rebalance date
    current = all_valid[0]
    selected.append(current)

    while True:
        # Find the index of current in dates_arr
        curr_pos = dates_arr.get_loc(current)
        next_pos = curr_pos + horizon_days
        if next_pos >= n:
            break
        next_date = dates_arr[next_pos]
        if next_date not in all_valid:
            break
        selected.append(next_date)
        current = next_date

    logger.info(
        "Rebalance schedule [%s, %s]: %d dates (first %s, last %s)",
        start_target_date.date(),
        end_target_date.date(),
        len(selected),
        selected[0].date() if selected else "n/a",
        selected[-1].date() if selected else "n/a",
    )

    return selected


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_lookback_dates(
    trading_dates: pd.DatetimeIndex,
    rebalance_date: pd.Timestamp,
    lookback_days: int = 126,
) -> pd.DatetimeIndex:
    """
    Return the trading dates in [t-lookback_days+1, …, t] (inclusive).

    Parameters
    ----------
    trading_dates : sorted pd.DatetimeIndex
    rebalance_date : pd.Timestamp
    lookback_days : int

    Returns
    -------
    pd.DatetimeIndex of length lookback_days.
    """
    dates_arr = trading_dates.sort_values()
    t_idx = dates_arr.get_loc(rebalance_date)
    start_idx = t_idx - lookback_days + 1
    if start_idx < 0:
        raise ValueError(
            f"Not enough history: need {lookback_days} days before {rebalance_date.date()}; "
            f"only {t_idx + 1} trading dates available."
        )
    return dates_arr[start_idx : t_idx + 1]


def get_horizon_dates(
    trading_dates: pd.DatetimeIndex,
    rebalance_date: pd.Timestamp,
    horizon_days: int = 21,
) -> pd.DatetimeIndex:
    """
    Return the trading dates in [t+1, …, t+horizon_days] (inclusive).

    Parameters
    ----------
    trading_dates : sorted pd.DatetimeIndex
    rebalance_date : pd.Timestamp
    horizon_days : int

    Returns
    -------
    pd.DatetimeIndex of length horizon_days.
    """
    dates_arr = trading_dates.sort_values()
    t_idx = dates_arr.get_loc(rebalance_date)
    end_idx = t_idx + horizon_days
    if end_idx >= len(dates_arr):
        raise ValueError(
            f"Not enough future dates: need {horizon_days} days after "
            f"{rebalance_date.date()}."
        )
    return dates_arr[t_idx + 1 : end_idx + 1]
