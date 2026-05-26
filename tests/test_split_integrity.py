"""
test_split_integrity.py
-----------------------
Verify split boundary and leakage rules (spec §35.2).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import build_non_overlapping_rebalance_dates


def _make_trading_dates(start="1998-01-01", end="2025-12-31", freq="B"):
    """Synthetic business-day trading calendar."""
    return pd.DatetimeIndex(pd.date_range(start, end, freq=freq))


TRADING_DATES = _make_trading_dates()

TRAIN_START = "2000-01-01"
TRAIN_END   = "2013-12-31"
VAL_START   = "2014-01-01"
VAL_END     = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2025-12-31"

LOOKBACK = 126
HORIZON  = 21


def _get_rebalance_dates(start, end):
    return build_non_overlapping_rebalance_dates(
        TRADING_DATES, start, end, LOOKBACK, HORIZON
    )


train_dates = _get_rebalance_dates(TRAIN_START, TRAIN_END)
val_dates   = _get_rebalance_dates(VAL_START, VAL_END)
test_dates  = _get_rebalance_dates(TEST_START, TEST_END)


def _target_window(t, dates_arr=TRADING_DATES):
    idx = dates_arr.get_loc(t)
    return dates_arr[idx + 1], dates_arr[idx + HORIZON]


def test_all_training_target_dates_within_2000_2013():
    end = pd.Timestamp(TRAIN_END)
    start = pd.Timestamp(TRAIN_START)
    for t in train_dates:
        t_start, t_end = _target_window(t)
        assert t_start >= start, f"Train target start {t_start} < {start}"
        assert t_end <= end,   f"Train target end {t_end} > {end}"


def test_all_validation_target_dates_within_2014_2020():
    end = pd.Timestamp(VAL_END)
    start = pd.Timestamp(VAL_START)
    for t in val_dates:
        t_start, t_end = _target_window(t)
        assert t_start >= start, f"Val target start {t_start} < {start}"
        assert t_end <= end,   f"Val target end {t_end} > {end}"


def test_all_test_target_dates_within_2021_2025():
    end = pd.Timestamp(TEST_END)
    start = pd.Timestamp(TEST_START)
    for t in test_dates:
        t_start, t_end = _target_window(t)
        assert t_start >= start, f"Test target start {t_start} < {start}"
        assert t_end <= end,   f"Test target end {t_end} > {end}"


def test_no_target_window_crosses_split_boundary():
    """No target window should straddle the train/val or val/test boundary."""
    val_start = pd.Timestamp(VAL_START)
    test_start = pd.Timestamp(TEST_START)

    for dates, start, end in [
        (train_dates, pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)),
        (val_dates,   pd.Timestamp(VAL_START),   pd.Timestamp(VAL_END)),
        (test_dates,  pd.Timestamp(TEST_START),  pd.Timestamp(TEST_END)),
    ]:
        for t in dates:
            t_s, t_e = _target_window(t)
            assert not (t_s < start and t_e > start), (
                f"Target window [{t_s}, {t_e}] crosses split start {start}"
            )
            assert not (t_s < end and t_e > end), (
                f"Target window [{t_s}, {t_e}] crosses split end {end}"
            )


def test_no_observation_assigned_to_multiple_splits():
    """No rebalance date should appear in more than one split."""
    all_sets = [set(d for d in train_dates), set(val_dates), set(test_dates)]
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            overlap = all_sets[i] & all_sets[j]
            assert len(overlap) == 0, f"Splits {i} and {j} share dates: {overlap}"


def test_non_overlapping_holding_periods():
    """Consecutive rebalance dates in the same split are exactly HORIZON apart."""
    for dates in [train_dates, val_dates, test_dates]:
        for i in range(len(dates) - 1):
            t_curr = TRADING_DATES.get_loc(dates[i])
            t_next = TRADING_DATES.get_loc(dates[i + 1])
            assert t_next - t_curr == HORIZON, (
                f"Non-overlapping violated between {dates[i].date()} "
                f"and {dates[i+1].date()} (gap={t_next - t_curr})"
            )
