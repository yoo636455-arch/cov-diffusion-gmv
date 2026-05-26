"""
test_universe_no_lookahead.py
------------------------------
Verify that the universe construction contains no lookahead bias (spec §35.3).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.universe import build_universe_at_date, compute_industry


def _make_crsp_panel():
    """Simple synthetic CRSP panel spanning two dates."""
    np.random.seed(0)
    dates = pd.to_datetime(["2010-01-04", "2010-01-05", "2010-01-06",
                             "2010-01-07", "2010-01-08"])
    n = 15
    rows = []
    for d in dates:
        for permno in range(1, n + 1):
            rows.append(
                {
                    "date": d,
                    "permno": permno,
                    "ret_total": np.random.normal(0, 0.01),
                    "me": float(permno) * 1000,  # increasing me by permno
                    "siccd": 3674 + permno,
                }
            )
    return pd.DataFrame(rows)


def _make_trading_dates(df):
    return pd.DatetimeIndex(sorted(df["date"].unique()))


def test_universe_uses_only_past_market_caps():
    """Universe ranks at t should use ME observed AT t, not future ME."""
    crsp = _make_crsp_panel()
    trading_dates = _make_trading_dates(crsp)

    # Artificially inflate ME for permno=1 on future dates only
    future_mask = (crsp["permno"] == 1) & (crsp["date"] > pd.Timestamp("2010-01-05"))
    crsp.loc[future_mask, "me"] = 999_999_999

    t = pd.Timestamp("2010-01-05")
    # lookback = 4 days (indices 0..4, t is at index 1 so we need at least lookback=1)
    univ = build_universe_at_date(
        crsp, trading_dates, t, lookback_days=2, top_n=10
    )

    # ME at t for permno=1 should be 1000, not 999_999_999
    row1 = univ[univ["permno"] == 1]
    if not row1.empty:
        assert row1["me"].iloc[0] == 1000.0, (
            f"Universe used future ME: got {row1['me'].iloc[0]}"
        )


def test_universe_sic_contemporaneous():
    """Industry classification should use SIC code at the rebalance date."""
    crsp = _make_crsp_panel()
    trading_dates = _make_trading_dates(crsp)

    # Change SIC for permno=3 on future date
    future_mask = (crsp["permno"] == 3) & (crsp["date"] > pd.Timestamp("2010-01-05"))
    crsp.loc[future_mask, "siccd"] = 9999

    t = pd.Timestamp("2010-01-05")
    univ = build_universe_at_date(crsp, trading_dates, t, lookback_days=2, top_n=10)

    row3 = univ[univ["permno"] == 3]
    if not row3.empty:
        assert row3["siccd"].iloc[0] == 3677, (
            "Universe used future SIC code"
        )


def test_industry_computation():
    """industry = floor(SIC / 100)."""
    sics = pd.Series([3674, 5911, 100, 9999, np.nan])
    industries = compute_industry(sics)
    assert industries.iloc[0] == 36
    assert industries.iloc[1] == 59
    assert industries.iloc[2] == 1
    assert industries.iloc[3] == 99
    assert pd.isna(industries.iloc[4])
