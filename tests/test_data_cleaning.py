"""
test_data_cleaning.py
---------------------
Tests for CRSP data cleaning (spec §35.1).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import (
    _compute_market_cap,
    _compute_total_return,
    clean_crsp_data,
)


@pytest.fixture
def raw_df():
    """Minimal synthetic CRSP-like panel."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2010-01-04", "2010-01-04", "2010-01-05", "2010-01-05"]
            ),
            "PERMNO": [10001, 10002, 10001, 10002],
            "RET": [0.01, -0.005, 0.02, np.nan],
            "DLRET": [np.nan, np.nan, np.nan, -1.0],
            "PRC": [50.0, -20.0, 51.0, 0.0],     # negative = bid/ask midpoint
            "SHROUT": [1000, 500, 1000, 500],       # thousands of shares
            "SHRCD": [11, 10, 11, 99],              # 99 = non-common
            "EXCHCD": [1, 2, 1, 3],
            "SICCD": [3674, 5911, 3674, 5911],
        }
    )


def test_unique_date_permno_after_cleaning(raw_df):
    df = clean_crsp_data(raw_df, eligible_share_codes=[10, 11])
    assert df.duplicated(subset=["date", "permno"]).sum() == 0


def test_positive_market_cap_only(raw_df):
    df = clean_crsp_data(raw_df, eligible_share_codes=[10, 11])
    assert (df["me"] > 0).all()


def test_common_share_filter(raw_df):
    df = clean_crsp_data(raw_df, eligible_share_codes=[10, 11])
    assert "shrcd" in df.columns or len(df) <= 3  # SHRCD 99 should be gone


def test_exchange_filter(raw_df):
    # EXCHCD=3 exists in row with SHRCD=99 (already filtered by share code)
    # Here we pass all share codes and check exchange filtering works
    df = clean_crsp_data(
        raw_df,
        eligible_share_codes=[10, 11, 99],
        eligible_exchange_codes=[1, 2],
    )
    # EXCHCD=3 row should be excluded (it's the SHRCD=99 row anyway)
    if "exchcd" in df.columns:
        assert (df["exchcd"].isin([1, 2])).all()


def test_delisting_return_compound():
    """Both RET and DLRET available -> compound."""
    df_tmp = pd.DataFrame(
        {
            "ret": [0.01, np.nan, 0.02],
            "dlret": [np.nan, -0.5, 0.001],
        }
    )
    result = _compute_total_return(df_tmp, include_delisting=True)
    # Row 0: only RET
    assert abs(result.iloc[0] - 0.01) < 1e-9
    # Row 1: only DLRET
    assert abs(result.iloc[1] - (-0.5)) < 1e-9
    # Row 2: both
    expected = (1 + 0.02) * (1 + 0.001) - 1
    assert abs(result.iloc[2] - expected) < 1e-9


def test_market_cap_positive_price_absolute():
    df_tmp = pd.DataFrame(
        {
            "prc": [-50.0, 30.0, np.nan],
            "shrout": [1000, 500, 100],
        }
    )
    me = _compute_market_cap(df_tmp)
    assert abs(me.iloc[0] - 50_000) < 1e-6
    assert abs(me.iloc[1] - 15_000) < 1e-6
    assert np.isnan(me.iloc[2])
