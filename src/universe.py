"""
universe.py
-----------
Construct the dynamic market-cap top-N universe at each rebalance date.

Spec §8:
* Universe = top-500 ordinary common shares by ME at the rebalance date.
* Requires complete past 126-day return history.
* Rebuilt dynamically at every rebalance date (no survivorship bias).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from .calendar import get_lookback_dates
from .config import get_config

logger = logging.getLogger(__name__)


def build_universe_at_date(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    rebalance_date: pd.Timestamp,
    lookback_days: int = 126,
    top_n: int = 500,
) -> pd.DataFrame:
    """
    Return the dynamic top-N universe at a single rebalance date.

    Eligibility criteria (spec §8.1):
    1. Security has a positive market cap (ME > 0) at the rebalance date.
    2. Security has complete daily returns over the previous 126 trading days.
    3. Ranked by ME at the rebalance date; top N retained.

    Parameters
    ----------
    crsp_df : pd.DataFrame
        Cleaned CRSP panel with columns: date, permno, ret_total, me, siccd
    trading_dates : pd.DatetimeIndex
        Sorted calendar of actual trading dates.
    rebalance_date : pd.Timestamp
        The rebalance formation date t.
    lookback_days : int
        Number of days in the input window.
    top_n : int
        Universe size (default 500).

    Returns
    -------
    pd.DataFrame with columns:
        permno, me, siccd, industry (2-digit SIC)
    Indexed by permno, ordered by descending ME.
    """
    # ---- 1. Identify the lookback window ------------------------------------
    try:
        lookback_window = get_lookback_dates(trading_dates, rebalance_date, lookback_days)
    except ValueError as exc:
        logger.warning("Skipping %s: %s", rebalance_date.date(), exc)
        return pd.DataFrame(columns=["permno", "me", "siccd", "industry"])

    # ---- 2. ME at the rebalance date ----------------------------------------
    snapshot_t = crsp_df[crsp_df["date"] == rebalance_date][
        ["permno", "me", "siccd"]
    ].copy()
    snapshot_t = snapshot_t[snapshot_t["me"] > 0].dropna(subset=["me"])

    if snapshot_t.empty:
        logger.warning("No securities with positive ME on %s", rebalance_date.date())
        return pd.DataFrame(columns=["permno", "me", "siccd", "industry"])

    # ---- 3. Complete-history filter -----------------------------------------
    # Securities with a non-NaN return on every lookback trading day
    lookback_panel = crsp_df[
        crsp_df["date"].isin(lookback_window) & crsp_df["permno"].isin(snapshot_t["permno"])
    ][["date", "permno", "ret_total"]].copy()

    # Count non-NaN return days per security
    valid_days = (
        lookback_panel.dropna(subset=["ret_total"])
        .groupby("permno")["date"]
        .nunique()
    )
    # Require all lookback_days present with a valid return
    complete_permnos = valid_days[valid_days == lookback_days].index

    snapshot_t = snapshot_t[snapshot_t["permno"].isin(complete_permnos)]

    if snapshot_t.empty:
        logger.warning(
            "No securities with complete %d-day history on %s",
            lookback_days, rebalance_date.date(),
        )
        return pd.DataFrame(columns=["permno", "me", "siccd", "industry"])

    # ---- 4. Top-N by ME ----------------------------------------------------
    snapshot_t = snapshot_t.sort_values("me", ascending=False).head(top_n)

    # ---- 5. Industry (2-digit SIC) ------------------------------------------
    snapshot_t["industry"] = compute_industry(snapshot_t["siccd"])

    snapshot_t = snapshot_t.reset_index(drop=True)
    logger.debug(
        "Universe at %s: %d securities", rebalance_date.date(), len(snapshot_t)
    )
    return snapshot_t


def compute_industry(siccd_series: pd.Series) -> pd.Series:
    """
    Map SIC codes to two-digit industry groups (floor(SIC / 100)).

    Returns NaN for missing SIC codes.
    """
    sic = pd.to_numeric(siccd_series, errors="coerce")
    return np.floor(sic / 100).astype("Int64")


def build_all_universes(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    rebalance_dates: List[pd.Timestamp],
    lookback_days: int = 126,
    top_n: int = 500,
) -> pd.DataFrame:
    """
    Build universes at all rebalance dates and return a combined DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        rebalance_date, permno, me, siccd, industry
    """
    records = []
    for t in rebalance_dates:
        univ = build_universe_at_date(
            crsp_df, trading_dates, t,
            lookback_days=lookback_days, top_n=top_n,
        )
        if univ.empty:
            continue
        univ["rebalance_date"] = t
        records.append(univ)

    if not records:
        logger.error("No universe records built across all rebalance dates.")
        return pd.DataFrame(
            columns=["rebalance_date", "permno", "me", "siccd", "industry"]
        )

    out = pd.concat(records, ignore_index=True)
    out = out[["rebalance_date", "permno", "me", "siccd", "industry"]]
    logger.info(
        "Built universe table: %d rows, %d unique rebalance dates",
        len(out), out["rebalance_date"].nunique(),
    )
    return out
