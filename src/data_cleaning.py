"""
data_cleaning.py
----------------
Load, filter, and standardize a raw CRSP daily equity panel.

Key responsibilities
--------------------
1.  Map raw CRSP column names to canonical names via column_mapping.yaml.
2.  Filter ordinary common shares (SHRCD ∈ {10, 11}).
3.  Filter major U.S. exchanges (EXCHCD ∈ {1, 2, 3}) if available.
4.  Construct delisting-adjusted daily total return.
5.  Construct daily market capitalization ME = |PRC| × SHROUT.
6.  Drop or flag malformed records.
7.  Enforce uniqueness by (date, permno).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import get_config, load_column_mapping, is_field_available

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """
    Map canonical field names to the actual column names present in *df*.

    Returns a dict  {canonical: actual_col | None}.
    If a column was not found but is listed as available=true in the mapping,
    a warning is emitted.
    """
    mapping = load_column_mapping()
    resolved: dict[str, Optional[str]] = {}

    for canonical, entry in mapping.items():
        raw = entry["raw_name"]
        # Try exact match first, then case-insensitive
        if raw in df.columns:
            resolved[canonical] = raw
        else:
            # Case-insensitive fallback
            lower_map = {c.lower(): c for c in df.columns}
            if raw.lower() in lower_map:
                resolved[canonical] = lower_map[raw.lower()]
                logger.info(
                    "Mapped '%s' -> '%s' (case-insensitive fallback)",
                    raw, resolved[canonical],
                )
            else:
                resolved[canonical] = None
                if entry.get("available", False):
                    logger.warning(
                        "Column '%s' (canonical: '%s') listed as available "
                        "in column_mapping.yaml but not found in the dataset.",
                        raw, canonical,
                    )

    return resolved


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_crsp_data(
    df: pd.DataFrame,
    *,
    eligible_share_codes: list[int] | None = None,
    eligible_exchange_codes: list[int] | None = None,
    include_delisting_returns: bool = True,
) -> pd.DataFrame:
    """
    Clean a raw CRSP daily panel and return the canonical cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CRSP daily panel as loaded from the parquet/csv extract.
    eligible_share_codes : list[int]
        Retain only these SHRCD values.  Default: [10, 11].
    eligible_exchange_codes : list[int] | None
        If not None, retain only these EXCHCD values.  Default: [1, 2, 3].
    include_delisting_returns : bool
        If True, compound RET and DLRET where both are available.

    Returns
    -------
    pd.DataFrame with canonical columns:
        date, permno, ret_total, me, shrcd, exchcd (if available), siccd (if available)
    """
    cfg = get_config()

    if eligible_share_codes is None:
        eligible_share_codes = cfg["data"]["eligible_share_codes"]
    if eligible_exchange_codes is None and cfg["data"]["restrict_major_us_exchanges"]:
        eligible_exchange_codes = cfg["data"]["eligible_exchange_codes"]

    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1 – Resolve columns
    # ------------------------------------------------------------------
    col_map = _resolve_columns(df)
    logger.info("Resolved column map: %s", col_map)

    # Rename available columns to canonical names
    rename = {v: k for k, v in col_map.items() if v is not None and v != k}
    df = df.rename(columns=rename)

    # ------------------------------------------------------------------
    # Step 2 – Parse date
    # ------------------------------------------------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("No 'date' column could be resolved.")

    # ------------------------------------------------------------------
    # Step 3 – Ensure permno is integer
    # ------------------------------------------------------------------
    if "permno" not in df.columns:
        raise ValueError("No 'permno' column could be resolved.")
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")

    # ------------------------------------------------------------------
    # Step 4 – Common share filter
    # ------------------------------------------------------------------
    if "shrcd" in df.columns and col_map.get("shrcd") is not None:
        df["shrcd"] = pd.to_numeric(df["shrcd"], errors="coerce")
        n_before = len(df)
        df = df[df["shrcd"].isin(eligible_share_codes)]
        logger.info(
            "Share-code filter (SHRCD ∈ %s): %d -> %d rows",
            eligible_share_codes, n_before, len(df),
        )
    else:
        logger.warning(
            "SHRCD not available – skipping common-share filter. "
            "Universe may include non-equity instruments."
        )

    # ------------------------------------------------------------------
    # Step 5 – Exchange filter
    # ------------------------------------------------------------------
    if eligible_exchange_codes is not None:
        if "exchcd" in df.columns and col_map.get("exchcd") is not None:
            df["exchcd"] = pd.to_numeric(df["exchcd"], errors="coerce")
            n_before = len(df)
            df = df[df["exchcd"].isin(eligible_exchange_codes)]
            logger.info(
                "Exchange filter (EXCHCD ∈ %s): %d -> %d rows",
                eligible_exchange_codes, n_before, len(df),
            )
        else:
            logger.warning(
                "EXCHCD not available – skipping exchange filter. "
                "Universe may include OTC/foreign-listed securities."
            )

    # ------------------------------------------------------------------
    # Step 6 – Parse returns
    # ------------------------------------------------------------------
    _parse_return_columns(df)

    # ------------------------------------------------------------------
    # Step 7 – Construct delisting-adjusted total return
    # ------------------------------------------------------------------
    df["ret_total"] = _compute_total_return(
        df,
        include_delisting=include_delisting_returns,
    )

    # ------------------------------------------------------------------
    # Step 8 – Construct market cap
    # ------------------------------------------------------------------
    df["me"] = _compute_market_cap(df)

    # ------------------------------------------------------------------
    # Step 9 – Drop rows where both return and ME are missing
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.dropna(subset=["date", "permno"])
    logger.info(
        "After dropping null date/permno: %d -> %d rows", n_before, len(df)
    )

    # ------------------------------------------------------------------
    # Step 10 – Deduplicate by (date, permno)
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.sort_values(["date", "permno"]).drop_duplicates(
        subset=["date", "permno"], keep="first"
    )
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning(
            "Dropped %d duplicate (date, permno) rows (kept first).", n_dupes
        )

    # ------------------------------------------------------------------
    # Step 11 – Keep only canonical columns that exist
    # ------------------------------------------------------------------
    keep_cols = ["date", "permno", "ret_total", "me"]
    optional_cols = ["shrcd", "exchcd", "siccd"]
    for oc in optional_cols:
        if oc in df.columns:
            keep_cols.append(oc)

    df = df[keep_cols].copy()
    df = df.sort_values(["date", "permno"]).reset_index(drop=True)

    logger.info(
        "Cleaned CRSP panel: %d rows, %d securities, date range %s – %s",
        len(df),
        df["permno"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )

    return df


# ---------------------------------------------------------------------------
# Return construction helpers
# ---------------------------------------------------------------------------

def _parse_return_columns(df: pd.DataFrame) -> None:
    """
    Convert RET and DLRET columns to numeric in-place.
    CRSP sometimes stores returns as strings ('C', 'B', …) for missing values.
    """
    for col in ["ret", "dlret"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _compute_total_return(
    df: pd.DataFrame,
    include_delisting: bool = True,
) -> pd.Series:
    """
    Construct delisting-adjusted daily total return.

    Rules (from spec Section 5.3):
    - Both RET and DLRET available  -> (1+ret)(1+dlret) - 1
    - Only RET                      -> ret
    - Only DLRET                    -> dlret
    - Neither                       -> NaN
    """
    has_ret = "ret" in df.columns
    has_dlret = "dlret" in df.columns and include_delisting

    if has_ret and has_dlret:
        ret = df["ret"]
        dlret = df["dlret"]

        both_avail = ret.notna() & dlret.notna()
        only_ret = ret.notna() & dlret.isna()
        only_dlr = ret.isna() & dlret.notna()

        total = pd.Series(np.nan, index=df.index, dtype=float)
        total[both_avail] = (1 + ret[both_avail]) * (1 + dlret[both_avail]) - 1
        total[only_ret] = ret[only_ret]
        total[only_dlr] = dlret[only_dlr]
        # Both missing -> NaN (spec: do not replace with zero)
        return total

    elif has_ret:
        logger.info("DLRET not used; using RET as total return.")
        return df["ret"].copy()

    elif has_dlret:
        logger.warning("RET not available; using DLRET as total return.")
        return df["dlret"].copy()

    else:
        logger.error("Neither RET nor DLRET found – all total returns will be NaN.")
        return pd.Series(np.nan, index=df.index, dtype=float)


def _compute_market_cap(df: pd.DataFrame) -> pd.Series:
    """
    ME = |PRC| × SHROUT.
    SHROUT in CRSP is in thousands of shares, so ME is in thousands of dollars.
    Returns NaN where PRC or SHROUT are missing.
    Set to NaN (not 0) where ME ≤ 0.
    """
    if "prc" not in df.columns or "shrout" not in df.columns:
        logger.warning(
            "PRC or SHROUT not available – market cap cannot be computed."
        )
        return pd.Series(np.nan, index=df.index, dtype=float)

    prc = pd.to_numeric(df["prc"], errors="coerce").abs()
    shrout = pd.to_numeric(df["shrout"], errors="coerce")

    me = prc * shrout  # thousands of dollars
    me[me <= 0] = np.nan
    return me


# ---------------------------------------------------------------------------
# Load from file
# ---------------------------------------------------------------------------

def load_raw_crsp(path: str | Path) -> pd.DataFrame:
    """Load a raw CRSP extract from a parquet or CSV file."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix in {".csv", ".gz", ".zip"}:
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
