"""
01_clean_crsp_data.py
---------------------
Phase 1: Load raw CRSP data, clean it, and save the canonical panel.

Usage (from project root):
    python scripts/01_clean_crsp_data.py [--raw-file PATH]

Outputs
-------
data/interim/cleaned_crsp_daily.parquet
config/column_mapping.yaml  (updated with resolved mapping)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Make src importable from the scripts folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, load_column_mapping
from src.data_cleaning import clean_crsp_data, load_raw_crsp
from src.utils import get_logger

logger = get_logger("01_clean_crsp_data", logging.INFO)


def main(raw_file: str | None = None) -> None:
    cfg = get_config()

    raw_path = Path(raw_file) if raw_file else Path(cfg["data"]["raw_file"])
    out_path = Path(cfg["data"]["cleaned_file"])

    logger.info("=" * 60)
    logger.info("STEP 1 – Clean CRSP daily data")
    logger.info("=" * 60)
    logger.info("Raw input:  %s", raw_path)
    logger.info("Output:     %s", out_path)

    # ---- Load raw data ---------------------------------------------------
    if not raw_path.exists():
        logger.error(
            "Raw file not found: %s\n"
            "Please place your CRSP daily parquet extract at that path "
            "or pass --raw-file <path>.",
            raw_path,
        )
        sys.exit(1)

    logger.info("Loading raw CRSP file …")
    df_raw = load_raw_crsp(raw_path)
    logger.info("Raw shape: %s", df_raw.shape)
    logger.info("Raw columns: %s", list(df_raw.columns))

    # ---- Inspect and write column mapping --------------------------------
    mapping = load_column_mapping()
    logger.info("Column mapping from config/column_mapping.yaml:")
    for canonical, entry in mapping.items():
        raw_col = entry["raw_name"]
        found = raw_col in df_raw.columns or raw_col.lower() in [
            c.lower() for c in df_raw.columns
        ]
        status = "✓" if found else "✗ MISSING"
        logger.info("  %-10s  raw=%-12s  %s", canonical, raw_col, status)

    # ---- Clean ----------------------------------------------------------
    logger.info("Cleaning …")
    df_clean = clean_crsp_data(
        df_raw,
        eligible_share_codes=cfg["data"]["eligible_share_codes"],
        eligible_exchange_codes=(
            cfg["data"]["eligible_exchange_codes"]
            if cfg["data"]["restrict_major_us_exchanges"]
            else None
        ),
        include_delisting_returns=cfg["data"]["include_delisting_returns"],
    )

    logger.info("Cleaned shape: %s", df_clean.shape)
    logger.info("Date range: %s – %s", df_clean["date"].min().date(), df_clean["date"].max().date())
    logger.info("Unique PERMNOs: %d", df_clean["permno"].nunique())

    # ---- Save ----------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out_path, index=False)
    logger.info("Saved cleaned panel to %s", out_path)

    # ---- Quick sanity checks -------------------------------------------
    dupes = df_clean.duplicated(subset=["date", "permno"]).sum()
    if dupes > 0:
        logger.error("SANITY CHECK FAILED: %d duplicate (date, permno) rows!", dupes)
    else:
        logger.info("SANITY CHECK PASSED: no duplicate (date, permno) rows.")

    neg_me = (df_clean["me"] <= 0).sum()
    if neg_me > 0:
        logger.warning(
            "%d rows have non-positive ME (should be 0 after cleaning).", neg_me
        )

    logger.info("Step 1 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw CRSP daily data.")
    parser.add_argument("--raw-file", type=str, default=None,
                        help="Override raw_file from base_config.yaml")
    args = parser.parse_args()
    main(raw_file=args.raw_file)
