"""
02_build_universe_and_groups.py
--------------------------------
Phase 2: Build trading calendar, rebalance dates, dynamic top-500
universe, training groups, and evaluation sleeves.

Outputs
-------
data/interim/trading_calendar.parquet
data/interim/rebalance_dates.parquet
data/interim/dynamic_top500_universe.parquet
data/interim/training_groups.parquet
data/interim/evaluation_sleeves.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import (
    build_non_overlapping_rebalance_dates,
    build_trading_calendar,
)
from src.config import get_config
from src.groups import build_all_evaluation_sleeves, build_all_training_groups
from src.universe import build_all_universes
from src.utils import get_logger

logger = get_logger("02_build_universe_and_groups", logging.INFO)


def main() -> None:
    cfg = get_config()

    cleaned_path = Path(cfg["data"]["cleaned_file"])
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 2 – Build universe and groups")
    logger.info("=" * 60)

    # ---- Load cleaned CRSP -----------------------------------------------
    logger.info("Loading cleaned CRSP panel from %s …", cleaned_path)
    crsp_df = pd.read_parquet(cleaned_path)
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    logger.info("Loaded %d rows", len(crsp_df))

    # ---- Trading calendar ------------------------------------------------
    logger.info("Building trading calendar …")
    trading_dates = build_trading_calendar(crsp_df)
    cal_df = pd.DataFrame({"date": trading_dates})
    cal_df.to_parquet(interim_dir / "trading_calendar.parquet", index=False)
    logger.info("Trading calendar: %d dates (%s – %s)",
                len(trading_dates),
                trading_dates.min().date(),
                trading_dates.max().date())

    lkb = cfg.rolling_windows["lookback_days"]
    hor = cfg.rolling_windows["horizon_days"]

    # ---- Rebalance dates for each split ----------------------------------
    splits = {
        "train": (cfg.periods["train"]["start"], cfg.periods["train"]["end"]),
        "validation": (cfg.periods["validation"]["start"], cfg.periods["validation"]["end"]),
        "test": (cfg.periods["test"]["start"], cfg.periods["test"]["end"]),
    }

    all_rebalance_rows = []
    split_rebalance_dates = {}

    for split_name, (start, end) in splits.items():
        dates = build_non_overlapping_rebalance_dates(
            trading_dates, start, end, lookback_days=lkb, horizon_days=hor
        )
        split_rebalance_dates[split_name] = dates
        for d in dates:
            all_rebalance_rows.append({"split": split_name, "rebalance_date": d})
        logger.info(
            "  %-12s: %d rebalance dates (%s – %s)",
            split_name, len(dates),
            dates[0].date() if dates else "n/a",
            dates[-1].date() if dates else "n/a",
        )

    reb_df = pd.DataFrame(all_rebalance_rows)
    reb_df.to_parquet(interim_dir / "rebalance_dates.parquet", index=False)
    logger.info("Rebalance dates saved.")

    # ---- Dynamic top-500 universe ----------------------------------------
    all_rebalance_dates = (
        split_rebalance_dates["train"]
        + split_rebalance_dates["validation"]
        + split_rebalance_dates["test"]
    )
    all_rebalance_dates = sorted(set(all_rebalance_dates))

    logger.info(
        "Building dynamic top-%d universe at %d rebalance dates …",
        cfg.market_cap_top_n, len(all_rebalance_dates),
    )
    universes_df = build_all_universes(
        crsp_df,
        trading_dates,
        all_rebalance_dates,
        lookback_days=lkb,
        top_n=cfg.market_cap_top_n,
    )
    universes_df.to_parquet(interim_dir / "dynamic_top500_universe.parquet", index=False)
    logger.info(
        "Universe table: %d rows, %d dates",
        len(universes_df), universes_df["rebalance_date"].nunique(),
    )

    # ---- Training groups (overlapping) -----------------------------------
    train_universe_df = universes_df[
        universes_df["rebalance_date"].isin(split_rebalance_dates["train"])
    ]
    logger.info(
        "Building overlapping training groups (%d train dates) …",
        train_universe_df["rebalance_date"].nunique(),
    )
    training_groups_df = build_all_training_groups(
        universes=train_universe_df,
        group_size=cfg.group_size,
        target_groups=cfg["groups"]["training"]["target_groups_per_rebalance_date"],
        seed=cfg["groups"]["training"]["deterministic_seed"],
    )
    training_groups_df.to_parquet(interim_dir / "training_groups.parquet", index=False)
    logger.info(
        "Training groups: %d total groups, %d dates",
        training_groups_df["group_id"].nunique(),
        training_groups_df["rebalance_date"].nunique(),
    )

    # ---- Evaluation sleeves (non-overlapping, val + test) ----------------
    eval_dates = (
        split_rebalance_dates["validation"] + split_rebalance_dates["test"]
    )
    eval_universe_df = universes_df[universes_df["rebalance_date"].isin(eval_dates)]

    logger.info(
        "Building non-overlapping evaluation sleeves (%d eval dates) …",
        eval_universe_df["rebalance_date"].nunique(),
    )
    eval_sleeves_df = build_all_evaluation_sleeves(
        universes=eval_universe_df,
        rebalance_dates=eval_dates,
        group_size=cfg.group_size,
    )
    eval_sleeves_df.to_parquet(interim_dir / "evaluation_sleeves.parquet", index=False)
    logger.info(
        "Evaluation sleeves: %d total sleeves, %d dates",
        eval_sleeves_df["sleeve_id"].nunique(),
        eval_sleeves_df["rebalance_date"].nunique(),
    )

    logger.info("Step 2 complete.")


if __name__ == "__main__":
    main()
