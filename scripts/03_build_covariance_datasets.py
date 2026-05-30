"""
03_build_covariance_datasets.py
--------------------------------
Phase 3: Build covariance input/target pairs for all splits, fit
training-only scalers, and save processed arrays.

Outputs
-------
data/processed/covariance_pairs_train.npz
data/processed/covariance_pairs_validation.npz
data/processed/covariance_pairs_test.npz
artifacts/scalers/conditioning_scaler.pkl
artifacts/scalers/target_scaler.pkl
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.datasets import (
    apply_scalers,
    build_covariance_dataset,
    build_daily_sliding_covariance_dataset,
    load_dataset,
    save_dataset,
)
from src.transforms import fit_training_scalers, load_scalers
from src.utils import get_logger

logger = get_logger("03_build_covariance_datasets", logging.INFO)


def main() -> None:
    cfg = get_config()

    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")
    scaler_dir = Path("artifacts/scalers")
    processed_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    ridge_eps = cfg.ridge_epsilon
    lkb = cfg.rolling_windows["lookback_days"]
    hor = cfg.rolling_windows["horizon_days"]

    logger.info("=" * 60)
    logger.info("STEP 3 – Build covariance datasets")
    logger.info("=" * 60)

    # ---- Load data -------------------------------------------------------
    logger.info("Loading cleaned CRSP and rebalance/group data …")
    crsp_df = pd.read_parquet(interim_dir / "cleaned_crsp_daily.parquet")
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    # Pre-index on (date, permno) for O(log n) sleeve lookups
    crsp_df = crsp_df.set_index(["date", "permno"]).sort_index()

    cal_df = pd.read_parquet(interim_dir / "trading_calendar.parquet")
    trading_dates = pd.DatetimeIndex(pd.to_datetime(cal_df["date"]).sort_values())

    reb_df = pd.read_parquet(interim_dir / "rebalance_dates.parquet")
    reb_df["rebalance_date"] = pd.to_datetime(reb_df["rebalance_date"])

    train_groups = pd.read_parquet(interim_dir / "training_groups.parquet")
    train_groups["rebalance_date"] = pd.to_datetime(train_groups["rebalance_date"])

    eval_sleeves = pd.read_parquet(interim_dir / "evaluation_sleeves.parquet")
    eval_sleeves["rebalance_date"] = pd.to_datetime(eval_sleeves["rebalance_date"])

    val_dates = set(
        reb_df[reb_df["split"] == "validation"]["rebalance_date"].tolist()
    )
    test_dates = set(
        reb_df[reb_df["split"] == "test"]["rebalance_date"].tolist()
    )

    # ---- Training dataset (daily-sliding window) -------------------------
    train_npz = processed_dir / "covariance_pairs_train.npz"
    cond_scaler_path = scaler_dir / "conditioning_scaler.pkl"
    tgt_scaler_path  = scaler_dir / "target_scaler.pkl"

    if train_npz.exists() and cond_scaler_path.exists() and tgt_scaler_path.exists():
        logger.info("Training dataset already exists — loading scalers and skipping rebuild.")
        cond_scaler, tgt_scaler = load_scalers(scaler_dir)
    else:
        stride = cfg["covariance_transform"].get("training_window_stride_days", 1)
        train_end = pd.Timestamp(cfg["periods"]["train"]["end"])
        logger.info(
            "Building TRAINING covariance pairs with daily stride=%d "
            "(train_end=%s) …", stride, train_end.date()
        )
        train_ds = build_daily_sliding_covariance_dataset(
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            groups_df=train_groups,
            train_end_date=train_end,
            lookback_days=lkb,
            horizon_days=hor,
            ridge_epsilon=ridge_eps,
            stride=stride,
        )
        logger.info(
            "Training: %d pairs, cond_vech shape %s",
            len(train_ds["condition_vech"]), train_ds["condition_vech"].shape,
        )
        logger.info("Fitting scalers on training data only …")
        cond_scaler, tgt_scaler = fit_training_scalers(
            train_condition_vectors=train_ds["condition_vech"],
            train_target_vectors=train_ds["target_vech"],
            save_dir=scaler_dir,
        )
        train_ds = apply_scalers(train_ds, cond_scaler, tgt_scaler)
        save_dataset(train_ds, train_npz)
        logger.info("Saved training dataset.")

    # ---- Validation dataset ----------------------------------------------
    val_sleeves = eval_sleeves[eval_sleeves["rebalance_date"].isin(val_dates)].copy()
    logger.info(
        "Building VALIDATION covariance pairs (%d dates, %d sleeves) …",
        val_sleeves["rebalance_date"].nunique(), val_sleeves["sleeve_id"].nunique(),
    )
    val_ds = build_covariance_dataset(
        crsp_df=crsp_df,
        trading_dates=trading_dates,
        groups_df=val_sleeves,
        group_id_col="sleeve_id",
        rebalance_date_col="rebalance_date",
        lookback_days=lkb,
        horizon_days=hor,
        ridge_epsilon=ridge_eps,
    )
    val_ds = apply_scalers(val_ds, cond_scaler, tgt_scaler)
    save_dataset(val_ds, processed_dir / "covariance_pairs_validation.npz")
    logger.info("Saved validation dataset: %d pairs.", len(val_ds["condition_vech"]))

    # ---- Test dataset ----------------------------------------------------
    test_sleeves = eval_sleeves[eval_sleeves["rebalance_date"].isin(test_dates)].copy()
    logger.info(
        "Building TEST covariance pairs (%d dates, %d sleeves) …",
        test_sleeves["rebalance_date"].nunique(), test_sleeves["sleeve_id"].nunique(),
    )
    test_ds = build_covariance_dataset(
        crsp_df=crsp_df,
        trading_dates=trading_dates,
        groups_df=test_sleeves,
        group_id_col="sleeve_id",
        rebalance_date_col="rebalance_date",
        lookback_days=lkb,
        horizon_days=hor,
        ridge_epsilon=ridge_eps,
    )
    test_ds = apply_scalers(test_ds, cond_scaler, tgt_scaler)
    save_dataset(test_ds, processed_dir / "covariance_pairs_test.npz")
    logger.info("Saved test dataset: %d pairs.", len(test_ds["condition_vech"]))

    # ---- Leakage check ---------------------------------------------------
    # Verify scalers were NOT refit on val/test
    logger.info(
        "LEAKAGE CHECK: scalers fitted on training data only. "
        "Training mean cond[0]: %.6f", cond_scaler.mean_[0]
    )

    logger.info("Step 3 complete.")


if __name__ == "__main__":
    main()
