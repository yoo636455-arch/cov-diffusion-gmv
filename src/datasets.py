"""
datasets.py
-----------
Build and manage the covariance input/target datasets for training,
validation, and testing.

Spec §12–13.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .covariance import build_covariance_pair, build_daily_sliding_pairs_for_group
from .groups import get_sleeve_permnos
from .transforms import (
    batch_covariance_to_log_vech,
    fit_training_scalers,
    load_scalers,
)

logger = logging.getLogger(__name__)

SPLIT_LABELS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Build covariance datasets
# ---------------------------------------------------------------------------

def build_covariance_dataset(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    groups_df: pd.DataFrame,
    group_id_col: str = "group_id",
    rebalance_date_col: str = "rebalance_date",
    lookback_days: int = 126,
    horizon_days: int = 21,
    ridge_epsilon: float = 1e-8,
) -> Dict:
    """
    For each group/sleeve at each rebalance date, compute:
    - historical 126-day sample covariance (conditioning)
    - future 21-day realized covariance proxy (target)

    Returns a dict:
    {
      "condition_raw":  np.ndarray (n, 10, 10),
      "target_raw":     np.ndarray (n, 10, 10),
      "condition_vech": np.ndarray (n, 55),
      "target_vech":    np.ndarray (n, 55),
      "metadata":       pd.DataFrame with columns:
                        [group_id, rebalance_date, industry, permno_list]
    }
    """
    conditions_raw = []
    targets_raw = []
    metadata_rows = []

    group_ids = groups_df[group_id_col].unique()

    for gid in sorted(group_ids):
        grp = groups_df[groups_df[group_id_col] == gid].sort_values("position")
        rebalance_date = grp[rebalance_date_col].iloc[0]
        permnos = grp["permno"].tolist()
        industry = grp["industry"].iloc[0]

        result = build_covariance_pair(
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            permnos=permnos,
            rebalance_date=rebalance_date,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
        )

        if result is None:
            logger.debug(
                "Skipping group %d at %s (missing returns)",
                gid, rebalance_date.date(),
            )
            continue

        S_hist, S_fwd, _, _ = result
        conditions_raw.append(S_hist)
        targets_raw.append(S_fwd)
        metadata_rows.append(
            {
                group_id_col: gid,
                rebalance_date_col: rebalance_date,
                "industry": industry,
                "permno_list": permnos,
            }
        )

    if not conditions_raw:
        logger.warning("No valid covariance pairs built.")
        return {
            "condition_raw": np.empty((0, 10, 10)),
            "target_raw": np.empty((0, 10, 10)),
            "condition_vech": np.empty((0, 55)),
            "target_vech": np.empty((0, 55)),
            "metadata": pd.DataFrame(),
        }

    cond_arr = np.stack(conditions_raw, axis=0)      # (n, 10, 10)
    tgt_arr = np.stack(targets_raw, axis=0)           # (n, 10, 10)
    cond_vech = batch_covariance_to_log_vech(cond_arr, ridge_epsilon)
    tgt_vech = batch_covariance_to_log_vech(tgt_arr, ridge_epsilon)

    meta = pd.DataFrame(metadata_rows)

    logger.info(
        "Built covariance dataset: %d pairs, cond shape %s, tgt shape %s",
        len(cond_arr), cond_vech.shape, tgt_vech.shape,
    )

    return {
        "condition_raw": cond_arr,
        "target_raw": tgt_arr,
        "condition_vech": cond_vech,
        "target_vech": tgt_vech,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Daily-sliding training dataset (20× more data)
# ---------------------------------------------------------------------------

def build_daily_sliding_covariance_dataset(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    groups_df: pd.DataFrame,
    train_end_date: pd.Timestamp,
    lookback_days: int = 126,
    horizon_days: int = 21,
    ridge_epsilon: float = 1e-8,
    stride: int = 1,
) -> Dict:
    """
    Build training covariance pairs using a *daily sliding window*.

    For each group formed at rebalance date ``t_k``, instead of computing
    one pair at ``t_k``, compute one pair for every trading day ``d`` in
    ``[t_k, t_{k+1})`` where both windows are fully available and the
    21-day forward window stays inside the training period.

    This gives ~21× more training observations vs. the 21-day stride.

    Parameters
    ----------
    groups_df      : training_groups.parquet DataFrame
    train_end_date : last calendar date of the training period (inclusive)
    stride         : 1 = daily, 21 = original (rebalance-date only)
    """
    import numpy as np

    # Pre-compute the sorted calendar array and a reverse lookup dict once
    td_sorted = trading_dates.sort_values()
    td_arr = td_sorted.to_numpy()          # numpy datetime64 for fast search
    # Integer position of every trading date
    date_to_idx: dict = {pd.Timestamp(d): i for i, d in enumerate(td_arr)}

    # Last valid anchor: forward window (horizon_days) must end ≤ train_end_date
    # Find the index such that td_arr[idx + horizon_days] <= train_end_date
    train_end_ts = pd.Timestamp(train_end_date)
    # latest anchor index whose fwd window is fully in training
    max_anchor_idx = (
        np.searchsorted(td_arr, np.datetime64(train_end_ts), side="right") - 1
        - horizon_days
    )

    # Build a sorted list of rebalance dates from the groups
    reb_dates = sorted(groups_df["rebalance_date"].unique())
    # Append a sentinel one step past the end
    reb_date_indices = [date_to_idx.get(pd.Timestamp(d)) for d in reb_dates]
    reb_date_indices = [i for i in reb_date_indices if i is not None]

    # For each (group, rebalance_date), identify all valid daily anchor indices
    # in [idx(t_k), idx(t_{k+1})) ∩ [lookback_days, max_anchor_idx]
    def _anchor_indices_for_period(start_idx: int, end_idx: int) -> list[int]:
        """Indices in [start_idx, end_idx) with stride, within valid range."""
        lo = max(start_idx, lookback_days)
        hi = min(end_idx, max_anchor_idx + 1)
        return list(range(lo, hi, stride))

    conditions_raw: list[np.ndarray] = []
    targets_raw:    list[np.ndarray] = []
    metadata_rows:  list[dict]       = []

    group_ids = sorted(groups_df["group_id"].unique())
    logger.info(
        "Building daily-sliding training dataset: %d groups, stride=%d …",
        len(group_ids), stride,
    )

    skipped_pairs = 0
    total_pairs   = 0

    for gid in group_ids:
        grp = groups_df[groups_df["group_id"] == gid].sort_values("position")
        reb_date  = pd.Timestamp(grp["rebalance_date"].iloc[0])
        permnos   = grp["permno"].tolist()
        industry  = grp["industry"].iloc[0]

        reb_idx = date_to_idx.get(reb_date)
        if reb_idx is None:
            continue

        # Holding period end = next rebalance date's index (exclusive)
        pos = reb_date_indices.index(reb_idx) if reb_idx in reb_date_indices else -1
        if pos >= 0 and pos + 1 < len(reb_date_indices):
            next_reb_idx = reb_date_indices[pos + 1]
        else:
            # Last rebalance date: use reb_idx + horizon_days as the end
            next_reb_idx = reb_idx + horizon_days

        anchor_idxs = _anchor_indices_for_period(reb_idx, next_reb_idx)
        if not anchor_idxs:
            continue

        pairs = build_daily_sliding_pairs_for_group(
            crsp_df=crsp_df,
            trading_dates=td_sorted,
            permnos=permnos,
            anchor_indices=anchor_idxs,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
        )

        for S_hist, S_fwd in pairs:
            conditions_raw.append(S_hist)
            targets_raw.append(S_fwd)
            metadata_rows.append(
                {
                    "group_id": gid,
                    "rebalance_date": reb_date,
                    "industry": industry,
                    "permno_list": permnos,
                }
            )

        total_pairs   += len(pairs)
        skipped_pairs += len(anchor_idxs) - len(pairs)

    logger.info(
        "Daily-sliding dataset: %d pairs built, %d anchor dates skipped "
        "(missing returns or boundary).",
        total_pairs, skipped_pairs,
    )

    if not conditions_raw:
        logger.warning("No valid sliding covariance pairs built.")
        return {
            "condition_raw":  np.empty((0, 10, 10)),
            "target_raw":     np.empty((0, 10, 10)),
            "condition_vech": np.empty((0, 55)),
            "target_vech":    np.empty((0, 55)),
            "metadata":       pd.DataFrame(),
        }

    cond_arr = np.stack(conditions_raw, axis=0)   # (n, 10, 10)
    tgt_arr  = np.stack(targets_raw,   axis=0)    # (n, 10, 10)
    cond_vech = batch_covariance_to_log_vech(cond_arr, ridge_epsilon)
    tgt_vech  = batch_covariance_to_log_vech(tgt_arr,  ridge_epsilon)
    meta = pd.DataFrame(metadata_rows)

    logger.info(
        "Sliding training pairs: %d | cond %s | tgt %s",
        len(cond_arr), cond_vech.shape, tgt_vech.shape,
    )
    return {
        "condition_raw":  cond_arr,
        "target_raw":     tgt_arr,
        "condition_vech": cond_vech,
        "target_vech":    tgt_vech,
        "metadata":       meta,
    }


# ---------------------------------------------------------------------------
# Apply scalers
# ---------------------------------------------------------------------------

def apply_scalers(
    dataset: Dict,
    conditioning_scaler,
    target_scaler,
) -> Dict:
    """
    Return a copy of dataset with standardized vech vectors added.

    Adds keys "condition_scaled" and "target_scaled".
    """
    ds = dict(dataset)
    ds["condition_scaled"] = conditioning_scaler.transform(ds["condition_vech"])
    ds["target_scaled"] = target_scaler.transform(ds["target_vech"])
    return ds


# ---------------------------------------------------------------------------
# Save / load NPZ
# ---------------------------------------------------------------------------

def save_dataset(dataset: Dict, path: str | Path) -> None:
    """Save a covariance dataset dict to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert metadata DataFrame to numpy arrays for NPZ storage
    meta = dataset["metadata"]
    save_dict = {
        "condition_raw": dataset["condition_raw"],
        "target_raw": dataset["target_raw"],
        "condition_vech": dataset["condition_vech"],
        "target_vech": dataset["target_vech"],
    }
    if "condition_scaled" in dataset:
        save_dict["condition_scaled"] = dataset["condition_scaled"]
    if "target_scaled" in dataset:
        save_dict["target_scaled"] = dataset["target_scaled"]

    np.savez(path, **save_dict)
    # Save metadata separately as parquet
    if not meta.empty:
        meta_path = path.with_suffix(".meta.parquet")
        # permno_list is a list of lists — convert to string for parquet
        meta2 = meta.copy()
        meta2["permno_list"] = meta2["permno_list"].apply(
            lambda x: ",".join(str(p) for p in x)
        )
        meta2.to_parquet(meta_path, index=False)

    logger.info("Saved dataset to %s", path)


def load_dataset(path: str | Path) -> Dict:
    """Load a covariance dataset dict from a .npz file."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    ds = {k: data[k] for k in data.files}

    meta_path = path.with_suffix(".meta.parquet")
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        meta["permno_list"] = meta["permno_list"].apply(
            lambda s: [int(p) for p in s.split(",")]
        )
        ds["metadata"] = meta
    else:
        ds["metadata"] = pd.DataFrame()

    return ds


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class CovariancePairDataset:
    """
    Minimal iterable dataset returning (condition_scaled, target_scaled) tensors.
    Avoids hard torch import at module level.
    """

    def __init__(self, dataset: Dict):
        import torch
        self.conditions = torch.tensor(
            dataset["condition_scaled"], dtype=torch.float32
        )
        self.targets = torch.tensor(
            dataset["target_scaled"], dtype=torch.float32
        )
        assert len(self.conditions) == len(self.targets)

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, idx: int):
        return self.conditions[idx], self.targets[idx]
