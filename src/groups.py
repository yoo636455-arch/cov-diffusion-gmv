"""
groups.py
---------
Construct training groups and evaluation sleeves.

Spec §10 – Overlapping training groups (50 per date, deterministic seed)
Spec §11 – Non-overlapping evaluation sleeves (sequential blocks of 10)
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training group sampling (overlapping allowed)
# ---------------------------------------------------------------------------

def sample_training_groups(
    universe_at_date: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    group_size: int = 10,
    target_groups: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample same-industry 10-stock groups for diffusion training.
    Overlapping groups are allowed.

    Ordering within each group: descending market cap.

    Parameters
    ----------
    universe_at_date : pd.DataFrame
        Universe snapshot with columns: permno, me, industry.
    rebalance_date : pd.Timestamp
    group_size : int
    target_groups : int
    seed : int

    Returns
    -------
    pd.DataFrame with columns:
        group_id, rebalance_date, industry, position (1..10),
        permno, market_cap
    """
    records = []
    group_counter = 0

    # Per-industry pools
    for industry, ind_df in universe_at_date.groupby("industry"):
        if pd.isna(industry):
            continue
        ind_df = ind_df.sort_values("me", ascending=False).reset_index(drop=True)
        n = len(ind_df)
        if n < group_size:
            continue

        # Deterministic RNG per (industry, date, seed)
        rng_seed = _make_seed(seed, industry, rebalance_date)
        rng = np.random.default_rng(rng_seed)

        # How many groups can we still add?
        remaining = target_groups - group_counter

        # Sample groups randomly (with replacement allowed within an industry)
        # Number of groups to sample from this industry: proportional
        # to industry size, but at most remaining and at least the floor.
        n_groups_here = max(1, min(remaining, int(np.floor(n / group_size))))

        for _ in range(n_groups_here):
            if group_counter >= target_groups:
                break
            # Sample group_size indices without replacement
            chosen_idx = rng.choice(n, size=group_size, replace=False)
            chosen = ind_df.iloc[np.sort(chosen_idx)].copy()
            # Order by descending ME within the group
            chosen = chosen.sort_values("me", ascending=False).reset_index(drop=True)

            for pos, (_, row) in enumerate(chosen.iterrows(), start=1):
                records.append(
                    {
                        "group_id": group_counter,
                        "rebalance_date": rebalance_date,
                        "industry": industry,
                        "position": pos,
                        "permno": row["permno"],
                        "market_cap": row["me"],
                    }
                )
            group_counter += 1

        if group_counter >= target_groups:
            break

    if not records:
        return pd.DataFrame(
            columns=["group_id", "rebalance_date", "industry",
                     "position", "permno", "market_cap"]
        )

    df = pd.DataFrame(records)
    logger.debug(
        "Training groups at %s: %d groups", rebalance_date.date(), group_counter
    )
    return df


def build_all_training_groups(
    universes: pd.DataFrame,
    group_size: int = 10,
    target_groups: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build training groups at all rebalance dates in *universes*.

    Parameters
    ----------
    universes : pd.DataFrame
        Output of build_all_universes, with columns:
        rebalance_date, permno, me, industry.

    Returns
    -------
    pd.DataFrame with a global group_id and all per-group columns.
    """
    parts = []
    global_id = 0

    for date, date_df in universes.groupby("rebalance_date"):
        grps = sample_training_groups(
            universe_at_date=date_df,
            rebalance_date=date,
            group_size=group_size,
            target_groups=target_groups,
            seed=seed,
        )
        if grps.empty:
            continue
        # Re-assign global group IDs
        old_ids = grps["group_id"].unique()
        id_map = {old: global_id + i for i, old in enumerate(sorted(old_ids))}
        grps["group_id"] = grps["group_id"].map(id_map)
        global_id += len(old_ids)
        parts.append(grps)

    if not parts:
        return pd.DataFrame(
            columns=["group_id", "rebalance_date", "industry",
                     "position", "permno", "market_cap"]
        )

    out = pd.concat(parts, ignore_index=True)
    logger.info(
        "Total training groups: %d across %d rebalance dates",
        out["group_id"].nunique(),
        out["rebalance_date"].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# Evaluation sleeve construction (non-overlapping)
# ---------------------------------------------------------------------------

def construct_evaluation_sleeves(
    universe_at_date: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    group_size: int = 10,
) -> pd.DataFrame:
    """
    Create deterministic non-overlapping same-industry sleeves.

    Within each industry:
    1. Sort by market cap descending.
    2. Partition into sequential blocks of exactly *group_size*.
    3. Drop residual stocks.

    Parameters
    ----------
    universe_at_date : pd.DataFrame
        Columns: permno, me, industry.
    rebalance_date : pd.Timestamp
    group_size : int

    Returns
    -------
    pd.DataFrame with columns:
        sleeve_id, rebalance_date, industry, position (1..10),
        permno, market_cap
    """
    records = []
    sleeve_counter = 0

    for industry, ind_df in universe_at_date.groupby("industry"):
        if pd.isna(industry):
            continue
        ind_df = ind_df.sort_values("me", ascending=False).reset_index(drop=True)
        n = len(ind_df)

        n_sleeves = n // group_size
        for s in range(n_sleeves):
            block = ind_df.iloc[s * group_size : (s + 1) * group_size]
            for pos, (_, row) in enumerate(block.iterrows(), start=1):
                records.append(
                    {
                        "sleeve_id": sleeve_counter,
                        "rebalance_date": rebalance_date,
                        "industry": industry,
                        "position": pos,
                        "permno": row["permno"],
                        "market_cap": row["me"],
                    }
                )
            sleeve_counter += 1

    if not records:
        return pd.DataFrame(
            columns=["sleeve_id", "rebalance_date", "industry",
                     "position", "permno", "market_cap"]
        )

    df = pd.DataFrame(records)
    logger.debug(
        "Evaluation sleeves at %s: %d sleeves", rebalance_date.date(), sleeve_counter
    )
    return df


def build_all_evaluation_sleeves(
    universes: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    group_size: int = 10,
) -> pd.DataFrame:
    """
    Build evaluation sleeves at all rebalance dates.

    Returns a DataFrame with a globally unique sleeve_id column.
    """
    parts = []
    global_sid = 0

    for date in rebalance_dates:
        date_df = universes[universes["rebalance_date"] == date]
        if date_df.empty:
            logger.warning("No universe data for rebalance date %s", date.date())
            continue
        sleeves = construct_evaluation_sleeves(date_df, date, group_size=group_size)
        if sleeves.empty:
            continue
        old_ids = sleeves["sleeve_id"].unique()
        id_map = {old: global_sid + i for i, old in enumerate(sorted(old_ids))}
        sleeves["sleeve_id"] = sleeves["sleeve_id"].map(id_map)
        global_sid += len(old_ids)
        parts.append(sleeves)

    if not parts:
        return pd.DataFrame(
            columns=["sleeve_id", "rebalance_date", "industry",
                     "position", "permno", "market_cap"]
        )

    out = pd.concat(parts, ignore_index=True)
    logger.info(
        "Total evaluation sleeves: %d across %d rebalance dates",
        out["sleeve_id"].nunique(),
        out["rebalance_date"].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _make_seed(base_seed: int, industry: int | float, date: pd.Timestamp) -> int:
    """Combine base seed, industry code, and date into a deterministic integer seed."""
    key = f"{base_seed}|{int(industry)}|{date.strftime('%Y%m%d')}"
    digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return digest % (2**31)


def get_sleeve_permnos(
    sleeves_df: pd.DataFrame,
    sleeve_id: int,
) -> List[int]:
    """Return ordered list of PERMNOs for a given sleeve_id."""
    sub = sleeves_df[sleeves_df["sleeve_id"] == sleeve_id].sort_values("position")
    return sub["permno"].tolist()
