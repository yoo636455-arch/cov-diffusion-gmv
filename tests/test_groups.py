"""
test_groups.py
--------------
Tests for sleeve construction (spec §35.4).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.groups import construct_evaluation_sleeves, sample_training_groups


def _make_universe(n_per_industry: int = 25, industries: list = None):
    if industries is None:
        industries = [36, 59, 73]
    rows = []
    permno = 1
    for ind in industries:
        for i in range(n_per_industry):
            rows.append(
                {
                    "permno": permno,
                    "me": float(n_per_industry - i) * 1000,  # descending ME
                    "industry": ind,
                    "siccd": ind * 100 + i,
                }
            )
            permno += 1
    return pd.DataFrame(rows)


DATE = pd.Timestamp("2010-01-04")


class TestEvaluationSleeves:
    def test_each_sleeve_has_exactly_10_stocks(self):
        univ = _make_universe(25)
        sleeves = construct_evaluation_sleeves(univ, DATE, group_size=10)
        counts = sleeves.groupby("sleeve_id").size()
        assert (counts == 10).all(), f"Sleeve sizes: {counts.to_dict()}"

    def test_each_sleeve_has_one_industry(self):
        univ = _make_universe(25)
        sleeves = construct_evaluation_sleeves(univ, DATE, group_size=10)
        n_industries = sleeves.groupby("sleeve_id")["industry"].nunique()
        assert (n_industries == 1).all()

    def test_positions_ordered_by_descending_market_cap(self):
        univ = _make_universe(10)
        sleeves = construct_evaluation_sleeves(univ, DATE, group_size=10)
        for sid, grp in sleeves.groupby("sleeve_id"):
            grp_sorted = grp.sort_values("position")
            mes = grp_sorted["market_cap"].values
            assert np.all(mes[:-1] >= mes[1:]), (
                f"Sleeve {sid}: positions not in descending ME order"
            )

    def test_no_overlap_between_sleeves_at_same_date(self):
        univ = _make_universe(30)
        sleeves = construct_evaluation_sleeves(univ, DATE, group_size=10)
        # No permno should appear in two different sleeves at same date
        perm_counts = sleeves.groupby("permno")["sleeve_id"].nunique()
        assert (perm_counts == 1).all(), "Permno appears in multiple sleeves"

    def test_residual_stocks_dropped(self):
        # 25 stocks / 10 = 2 sleeves, 5 residual
        univ = _make_universe(n_per_industry=25, industries=[36])
        sleeves = construct_evaluation_sleeves(univ, DATE, group_size=10)
        assert sleeves["sleeve_id"].nunique() == 2
        assert len(sleeves) == 20


class TestTrainingGroups:
    def test_training_groups_have_exactly_10_stocks(self):
        univ = _make_universe(30)
        groups = sample_training_groups(univ, DATE, group_size=10, target_groups=5, seed=42)
        counts = groups.groupby("group_id").size()
        assert (counts == 10).all()

    def test_training_groups_same_industry(self):
        univ = _make_universe(30)
        groups = sample_training_groups(univ, DATE, group_size=10, target_groups=5, seed=42)
        n_industries = groups.groupby("group_id")["industry"].nunique()
        assert (n_industries == 1).all()

    def test_training_deterministic_seed(self):
        univ = _make_universe(30)
        g1 = sample_training_groups(univ, DATE, group_size=10, seed=42)
        g2 = sample_training_groups(univ, DATE, group_size=10, seed=42)
        pd.testing.assert_frame_equal(g1.reset_index(drop=True), g2.reset_index(drop=True))
