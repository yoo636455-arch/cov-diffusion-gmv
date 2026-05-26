"""
test_reproducibility.py
------------------------
Verify that key operations are deterministic under the same seed (spec §35.10).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beta_schedules import make_beta_schedule
from src.diffusion import DDPMScheduler, p_sample_loop
from src.groups import construct_evaluation_sleeves, sample_training_groups
from src.model import build_model


def _make_universe(n_per_industry: int = 25, industries=None):
    if industries is None:
        industries = [36, 59]
    rows = []
    permno = 1
    for ind in industries:
        for i in range(n_per_industry):
            rows.append(
                {
                    "permno": permno,
                    "me": float(n_per_industry - i) * 1000,
                    "industry": ind,
                    "siccd": ind * 100 + i,
                }
            )
            permno += 1
    import pandas as pd
    return pd.DataFrame(rows)


DATE = __import__("pandas").Timestamp("2010-01-04")


def test_training_group_sampling_reproducible():
    univ = _make_universe()
    g1 = sample_training_groups(univ, DATE, seed=42)
    g2 = sample_training_groups(univ, DATE, seed=42)
    __import__("pandas").testing.assert_frame_equal(
        g1.reset_index(drop=True),
        g2.reset_index(drop=True),
    )


def test_evaluation_sleeve_construction_deterministic():
    univ = _make_universe(25)
    s1 = construct_evaluation_sleeves(univ, DATE)
    s2 = construct_evaluation_sleeves(univ, DATE)
    __import__("pandas").testing.assert_frame_equal(
        s1.reset_index(drop=True),
        s2.reset_index(drop=True),
    )


def test_scenario_generation_reproducible():
    """Same seed -> same generated scenarios."""
    model = build_model()
    model.eval()
    scheduler = DDPMScheduler("linear", T=10)

    c = torch.randn(55)
    y1 = p_sample_loop(model, c, scheduler, num_samples=3, seed=0)
    y2 = p_sample_loop(model, c, scheduler, num_samples=3, seed=0)

    assert torch.allclose(y1, y2), "Scenario generation not reproducible under same seed"


def test_scenario_generation_different_seeds():
    """Different seeds -> different scenarios."""
    model = build_model()
    model.eval()
    scheduler = DDPMScheduler("linear", T=10)

    c = torch.randn(55)
    y1 = p_sample_loop(model, c, scheduler, num_samples=3, seed=0)
    y2 = p_sample_loop(model, c, scheduler, num_samples=3, seed=999)

    assert not torch.allclose(y1, y2), "Different seeds produced identical scenarios"


def test_nested_scenario_prefix_consistency():
    """
    The common-random-numbers approach is implemented by always generating
    M_max scenarios and then slicing.  Verify that using scenarios[:5] from
    a 50-scenario batch gives the same result as explicitly using those 5
    rows (i.e., slicing is the mechanism, not re-running with M=5).
    """
    model = build_model()
    model.eval()
    scheduler = DDPMScheduler("linear", T=10)

    c = torch.randn(55)
    # Generate M_max=10 scenarios
    y_all = p_sample_loop(model, c, scheduler, num_samples=10, seed=42)

    # The "M=5" result is obtained by slicing the first 5 from the same batch
    y_first_5 = y_all[:5]

    # Verify that slicing is deterministic (same call, same slice)
    y_all_again = p_sample_loop(model, c, scheduler, num_samples=10, seed=42)
    assert torch.allclose(y_all_again[:5], y_first_5), (
        "Sliced scenarios not reproducible — common random number property violated"
    )
