"""
plotting.py
-----------
Generate all required figures.

Spec §37.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DIR = Path("results/figures")


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# Validation figures
# ---------------------------------------------------------------------------

def plot_validation_configuration_ranking(
    results_df: pd.DataFrame,
    top_k: int = 20,
) -> None:
    """
    Bar chart of the top-k validation configurations sorted by volatility.
    """
    df = results_df.sort_values("validation_annualized_realized_volatility").head(top_k)
    labels = [
        f"{r['beta_schedule_type'][:3]}_T{r['diffusion_steps_T']}\nα={r['alpha']}_M={r['scenario_count_M']}"
        if not r.get("is_sample_covariance_boundary", False)
        else "SampleCov\nα=1"
        for _, r in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.6), 5))
    bars = ax.bar(range(len(df)), df["validation_annualized_realized_volatility"])
    # Highlight selected
    for i, (_, row) in enumerate(df.iterrows()):
        if row.get("selected_primary_model", False):
            bars[i].set_color("crimson")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Ann. Realized Vol (Validation)")
    ax.set_title("Validation Configuration Ranking\n(crimson = selected)", fontsize=11)
    ax.set_xlabel("Configuration")
    fig.tight_layout()
    _save(fig, "validation_configuration_volatility_ranking.png")


def plot_validation_alpha_sensitivity(results_df: pd.DataFrame) -> None:
    """Median validation vol by alpha."""
    df = results_df[~results_df.get("is_sample_covariance_boundary", False).fillna(False)]
    grouped = df.groupby("alpha")["validation_annualized_realized_volatility"].median()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grouped.index, grouped.values, marker="o")
    ax.set_xlabel("Alpha (historical covariance weight)")
    ax.set_ylabel("Median Ann. Vol (Validation)")
    ax.set_title("Validation Alpha Sensitivity")
    ax.grid(True, alpha=0.3)
    _save(fig, "validation_alpha_sensitivity.png")


def plot_validation_M_sensitivity(results_df: pd.DataFrame) -> None:
    """Median validation vol by M."""
    df = results_df[~results_df.get("is_sample_covariance_boundary", False).fillna(False)]
    grouped = df.groupby("scenario_count_M")["validation_annualized_realized_volatility"].median()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grouped.index, grouped.values, marker="o")
    ax.set_xlabel("M (number of scenarios)")
    ax.set_ylabel("Median Ann. Vol (Validation)")
    ax.set_title("Validation M (Scenario Count) Sensitivity")
    ax.grid(True, alpha=0.3)
    _save(fig, "validation_M_sensitivity.png")


# ---------------------------------------------------------------------------
# Test figures
# ---------------------------------------------------------------------------

def _daily_returns_to_wealth(daily_returns: List[float]) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(daily_returns, dtype=float))


def plot_test_cumulative_wealth(
    returns_dict: Dict[str, List[float]],
    dates: Optional[List[pd.Timestamp]] = None,
    net: bool = False,
) -> None:
    """
    Cumulative wealth curves for all methods.
    """
    suffix = "net_10bps" if net else "gross"
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, rets in returns_dict.items():
        wealth = _daily_returns_to_wealth(rets)
        x = dates[:len(wealth)] if dates is not None else np.arange(len(wealth))
        ax.plot(x, wealth, label=label, linewidth=1.5)

    ax.set_title(
        f"Cumulative Wealth — Test Period (2021–2025), {'Net 10bps' if net else 'Gross'}\n"
        "FINAL UNTOUCHED OUT-OF-SAMPLE RESULTS"
    )
    ax.set_xlabel("Date" if dates is not None else "Trading Day")
    ax.set_ylabel("Cumulative Wealth ($1 initial)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    _save(fig, f"test_cumulative_wealth_{suffix}.png")


def plot_test_rolling_volatility(
    returns_dict: Dict[str, List[float]],
    dates: Optional[List[pd.Timestamp]] = None,
    window: int = 63,
) -> None:
    """Rolling realized volatility."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, rets in returns_dict.items():
        r = pd.Series(rets)
        rolling_vol = r.rolling(window).std() * np.sqrt(252)
        x = dates[:len(rolling_vol)] if dates is not None else rolling_vol.index
        ax.plot(x, rolling_vol.values, label=label, linewidth=1.2)

    ax.set_title(
        f"Rolling {window}-Day Ann. Volatility — Test Period\n"
        "FINAL UNTOUCHED OUT-OF-SAMPLE RESULTS"
    )
    ax.set_ylabel("Ann. Volatility")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if dates is not None:
        fig.autofmt_xdate()
    _save(fig, "test_rolling_63day_volatility.png")


def plot_test_turnover(
    turnovers_dict: Dict[str, List[float]],
    rebalance_dates: Optional[List[pd.Timestamp]] = None,
) -> None:
    """Turnover per rebalance comparison."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, tos in turnovers_dict.items():
        x = rebalance_dates[:len(tos)] if rebalance_dates is not None else range(len(tos))
        ax.plot(x, tos, label=label, marker=".", markersize=4, linewidth=1)
    ax.set_title("Portfolio Turnover per Rebalance — Test Period")
    ax.set_ylabel("One-Way Turnover")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if rebalance_dates is not None:
        fig.autofmt_xdate()
    _save(fig, "test_turnover_comparison.png")


def plot_test_weight_concentration(
    hhi_dict: Dict[str, List[float]],
    rebalance_dates: Optional[List[pd.Timestamp]] = None,
) -> None:
    """Average HHI comparison over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, hhis in hhi_dict.items():
        x = rebalance_dates[:len(hhis)] if rebalance_dates is not None else range(len(hhis))
        ax.plot(x, hhis, label=label, linewidth=1.2)
    ax.set_title("Weight Concentration (HHI) — Test Period")
    ax.set_ylabel("Aggregate HHI")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if rebalance_dates is not None:
        fig.autofmt_xdate()
    _save(fig, "test_weight_concentration_comparison.png")


def plot_test_scenario_dispersion(
    dispersion_df: pd.DataFrame,
) -> None:
    """Scenario dispersion over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    grouped = dispersion_df.groupby("date")["scenario_dispersion"].mean()
    ax.plot(grouped.index, grouped.values, linewidth=1)
    ax.set_title("Mean Scenario Dispersion — Test Period")
    ax.set_ylabel("Mean Frobenius² Dispersion")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _save(fig, "test_scenario_dispersion_over_time.png")
