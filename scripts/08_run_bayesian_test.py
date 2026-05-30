"""
08_run_bayesian_test.py
-----------------------
Bayesian Inverse-Wishart covariance estimation backtest (2021-2025 test period).

Implements the 5-step Bayesian IW approach:
    1. Prior:      Ψ = 252-day sample covariance,  ν = 252
    2. Likelihood: S = scatter matrix from 21-day horizon returns
                   S | Σ ~ Wishart(Σ, 21)
    3. Posterior:  Σ|S ~ IW(Ψ + S, ν + 21)
    4. Forecast:   posterior mean = (Ψ + S) / (ν + 21 - N - 1)
    5. Portfolio:  posterior mean → long-only GMV → weights

Compared against:
    - Sample Covariance GMV (126-day lookback)
    - Ledoit-Wolf GMV (126-day lookback)
    - Equal Weight

NOTE: The Bayesian update uses the most recent 21-day historical returns
      available at the rebalance date. This is intended as an operational
      Bayesian IW covariance estimator.

Run from the repo root:
    python scripts/08_run_bayesian_test.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_one_rebalance
from src.bayesian_covariance import bayesian_iw_gmv
from src.benchmarks import equal_weight, ledoit_wolf_linear_gmv
from src.calendar import get_lookback_dates
from src.covariance import compute_return_matrix, compute_sample_covariance
from src.config import get_config
from src.gmv import solve_long_only_gmv
from src.groups import get_sleeve_permnos
from src.metrics import compute_all_metrics, volatility_reduction_vs_sample_gmv
from src.turnover import apply_transaction_costs
from src.utils import get_logger, set_global_seed

logger = get_logger("08_run_bayesian_test", logging.INFO)

LOOKBACK_DAYS = 126   # Ψ and ν use 126-day lookback (matches diffusion model)
LIKELIHOOD_DAYS = 21  # likelihood uses the most recent 21 days of known returns
HORIZON_DAYS  = 21    # holding period horizon for the backtest


# ---------------------------------------------------------------------------
# Generic backtest runner (mirrors script 06)
# ---------------------------------------------------------------------------

def run_method_backtest(
    method_name: str,
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    test_dates: List[pd.Timestamp],
    test_sleeves_df: pd.DataFrame,
    weight_fn,
    horizon_days: int = 21,
    cost_bps: float = 10.0,
) -> Dict:
    all_gross, all_weights, all_turnovers = [], [], []
    rebalance_idxs = []
    repair_log = []
    prev_drifted = None

    for date in test_dates:
        try:
            sleeve_wts, sleeve_perms = weight_fn(date)
        except Exception as exc:
            logger.warning("weight_fn error at %s: %s", date.date(), exc)
            continue
        if not sleeve_wts:
            continue

        daily_rets, agg_wts, drifted, to = run_one_rebalance(
            sleeve_weights=sleeve_wts,
            sleeve_permnos=sleeve_perms,
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            rebalance_date=date,
            horizon_days=horizon_days,
            previous_drifted_weights=prev_drifted,
            repair_log=repair_log,
        )
        if not daily_rets:
            continue

        rebalance_idxs.append(len(all_gross))
        all_gross.extend(daily_rets)
        all_weights.append((date, agg_wts))
        all_turnovers.append((date, to))
        prev_drifted = drifted

    net_returns = apply_transaction_costs(
        daily_returns=all_gross,
        rebalance_indices=rebalance_idxs,
        turnovers=[to for _, to in all_turnovers],
        cost_bps=cost_bps,
    )
    weight_arrays = [np.array(list(wt.values())) for _, wt in all_weights]
    metrics_gross = compute_all_metrics(all_gross, weight_arrays)
    metrics_10bps = compute_all_metrics(net_returns, weight_arrays)
    metrics_gross["average_turnover"] = (
        np.mean([to for _, to in all_turnovers]) if all_turnovers else np.nan
    )

    return {
        "method":            method_name,
        "gross_returns":     all_gross,
        "net_returns_10bps": net_returns,
        "metrics_gross":     metrics_gross,
        "metrics_10bps":     metrics_10bps,
        "weights":           all_weights,
        "turnovers":         all_turnovers,
        "repair_log":        repair_log,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = get_config()

    interim_dir = Path("data/interim")
    out_dir     = Path("results/bayesian")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.random_seed)

    # ---- Load data --------------------------------------------------------
    logger.info("Loading CRSP data and calendar …")
    crsp_df = pd.read_parquet(interim_dir / "cleaned_crsp_daily.parquet")
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    # Indexed version for fast compute_return_matrix lookups
    crsp_indexed = crsp_df.set_index(["date", "permno"]).sort_index()

    cal_df = pd.read_parquet(interim_dir / "trading_calendar.parquet")
    trading_dates = pd.DatetimeIndex(pd.to_datetime(cal_df["date"]).sort_values())

    reb_df = pd.read_parquet(interim_dir / "rebalance_dates.parquet")
    reb_df["rebalance_date"] = pd.to_datetime(reb_df["rebalance_date"])
    test_dates = sorted(reb_df[reb_df["split"] == "test"]["rebalance_date"].tolist())

    logger.info(
        "Test rebalance dates: %d  (first: %s, last: %s)",
        len(test_dates), test_dates[0].date(), test_dates[-1].date(),
    )

    eval_sleeves = pd.read_parquet(interim_dir / "evaluation_sleeves.parquet")
    eval_sleeves["rebalance_date"] = pd.to_datetime(eval_sleeves["rebalance_date"])
    test_sleeves_df = eval_sleeves[
        eval_sleeves["rebalance_date"].isin(test_dates)
    ].copy()

    # ---- Pre-compute sleeve data (252-day lookback + available historical likelihood) ------
    logger.info("Pre-computing sleeve data  (lookback=%d, likelihood=%d, horizon=%d) …",
                LOOKBACK_DAYS, LIKELIHOOD_DAYS, HORIZON_DAYS)
    sleeve_data: Dict[Tuple, Dict] = {}
    skipped = 0

    for date in test_dates:
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        for sid in date_sleeves["sleeve_id"].unique():
            permnos = get_sleeve_permnos(date_sleeves, sid)
            try:
                lb_dates = get_lookback_dates(trading_dates, date, LOOKBACK_DAYS)
            except ValueError:
                skipped += 1
                continue

            R_hist = compute_return_matrix(
                crsp_indexed, trading_dates, permnos, date, lb_dates
            )
            if R_hist is None:
                skipped += 1
                continue

            S_hist = compute_sample_covariance(R_hist)
            R_lik = R_hist[-LIKELIHOOD_DAYS:]

            sleeve_data[(date, sid)] = {
                "permnos": permnos,
                "S_hist":  S_hist,   # (N,N) 252-day sample cov  = Ψ
                "R_hist":  R_hist,   # (252, N) lookback returns  (for LW)
                "R_lik":   R_lik,    # (21,  N) most recent known returns for likelihood
            }

    logger.info(
        "Pre-computed %d sleeve-date pairs  (%d skipped due to missing data).",
        len(sleeve_data), skipped,
    )

    # ---- Weight functions -------------------------------------------------

    def bayesian_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            d = sleeve_data[key]
            wts[sid]  = bayesian_iw_gmv(
                psi=d["S_hist"],
                nu=LOOKBACK_DAYS,
                r_lik=d["R_lik"],
            )
            perms[sid] = d["permnos"]
        return wts, perms

    def sample_gmv_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            wts[sid]  = solve_long_only_gmv(sleeve_data[key]["S_hist"])
            perms[sid] = sleeve_data[key]["permnos"]
        return wts, perms

    def lw_gmv_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            wts[sid]  = ledoit_wolf_linear_gmv(sleeve_data[key]["R_hist"])
            perms[sid] = sleeve_data[key]["permnos"]
        return wts, perms

    def equal_weight_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            wts[sid]  = equal_weight(len(sleeve_data[key]["permnos"]))
            perms[sid] = sleeve_data[key]["permnos"]
        return wts, perms

    methods = {
        "Bayesian_IW_GMV":      bayesian_fn,
        "Sample_Cov_GMV_126d":  sample_gmv_fn,
        "LedoitWolf_GMV_126d":  lw_gmv_fn,
        "Equal_Weight":         equal_weight_fn,
    }

    # ---- Run backtests ----------------------------------------------------
    all_results = {}
    for method_name, weight_fn in methods.items():
        logger.info("Running: %s …", method_name)
        result = run_method_backtest(
            method_name=method_name,
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            test_dates=test_dates,
            test_sleeves_df=test_sleeves_df,
            weight_fn=weight_fn,
            horizon_days=HORIZON_DAYS,
            cost_bps=10.0,
        )
        all_results[method_name] = result
        pd.DataFrame({
            "gross_return":     result["gross_returns"],
            "net_return_10bps": result["net_returns_10bps"],
        }).to_csv(out_dir / f"{method_name}_returns.csv", index=False)
        logger.info(
            "  → vol_gross=%.4f  ret_gross=%.4f  sharpe=%.2f",
            result["metrics_gross"].get("annualized_volatility_gross", float("nan")),
            result["metrics_gross"].get("annualized_return_gross", float("nan")),
            result["metrics_gross"].get("sharpe_gross", float("nan")),
        )

    # ---- Summary table ----------------------------------------------------
    sample_vol = all_results["Sample_Cov_GMV_126d"]["metrics_gross"].get(
        "annualized_volatility_gross", np.nan
    )
    rows = []
    for method_name, result in all_results.items():
        mg = result["metrics_gross"]
        mn = result["metrics_10bps"]
        rows.append({
            "method":                              method_name,
            "annualized_volatility_gross":         mg.get("annualized_volatility_gross"),
            "volatility_reduction_vs_sample_gmv":  volatility_reduction_vs_sample_gmv(
                mg.get("annualized_volatility_gross", np.nan), sample_vol
            ),
            "annualized_return_gross":             mg.get("annualized_return_gross"),
            "sharpe_gross":                        mg.get("sharpe_gross"),
            "cvar_95_daily_gross":                 mg.get("cvar_95_daily_gross"),
            "maximum_drawdown_gross":              mg.get("maximum_drawdown_gross"),
            "average_turnover":                    mg.get("average_turnover"),
            "annualized_return_net_10bps":         mn.get("annualized_return_gross"),
            "annualized_volatility_net_10bps":     mn.get("annualized_volatility_gross"),
            "average_max_stock_weight":            mg.get("average_max_stock_weight"),
            "average_weight_hhi":                  mg.get("average_weight_hhi"),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "bayesian_performance_summary.csv", index=False)

    logger.info("\n=== BAYESIAN IW vs BASELINES  (2021-2025 test period) ===")
    logger.info("\n%s", summary_df[[
        "method", "annualized_volatility_gross",
        "annualized_return_gross", "sharpe_gross",
    ]].to_string(index=False))
    logger.info("\nResults saved to: %s", out_dir)


if __name__ == "__main__":
    main()
