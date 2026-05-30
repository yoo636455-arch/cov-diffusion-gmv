"""
09_compare_diffusion_vs_bayesian.py
-------------------------------------
Direct comparison of brhie's SDEdit diffusion model vs Bayesian IW GMV,
both run on the same 2021–2025 test sleeves under identical conditions.

Methods compared:
  - brhie_SDEdit_GMV     : brhie unconditional DDPM, rho=0.1, T=50, M=20, quadratic
  - Bayesian_IW_GMV      : Inverse-Wishart conjugate posterior mean
  - Sample_Cov_GMV_126d  : raw 126-day sample covariance
  - LedoitWolf_GMV_126d  : Ledoit-Wolf linear shrinkage
  - Equal_Weight

Run from repo root:
    python scripts/09_compare_diffusion_vs_bayesian.py
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_one_rebalance
from src.bayesian_covariance import bayesian_iw_gmv
from src.benchmarks import equal_weight, ledoit_wolf_linear_gmv
from src.calendar import get_lookback_dates
from src.covariance import compute_return_matrix, compute_sample_covariance
from src.config import get_config
from src.diffusion import DDPMScheduler
from src.generate import generate_denoised_covariances, deterministic_scenario_seed
from src.gmv import solve_long_only_gmv
from src.groups import get_sleeve_permnos
from src.metrics import compute_all_metrics, volatility_reduction_vs_sample_gmv
from src.model import UnconditionalMLPDenoiser, build_denoising_model
from src.transforms import covariance_to_log_vech
from src.turnover import apply_transaction_costs
from src.utils import get_logger, set_global_seed

logger = get_logger("09_compare_diffusion_vs_bayesian", logging.INFO)

LOOKBACK_DAYS   = 126
LIKELIHOOD_DAYS = 21
HORIZON_DAYS    = 21
RIDGE_EPS       = 1e-8

# brhie selected model config
BRHIE_SCHEDULE  = "quadratic"
BRHIE_T         = 50
BRHIE_RHO       = 0.1
BRHIE_M         = 20


# ---------------------------------------------------------------------------
# Backtest runner (same structure as script 08)
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
    set_global_seed(cfg.random_seed)

    interim_dir  = Path("data/interim")
    brhie_dir    = Path("artifacts/brhie")
    out_dir      = Path("results/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load brhie model and scaler ----------------------------------------
    logger.info("Loading brhie SDEdit model …")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = build_denoising_model()
    state = torch.load(brhie_dir / "selected_denoising_model.pt", map_location="cpu", weights_only=False)
    # state may be a plain state_dict or a dict with a "model_state_dict" key
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    with open(brhie_dir / "conditioning_scaler.pkl", "rb") as f:
        conditioning_scaler = pickle.load(f)

    scheduler = DDPMScheduler(
        schedule_type=BRHIE_SCHEDULE,
        T=BRHIE_T,
        device=device,
    )
    logger.info(
        "brhie model loaded: schedule=%s  T=%d  rho=%.2f  M=%d  device=%s",
        BRHIE_SCHEDULE, BRHIE_T, BRHIE_RHO, BRHIE_M, device,
    )

    # ---- Load data ----------------------------------------------------------
    logger.info("Loading CRSP data and evaluation sleeves …")
    crsp_df = pd.read_parquet(interim_dir / "cleaned_crsp_daily.parquet")
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    crsp_indexed = crsp_df.set_index(["date", "permno"]).sort_index()

    cal_df = pd.read_parquet(interim_dir / "trading_calendar.parquet")
    trading_dates = pd.DatetimeIndex(pd.to_datetime(cal_df["date"]).sort_values())

    reb_df = pd.read_parquet(interim_dir / "rebalance_dates.parquet")
    reb_df["rebalance_date"] = pd.to_datetime(reb_df["rebalance_date"])
    test_dates = sorted(reb_df[reb_df["split"] == "test"]["rebalance_date"].tolist())
    logger.info("Test dates: %d  (%s → %s)",
                len(test_dates), test_dates[0].date(), test_dates[-1].date())

    eval_sleeves = pd.read_parquet(interim_dir / "evaluation_sleeves.parquet")
    eval_sleeves["rebalance_date"] = pd.to_datetime(eval_sleeves["rebalance_date"])
    test_sleeves_df = eval_sleeves[eval_sleeves["rebalance_date"].isin(test_dates)].copy()

    # ---- Pre-compute sleeve data -------------------------------------------
    logger.info("Pre-computing sleeve data (lookback=%d) …", LOOKBACK_DAYS)
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

            R_hist = compute_return_matrix(crsp_indexed, trading_dates, permnos, date, lb_dates)
            if R_hist is None:
                skipped += 1
                continue

            S_hist = compute_sample_covariance(R_hist)
            cond_vec = covariance_to_log_vech(S_hist, RIDGE_EPS)

            sleeve_data[(date, sid)] = {
                "permnos":  permnos,
                "S_hist":   S_hist,
                "R_hist":   R_hist,
                "R_lik":    R_hist[-LIKELIHOOD_DAYS:],
                "cond_vec": cond_vec,
            }

    logger.info("Pre-computed %d sleeve-dates (%d skipped).", len(sleeve_data), skipped)

    # ---- Weight functions --------------------------------------------------

    def brhie_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            d = sleeve_data[key]
            seed = deterministic_scenario_seed(BRHIE_SCHEDULE, BRHIE_T, date, sid)
            denoised = generate_denoised_covariances(
                model=model,
                scheduler=scheduler,
                condition_vector_raw=d["cond_vec"],
                conditioning_scaler=conditioning_scaler,
                rho=BRHIE_RHO,
                num_draws=BRHIE_M,
                seed=seed,
                device=device,
            )
            cov_mean = np.mean(np.stack(denoised, axis=0), axis=0)
            cov_mean = 0.5 * (cov_mean + cov_mean.T)
            wts[sid]  = solve_long_only_gmv(cov_mean)
            perms[sid] = d["permnos"]
        return wts, perms

    def bayesian_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            d = sleeve_data[key]
            wts[sid]  = bayesian_iw_gmv(psi=d["S_hist"], nu=LOOKBACK_DAYS, r_lik=d["R_lik"])
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

    def lw_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            wts[sid]  = ledoit_wolf_linear_gmv(sleeve_data[key]["R_hist"])
            perms[sid] = sleeve_data[key]["permnos"]
        return wts, perms

    def ew_fn(date):
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
        "brhie_SDEdit_GMV":    brhie_fn,
        "Bayesian_IW_GMV":     bayesian_fn,
        "Sample_Cov_GMV_126d": sample_gmv_fn,
        "LedoitWolf_GMV_126d": lw_fn,
        "Equal_Weight":        ew_fn,
    }

    # ---- Run backtests -----------------------------------------------------
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
            "  → vol=%.4f  ret=%.4f  sharpe=%.3f",
            result["metrics_gross"].get("annualized_volatility_gross", float("nan")),
            result["metrics_gross"].get("annualized_return_gross", float("nan")),
            result["metrics_gross"].get("sharpe_gross", float("nan")),
        )

    # ---- Summary table -----------------------------------------------------
    sample_vol = all_results["Sample_Cov_GMV_126d"]["metrics_gross"].get(
        "annualized_volatility_gross", np.nan
    )
    rows = []
    for name, result in all_results.items():
        mg = result["metrics_gross"]
        mn = result["metrics_10bps"]
        rows.append({
            "method":                             name,
            "annualized_volatility_gross":        mg.get("annualized_volatility_gross"),
            "volatility_reduction_vs_sample_gmv": volatility_reduction_vs_sample_gmv(
                mg.get("annualized_volatility_gross", np.nan), sample_vol
            ),
            "annualized_return_gross":            mg.get("annualized_return_gross"),
            "sharpe_gross":                       mg.get("sharpe_gross"),
            "cvar_95_daily_gross":                mg.get("cvar_95_daily_gross"),
            "maximum_drawdown_gross":             mg.get("maximum_drawdown_gross"),
            "average_turnover":                   mg.get("average_turnover"),
            "annualized_return_net_10bps":        mn.get("annualized_return_gross"),
            "annualized_volatility_net_10bps":    mn.get("annualized_volatility_gross"),
            "average_max_stock_weight":           mg.get("average_max_stock_weight"),
            "average_weight_hhi":                 mg.get("average_weight_hhi"),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "comparison_performance_summary.csv", index=False)

    logger.info("\n=== DIFFUSION vs BAYESIAN vs BENCHMARKS  (2021-2025) ===")
    logger.info("\n%s", summary_df[[
        "method", "annualized_volatility_gross",
        "annualized_return_gross", "sharpe_gross",
    ]].to_string(index=False))
    logger.info("\nResults saved to: %s", out_dir)


if __name__ == "__main__":
    main()
