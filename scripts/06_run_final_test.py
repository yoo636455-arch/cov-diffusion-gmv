"""
06_run_final_test.py
---------------------
Phase 6: Run the FINAL UNTOUCHED test backtest (2021–2025).

Evaluates:
- Proposed method (validation-selected configuration)
- Equal Weight benchmark
- Sample Covariance GMV benchmark
- Ledoit-Wolf Linear Shrinkage GMV benchmark
- Required ablations (pure diffusion, single scenario, pure sample cov)

DO NOT modify this script after inspecting test results.

Outputs
-------
results/test/test_performance_summary.csv
results/portfolios/  (daily returns, weights)
results/diagnostics/  (scenario dispersion, forecast losses, repairs)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_one_rebalance
from src.benchmarks import (
    equal_weight,
    ledoit_wolf_linear_gmv,
    sample_covariance_gmv,
)
from src.covariance import build_covariance_pair, get_future_returns
from src.diagnostics import collect_diagnostics_record
from src.diffusion import DDPMScheduler
from src.generate import (
    combine_covariances,
    deterministic_scenario_seed,
    generate_covariance_scenarios,
)
from src.gmv import solve_long_only_gmv
from src.groups import get_sleeve_permnos
from src.metrics import (
    annualized_volatility,
    compute_all_metrics,
    volatility_reduction_vs_sample_gmv,
)
from src.train import load_trained_model
from src.transforms import covariance_to_log_vech, load_scalers
from src.turnover import apply_transaction_costs, compute_turnover
from src.config import get_config
from src.utils import get_device, get_logger, set_global_seed

logger = get_logger("06_run_final_test", logging.INFO)


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
    """
    Generic backtest runner. weight_fn(date, sleeve_id) -> (weights_dict, permnos_dict).
    """
    all_gross = []
    all_weights = []
    all_turnovers = []
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

    # Compute metrics at multiple cost levels
    weight_arrays = [np.array(list(wt.values())) for _, wt in all_weights]
    metrics_gross = compute_all_metrics(all_gross, weight_arrays)
    metrics_10bps = compute_all_metrics(net_returns, weight_arrays)

    avg_turnover = np.mean([to for _, to in all_turnovers]) if all_turnovers else np.nan
    metrics_gross["average_turnover"] = avg_turnover

    return {
        "method": method_name,
        "gross_returns": all_gross,
        "net_returns_10bps": net_returns,
        "metrics_gross": metrics_gross,
        "metrics_10bps": metrics_10bps,
        "weights": all_weights,
        "turnovers": all_turnovers,
        "repair_log": repair_log,
    }


def main() -> None:
    cfg = get_config()

    interim_dir = Path("data/interim")
    selected_dir = Path("artifacts/selected_model")
    scaler_dir = Path("artifacts/scalers")
    test_dir = Path("results/test")
    portfolio_dir = Path("results/portfolios")
    diag_dir = Path("results/diagnostics")
    ablation_dir = Path("results/ablations")

    for d in [test_dir, portfolio_dir, diag_dir, ablation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.random_seed)
    device = get_device()

    logger.info("=" * 60)
    logger.info("STEP 6 – FINAL TEST BACKTEST (2021–2025)")
    logger.info("THIS DATA HAS NEVER BEEN SEEN BEFORE.")
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    # ---- Load selected config --------------------------------------------
    with open(selected_dir / "selected_model_config.yaml") as fh:
        sel_cfg = yaml.safe_load(fh)["selected_model"]

    logger.info("Selected configuration: %s", sel_cfg)

    # ---- Load data -------------------------------------------------------
    crsp_df = pd.read_parquet(interim_dir / "cleaned_crsp_daily.parquet")
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    cal_df = pd.read_parquet(interim_dir / "trading_calendar.parquet")
    trading_dates = pd.DatetimeIndex(pd.to_datetime(cal_df["date"]).sort_values())
    reb_df = pd.read_parquet(interim_dir / "rebalance_dates.parquet")
    reb_df["rebalance_date"] = pd.to_datetime(reb_df["rebalance_date"])

    test_dates = sorted(
        reb_df[reb_df["split"] == "test"]["rebalance_date"].tolist()
    )
    logger.info("Test rebalance dates: %d (first: %s, last: %s)",
                len(test_dates), test_dates[0].date(), test_dates[-1].date())

    eval_sleeves = pd.read_parquet(interim_dir / "evaluation_sleeves.parquet")
    eval_sleeves["rebalance_date"] = pd.to_datetime(eval_sleeves["rebalance_date"])
    test_sleeves_df = eval_sleeves[
        eval_sleeves["rebalance_date"].isin(test_dates)
    ].copy()

    cond_scaler, tgt_scaler = load_scalers(scaler_dir)
    lkb = cfg.rolling_windows["lookback_days"]
    hor = cfg.rolling_windows["horizon_days"]

    # ---- Pre-compute sleeve data -----------------------------------------
    logger.info("Pre-computing test sleeve data …")
    sleeve_data: Dict[Tuple, Dict] = {}

    for date in test_dates:
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        for sid in date_sleeves["sleeve_id"].unique():
            permnos = get_sleeve_permnos(date_sleeves, sid)
            result = build_covariance_pair(
                crsp_df, trading_dates, permnos, date, lookback_days=lkb, horizon_days=hor
            )
            if result is None:
                continue
            S_hist, S_fwd, R_hist, _ = result
            sleeve_data[(date, sid)] = {
                "permnos": permnos,
                "S_hist": S_hist,
                "S_fwd": S_fwd,
                "R_hist": R_hist,
            }

    # ---- Load selected model and generate scenarios ----------------------
    sched = sel_cfg["beta_schedule_type"]
    T_sel = sel_cfg["diffusion_steps_T"]
    alpha_star = sel_cfg["alpha"]
    M_star = sel_cfg["scenario_count_M"]

    is_pure_sample = (sched == "not_applicable") or (alpha_star == 1.0)

    diffusion_scenarios: Dict[Tuple, List[np.ndarray]] = {}
    diagnostics_records = []

    if not is_pure_sample:
        ckpt_path = selected_dir / "selected_model.pt"
        if not ckpt_path.exists():
            # Fall back to model dir
            ckpt_path = Path("artifacts/models") / f"ddpm_schedule-{sched}_T-{T_sel}_seed-{cfg.random_seed}.pt"

        logger.info("Loading selected model from %s …", ckpt_path)
        model, _ = load_trained_model(ckpt_path, device=str(device))
        model.eval()
        scheduler = DDPMScheduler(
            schedule_type=sched, T=T_sel,
            beta_min=cfg.training["beta_min"],
            beta_max=cfg.training["beta_max"],
            device=device,
        )

        logger.info("Generating scenarios for test sleeves (M_max=50) …")
        M_max = 50
        for (date, sid), data in sleeve_data.items():
            try:
                cov_vech = covariance_to_log_vech(data["S_hist"], cfg.ridge_epsilon)
            except Exception:
                continue
            gen_seed = deterministic_scenario_seed(sched, T_sel, date, sid, cfg.random_seed)
            scenarios = generate_covariance_scenarios(
                model=model, scheduler=scheduler,
                condition_vector_raw=cov_vech,
                conditioning_scaler=cond_scaler,
                target_scaler=tgt_scaler,
                num_scenarios=M_max,
                seed=gen_seed,
            )
            diffusion_scenarios[(date, sid)] = scenarios

            # Diagnostics
            combined = combine_covariances(data["S_hist"], scenarios[:M_star], alpha_star)
            rec = collect_diagnostics_record(
                date=date, sleeve_id=sid,
                generated_covariances=scenarios[:M_star],
                combined_cov=combined,
                sample_cov=data["S_hist"],
                realized_cov=data["S_fwd"],
                ridge_epsilon=cfg.ridge_epsilon,
            )
            diagnostics_records.append(rec)

        diag_df = pd.DataFrame(diagnostics_records)
        diag_df.to_csv(diag_dir / "scenario_dispersion.csv", index=False)
        diag_df.to_csv(diag_dir / "covariance_forecast_losses.csv", index=False)
        logger.info("Saved diagnostics.")

    # ---- Define weight functions -----------------------------------------

    def make_proposed_weight_fn(alpha, M):
        def fn(date):
            date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
            wts, perms = {}, {}
            for sid in date_sleeves["sleeve_id"].unique():
                key = (date, sid)
                if key not in sleeve_data:
                    continue
                data = sleeve_data[key]
                if is_pure_sample or key not in diffusion_scenarios:
                    cov = data["S_hist"]
                else:
                    scenarios_M = diffusion_scenarios[key][:M]
                    cov = combine_covariances(data["S_hist"], scenarios_M, alpha)
                w = solve_long_only_gmv(cov)
                wts[sid] = w
                perms[sid] = data["permnos"]
            return wts, perms
        return fn

    def equal_weight_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            n = len(sleeve_data[key]["permnos"])
            wts[sid] = equal_weight(n)
            perms[sid] = sleeve_data[key]["permnos"]
        return wts, perms

    def sample_gmv_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            w = solve_long_only_gmv(sleeve_data[key]["S_hist"])
            wts[sid] = w
            perms[sid] = sleeve_data[key]["permnos"]
        return wts, perms

    def lw_gmv_fn(date):
        date_sleeves = test_sleeves_df[test_sleeves_df["rebalance_date"] == date]
        wts, perms = {}, {}
        for sid in date_sleeves["sleeve_id"].unique():
            key = (date, sid)
            if key not in sleeve_data:
                continue
            R_hist = sleeve_data[key]["R_hist"]
            w = ledoit_wolf_linear_gmv(R_hist)
            wts[sid] = w
            perms[sid] = sleeve_data[key]["permnos"]
        return wts, perms

    # ---- Run all methods --------------------------------------------------
    methods = {
        "Stabilized_Diffusion_GMV": make_proposed_weight_fn(alpha_star, M_star),
        "Equal_Weight": equal_weight_fn,
        "Sample_Covariance_GMV": sample_gmv_fn,
        "LedoitWolf_Linear_GMV": lw_gmv_fn,
    }

    # Ablations
    if not is_pure_sample:
        methods["Ablation_Pure_Diffusion_alpha0"] = make_proposed_weight_fn(0.0, M_star)
        methods["Ablation_Single_Scenario_M1"] = make_proposed_weight_fn(alpha_star, 1)

    methods["Ablation_Pure_Sample_alpha1"] = sample_gmv_fn

    all_results = {}
    for method_name, weight_fn in methods.items():
        logger.info("\nRunning: %s …", method_name)
        result = run_method_backtest(
            method_name=method_name,
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            test_dates=test_dates,
            test_sleeves_df=test_sleeves_df,
            weight_fn=weight_fn,
            horizon_days=hor,
            cost_bps=10.0,
        )
        all_results[method_name] = result

        # Save daily returns
        pd.DataFrame({
            "gross_return": result["gross_returns"],
            "net_return_10bps": result["net_returns_10bps"],
        }).to_csv(portfolio_dir / f"{method_name}_returns.csv", index=False)

    # ---- Build summary table ---------------------------------------------
    sample_vol = all_results["Sample_Covariance_GMV"]["metrics_gross"].get(
        "annualized_volatility_gross", np.nan
    )
    rows = []
    for method_name, result in all_results.items():
        mg = result["metrics_gross"]
        mn = result["metrics_10bps"]
        row = {
            "method": method_name,
            "annualized_volatility_gross": mg.get("annualized_volatility_gross"),
            "volatility_reduction_vs_sample_gmv": volatility_reduction_vs_sample_gmv(
                mg.get("annualized_volatility_gross", np.nan), sample_vol
            ),
            "annualized_return_gross": mg.get("annualized_return_gross"),
            "sharpe_gross": mg.get("sharpe_gross"),
            "cvar_95_daily_gross": mg.get("cvar_95_daily_gross"),
            "maximum_drawdown_gross": mg.get("maximum_drawdown_gross"),
            "average_turnover": mg.get("average_turnover"),
            "annualized_return_net_10bps": mn.get("annualized_return_gross"),
            "annualized_volatility_net_10bps": mn.get("annualized_volatility_gross"),
            "average_max_stock_weight": mg.get("average_max_stock_weight"),
            "average_weight_hhi": mg.get("average_weight_hhi"),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(test_dir / "test_performance_summary.csv", index=False)

    logger.info("\n=== TEST PERFORMANCE SUMMARY ===")
    logger.info(
        summary_df[["method", "annualized_volatility_gross",
                     "annualized_return_gross", "sharpe_gross"]].to_string()
    )

    # Save repair log
    all_repairs = []
    for method_name, result in all_results.items():
        for rec in result["repair_log"]:
            rec["method"] = method_name
            all_repairs.append(rec)
    if all_repairs:
        pd.DataFrame(all_repairs).to_csv(diag_dir / "covariance_repairs.csv", index=False)

    logger.info("Step 6 complete.")


if __name__ == "__main__":
    main()
