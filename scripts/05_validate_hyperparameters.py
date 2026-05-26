"""
05_validate_hyperparameters.py
--------------------------------
Phase 5: Evaluate all 81 effective validation configurations.

For each of 4 trained models (linear schedule, T ∈ {400, 800, 1200, 2000})
× (4 alpha × 5 M) = 80 diffusion-blend configurations,
plus 1 sample-covariance boundary (alpha=1):
  - Generate M_max=50 scenarios per validation sleeve per model.
  - Construct combined covariances using nested M subsets.
  - Solve long-only GMV.
  - Compute annualized realized portfolio volatility.
  - Rank and select the single best configuration.

Disk caching
------------
Expensive intermediate results are saved so re-runs skip recomputation:

  results/validation/cache/sample_covs.pkl
      Sample covariances + permno lists for all validation sleeve-dates.

  results/validation/cache/scenarios_schedule-<sched>_T-<T>.pkl
      50 generated scenario covariance matrices per sleeve-date, per model.

Delete any of these files to force full recomputation of that stage.

Outputs
-------
results/validation/validation_grid_results.csv
results/validation/top5_validation_configurations.csv
artifacts/selected_model/selected_model_config.yaml
artifacts/selected_model/selected_model.pt  (copy of selected checkpoint)
"""

from __future__ import annotations

import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from tqdm.auto import tqdm as _tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_one_rebalance
from src.config import get_config
from src.covariance import build_covariance_pair
from src.diffusion import DDPMScheduler
from src.generate import (
    combine_covariances,
    deterministic_scenario_seed,
    generate_covariance_scenarios,
)
from src.gmv import solve_long_only_gmv
from src.groups import get_sleeve_permnos
from src.metrics import annualized_volatility
from src.train import load_trained_model
from src.transforms import covariance_to_log_vech, load_scalers
from src.utils import get_device, get_logger, set_global_seed

logger = get_logger("05_validate_hyperparameters", logging.INFO)


# ---------------------------------------------------------------------------
# Helper: run portfolio backtest for given weights
# ---------------------------------------------------------------------------

def run_validation_backtest(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    val_rebalance_dates: List[pd.Timestamp],
    sleeve_weights_by_date: Dict[pd.Timestamp, Dict[int, np.ndarray]],
    sleeve_permnos_by_date: Dict[pd.Timestamp, Dict[int, List[int]]],
    horizon_days: int = 21,
) -> List[float]:
    """Compute the full validation daily return stream for given weights."""
    all_returns: List[float] = []
    for date in val_rebalance_dates:
        if date not in sleeve_weights_by_date:
            continue
        wts = sleeve_weights_by_date[date]
        perms = sleeve_permnos_by_date.get(date, {})
        if not wts:
            continue
        daily_rets, _, _, _ = run_one_rebalance(
            sleeve_weights=wts,
            sleeve_permnos=perms,
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            rebalance_date=date,
            horizon_days=horizon_days,
        )
        all_returns.extend(daily_rets)
    return all_returns


# ---------------------------------------------------------------------------
# Helper: pretty-print current leaderboard
# ---------------------------------------------------------------------------

def _print_leaderboard(results: list[dict], top_n: int = 5) -> None:
    if not results:
        return
    df = pd.DataFrame(results).sort_values("validation_annualized_realized_volatility")
    top = df.head(top_n)
    lines = ["", "  ── Current top-{} ──────────────────────────────────".format(top_n)]
    lines.append("  {:>3}  {:>12}  {:>4}  {:>5}  {:>3}  {:>9}".format(
        "rk", "schedule", "T", "alpha", "M", "val_vol"
    ))
    lines.append("  " + "-" * 48)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        sched = str(row["beta_schedule_type"])[:12]
        T_    = int(row["diffusion_steps_T"]) if pd.notna(row.get("diffusion_steps_T")) else "—"
        alpha = row["alpha"]
        M_    = int(row["scenario_count_M"]) if pd.notna(row.get("scenario_count_M")) else "—"
        vol   = row["validation_annualized_realized_volatility"]
        lines.append("  {:>3}  {:>12}  {:>4}  {:>5.2f}  {:>3}  {:>9.6f}".format(
            rank, sched, T_, alpha, M_, vol
        ))
    lines.append("")
    msg = "\n".join(lines)
    if _has_tqdm:
        _tqdm.write(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = get_config()

    interim_dir     = Path("data/interim")
    model_dir       = Path("artifacts/models")
    scaler_dir      = Path("artifacts/scalers")
    val_results_dir = Path("results/validation")
    selected_dir    = Path("artifacts/selected_model")
    cache_dir       = val_results_dir / "cache"

    for d in [val_results_dir, selected_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.random_seed)
    device = get_device()

    logger.info("=" * 60)
    logger.info("STEP 5 – Validate hyperparameters (2014–2020)")
    logger.info("Device: %s", device)
    logger.info("Cache dir: %s", cache_dir)
    logger.info("=" * 60)

    # ---- Load data -------------------------------------------------------
    crsp_df = pd.read_parquet(interim_dir / "cleaned_crsp_daily.parquet")
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    cal_df  = pd.read_parquet(interim_dir / "trading_calendar.parquet")
    trading_dates = pd.DatetimeIndex(pd.to_datetime(cal_df["date"]).sort_values())
    reb_df  = pd.read_parquet(interim_dir / "rebalance_dates.parquet")
    reb_df["rebalance_date"] = pd.to_datetime(reb_df["rebalance_date"])

    val_dates_all = sorted(
        reb_df[reb_df["split"] == "validation"]["rebalance_date"].tolist()
    )

    eval_sleeves = pd.read_parquet(interim_dir / "evaluation_sleeves.parquet")
    eval_sleeves["rebalance_date"] = pd.to_datetime(eval_sleeves["rebalance_date"])
    val_sleeves_df = eval_sleeves[
        eval_sleeves["rebalance_date"].isin(val_dates_all)
    ].copy()

    cond_scaler, tgt_scaler = load_scalers(scaler_dir)

    lkb = cfg.rolling_windows["lookback_days"]
    hor = cfg.rolling_windows["horizon_days"]

    schedule_grid = cfg.training["beta_schedule_grid"]
    T_grid        = cfg.training["diffusion_steps_grid"]
    alpha_grid    = [0.00, 0.25, 0.50, 0.75]
    M_grid        = cfg.generation["scenario_count_grid"]
    M_max         = cfg.generation["maximum_scenarios_generated"]

    # =====================================================================
    # STAGE 1 – Pre-compute sample covariances (cached)
    # =====================================================================
    sample_cov_cache_path = cache_dir / "sample_covs.pkl"

    if sample_cov_cache_path.exists():
        logger.info("✓ Loading cached sample covariances from %s", sample_cov_cache_path)
        with open(sample_cov_cache_path, "rb") as fh:
            _sc = pickle.load(fh)
        sleeve_permnos_by_date    = _sc["sleeve_permnos_by_date"]
        sleeve_sample_cov_by_date = _sc["sleeve_sample_cov_by_date"]
        sleeve_hist_ret_by_date   = _sc["sleeve_hist_ret_by_date"]
        n_sleeves = sum(len(v) for v in sleeve_sample_cov_by_date.values())
        logger.info("  Loaded %d dates, %d sleeves.", len(sleeve_permnos_by_date), n_sleeves)
    else:
        logger.info(
            "Pre-computing sample covariances for %d validation dates …",
            len(val_dates_all),
        )
        sleeve_permnos_by_date:    Dict[pd.Timestamp, Dict[int, List[int]]]   = {}
        sleeve_sample_cov_by_date: Dict[pd.Timestamp, Dict[int, np.ndarray]]  = {}
        sleeve_hist_ret_by_date:   Dict[pd.Timestamp, Dict[int, np.ndarray]]  = {}

        date_iter = (
            _tqdm(val_dates_all, desc="Sample covs", unit="date", dynamic_ncols=True)
            if _has_tqdm else val_dates_all
        )
        for date in date_iter:
            date_sleeves = val_sleeves_df[val_sleeves_df["rebalance_date"] == date]
            sleeve_permnos_by_date[date]    = {}
            sleeve_sample_cov_by_date[date] = {}
            sleeve_hist_ret_by_date[date]   = {}

            for sid in date_sleeves["sleeve_id"].unique():
                permnos = get_sleeve_permnos(date_sleeves, sid)
                result  = build_covariance_pair(
                    crsp_df, trading_dates, permnos, date,
                    lookback_days=lkb, horizon_days=hor,
                )
                if result is None:
                    continue
                S_hist, _, R_hist, _ = result
                sleeve_permnos_by_date[date][sid]    = permnos
                sleeve_sample_cov_by_date[date][sid] = S_hist
                sleeve_hist_ret_by_date[date][sid]   = R_hist

        with open(sample_cov_cache_path, "wb") as fh:
            pickle.dump(
                {
                    "sleeve_permnos_by_date":    sleeve_permnos_by_date,
                    "sleeve_sample_cov_by_date": sleeve_sample_cov_by_date,
                    "sleeve_hist_ret_by_date":   sleeve_hist_ret_by_date,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        n_sleeves = sum(len(v) for v in sleeve_sample_cov_by_date.values())
        logger.info("✓ Saved sample covariance cache  (%d sleeves)  → %s",
                    n_sleeves, sample_cov_cache_path)

    validation_results: list[dict] = []

    # =====================================================================
    # STAGE 2 – alpha=1 boundary (sample covariance GMV) — evaluated once
    # =====================================================================
    logger.info("\nEvaluating alpha=1 (sample covariance GMV) …")
    sample_wts_by_date: Dict[pd.Timestamp, Dict[int, np.ndarray]] = {}
    for date in val_dates_all:
        sample_wts_by_date[date] = {}
        for sid, S_hist in sleeve_sample_cov_by_date.get(date, {}).items():
            try:
                sample_wts_by_date[date][sid] = solve_long_only_gmv(S_hist)
            except Exception:
                pass

    sample_returns = run_validation_backtest(
        crsp_df, trading_dates, val_dates_all,
        sample_wts_by_date, sleeve_permnos_by_date, hor,
    )
    sample_vol = annualized_volatility(np.array(sample_returns)) if sample_returns else np.nan
    validation_results.append({
        "beta_schedule_type":                     "not_applicable",
        "diffusion_steps_T":                      None,
        "alpha":                                  1.0,
        "scenario_count_M":                       None,
        "validation_annualized_realized_volatility": sample_vol,
        "is_sample_covariance_boundary":          True,
    })
    logger.info("  alpha=1.00  M=—  →  val_vol=%.6f  (sample covariance boundary)", sample_vol)

    # =====================================================================
    # STAGE 3 – Diffusion-blend configurations
    # =====================================================================
    total_models   = len(schedule_grid) * len(T_grid)
    total_combos   = total_models * len(alpha_grid) * len(M_grid)
    combos_done    = 0

    logger.info(
        "\nEvaluating %d diffusion models × %d alpha × %d M = %d configurations …",
        total_models, len(alpha_grid), len(M_grid), total_models * len(alpha_grid) * len(M_grid),
    )

    model_pairs = [(s, T) for s in schedule_grid for T in T_grid]
    model_iter  = (
        _tqdm(model_pairs, desc="Models", unit="model", dynamic_ncols=True)
        if _has_tqdm else model_pairs
    )

    for schedule_type, T in model_iter:
        seed      = cfg.random_seed
        ckpt_path = model_dir / f"ddpm_schedule-{schedule_type}_T-{T}_seed-{seed}.pt"
        if not ckpt_path.exists():
            logger.warning("  Checkpoint not found: %s – SKIPPING.", ckpt_path)
            combos_done += len(alpha_grid) * len(M_grid)
            continue

        header = f"schedule={schedule_type}  T={T}"
        if _has_tqdm:
            _tqdm.write(f"\n── {header} ──────────────────────────────────────")
        else:
            print(f"\n── {header} ──────────────────────────────────────")

        # ------------------------------------------------------------------
        # STAGE 3a – Generate (or load cached) scenarios
        # ------------------------------------------------------------------
        scenario_cache_path = cache_dir / f"scenarios_schedule-{schedule_type}_T-{T}.pkl"

        if scenario_cache_path.exists():
            if _has_tqdm:
                _tqdm.write(f"  ✓ Loading cached scenarios from {scenario_cache_path.name}")
            else:
                logger.info("  ✓ Loading cached scenarios from %s", scenario_cache_path)
            with open(scenario_cache_path, "rb") as fh:
                scenario_cache: Dict[Tuple, List[np.ndarray]] = pickle.load(fh)
        else:
            logger.info("  Loading model checkpoint …")
            model, _ = load_trained_model(ckpt_path, device=str(device))
            model.eval()
            scheduler = DDPMScheduler(
                schedule_type=schedule_type, T=T,
                beta_min=cfg.training["beta_min"],
                beta_max=cfg.training["beta_max"],
                device=device,
            )

            scenario_cache: Dict[Tuple, List[np.ndarray]] = {}

            # Count total sleeve-dates to generate
            all_sleeve_dates = [
                (date, sid)
                for date in val_dates_all
                for sid in sleeve_sample_cov_by_date.get(date, {})
            ]

            gen_iter = (
                _tqdm(all_sleeve_dates, desc="  Generating", unit="sleeve",
                      dynamic_ncols=True, leave=False)
                if _has_tqdm else all_sleeve_dates
            )

            for date, sid in gen_iter:
                S_hist = sleeve_sample_cov_by_date[date][sid]
                try:
                    cov_vech = covariance_to_log_vech(S_hist, cfg.ridge_epsilon)
                except Exception as exc:
                    logger.debug("log_vech failed for sleeve %d at %s: %s", sid, date.date(), exc)
                    continue

                gen_seed = deterministic_scenario_seed(
                    schedule_type, T, date, sid, base_seed=seed
                )
                scenarios = generate_covariance_scenarios(
                    model=model,
                    scheduler=scheduler,
                    condition_vector_raw=cov_vech,
                    conditioning_scaler=cond_scaler,
                    target_scaler=tgt_scaler,
                    num_scenarios=M_max,
                    seed=gen_seed,
                )
                scenario_cache[(date, sid)] = scenarios

            with open(scenario_cache_path, "wb") as fh:
                pickle.dump(scenario_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                "  ✓ Saved scenario cache (%d sleeve-dates) → %s",
                len(scenario_cache), scenario_cache_path,
            )

        # ------------------------------------------------------------------
        # STAGE 3b – Evaluate every (alpha, M) combo for this model
        # ------------------------------------------------------------------
        combo_pairs = [(alpha, M) for M in M_grid for alpha in alpha_grid]
        combo_iter  = (
            _tqdm(combo_pairs, desc="  Configs", unit="cfg",
                  dynamic_ncols=True, leave=False)
            if _has_tqdm else combo_pairs
        )

        model_best_vol = float("inf")
        model_results  = []

        for alpha, M in combo_iter:
            wts_by_date: Dict[pd.Timestamp, Dict[int, np.ndarray]] = {}
            for date in val_dates_all:
                wts_by_date[date] = {}
                for sid, S_hist in sleeve_sample_cov_by_date.get(date, {}).items():
                    key = (date, sid)
                    if key not in scenario_cache:
                        continue
                    combined_cov = combine_covariances(
                        S_hist, scenario_cache[key][:M], alpha
                    )
                    try:
                        wts_by_date[date][sid] = solve_long_only_gmv(combined_cov)
                    except Exception:
                        pass

            daily_rets = run_validation_backtest(
                crsp_df, trading_dates, val_dates_all,
                wts_by_date, sleeve_permnos_by_date, hor,
            )
            vol = annualized_volatility(np.array(daily_rets)) if daily_rets else np.nan

            row = {
                "beta_schedule_type":                     schedule_type,
                "diffusion_steps_T":                      T,
                "alpha":                                  alpha,
                "scenario_count_M":                       M,
                "validation_annualized_realized_volatility": vol,
                "is_sample_covariance_boundary":          False,
            }
            validation_results.append(row)
            model_results.append(row)
            combos_done += 1
            model_best_vol = min(model_best_vol, vol)

            # ── Live result line ──────────────────────────────────────────
            marker = " ◀ best so far" if vol <= min(
                r["validation_annualized_realized_volatility"]
                for r in validation_results
            ) else ""
            line = (
                f"    schedule={schedule_type:<12}  T={T:>3}  "
                f"alpha={alpha:.2f}  M={M:>2}  →  val_vol={vol:.6f}  "
                f"({vol*100:.4f}%){marker}"
            )
            if _has_tqdm:
                _tqdm.write(line)
            else:
                print(line)

            if _has_tqdm and hasattr(combo_iter, "set_postfix"):
                combo_iter.set_postfix(
                    alpha=f"{alpha:.2f}", M=M, vol=f"{vol:.5f}",
                    best=f"{model_best_vol:.5f}",
                )

        # Summary for this model
        model_df = pd.DataFrame(model_results).sort_values(
            "validation_annualized_realized_volatility"
        )
        best_row = model_df.iloc[0]
        summary = (
            f"\n  ✓ {header}  done — "
            f"best this model: alpha={best_row['alpha']:.2f}  "
            f"M={int(best_row['scenario_count_M'])}  "
            f"vol={best_row['validation_annualized_realized_volatility']:.6f}  "
            f"[{combos_done}/{total_combos + 1} configs evaluated]"
        )
        if _has_tqdm:
            _tqdm.write(summary)
        else:
            print(summary)

        # Print running top-5 leaderboard every time a model finishes
        _print_leaderboard(validation_results, top_n=5)

    # =====================================================================
    # STAGE 4 – Rank and select
    # =====================================================================
    results_df = pd.DataFrame(validation_results)
    results_df = results_df.sort_values(
        "validation_annualized_realized_volatility"
    ).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1
    results_df["selected_primary_model"] = False

    # Only linear schedule is active; kept as a dict for forward-compatibility.
    schedule_priority = {"linear": 0, "not_applicable": 1}

    best_vol = results_df["validation_annualized_realized_volatility"].iloc[0]
    tied = results_df[
        (results_df["validation_annualized_realized_volatility"] - best_vol).abs() < 1e-8
    ].copy()

    def tie_sort_key(row):
        alpha = row["alpha"] if pd.notna(row["alpha"]) else -999.0
        M     = row["scenario_count_M"] if pd.notna(row["scenario_count_M"]) else 999
        T_val = row["diffusion_steps_T"] if pd.notna(row["diffusion_steps_T"]) else 999
        sched = schedule_priority.get(str(row["beta_schedule_type"]), 999)
        return (-alpha, M, T_val, sched)

    tied_sorted = tied.apply(tie_sort_key, axis=1).sort_values()
    best_idx    = tied_sorted.index[0]
    results_df.loc[best_idx, "selected_primary_model"] = True

    selected = results_df.loc[best_idx].to_dict()

    # ── Final leaderboard ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL VALIDATION RANKING (top 10)")
    logger.info("=" * 60)
    top10 = results_df.head(10)
    for _, row in top10.iterrows():
        star = " ◀ SELECTED" if row["selected_primary_model"] else ""
        T_str = f"T={int(row['diffusion_steps_T'])}" if pd.notna(row.get("diffusion_steps_T")) else "T=—"
        M_str = f"M={int(row['scenario_count_M'])}" if pd.notna(row.get("scenario_count_M")) else "M=—"
        logger.info(
            "  #%2d  %-12s  %5s  alpha=%.2f  %-4s  vol=%.6f%s",
            int(row["rank"]),
            row["beta_schedule_type"],
            T_str,
            row["alpha"],
            M_str,
            row["validation_annualized_realized_volatility"],
            star,
        )

    logger.info("\n=== SELECTED CONFIGURATION ===")
    for k, v in selected.items():
        logger.info("  %-48s  %s", k, v)

    # ---- Save results -----------------------------------------------------
    results_df.to_csv(val_results_dir / "validation_grid_results.csv", index=False)
    top5 = results_df.head(5)
    top5.to_csv(val_results_dir / "top5_validation_configurations.csv", index=False)
    logger.info(
        "\nTop-5 configurations:\n%s",
        top5[["beta_schedule_type", "diffusion_steps_T", "alpha",
              "scenario_count_M", "validation_annualized_realized_volatility"]].to_string()
    )

    # ---- Save selected config YAML ----------------------------------------
    selected_config = {
        "selected_model": {
            "beta_schedule_type": str(selected["beta_schedule_type"]),
            "diffusion_steps_T": (
                int(selected["diffusion_steps_T"])
                if pd.notna(selected.get("diffusion_steps_T")) else None
            ),
            "alpha": float(selected["alpha"]),
            "scenario_count_M": (
                int(selected["scenario_count_M"])
                if pd.notna(selected.get("scenario_count_M")) else None
            ),
            "validation_metric": "gross_annualized_realized_gmv_portfolio_volatility",
            "validation_annualized_realized_volatility": float(
                selected["validation_annualized_realized_volatility"]
            ),
            "validation_period":   ["2014-01-01", "2020-12-31"],
            "test_period_locked":  ["2021-01-01", "2025-12-31"],
        }
    }

    with open(selected_dir / "selected_model_config.yaml", "w") as fh:
        yaml.dump(selected_config, fh, default_flow_style=False)

    # ---- Copy selected checkpoint -----------------------------------------
    sched = selected["beta_schedule_type"]
    T_sel = selected["diffusion_steps_T"]
    if pd.notna(sched) and sched != "not_applicable" and pd.notna(T_sel):
        src_ckpt = model_dir / f"ddpm_schedule-{sched}_T-{int(T_sel)}_seed-{cfg.random_seed}.pt"
        if src_ckpt.exists():
            shutil.copy2(src_ckpt, selected_dir / "selected_model.pt")
            logger.info("Copied selected model checkpoint → %s", selected_dir)

    logger.info("\nStep 5 complete.")


if __name__ == "__main__":
    main()
