"""
07_generate_report.py
----------------------
Phase 7: Generate all figures, result tables, and the final Markdown report.

Outputs
-------
results/figures/*.png
reports/final_results.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plotting import (
    plot_test_cumulative_wealth,
    plot_test_rolling_volatility,
    plot_test_scenario_dispersion,
    plot_test_turnover,
    plot_test_weight_concentration,
    plot_validation_alpha_sensitivity,
    plot_validation_configuration_ranking,
    plot_validation_M_sensitivity,
)
from src.utils import get_logger

logger = get_logger("07_generate_report", logging.INFO)


def main() -> None:
    figures_dir = Path("results/figures")
    reports_dir = Path("reports")
    val_dir = Path("results/validation")
    test_dir = Path("results/test")
    portfolio_dir = Path("results/portfolios")
    diag_dir = Path("results/diagnostics")
    selected_dir = Path("artifacts/selected_model")

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 7 – Generate report and figures")
    logger.info("=" * 60)

    # ---- Load validation results -----------------------------------------
    val_grid_path = val_dir / "validation_grid_results.csv"
    if val_grid_path.exists():
        val_df = pd.read_csv(val_grid_path)
        plot_validation_configuration_ranking(val_df)
        plot_validation_alpha_sensitivity(val_df)
        plot_validation_M_sensitivity(val_df)
        logger.info("Generated validation figures.")
    else:
        logger.warning("validation_grid_results.csv not found – skipping val figures.")

    # ---- Load test returns -----------------------------------------------
    method_files = list(portfolio_dir.glob("*_returns.csv"))
    returns_dict = {}
    for f in method_files:
        method_name = f.stem.replace("_returns", "")
        df_r = pd.read_csv(f)
        returns_dict[method_name] = df_r["gross_return"].tolist()

    net_returns_dict = {}
    for f in method_files:
        method_name = f.stem.replace("_returns", "")
        df_r = pd.read_csv(f)
        net_returns_dict[method_name] = df_r["net_return_10bps"].tolist()

    if returns_dict:
        plot_test_cumulative_wealth(returns_dict, net=False)
        plot_test_cumulative_wealth(net_returns_dict, net=True)
        plot_test_rolling_volatility(returns_dict)
        logger.info("Generated test cumulative wealth and rolling vol figures.")

    # ---- Scenario dispersion figure --------------------------------------
    disp_path = diag_dir / "scenario_dispersion.csv"
    if disp_path.exists():
        disp_df = pd.read_csv(disp_path)
        if "scenario_dispersion" in disp_df.columns:
            plot_test_scenario_dispersion(disp_df)

    # ---- Load selected config --------------------------------------------
    sel_cfg = {}
    sel_path = selected_dir / "selected_model_config.yaml"
    if sel_path.exists():
        with open(sel_path) as fh:
            sel_cfg = yaml.safe_load(fh).get("selected_model", {})

    # ---- Load test performance summary -----------------------------------
    test_summary_path = test_dir / "test_performance_summary.csv"
    test_summary_df = pd.DataFrame()
    if test_summary_path.exists():
        test_summary_df = pd.read_csv(test_summary_path)

    # ---- Write Markdown report -------------------------------------------
    logger.info("Writing final_results.md …")
    _write_report(sel_cfg, test_summary_df, reports_dir)
    logger.info("Step 7 complete.")


def _write_report(sel_cfg: dict, test_df: pd.DataFrame, reports_dir: Path) -> None:
    report_path = reports_dir / "final_results.md"

    selected_str = "\n".join(f"- **{k}**: {v}" for k, v in sel_cfg.items())

    test_table = (
        test_df.to_markdown(index=False, floatfmt=".4f")
        if not test_df.empty and hasattr(test_df, "to_markdown")
        else "*test_performance_summary.csv not yet generated*"
    )

    report = f"""# Final Results: Stabilized Conditional Diffusion Forecasting for GMV Portfolio Optimization

---

## 1. Research Question

Does a stabilized conditional diffusion covariance estimator produce lower future realized GMV portfolio
volatility than conventional covariance estimators (sample covariance, Ledoit-Wolf shrinkage)?

---

## 2. Motivation and Methodological Framing

The historical sample covariance is estimated from a finite window of daily returns and is therefore
noisy. The relevant covariance for the next monthly holding period may differ from the historical
estimate. We use a conditional DDPM to generate multiple plausible next-month covariance matrices
conditional on the historical 126-day sample covariance. The mean of these generated matrices is
blended with the historical sample covariance through a stability weight α to form a stabilized
combined covariance estimator.

> **Correct terminology:** The model is a conditional diffusion covariance forecaster with
> portfolio-objective-aligned hyperparameter selection.

---

## 3. CRSP Daily Data and Dynamic Top-500 Universe

- **Source:** CRSP daily U.S. equity data from WRDS
- **Security filter:** SHRCD ∈ {{10, 11}} (ordinary common shares); EXCHCD ∈ {{1, 2, 3}} (major U.S. exchanges)
- **Return construction:** Delisting-adjusted daily total returns
- **Market cap:** ME = |PRC| × SHROUT
- **Universe:** Dynamic top-500 by ME at each rebalance date

---

## 4. Same-Industry Sleeve Construction

- **Industry:** 2-digit SIC (floor(SIC/100))
- **Sleeve size:** 10 stocks per sleeve (N=10)
- **Ordering within sleeve:** Descending market cap at formation date
- **Validation/Test sleeves:** Deterministic non-overlapping sequential blocks within each industry
- **Capital allocation:** Equal weight across sleeves (1/G_t per sleeve)

---

## 5. Covariance Input and Future Target Construction

- **Historical conditioning covariance:** S_t^126 = Cov(returns over 126 trading days ending at t)
- **Future realized covariance proxy:** S_{{t+1:t+21}}^21 = Cov(returns over next 21 trading days)

> The 21-day future covariance matrix is an ex-post realized covariance proxy for the holding period,
> not an observable latent true covariance matrix. It is noisy because it is estimated from a limited
> number of daily returns.

**Representation:** Ridge-stabilized (ε=1e-8) matrix-log vech vectorization → R^55

---

## 6. Conditional Diffusion Model

- **Architecture:** Conditional MLP with sinusoidal time embedding
- **Input:** [y_s (55), e(s) (32), c̃ (55)] = 142 dimensions
- **Hidden:** 3 layers × 128 units, SiLU activation
- **Output:** 55 dimensions (predicted noise)
- **Training:** DDPM noise prediction loss, 200 epochs, Adam (lr=1e-3, wd=1e-5), batch 128

---

## 7. Stabilized Covariance Estimator and the Role of Alpha

$$\\hat{{\\Sigma}}_{{g,t+1}}^{{\\text{{combined}}}} = \\alpha S_{{g,t}}^{{126}} + (1-\\alpha) \\hat{{\\Sigma}}_{{g,t+1}}^{{\\text{{diff}}}}$$

Any performance improvement should be attributed to a **stabilized conditional diffusion estimator**,
not necessarily to pure diffusion alone.

---

## 8. GMV Portfolio Construction

- **Objective:** Long-only, fully invested minimum variance per sleeve
- **Solver:** CLARABEL via CVXPY
- **Aggregate weights:** Equal capital allocation across non-overlapping sleeves

---

## 9. Training Design: 2000–2013

- Training observations: future holding-window entirely within 2000–2013
- ~40–50 overlapping same-industry 10-stock groups per rebalance date (deterministic seed=42)
- 4 candidate diffusion models: linear schedule × T ∈ {400, 800, 1200, 2000}

---

## 10. Validation and Hyperparameter Selection: 2014–2020

- **Sole selection metric:** Annualized realized GMV portfolio volatility
- 81 effective configurations: 4 models × 4α × 5M + 1 (α=1 boundary)

---

## 11. Selected Primary Configuration

{selected_str if selected_str else "*See artifacts/selected_model/selected_model_config.yaml*"}

---

## 12. Final Untouched Test Results: 2021–2025

{test_table}

---

## 13. Benchmark Comparison

See test_performance_summary.csv for full comparison against:
- Equal Weight
- Sample Covariance GMV (α=1)
- Ledoit-Wolf Linear Shrinkage GMV

---

## 14. Required Ablation Results

| Ablation | Alpha | M | Question |
|---|---|---|---|
| Pure Diffusion | 0.0 | M* | Did stability blending help? |
| Single Scenario | α* | 1 | Did scenario averaging help? |
| Pure Sample Cov | 1.0 | — | Did diffusion add value? |

See results/ablations/ for detailed ablation metrics.

---

## 15. Transaction-Cost Sensitivity

Primary reported cost: 10 bps per one-way turnover.
Sensitivity reported at 0, 5, 10, 20 bps.

---

## 16. Diffusion Scenario Diagnostics

See results/diagnostics/scenario_dispersion.csv and covariance_forecast_losses.csv.

---

## 17. Numerical Stability Diagnostics

See results/diagnostics/covariance_repairs.csv.

---

## 18. Limitations

### 18.1 Noisy Future Covariance Proxy
The subsequent 21-day realized covariance matrix is estimated from only 21 daily observations for
10 assets and is therefore a noisy proxy for the future covariance relevant to the holding period.

### 18.2 Diffusion Is Not Directly Recovering True Covariance
The model learns the conditional distribution of realized future covariance proxies. It does not
directly observe or recover a latent true covariance matrix.

### 18.3 Standard GMV Uses Only the Scenario Mean
Although diffusion generates multiple covariance scenarios, the main GMV portfolio uses their
arithmetic mean before blending with historical sample covariance. The main strategy uses the
diffusion distribution through its implied expected covariance, not through an explicit
tail-risk optimization criterion.

### 18.4 Alpha Makes the Method a Hybrid Estimator
Any performance improvement should be attributed to the stabilized conditional diffusion estimator,
not necessarily to pure diffusion alone.

### 18.5 Dependence Across Training Groups
Training observations produced from overlapping groups are cross-sectionally dependent because
groups may share stocks and common shocks.

### 18.6 Hyperparameter Search Risk
Even with a seven-year validation period, selecting among 81 effective configurations creates
a nontrivial risk of validation overfitting. The test period remains fully untouched.

### 18.7 Strong Benchmark Possibility
Ledoit-Wolf shrinkage is a strong conventional covariance benchmark. If it outperforms the
diffusion method in the final test period, this result is reported honestly.

---

## 19. Conclusion

This study evaluates a stabilized conditional diffusion covariance estimator for global
minimum-variance portfolio construction. Using CRSP daily U.S. equity data, a dynamic
market-cap top-500 universe was formed at each non-overlapping 21-trading-day rebalance date.
Same-industry 10-stock sleeves were constructed, and each sleeve's covariance matrix estimated
from the previous 126 trading days served as the conditioning input. A conditional diffusion model
was trained on 2000–2013 data to generate multiple plausible covariance matrices for the
subsequent 21-trading-day holding period. The generated covariance scenarios were averaged in
covariance-matrix space to obtain a diffusion-implied expected future covariance, which was then
blended with the observed historical sample covariance through a stability weight α. The beta
schedule family, number of diffusion steps T, scenario count M, and stability weight α were
selected solely according to annualized realized GMV portfolio volatility during 2014–2020. The
single best validation-selected specification was evaluated once on the untouched 2021–2025 test
period against equal-weight, raw sample-covariance GMV, and Ledoit-Wolf shrinkage benchmarks.
"""

    with open(report_path, "w") as fh:
        fh.write(report)

    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
