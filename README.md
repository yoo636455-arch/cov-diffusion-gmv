# Stabilized Conditional Diffusion Forecasting of Next-Month Covariance Distributions for Global Minimum-Variance Portfolio Optimization

## Overview

This repository implements a complete, reproducible quantitative research pipeline that tests
whether a **stabilized conditional diffusion covariance estimator** improves the realized
out-of-sample volatility of Global Minimum-Variance (GMV) portfolios constructed from CRSP
daily U.S. equity data.

### Central Question

> Does the stabilized conditional diffusion covariance estimator produce lower future realized GMV volatility than conventional estimators?

---

## Repository Structure

```
final_sprint_cov_diffusion/
├── config/
│   ├── base_config.yaml       # All hyperparameters and settings
│   └── column_mapping.yaml    # CRSP column name resolution
├── data/
│   ├── raw/                   # Place crsp_daily.parquet here
│   ├── interim/               # Cleaned data, universe, groups
│   └── processed/             # Covariance train/val/test NPZ files
├── src/                       # Core library modules
├── scripts/                   # Numbered pipeline stages (01–07)
├── artifacts/                 # Trained models, scalers, selected config
├── results/                   # Validation/test results, diagnostics, figures
├── reports/                   # Final report and implementation notes
└── tests/                     # Unit tests
```

---

## Data Setup (WRDS)

You have WRDS access. Pull the CRSP daily stock file with at minimum these variables:

```
PERMNO, date, RET, DLRET, PRC, SHROUT, SHRCD, EXCHCD, SICCD
```

Save as: `data/raw/crsp_daily.parquet`

Recommended WRDS query (SAS/Python):
```python
import wrds
db = wrds.Connection()
crsp = db.raw_sql("""
    SELECT a.permno, a.date, a.ret, a.dlret, a.prc, a.shrout, b.shrcd, b.exchcd, b.siccd
    FROM crsp.dsf AS a
    LEFT JOIN crsp.dsenames AS b
        ON a.permno = b.permno
        AND b.namedt <= a.date
        AND a.date <= b.nameendt
    WHERE a.date BETWEEN '1998-01-01' AND '2025-12-31'
""", date_cols=['date'])
crsp.to_parquet('data/raw/crsp_daily.parquet', index=False)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Execute scripts in order from the project root:

```bash
# Phase 1: Clean data
python scripts/01_clean_crsp_data.py

# Phase 2: Build universe and groups
python scripts/02_build_universe_and_groups.py

# Phase 3: Build covariance datasets (fits training-only scalers)
python scripts/03_build_covariance_datasets.py

# Phase 4: Train 9 diffusion models (~hours on CPU, ~30min on GPU)
python scripts/04_train_diffusion_models.py

# Phase 5: Validate 181 configurations (select primary model)
python scripts/05_validate_hyperparameters.py

# Phase 6: Final test backtest (run ONCE, no further model changes)
python scripts/06_run_final_test.py

# Phase 7: Generate figures and report
python scripts/07_generate_report.py
```

---

## Running Unit Tests

```bash
cd final_sprint_cov_diffusion
pytest tests/ -v
```

---

## Design Decisions

| Component | Choice |
|---|---|
| Covariance representation | Ridge-stabilized matrix-log vech (R^55) |
| Diffusion model | Conditional DDPM with MLP denoiser |
| Portfolio objective | Long-only GMV via CVXPY/CLARABEL |
| Hyperparameter selection | Sole metric: validation annualized realized vol |
| Test period | Fully untouched (2021–2025) |
| Benchmarks | EW, Sample Cov GMV, LW Linear Shrinkage GMV |

---

## Time Splits

| Split | Dates | Purpose |
|---|---|---|
| Training | 2000–2013 | Fit diffusion models |
| Validation | 2014–2020 | Select β schedule, α, T, M |
| Test | 2021–2025 | Final untouched evaluation |

**Critical rule:** Observation assignment uses the future holding-window dates, not the
input lookback-window dates. The input lookback window may extend into the prior split.

---

## Limitations

See `reports/final_results.md` §18 and `reports/implementation_notes.md` for all limitations
and implementation deviations.

---

## Citation

Please cite the original CRSP data source:
> Center for Research in Security Prices (CRSP). CRSP US Stock Database. University of Chicago Booth School of Business.
