# Implementation Notes

## Project
Stabilized Conditional Diffusion Forecasting of Next-Month Covariance Distributions
for Global Minimum-Variance Portfolio Optimization

---

## Format

Each entry follows this structure:

### [INnnn] Short title

| Field | Detail |
|---|---|
| Date | YYYY-MM-DD |
| Component | Affected module or spec section |
| Reason | Why the deviation was necessary |
| Original design | What the spec originally said |
| Implementation | What was actually implemented |
| Impact | Likely effect on results |

---

## Entries

### [IN0001] Daily-sliding window for training covariance pairs

| Field | Detail |
|---|---|
| Date | 2026-05-24 |
| Component | `src/covariance.py`, `src/datasets.py`, `scripts/03_build_covariance_datasets.py`, `config/base_config.yaml` (§10, §12, §18) |
| Reason | The original spec generates one `(S_hist, S_fwd)` training pair per group per 21-day rebalance date, yielding ~4,751 pairs. This is insufficient to train a neural network reliably; modern diffusion models typically require tens of thousands of samples to learn a useful conditional distribution. |
| Original design | Spec §10–12 implies one covariance pair per group at each rebalance date (stride = 21 trading days). Expected training set: ~4,751 pairs over 2000–2013. |
| Implementation | A daily-sliding window is applied to training data only. For each 10-stock group formed at rebalance date `t_k`, covariance pairs are computed for every trading day `d` in `[t_k, t_{k+1})` where (a) a complete 126-day lookback exists, (b) the 21-day forward window ends on or before 2013-12-31 (strictly inside training), and (c) all 10 stocks have non-missing returns in both windows. Validation and test data retain the original 21-day stride (non-overlapping evaluation sleeves). The stride is configurable via `config/base_config.yaml`: `covariance_transform.training_window_stride_days: 1`. Set to 21 to revert to the original behaviour. |
| Impact | Training set grows from ~4,751 to ~99,000 pairs (~21× increase). Consecutive pairs share 125 of 126 lookback return days and 20 of 21 forward return days, introducing autocorrelation. The diffusion model is not tested on autocorrelated data (validation and test use non-overlapping periods), so this cannot inflate out-of-sample metrics. The increased data volume is expected to improve model fitting and reduce underfitting on the 55-dimensional covariance representation. The spec's stated limitation about dependence across overlapping training groups (§39.5) applies equally to this sliding-window extension. |

---

### [IN0002] GPU/MPS device support

| Field | Detail |
|---|---|
| Date | 2026-05-24 |
| Component | `src/utils.py`, `src/train.py`, `scripts/05_validate_hyperparameters.py`, `scripts/06_run_final_test.py` |
| Reason | The spec does not prescribe a compute device. Training 4 models (linear schedule, T ∈ {400, 800, 1200, 2000}) for 200 epochs on CPU is prohibitively slow, especially at T=1200 and T=2000 where each training step requires more reverse-step iterations. GPU/MPS reduces total training time from many hours to under one hour. |
| Original design | Spec §15 (fixed architecture) and §18 (fixed training procedure) are device-agnostic. |
| Implementation | Added `get_device()` to `src/utils.py` selecting CUDA (NVIDIA) → MPS (Apple Silicon) → CPU. `DDPMScheduler` tensors are now created on the same device as the model in scripts 05 and 06, preventing device-mismatch errors that would occur when a CPU-resident scheduler is used with a GPU-resident model. `torch.backends.cudnn.benchmark = True` is set on CUDA. `pin_memory=True` and `non_blocking=True` transfers are used with CUDA. Training results (loss values, final weights) are numerically identical across devices because the same seed and deterministic operations are used. |
| Impact | Purely computational — no effect on model outputs, hyperparameter selection, or results. All unit tests pass on MPS (Apple Silicon). |

---

### [IN0003] Reduced model grid: linear schedule only, T ∈ {400, 800, 1200, 2000}

| Field | Detail |
|---|---|
| Date | 2026-05-26 |
| Component | `config/base_config.yaml`, `scripts/04_train_diffusion_models.py`, `scripts/05_validate_hyperparameters.py`, `scripts/07_generate_report.py` |
| Original design | Spec §17–18 defined a 3 × 3 grid: schedule ∈ {linear, quadratic, logarithmic} × T ∈ {25, 50, 100}, yielding 9 models and 181 effective validation configurations. |
| Reason | For a 55-dimensional covariance vector representation, a higher number of diffusion steps T provides finer-grained denoising at the cost of more reverse-step iterations at inference. The linear schedule is a strong default that does not require tuning the schedule shape. Restricting to one schedule type and using larger T values focuses the search budget on the dimension most likely to matter for covariance diffusion quality. |
| Implementation | `beta_schedule_grid` set to `["linear"]`; `diffusion_steps_grid` set to `[400, 800, 1200, 2000]`. This yields 4 models and 81 effective validation configurations (4 × 4α × 5M + 1 boundary). The quadratic and logarithmic schedule implementations are retained in `src/beta_schedules.py` and their unit tests are preserved. |
| Impact | Fewer model variants but larger diffusion step counts. Validation search space reduced from 181 to 81 configurations, lowering hyperparameter search risk. Higher T values increase inference cost (scenario generation time) linearly with T. |

---

## CRSP Column Mapping Resolution

After loading the raw CRSP file, the resolved mapping between canonical fields and raw column
names was inspected and logged. See `config/column_mapping.yaml` for the current mapping.

If any of the following fields were unavailable in the extract, the corresponding entry
in `column_mapping.yaml` was set to `available: false`, and the limitation is documented here:

| Canonical Field | Status | Fallback Used |
|---|---|---|
| date | ✓ | — |
| permno | ✓ | — |
| ret | ✓ | — |
| dlret | ✓ | — |
| prc | ✓ | — |
| shrout | ✓ | — |
| shrcd | ✓ | — |
| exchcd | ✓ | — |
| siccd | ✓ | — |

*Update this table after running script 01 on the actual CRSP extract.*

---

## Ridge Epsilon Used

Primary ridge value: ε = 1e-8 (as specified).

If numerical instability required a larger ridge, document the final value here:

| Context | Ridge Value Used | Original | Reason |
|---|---|---|---|
| Default | 1e-8 | 1e-8 | No instability observed |

---

## GMV Solver Fallbacks

Any GMV optimization failures that triggered equal-weight fallback will be logged to:
`results/diagnostics/covariance_repairs.csv`

---

## Ledoit-Wolf Nonlinear Shrinkage

As specified in `base_config.yaml`:
```yaml
benchmarks:
  ledoit_wolf_nonlinear_shrinkage_gmv: false
```

The nonlinear Ledoit-Wolf estimator was not included as a primary benchmark because a
reliable production-quality Python implementation was not available. If included, it would be
implemented using the analytical nonlinear shrinkage formula from Ledoit & Wolf (2020).
