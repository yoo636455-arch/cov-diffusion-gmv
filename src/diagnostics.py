"""
diagnostics.py
--------------
Diffusion model diagnostics.

Spec §29 – scenario dispersion, forecast losses, numerical stability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import logm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario dispersion
# ---------------------------------------------------------------------------

def compute_scenario_dispersion(
    generated_covariances: List[np.ndarray],
) -> float:
    """
    D = (1/M) * sum_m ||Sigma^(m) - mean(Sigma)||_F^2

    Spec §29.1.
    """
    M = len(generated_covariances)
    if M == 0:
        return np.nan
    stack = np.stack(generated_covariances, axis=0)  # (M, N, N)
    mean_cov = stack.mean(axis=0)                     # (N, N)
    diffs = stack - mean_cov[np.newaxis, :, :]        # (M, N, N)
    frob_sq = (diffs ** 2).sum(axis=(1, 2))           # (M,)
    return float(frob_sq.mean())


# ---------------------------------------------------------------------------
# Frobenius and log-covariance forecast losses
# ---------------------------------------------------------------------------

def frobenius_loss(
    predicted_cov: np.ndarray,
    realized_cov: np.ndarray,
) -> float:
    """||Sigma_hat - S_realized||_F (spec §29.2)."""
    return float(np.linalg.norm(predicted_cov - realized_cov, "fro"))


def log_covariance_loss(
    predicted_cov: np.ndarray,
    realized_cov: np.ndarray,
    ridge_epsilon: float = 1e-8,
) -> float:
    """
    ||logm(Sigma_hat) - logm(S_realized + eps*I)||_F

    Spec §29.2.
    """
    n = predicted_cov.shape[0]
    try:
        A_pred = logm(predicted_cov + ridge_epsilon * np.eye(n))
        A_real = logm(realized_cov + ridge_epsilon * np.eye(n))
        A_pred = 0.5 * (A_pred + A_pred.T).real
        A_real = 0.5 * (A_real + A_real.T).real
        return float(np.linalg.norm(A_pred - A_real, "fro"))
    except Exception as exc:
        logger.debug("log_covariance_loss failed: %s", exc)
        return np.nan


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def covariance_condition_number(cov: np.ndarray) -> float:
    """Return the condition number of the covariance matrix."""
    eigvals = np.linalg.eigvalsh(cov)
    eig_min = eigvals.min()
    eig_max = eigvals.max()
    if eig_min <= 0:
        return np.inf
    return float(eig_max / eig_min)


def covariance_min_eigenvalue(cov: np.ndarray) -> float:
    """Return the minimum eigenvalue."""
    return float(np.linalg.eigvalsh(cov).min())


# ---------------------------------------------------------------------------
# Aggregate diagnostics over a period
# ---------------------------------------------------------------------------

def collect_diagnostics_record(
    date: pd.Timestamp,
    sleeve_id: int,
    generated_covariances: List[np.ndarray],
    combined_cov: np.ndarray,
    sample_cov: np.ndarray,
    realized_cov: Optional[np.ndarray] = None,
    ridge_epsilon: float = 1e-8,
) -> dict:
    """
    Collect all diagnostics for one sleeve-date into a flat dict.
    """
    rec = {
        "date": date,
        "sleeve_id": sleeve_id,
        "scenario_dispersion": compute_scenario_dispersion(generated_covariances),
        "combined_min_eigenvalue": covariance_min_eigenvalue(combined_cov),
        "combined_condition_number": covariance_condition_number(combined_cov),
        "sample_min_eigenvalue": covariance_min_eigenvalue(sample_cov),
    }

    if realized_cov is not None:
        rec["frobenius_loss"] = frobenius_loss(combined_cov, realized_cov)
        rec["log_covariance_loss"] = log_covariance_loss(
            combined_cov, realized_cov, ridge_epsilon
        )

    return rec
