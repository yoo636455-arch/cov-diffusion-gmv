"""
benchmarks.py
-------------
Covariance estimation benchmarks for the GMV portfolio comparison.

Spec §26:
- Equal Weight (no covariance estimation)
- Sample Covariance GMV
- Ledoit-Wolf Linear Shrinkage GMV
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Equal Weight
# ---------------------------------------------------------------------------

def equal_weight(n: int) -> np.ndarray:
    """Return equal weights for n assets."""
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Sample Covariance GMV
# ---------------------------------------------------------------------------

def sample_covariance_gmv(
    sample_cov: np.ndarray,
    repair_log: Optional[list] = None,
    log_context: Optional[dict] = None,
) -> np.ndarray:
    """
    Long-only GMV using the plain 126-day sample covariance matrix.
    Corresponds to alpha=1.0 in the diffusion framework.
    """
    from .gmv import solve_long_only_gmv
    return solve_long_only_gmv(sample_cov, repair_log=repair_log, log_context=log_context)


# ---------------------------------------------------------------------------
# Ledoit-Wolf Linear Shrinkage GMV
# ---------------------------------------------------------------------------

def ledoit_wolf_shrinkage_covariance(
    return_matrix: np.ndarray,
) -> np.ndarray:
    """
    Fit a Ledoit-Wolf linear shrinkage covariance estimator.

    Parameters
    ----------
    return_matrix : (T, N) matrix of daily returns, T >= N+1.

    Returns
    -------
    (N, N) shrinkage covariance matrix.
    """
    lw = LedoitWolf(assume_centered=False)
    lw.fit(return_matrix)
    return lw.covariance_


def ledoit_wolf_linear_gmv(
    return_matrix: np.ndarray,
    repair_log: Optional[list] = None,
    log_context: Optional[dict] = None,
) -> np.ndarray:
    """
    Long-only GMV using Ledoit-Wolf linear shrinkage covariance.

    Parameters
    ----------
    return_matrix : (T, N) – past 126 daily returns

    Returns
    -------
    (N,) weight vector
    """
    from .gmv import solve_long_only_gmv

    try:
        lw_cov = ledoit_wolf_shrinkage_covariance(return_matrix)
    except Exception as exc:
        logger.warning(
            "Ledoit-Wolf fitting failed: %s. Falling back to sample covariance.", exc
        )
        lw_cov = np.cov(return_matrix.T)

    return solve_long_only_gmv(lw_cov, repair_log=repair_log, log_context=log_context)
