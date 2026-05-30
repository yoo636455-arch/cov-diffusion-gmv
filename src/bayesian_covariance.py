"""
bayesian_covariance.py
-----------------------
Bayesian Inverse-Wishart covariance estimation for GMV portfolios.

Model:
    Prior:      Σ ~ IW(Ψ, ν)
    Likelihood: S | Σ ~ Wishart(Σ, h)   where S = X'X from the most recent known returns
    Posterior:  Σ|S  ~ IW(Ψ + S, ν + h)
    Post. mean: (Ψ + S) / (ν + h - N - 1)

NOTE: The likelihood uses historical returns available at the rebalance date.
      This is intended as an operational Bayesian IW covariance estimator.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def bayesian_iw_posterior_cov(
    psi: np.ndarray,
    nu: int,
    r_lik: np.ndarray,
) -> np.ndarray:
    """
    Posterior mean covariance under the IW-Wishart conjugate model.

    Parameters
    ----------
    psi   : (N, N) prior scale matrix — set to the historical sample covariance
    nu    : prior degrees of freedom — set to lookback_days (e.g. 252)
    r_lik : (h, N) return matrix used as the likelihood observation,
            available at the rebalance date

    Returns
    -------
    (N, N) posterior mean covariance matrix
    """
    N = psi.shape[0]
    h = r_lik.shape[0]

    # Scatter matrix S = X'X from demeaned horizon returns: S ~ Wishart(Σ, h)
    X_h = r_lik - r_lik.mean(axis=0)
    S = X_h.T @ X_h  # (N, N)

    psi_post = psi + S
    nu_post  = nu + h

    dof_eff = nu_post - N - 1
    if dof_eff <= 0:
        logger.warning(
            "Effective dof %d <= 0 (N=%d, nu=%d, h=%d). Falling back to psi_post / nu_post.",
            dof_eff, N, nu, h,
        )
        return psi_post / nu_post

    posterior_mean = psi_post / dof_eff
    return 0.5 * (posterior_mean + posterior_mean.T)


def bayesian_iw_gmv(
    psi: np.ndarray,
    nu: int,
    r_lik: np.ndarray,
    repair_log: Optional[list] = None,
    log_context: Optional[dict] = None,
) -> np.ndarray:
    """
    Long-only GMV portfolio using the Bayesian IW posterior mean covariance.

    Parameters
    ----------
    psi       : (N, N) prior scale matrix (historical sample covariance)
    nu        : prior degrees of freedom (lookback_days)
    r_lik     : (h, N) return matrix for the likelihood update available at rebalance date
    repair_log: optional list to collect numerical repair records
    log_context: optional metadata for repair logging

    Returns
    -------
    (N,) portfolio weight vector
    """
    from .gmv import solve_long_only_gmv

    posterior_cov = bayesian_iw_posterior_cov(psi, nu, r_lik)
    return solve_long_only_gmv(
        posterior_cov, repair_log=repair_log, log_context=log_context
    )
