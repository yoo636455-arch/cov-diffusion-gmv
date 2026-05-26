"""
gmv.py
------
Long-only GMV portfolio optimization.

Spec §21.
"""

from __future__ import annotations

import logging
from typing import Optional

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def solve_long_only_gmv(
    covariance: np.ndarray,
    repair_log: Optional[list] = None,
    log_context: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute long-only, fully invested GMV weights.

    Problem:
        minimize    w' Sigma w
        subject to  sum(w) == 1
                    w >= 0

    Parameters
    ----------
    covariance : (N, N) combined covariance matrix
    repair_log : if not None, append a dict to this list when jitter is applied
    log_context : dict with contextual info for repair logging (date, sleeve_id, etc.)

    Returns
    -------
    np.ndarray, shape (N,) – portfolio weights summing to 1, all >= 0
    """
    covariance = 0.5 * (covariance + covariance.T)  # symmetrize

    # Check positive definiteness; apply jitter if needed
    eig_min = np.linalg.eigvalsh(covariance).min()
    jitter_applied = 0.0

    if eig_min <= 1e-10:
        jitter_applied = abs(eig_min) + 1e-8
        covariance = covariance + jitter_applied * np.eye(covariance.shape[0])
        logger.debug(
            "Applied jitter %.2e to covariance (min_eig was %.2e)",
            jitter_applied, eig_min,
        )

        if repair_log is not None:
            entry = {
                "min_eig_before_repair": eig_min,
                "jitter_added": jitter_applied,
            }
            if log_context:
                entry.update(log_context)
            repair_log.append(entry)

    n_assets = covariance.shape[0]
    w = cp.Variable(n_assets)

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(covariance)))
    constraints = [cp.sum(w) == 1, w >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    if w.value is None:
        # Fall back to equal weight
        logger.warning(
            "GMV optimization failed (status: %s). Falling back to equal weight.",
            problem.status,
        )
        return np.full(n_assets, 1.0 / n_assets)

    weights = np.asarray(w.value).reshape(-1)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights = np.clip(weights, 0.0, None)  # remove tiny negatives from solver
    weights = weights / weights.sum()

    return weights
