"""
beta_schedules.py
-----------------
Beta schedule factory for DDPM training and generation.

Spec §16.

Active configuration: linear schedule only, T ∈ {400, 800, 1200, 2000}.
The quadratic and logarithmic implementations are retained for completeness
and test coverage but are not used in the current training grid.
"""

from __future__ import annotations

import numpy as np


def make_beta_schedule(
    schedule_type: str,
    T: int,
    beta_min: float = 1.0e-4,
    beta_max: float = 2.0e-2,
) -> np.ndarray:
    """
    Create a beta schedule for DDPM training and generation.

    Supported schedule types: 'linear', 'quadratic', 'logarithmic'.
    Active in current configuration: 'linear' only.

    Parameters
    ----------
    schedule_type : str
    T : int
        Number of diffusion steps (must be >= 2).
    beta_min : float
        Minimum beta value (fixed at 1e-4 per spec).
    beta_max : float
        Maximum beta value (fixed at 0.02 per spec).

    Returns
    -------
    np.ndarray of shape (T,) with values strictly in (0, 1).
    """
    if T < 2:
        raise ValueError("T must be at least 2.")

    if schedule_type == "linear":
        betas = np.linspace(beta_min, beta_max, T)

    elif schedule_type == "quadratic":
        betas = np.linspace(np.sqrt(beta_min), np.sqrt(beta_max), T) ** 2

    elif schedule_type == "logarithmic":
        betas = np.exp(np.linspace(np.log(beta_min), np.log(beta_max), T))

    else:
        raise ValueError(
            f"Unsupported beta schedule type: '{schedule_type}'. "
            "Choose from: 'linear', 'quadratic', 'logarithmic'."
        )

    if betas.shape != (T,):
        raise ValueError("Incorrect beta schedule shape.")

    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("Every beta value must be strictly between 0 and 1.")

    return betas


def compute_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product alpha_bar_s = prod_{j=1}^{s} alpha_j
    where alpha_j = 1 - beta_j.

    Parameters
    ----------
    betas : np.ndarray, shape (T,)

    Returns
    -------
    np.ndarray, shape (T,), with alpha_bar[s] = prod alpha_j for j <= s.
    """
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return alpha_bar
