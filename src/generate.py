"""
generate.py
-----------
Generate conditional covariance scenarios for validation and test sleeves.

Spec §19 – common random numbers, nested scenario prefixes.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

import numpy as np
import torch

from .diffusion import DDPMScheduler, p_sample_loop
from .transforms import load_scalers, log_vech_to_covariance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic seed per (model config, rebalance date, sleeve id)
# ---------------------------------------------------------------------------

def deterministic_scenario_seed(
    schedule_type: str,
    T: int,
    rebalance_date,
    sleeve_id: int,
    base_seed: int = 42,
) -> int:
    """
    Produce a deterministic integer seed for scenario generation.

    The seed depends on the model configuration, date, and sleeve, ensuring
    that common-random-numbers comparisons across alpha/M are valid.
    """
    key = (
        f"{base_seed}|{schedule_type}|{T}|"
        f"{str(rebalance_date)[:10]}|{sleeve_id}"
    )
    digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return digest % (2**31)


# ---------------------------------------------------------------------------
# Generate M_max scenarios for one sleeve
# ---------------------------------------------------------------------------

def generate_covariance_scenarios(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    condition_vector_raw: np.ndarray,
    conditioning_scaler,
    target_scaler,
    num_scenarios: int = 50,
    seed: Optional[int] = None,
    device: Optional[str | torch.device] = None,
) -> List[np.ndarray]:
    """
    Generate *num_scenarios* plausible next-month SPD covariance matrices
    conditional on the historical sample covariance.

    Parameters
    ----------
    model : ConditionalMLPDenoiser (already in eval mode on device)
    scheduler : DDPMScheduler
    condition_vector_raw : (55,) – log-vech of historical sample covariance
                           (NOT yet standardized)
    conditioning_scaler : fitted sklearn StandardScaler
    target_scaler : fitted sklearn StandardScaler
    num_scenarios : int
    seed : deterministic seed (for common random numbers)
    device : torch device string or None

    Returns
    -------
    List of *num_scenarios* np.ndarray each of shape (10, 10).
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    # Standardize conditioning vector
    c_raw = condition_vector_raw.reshape(1, -1)
    c_scaled = conditioning_scaler.transform(c_raw)  # (1, 55)
    c_tensor = torch.tensor(c_scaled, dtype=torch.float32, device=device)

    # Generate standardized target vectors
    y_generated = p_sample_loop(
        model=model,
        c_tilde=c_tensor,
        scheduler=scheduler,
        num_samples=num_scenarios,
        seed=seed,
    )  # (num_scenarios, 55)

    y_np = y_generated.cpu().numpy()  # (num_scenarios, 55)

    # Inverse-standardize
    y_unscaled = target_scaler.inverse_transform(y_np)  # (num_scenarios, 55)

    # Reconstruct SPD covariance matrices
    covariances: List[np.ndarray] = []
    for i in range(num_scenarios):
        try:
            cov = log_vech_to_covariance(y_unscaled[i])
            covariances.append(cov)
        except Exception as exc:
            logger.warning(
                "Scenario %d covariance reconstruction failed: %s. "
                "Using identity fallback.", i, exc
            )
            covariances.append(np.eye(10) * 1e-4)

    return covariances


# ---------------------------------------------------------------------------
# Combine generated covariances with historical sample covariance
# ---------------------------------------------------------------------------

def combine_covariances(
    sample_cov: np.ndarray,
    generated_covariances: List[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """
    Combine observable historical sample covariance and the arithmetic
    mean of generated next-month covariance scenarios.

    Spec §20.2:
      combined = alpha * sample_cov + (1 - alpha) * mean(generated_covariances)

    NOTE: average in covariance space, not log-covariance space.

    Parameters
    ----------
    sample_cov : (N, N) historical 126-day sample covariance
    generated_covariances : list of (N, N) generated next-month covariances
    alpha : float in [0, 1]

    Returns
    -------
    (N, N) symmetric combined covariance matrix
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}.")

    if alpha == 1.0:
        combined_cov = sample_cov.copy()
    else:
        diffusion_expected_cov = np.mean(
            np.stack(generated_covariances, axis=0),
            axis=0,
        )  # (N, N) – arithmetic mean in covariance space
        combined_cov = (
            alpha * sample_cov
            + (1.0 - alpha) * diffusion_expected_cov
        )

    # Symmetrize numerically
    combined_cov = 0.5 * (combined_cov + combined_cov.T)
    return combined_cov
