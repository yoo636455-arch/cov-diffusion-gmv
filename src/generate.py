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
from .model import UnconditionalMLPDenoiser
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


# ---------------------------------------------------------------------------
# SDEdit-style partial noising + reverse denoising (unconditional)
# ---------------------------------------------------------------------------

@torch.no_grad()
def p_sample_loop_partial(
    model: UnconditionalMLPDenoiser,
    scheduler: DDPMScheduler,
    y_start: torch.Tensor,
    s_start: int,
) -> torch.Tensor:
    """
    Run the unconditional DDPM reverse chain from s_start down to 0.
    """
    model.eval()
    device = next(model.parameters()).device
    y = y_start.to(device)
    num_draws = y.shape[0]

    for s_idx in range(s_start, 0, -1):
        s_tensor = torch.full((num_draws,), s_idx, dtype=torch.long, device=device)
        eps_hat = model(y, s_tensor)

        alpha_s    = scheduler.alphas[s_idx - 1]
        alpha_bar_s = scheduler.alpha_bar[s_idx - 1]
        sqrt_1_ab  = scheduler.sqrt_one_minus_alpha_bar[s_idx - 1]

        x0_hat = (y - sqrt_1_ab * eps_hat) / torch.sqrt(alpha_bar_s)
        x0_hat = x0_hat.clamp(-10.0, 10.0)

        if s_idx > 1:
            alpha_bar_prev = scheduler.alpha_bar[s_idx - 2]
            posterior_mean = (
                torch.sqrt(alpha_bar_prev) * scheduler.betas[s_idx - 1] / (1.0 - alpha_bar_s) * x0_hat
                + torch.sqrt(alpha_s) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_s) * y
            )
            posterior_var = scheduler.posterior_variance[s_idx - 1]
            y = posterior_mean + torch.sqrt(posterior_var) * torch.randn_like(y)
        else:
            y = x0_hat

    return y


def generate_denoised_covariances(
    model: UnconditionalMLPDenoiser,
    scheduler: DDPMScheduler,
    condition_vector_raw: np.ndarray,
    conditioning_scaler,
    rho: float,
    num_draws: int = 20,
    seed: Optional[int] = None,
    device: Optional[str | torch.device] = None,
) -> List[np.ndarray]:
    """
    SDEdit-style partial denoising of an observed historical covariance.

    Takes the 126-day sample covariance log-vech, corrupts it to step
    s_start = round(rho * T), then reverse-denoises with the unconditional
    model. Returns num_draws independent denoised SPD covariance matrices.
    The caller should average these in covariance space.
    """
    if rho <= 0.0:
        raise ValueError("rho must be > 0.")

    s_start = max(1, round(rho * scheduler.T))

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    # Standardize observed covariance log-vech
    c_raw = condition_vector_raw.reshape(1, -1)
    y0_scaled = conditioning_scaler.transform(c_raw)
    y0_tensor = torch.tensor(y0_scaled, dtype=torch.float32, device=device)
    y0_tensor = y0_tensor.expand(num_draws, -1).clone()

    if seed is not None:
        torch.manual_seed(seed)

    # Forward-corrupt to s_start
    alpha_bar_s = scheduler.alpha_bar[s_start - 1]
    eps_fwd = torch.randn_like(y0_tensor)
    y_corrupted = torch.sqrt(alpha_bar_s) * y0_tensor + torch.sqrt(1.0 - alpha_bar_s) * eps_fwd

    # Reverse-denoise
    y_denoised = p_sample_loop_partial(model, scheduler, y_corrupted, s_start)

    # Inverse-standardize with the same conditioning scaler
    y_np = y_denoised.cpu().numpy()
    y_unscaled = conditioning_scaler.inverse_transform(y_np)

    covariances: List[np.ndarray] = []
    for i in range(num_draws):
        try:
            cov = log_vech_to_covariance(y_unscaled[i])
            covariances.append(cov)
        except Exception as exc:
            logger.warning("Denoised covariance %d failed: %s. Using identity.", i, exc)
            covariances.append(np.eye(10) * 1e-4)

    return covariances
