"""
diffusion.py
------------
DDPM forward diffusion, DDPM training step, and DDPM reverse generation.

Spec §14, §19.
"""

from __future__ import annotations

import numpy as np
import torch

from .beta_schedules import compute_alpha_bar, make_beta_schedule
from .model import ConditionalMLPDenoiser


# ---------------------------------------------------------------------------
# DDPMScheduler – holds all schedule tensors on the right device
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """Pre-compute and cache all schedule tensors for a given (schedule_type, T)."""

    def __init__(
        self,
        schedule_type: str,
        T: int,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        device: str | torch.device = "cpu",
    ):
        self.T = T
        self.schedule_type = schedule_type
        self.device = torch.device(device)

        betas_np = make_beta_schedule(schedule_type, T, beta_min, beta_max)
        alphas_np = 1.0 - betas_np
        alpha_bar_np = compute_alpha_bar(betas_np)

        # Store as torch tensors (shape: (T,))
        self.betas = torch.tensor(betas_np, dtype=torch.float32, device=self.device)
        self.alphas = torch.tensor(alphas_np, dtype=torch.float32, device=self.device)
        self.alpha_bar = torch.tensor(alpha_bar_np, dtype=torch.float32, device=self.device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # For DDPM reverse step:
        #   posterior variance: beta_tilde_s = beta_s * (1 - alpha_bar_{s-1}) / (1 - alpha_bar_s)
        alpha_bar_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alpha_bar[:-1]]
        )
        self.posterior_variance = (
            self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_variance = self.posterior_variance.clamp(min=1e-20)

    def to(self, device: str | torch.device) -> "DDPMScheduler":
        """Move all tensors to a new device and return self."""
        self.device = torch.device(device)
        for attr in [
            "betas", "alphas", "alpha_bar", "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(self.device))
        return self


# ---------------------------------------------------------------------------
# Forward diffusion (training)
# ---------------------------------------------------------------------------

def q_sample(
    y0: torch.Tensor,
    s: torch.Tensor,
    scheduler: DDPMScheduler,
    eps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to y0 at diffusion step s using the closed-form:
      y_s = sqrt(alpha_bar_s) * y0 + sqrt(1 - alpha_bar_s) * eps

    Parameters
    ----------
    y0  : (batch, 55) – clean target vector
    s   : (batch,)    – 1-indexed diffusion step
    scheduler : DDPMScheduler
    eps : (batch, 55) or None – if None, sample from N(0, I)

    Returns
    -------
    (y_s, eps) both of shape (batch, 55)
    """
    if eps is None:
        eps = torch.randn_like(y0)

    # s is 1-indexed; index into alpha_bar with (s-1)
    sqrt_ab = scheduler.sqrt_alpha_bar[s - 1][:, None]           # (batch, 1)
    sqrt_1_ab = scheduler.sqrt_one_minus_alpha_bar[s - 1][:, None]  # (batch, 1)

    y_s = sqrt_ab * y0 + sqrt_1_ab * eps
    return y_s, eps


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def ddpm_training_step(
    model: ConditionalMLPDenoiser,
    y0: torch.Tensor,
    c_tilde: torch.Tensor,
    scheduler: DDPMScheduler,
) -> torch.Tensor:
    """
    Single DDPM training step.

    1. Sample s ~ Uniform{1, ..., T}
    2. Sample eps ~ N(0, I)
    3. Compute y_s via forward diffusion
    4. Predict eps_hat via the denoiser
    5. Return MSE(eps, eps_hat)

    Parameters
    ----------
    model    : ConditionalMLPDenoiser
    y0       : (batch, 55) – standardized target vectors
    c_tilde  : (batch, 55) – standardized conditioning vectors
    scheduler : DDPMScheduler

    Returns
    -------
    torch.Tensor scalar – mean DDPM noise prediction loss
    """
    batch_size = y0.shape[0]
    device = y0.device

    # Sample random diffusion steps (1-indexed)
    s = torch.randint(1, scheduler.T + 1, (batch_size,), device=device)

    # Forward diffusion
    y_s, eps = q_sample(y0, s, scheduler)

    # Predict noise
    eps_hat = model(y_s, s, c_tilde)

    # MSE loss
    loss = ((eps - eps_hat) ** 2).mean()
    return loss


# ---------------------------------------------------------------------------
# Reverse diffusion (inference / generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def p_sample_loop(
    model: ConditionalMLPDenoiser,
    c_tilde: torch.Tensor,
    scheduler: DDPMScheduler,
    num_samples: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate *num_samples* covariance vectors conditional on c_tilde.

    Uses DDPM ancestral sampling (p_theta sampling).

    Parameters
    ----------
    model      : ConditionalMLPDenoiser (in eval mode)
    c_tilde    : (55,) or (1, 55) – conditioning vector for ONE sleeve
    scheduler  : DDPMScheduler
    num_samples : number of scenarios M to generate
    seed       : if not None, fix torch RNG for reproducibility

    Returns
    -------
    torch.Tensor, shape (num_samples, 55) – generated standardized target vectors
    """
    model.eval()
    device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)

    # Broadcast conditioning vector to (num_samples, 55)
    if c_tilde.dim() == 1:
        c_tilde = c_tilde.unsqueeze(0)
    c_tilde = c_tilde.expand(num_samples, -1).to(device)

    # Start from pure Gaussian noise
    y = torch.randn(num_samples, c_tilde.shape[-1], device=device)

    for s_idx in range(scheduler.T, 0, -1):
        s_tensor = torch.full((num_samples,), s_idx, dtype=torch.long, device=device)

        # Predict noise
        eps_hat = model(y, s_tensor, c_tilde)

        # DDPM reverse step
        alpha_s = scheduler.alphas[s_idx - 1]
        alpha_bar_s = scheduler.alpha_bar[s_idx - 1]
        sqrt_one_minus_ab = scheduler.sqrt_one_minus_alpha_bar[s_idx - 1]

        # x0 estimate
        x0_hat = (y - sqrt_one_minus_ab * eps_hat) / torch.sqrt(alpha_bar_s)
        x0_hat = x0_hat.clamp(-10.0, 10.0)  # stability clamp

        if s_idx > 1:
            alpha_bar_prev = scheduler.alpha_bar[s_idx - 2]
            posterior_mean = (
                torch.sqrt(alpha_bar_prev) * scheduler.betas[s_idx - 1] / (1.0 - alpha_bar_s) * x0_hat
                + torch.sqrt(alpha_s) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_s) * y
            )
            posterior_var = scheduler.posterior_variance[s_idx - 1]
            noise = torch.randn_like(y)
            y = posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            # Last step: no noise added (s=1)
            y = x0_hat

    return y  # (num_samples, 55)
