"""
model.py
--------
Conditional MLP denoiser for DDPM.

Architecture (spec §15):
  Input  : [y_s (55), e(s) (32), c̃ (55)] = 142 dimensions
  Hidden : 3 layers × 128 units, SiLU activation
  Output : 55 dimensions (predicted noise)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal time-step embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Produces a sinusoidal embedding of the diffusion time-step.

    Output dimension: embed_dim.
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor, shape (batch,) – integer diffusion step (1-indexed)

        Returns
        -------
        torch.Tensor, shape (batch, embed_dim)
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )  # (half,)
        args = t[:, None].float() * freqs[None, :]  # (batch, half)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, embed_dim)
        if self.embed_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


# ---------------------------------------------------------------------------
# Conditional MLP denoiser
# ---------------------------------------------------------------------------

class ConditionalMLPDenoiser(nn.Module):
    """
    Predict the noise epsilon given (y_s, s, c̃).

    Parameters
    ----------
    noised_dim       : int  – dimension of noised target vector (55)
    condition_dim    : int  – dimension of conditioning vector (55)
    time_embed_dim   : int  – sinusoidal time embedding dimension (32)
    hidden_dim       : int  – width of each hidden layer (128)
    num_hidden       : int  – number of hidden layers (3)
    dropout          : float – dropout probability (0.0)
    """

    def __init__(
        self,
        noised_dim: int = 55,
        condition_dim: int = 55,
        time_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(embed_dim=time_embed_dim)

        input_dim = noised_dim + time_embed_dim + condition_dim

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, noised_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        y_s: torch.Tensor,
        t: torch.Tensor,
        c_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_s     : (batch, 55) – noised target at step s
        t       : (batch,)    – diffusion step indices (1-indexed integers)
        c_tilde : (batch, 55) – standardized conditioning vector

        Returns
        -------
        eps_hat : (batch, 55) – predicted noise
        """
        t_emb = self.time_embedding(t)          # (batch, 32)
        x = torch.cat([y_s, t_emb, c_tilde], dim=-1)  # (batch, 142)
        return self.net(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    noised_dim: int = 55,
    condition_dim: int = 55,
    time_embed_dim: int = 32,
    hidden_dim: int = 128,
    num_hidden: int = 3,
    dropout: float = 0.0,
) -> ConditionalMLPDenoiser:
    """Construct and return a ConditionalMLPDenoiser."""
    return ConditionalMLPDenoiser(
        noised_dim=noised_dim,
        condition_dim=condition_dim,
        time_embed_dim=time_embed_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        dropout=dropout,
    )
