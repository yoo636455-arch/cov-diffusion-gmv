"""
test_diffusion_shapes.py
-------------------------
Tests for model dimensionality and generated covariance validity (spec §35.7).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beta_schedules import make_beta_schedule
from src.diffusion import DDPMScheduler, p_sample_loop, q_sample
from src.model import ConditionalMLPDenoiser, build_model
from src.transforms import log_vech_to_covariance


@pytest.fixture
def model():
    return build_model()


@pytest.fixture
def scheduler():
    return DDPMScheduler("linear", T=25)


class TestModelShapes:
    def test_output_dimension(self, model):
        batch = 4
        y_s = torch.randn(batch, 55)
        t = torch.randint(1, 26, (batch,))
        c = torch.randn(batch, 55)
        out = model(y_s, t, c)
        assert out.shape == (batch, 55), f"Output shape {out.shape} != (4, 55)"

    def test_input_dimensions_correct(self, model):
        """Verify model accepts the correct concatenated input width."""
        # noised(55) + time_emb(32) + cond(55) = 142
        # This is tested indirectly through a forward pass
        y = torch.randn(2, 55)
        t = torch.tensor([1, 5])
        c = torch.randn(2, 55)
        out = model(y, t, c)
        assert out.shape == (2, 55)


class TestDiffusionGeneration:
    def test_generated_vector_shape(self, model, scheduler):
        c = torch.randn(55)
        y_gen = p_sample_loop(model, c, scheduler, num_samples=3, seed=0)
        assert y_gen.shape == (3, 55), f"Shape {y_gen.shape}"

    def test_reconstructed_covariance_shape(self, model, scheduler):
        c = torch.randn(55)
        y_gen = p_sample_loop(model, c, scheduler, num_samples=2, seed=42)
        y_np = y_gen.detach().cpu().numpy()
        for i in range(2):
            cov = log_vech_to_covariance(y_np[i])
            assert cov.shape == (10, 10), f"Covariance shape {cov.shape}"

    def test_generated_covariance_is_spd(self, model, scheduler):
        torch.manual_seed(0)
        c = torch.randn(55)
        y_gen = p_sample_loop(model, c, scheduler, num_samples=5, seed=1)
        y_np = y_gen.detach().cpu().numpy()
        for i in range(5):
            cov = log_vech_to_covariance(y_np[i])
            eigvals = np.linalg.eigvalsh(cov)
            assert eigvals.min() > 0, (
                f"Generated covariance {i} is not PD: min eig {eigvals.min():.2e}"
            )

    def test_generated_covariance_is_symmetric(self, model, scheduler):
        c = torch.randn(55)
        y_gen = p_sample_loop(model, c, scheduler, num_samples=3, seed=99)
        y_np = y_gen.detach().cpu().numpy()
        for i in range(3):
            cov = log_vech_to_covariance(y_np[i])
            assert np.allclose(cov, cov.T, atol=1e-8)


class TestForwardDiffusion:
    def test_q_sample_shape(self, scheduler):
        y0 = torch.randn(8, 55)
        s = torch.randint(1, 26, (8,))
        y_s, eps = q_sample(y0, s, scheduler)
        assert y_s.shape == (8, 55)
        assert eps.shape == (8, 55)
