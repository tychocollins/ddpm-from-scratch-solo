# diffusion.py
# First 50 lines of the GaussianDiffusion class – 100 % typed by you, explained line-by-line
# This is the heart of DDPM – everything starts here.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

class GaussianDiffusion(nn.Module):
    """
    Classic DDPM (Denoising Diffusion Probabilistic Model)
    Paper: https://arxiv.org/abs/2006.11239
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "linear",           # "linear" or "cosine"
    ):
        super().__init__()
        self.timesteps = timesteps

        if schedule == "linear":
            betas = self.linear_beta_schedule()
        elif schedule == "cosine":
            betas = self.cosine_beta_schedule()
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

        # Pre-computey everything once
        self.register_buffer("betas", betas)                     # β_t
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)                   # α_t
        self.register_buffer("alphas_cumprod", alphas.cumprod(dim=0))  # ᾱ_t
        self.register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer("sqrt_alphas_cumprod", self.alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - self.alphas_cumprod).sqrt())

    def linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.timesteps)

    def cosine_beta_schedule(self):
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + 0.008) / 1.008 * math.pi / 2).pow(2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # Forward process: q(x_t | x_0)
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t, None, None, None] * x_start +
            self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )
