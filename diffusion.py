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
        model,
        timesteps: int = 1000,
        schedule: str = "linear",           # "linear" or "cosine"
    ):
        super().__init__()
        self.model = model
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


# ================================================
    # DAY 1 – THE REAL TRAINING STARTS HERE
    # You are typing every line below yourself
    # ================================================

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        #The simplified training objective from the paper (Equation 12)
        #This is literally the ONLY loss we use — everything else is just setup
        
        if noise is None:
            noise = torch.randn_like(x_start)

            # Add noise according to q(x_t | x_0) — this is the forward process
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

            # Model that predicts the noise that was added
            predicted_noise = self.model(x_noisy,  t)

            #MSE between real noise and predicted noise - this trains the model
            loss = F.mse_loss(predicted_noise, noise)
            return loss
    
    # -----------------------------
    # One reverse process step (sampling)
    # -----------------------------

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, t_index: int):
        """ Take one step from x_t to x_{t-1}"""
        betas_t = self.betas[t, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])

        #Equation 11 from the paper - the mean of the Reverse Step
        predicted_noise = self.model(x, t)
        mean = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise
        )

        if t_index == 0:
            return mean
        else:
            posterior_variance_t = self.posterior_variance[t, None, None, None]
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise
        
        @torch.no_grad()
        def p_sample_loop(self, shape):
            """ Full sampling loop - start from pure noise (x_T) 
            to final image (x_0)"""
            device = self.betas.device
            b = shape[0]
            img = torch.randn(shape, device=device)
            for i in reversed(range(0, self.timesteps)):
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t, i)
                return img

    @torch.no_grad()
    def sample(self, batch_size=16, img_size=64):
         #Easy Public function to generate images
            return self.p_sample_loop(shape=(batch_size, 3, img_size, img_size))

