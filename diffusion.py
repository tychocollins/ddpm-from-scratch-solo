# diffusion.py — FIXED VERSION (Time Embedding & Reverse Step)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional


def get_timestep_embedding(timesteps, embedding_dim, device):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps: int = 1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Time embedding
        self.time_emb_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
        )

        # Linear beta schedule
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        
        # New buffers for easier access to alpha coefficients (Fix 2: Mean Calculation)
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer("posterior_mean_coef1", betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)) # Not used if predicting noise/x_0, but useful to have.
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)) # Not used if predicting noise/x_0, but useful to have.


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t, None, None, None] * x_start +
            self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )
# ================================================
    # DAY 1 – THE REAL TRAINING STARTS HERE  
    # ================================================

    #THE FORWARD PROCESS
    #NOTE: p_losses the ONLY function that actually trains the diffusion model — it takes a clean image, 
    # adds noise at a random timestep, asks the neural network “what noise did I just add?”, 
    # and punishes it with MSE loss for being wrong.


    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # Time embedding (FIX 1: Apply to model, not x_noisy)
        time_emb = get_timestep_embedding(t, self.time_emb_dim, x_noisy.device)
        time_emb = self.time_mlp(time_emb)
        # Note: We pass time_emb (shape B, D) to the model. 
        # The model must apply it, typically via FiLM/AdaptiveNorm layers.
        
        predicted_noise = self.model(x_noisy, time_emb) # <--- UPDATED: passing time_emb
        return F.mse_loss(predicted_noise, noise)

  # -----------------------------
    # One REVERSE process step (sampling)
    # -----------------------------

     #NOTE:
     #This function takes a super-noisy image at timestep t and asks the model: “What was the image like one tiny step ago?” — then removes one tiny bit of noise.
     #That line is literally how Stable Diffusion turns pure static into a perfect face, one step at a time.


    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        # Time embedding (FIX 1: Apply to model, not x)
        time_emb = get_timestep_embedding(t, self.time_emb_dim, x.device)
        time_emb = self.time_mlp(time_emb)
        
        predicted_noise = self.model(x, time_emb) # <--- UPDATED: passing time_emb
        
        # FIX 2: Correct Mean Calculation (Based on canonical DDPM formula)
        # mu_theta(x_t, t) = 1/sqrt(alpha_t) * [ x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t) ]
        
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t, None, None, None] # 1 / sqrt(alpha_t)
        
        # Coefficient for the predicted noise: beta_t / sqrt(1 - alpha_bar_t)
        noise_coef = self.betas[t, None, None, None] / self.sqrt_one_minus_alphas_cumprod[t, None, None, None]

        mean = sqrt_recip_alpha_t * (x - noise_coef * predicted_noise)

        if t_index == 0:
            return mean
        else:
            variance = self.posterior_variance[t, None, None, None]
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()


      #NOTE:
      # p_sample_loop is the magic loop that turns pure random static into a beautiful new face by calling p_sample 
      #(remove one tiny step of noise) 1000 times — starting from total garbage and ending with a real-looking image.
    def p_sample_loop(self, shape):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, img_size=64):
        return self.p_sample_loop((batch_size, 3, img_size, img_size))