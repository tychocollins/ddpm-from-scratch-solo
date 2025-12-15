# diffusion.py — FINAL WORKING VERSION (Fixes Time MLP and Channel Mismatch)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# --- Helper Functions ---

def get_timestep_embedding(timesteps, embedding_dim, device):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extracts the values of a at index t, and reshapes it to broadcast across x."""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


#BETA SCHEDULE

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in paper: 'Improved Denoising Diffusion Probabilistic Models'
    s is a small offset to prevent beta_t from being too close to 1 at t=0.
    """
    # Create a tensor for the timesteps
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # Calculate the alpha_cumprod values using the cosine function
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Calculate betas from the ratio of consecutive alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # Clip betas to ensure stability (preventing very small/large values)
    return torch.clip(betas, 0.0001, 0.999)
# --- Main Diffusion Class ---

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps: int = 1000, time_emb_dim: int = 128): # Use 128 as defined in your UNet
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Time embedding MLP — DEFINED HERE (on the Diffusion object)
        self.time_emb_dim = time_emb_dim 
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
        )

        # Cosine beta schedule (CRITICAL CHANGE)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register all coefficients as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

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

        # FIX 1: Use the UNet's static method for embedding, but get the dimension from self.time_mlp
        time_emb = self.model._sinusoidal_embedding(t, self.time_mlp[0].in_features)
        
        # FIX 2: Project the embedding using the Diffusion object's MLP
        time_emb = self.time_mlp(time_emb)
        
        predicted_noise = self.model(x_noisy, time_emb) 
        return F.mse_loss(predicted_noise, noise)


   # -----------------------------
    # One REVERSE process step (sampling)
    # -----------------------------

     #NOTE:
     #This function takes a super-noisy image at timestep t and asks the model: “What was the image like one tiny step ago?” — then removes one tiny bit of noise.
     #That line is literally how Stable Diffusion turns pure static into a perfect face, one step at a time.

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        # FIX 1: Use the UNet's static method for embedding, but get the dimension from self.time_mlp
        time_emb = self.model._sinusoidal_embedding(t, self.time_mlp[0].in_features)
        
        # FIX 2: Project the embedding using the Diffusion object's MLP
        time_emb = self.time_mlp(time_emb)
        
        predicted_noise = self.model(x, time_emb) 
        
        # Extract coefficients using the helper for consistency
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        # Mean Calculation: mu_theta(x_t, t)
        noise_coef = beta_t / sqrt_one_minus_alpha_t
        mean = sqrt_recip_alpha_t * (x - noise_coef * predicted_noise)

        if t_index == 0:
            return mean
        else:
            variance = extract(self.posterior_variance, t, x.shape)
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
        # FIX 3: Retrieve the channel count dynamically from the UNet model
        num_channels = self.model.in_channels 
        return self.p_sample_loop((batch_size, num_channels, img_size, img_size))