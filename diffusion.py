import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Functions (Keep as is) ---
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(a.device).long())
    return out.view(batch_size, 1, 1, 1)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)

def get_timestep_embedding(timesteps, embedding_dim, device):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

# --- Main Diffusion Class ---
class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, time_emb_dim=256):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.time_emb_dim = time_emb_dim

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer("betas", betas)
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

    def p_losses(self, x_start, t, t_emb=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        if t_emb is None:
            t_emb = get_timestep_embedding(t, self.time_emb_dim, t.device)
        predicted_noise = self.model(x_noisy, t_emb)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        t_emb = get_timestep_embedding(t, self.time_emb_dim, t.device)
        pred_noise = self.model(x, t_emb)
        
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        # 1. Calculate the mean and clamp it to keep values stable
        mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_t) * pred_noise)
        mean = torch.clamp(mean, -1.0, 1.0) # <--- NEW: Mean Stabilizer
        
        if t_index == 0:
            return mean
        else:
            variance = extract(self.posterior_variance, t, x.shape)
            # 2. Add slightly less noise (0.7 scale) to prevent color blowout
            return mean + (torch.sqrt(variance) * 0.7) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, batch_size=16, img_size=64):
        device = self.betas.device
        img = torch.randn((batch_size, 3, img_size, img_size), device=device)
        
        print(f"Sampling {batch_size} images with stabilization...")
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            
            # Final safety clamp at each step
            img = torch.clamp(img, -1.0, 1.0)
            
            if i % 100 == 0:
                print(f"Step {i} remaining...")
                
        return img