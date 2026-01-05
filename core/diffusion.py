import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(a.device).long())
    return out.view(batch_size, 1, 1, 1)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, time_emb_dim=256):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.time_emb_dim = time_emb_dim

        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer("betas", betas)
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_noisy = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        t_emb = self.model._sinusoidal_embedding(t, self.time_emb_dim)
        predicted_noise = self.model(x_noisy, t_emb)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        t_emb = self.model._sinusoidal_embedding(t, self.time_emb_dim)
        pred_noise = self.model(x, t_emb)
        
        alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        model_mean = alpha_t * (x - (beta_t / sqrt_one_minus_alpha_t) * pred_noise)
        if t_index == 0: return model_mean
        variance = extract(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(variance) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, batch_size=4, img_size=64):
        device = self.betas.device
        img = torch.randn((batch_size, 3, img_size, img_size), device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        return img