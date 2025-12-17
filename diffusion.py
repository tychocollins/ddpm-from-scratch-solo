import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Functions ---

def extract(a, t, x_shape):
    """
    Safely extracts coefficients from a schedule tensor 'a' 
    using indices 't' and reshapes for (B, C, H, W) broadcasting.
    """
    batch_size = t.shape[0]
    # Ensure t is long and on the same device as the schedule
    out = a.gather(-1, t.to(a.device).long())
    # Reshape to (Batch, 1, 1, 1) so it can multiply against (B, 3, 64, 64)
    return out.view(batch_size, 1, 1, 1)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Improved schedule that prevents the 'noise explosion' at early steps.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)

def get_timestep_embedding(timesteps, embedding_dim, device):
    """
    Creates sinusoidal positional embeddings for the time dimension.
    """
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

        # Project the sinusoidal embedding into a higher-dimensional space
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
        )

        # Generate Beta Schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (automatically moves to MPS/GPU with .to(device))
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer("betas", betas)
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """Forward Diffusion: Add noise to the clean image."""
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

    def p_losses(self, x_start, t, t_emb=None, noise=None):
        """Calculate MSE loss between added noise and predicted noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to image
        x_noisy = self.q_sample(x_start, t, noise)
        
        # If training script didn't pass t_emb, create and project it
        if t_emb is None:
            t_emb = get_timestep_embedding(t, self.time_emb_dim, t.device)
        
        # Always project through the local MLP
        t_emb_proj = self.time_mlp(t_emb)
        
        # Neural Network predicts the noise
        predicted_noise = self.model(x_noisy, t_emb_proj)
        
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()


      # -----------------------------
    # One REVERSE process step (sampling)
    # -----------------------------

     #NOTE:
     #This function takes a super-noisy image at timestep t and asks the model: “What was the image like one tiny step ago?” — then removes one tiny bit of noise.
     #That line is literally how Stable Diffusion turns pure static into a perfect face, one step at a time.

    def p_sample(self, x, t, t_index):
        """Reverse Diffusion: Remove one step of noise."""
        t_emb = get_timestep_embedding(t, self.time_emb_dim, t.device)
        t_emb_proj = self.time_mlp(t_emb)
        
        pred_noise = self.model(x, t_emb_proj)
        
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        # Calculate mean of previous step
        mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_t) * pred_noise)
        
        if t_index == 0:
            return mean
        else:
            variance = extract(self.posterior_variance, t, x.shape)
            return mean + torch.sqrt(variance) * torch.randn_like(x)
 #NOTE:
      # p_sample_loop is the magic loop that turns pure random static into a beautiful new face by calling p_sample 
      #(remove one tiny step of noise) 1000 times — starting from total garbage and ending with a real-looking image.
      
    @torch.no_grad()
    def sample(self, batch_size=16, img_size=64):
        """Generate images from pure noise."""
        device = self.betas.device
        img = torch.randn((batch_size, 3, img_size, img_size), device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        return img