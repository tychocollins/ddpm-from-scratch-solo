# unet.py — FINAL WORKING VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- NEW COMPONENT: AdaGroupNorm (Adaptive Group Normalization) ---
class AdaGroupNorm(nn.Module):
    """
    Adaptive Group Normalization to condition the features on the time embedding.
    """
    def __init__(self, channels, time_embed_dim=128, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False)
        # Two linear layers to project time embedding into scale (gamma) and shift (beta)
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * channels)
        )

    def forward(self, x, time_emb):
        # Normalize the input features
        x = self.norm(x)
        
        # Project time embedding to gamma and beta
        gamma_beta = self.projection(time_emb)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        
        # Apply modulation: gamma * x + beta
        return x * (1 + gamma) + beta


# --- MODIFIED BLOCK: ConditionalConvBlock ---
class ConditionalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=128):
        super().__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaGroupNorm(out_channels, time_embed_dim)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm(out_channels, time_embed_dim)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, time_emb):
        # Conv 1
        x = self.conv1(x)
        x = self.norm1(x, time_emb)
        x = self.relu1(x)
        
        # Conv 2
        x = self.conv2(x)
        x = self.norm2(x, time_emb)
        x = self.relu2(x)
        return x


# --- MODIFIED UNet ---
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_embed_dim=128):
        super().__init__()
        self.in_channels = in_channels # <-- CRITICAL: Saves channel count for diffusion.py
        self.time_embed_dim = time_embed_dim

        # Encoder - Now uses ConditionalConvBlock
        self.down1 = ConditionalConvBlock(in_channels, 64, time_embed_dim)
        self.down2 = ConditionalConvBlock(64, 128, time_embed_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConditionalConvBlock(128, 256, time_embed_dim)

        # Decoder (using ConvTranspose2d for upsampling)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = ConditionalConvBlock(256, 128, time_embed_dim)   # 128 (up) + 128 (skip)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = ConditionalConvBlock(128, 64, time_embed_dim)    # 64 + 64

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    # *** FIX: ADDED SINUSOIDAL EMBEDDING METHOD ***
    @staticmethod
    def _sinusoidal_embedding(timesteps, embedding_dim):
        """
        Calculates the sinusoidal positional encoding for timesteps.
        (This method is called by diffusion.py's p_losses and p_sample)
        """
        device = timesteps.device
        
        half_dim = embedding_dim // 2
        
        # Calculate frequency terms
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        
        # Calculate arguments (t * freq) and combine sin/cos
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Pad if dimension is odd
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, time_emb): # <--- UPDATED SIGNATURE: Accepts time_emb
        # Encoder
        d1 = self.down1(x, time_emb)
        d2 = self.down2(self.pool(d1), time_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(d2), time_emb)

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1, time_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2, time_emb)

        return self.final(u2)