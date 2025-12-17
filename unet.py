import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Component 1: Adaptive Group Normalization ---
class AdaGroupNorm(nn.Module):
    def __init__(self, channels, time_embed_dim=128, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False)
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * channels)
        )

    def forward(self, x, time_emb):
        x = self.norm(x)
        gamma_beta = self.projection(time_emb)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        return x * (1 + gamma) + beta

# --- Component 2: Conditional Convolutional Block ---
class ConditionalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaGroupNorm(out_channels, time_embed_dim)
        self.relu1 = nn.SiLU() # SiLU is standard for SOTA Diffusion
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm(out_channels, time_embed_dim)
        self.relu2 = nn.SiLU()

    def forward(self, x, time_emb):
        x = self.relu1(self.norm1(self.conv1(x), time_emb))
        x = self.relu2(self.norm2(self.conv2(x), time_emb))
        return x

# --- Component 3: Self-Attention (Crucial for 64x64 faces) ---
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        x_in = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x_in)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_in
        attention_value = self.ff_self(attention_value) + attention_value
        # Restore: (B, H*W, C) -> (B, C, H, W)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

# --- Full UNet Architecture ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_embed_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim

        # Encoder
        self.down1 = ConditionalConvBlock(in_channels, 64, time_embed_dim)
        self.down2 = ConditionalConvBlock(64, 128, time_embed_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck (Attention lives here)
        self.bottleneck_conv = ConditionalConvBlock(128, 256, time_embed_dim)
        self.bottleneck_att = SelfAttention(256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = ConditionalConvBlock(256, 128, time_embed_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = ConditionalConvBlock(128, 64, time_embed_dim)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    @staticmethod
    def _sinusoidal_embedding(timesteps, embedding_dim):
        device = timesteps.device
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, time_emb):
        # Time embedding is already projected by diffusion.py 
        # or can be projected here if using raw t
        
        # Encoder
        d1 = self.down1(x, time_emb)
        d2 = self.down2(self.pool(d1), time_emb)

        # Bottleneck with Global Attention
        
        b = self.bottleneck_conv(self.pool(d2), time_emb)
        b = self.bottleneck_att(b)

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1, time_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2, time_emb)

        return self.final(u2)