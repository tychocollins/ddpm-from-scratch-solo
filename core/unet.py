import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.SiLU()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.relu(self.norm1(self.conv1(x)))
        h += self.time_mlp(t)[:, :, None, None]
        h = self.relu(self.norm2(self.conv2(h)))
        return h + self.residual_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_embed_dim=256):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Downsampling path
        self.down1 = ResidualBlock(64, 128, time_embed_dim)
        self.down2 = ResidualBlock(128, 256, time_embed_dim)
        self.down3 = ResidualBlock(256, 512, time_embed_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(512, 512, time_embed_dim)
        
        # Upsampling path
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res_up1 = ResidualBlock(512, 256, time_embed_dim) 
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res_up2 = ResidualBlock(256, 128, time_embed_dim)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res_up3 = ResidualBlock(128, 64, time_embed_dim)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def _sinusoidal_embedding(timesteps, embedding_dim):
        device = timesteps.device
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def forward(self, x, t_emb):
        x1 = self.init_conv(x)
        x2 = self.down1(self.pool(x1), t_emb)
        x3 = self.down2(self.pool(x2), t_emb)
        x4 = self.down3(self.pool(x3), t_emb)
        mid = self.bottleneck(x4, t_emb)
        u1 = self.res_up1(torch.cat([self.up1(mid), x3], dim=1), t_emb)
        u2 = self.res_up2(torch.cat([self.up2(u1), x2], dim=1), t_emb)
        u3 = self.res_up3(torch.cat([self.up3(u2), x1], dim=1), t_emb)
        return self.final_conv(u3)