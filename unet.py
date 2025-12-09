# unet.py — FINAL WORKING VERSION FOR MNIST (28×28)
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)   # 128 (up) + 128 (skip)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)    # 64 + 64

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t=None):
        # Encoder
        d1 = self.down1(x)          # 28×28 → 28×28
        d2 = self.down2(self.pool(d1))  # 14×14

        # Bottleneck
        b = self.bottleneck(self.pool(d2))  # 7×7

        # Decoder
        u1 = self.up1(b)                     # 7×7 → 14×14
        u1 = torch.cat([u1, d2], dim=1)      # skip connection
        u1 = self.conv1(u1)                  # 14×14

        u2 = self.up2(u1)                    # 14×14 → 28×28
        u2 = torch.cat([u2, d1], dim=1)      # skip connection
        u2 = self.conv2(u2)                  # 28×28

        return self.final(u2)                # 28×28 × 1