# train_mnist.py — Day 1 test
# This file proves your DDPM code actually works

from diffusion import GaussianDiffusion
from datasets import get_mnist_loader
import torch

# -------------------------------
# Dummy model for now (we replace this tomorrow with real U-Net)
# -------------------------------
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        # Just returns random noise — loss will be ~1.0
        return torch.randn_like(x)

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
# THESE 4 LINES MUST BE OUTSIDE / UNINDENTED FROM THE CLASS!
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

#NOTE: “This loop grabs real digits, randomly noises them, asks our dummy model to guess the noise, 
# and prints how wrong it was — proving the entire DDPM pipeline works.”

model = DummyModel()
diffusion = GaussianDiffusion(model, timesteps=1000)
diffusion.to('mps')                     # remove this line if not on Mac

# Load MNIST
loader = get_mnist_loader(batch_size=128)

print("Starting Day 1 training test...")
print("-" * 50)

# Run 20 batches
for batch_idx, (x, _) in enumerate(loader):
    x = x.to('mps')
    t = torch.randint(0, 1000, (x.shape[0],), device='mps').long()

    loss = diffusion.p_losses(x, t)
    print(f"Batch {batch_idx:2d} → loss: {loss.item():.4f}")

    if batch_idx == 19:
        break

print("-" * 50)
print("DAY 1 SUCCESS — your DDPM is alive!")