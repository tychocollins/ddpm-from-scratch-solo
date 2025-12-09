# train_mnist.py — Day 2 (real U-Net)
from diffusion import GaussianDiffusion
from datasets import get_mnist_loader
from unet import UNet
import torch

# Real U-Net (the one you just typed)
model = UNet(in_channels=1, out_channels=1)

diffusion = GaussianDiffusion(model, timesteps=1000)
diffusion.to('mps')  # remove if not on Mac

loader = get_mnist_loader(batch_size=128)

print("Starting Day 2 training test — REAL U-NET")
print("-" * 50)

for batch_idx, (x, _) in enumerate(loader):
    x = x.to('mps')
    t = torch.randint(0, 1000, (x.shape[0],), device='mps').long()

    loss = diffusion.p_losses(x, t)
    print(f"Batch {batch_idx:2d} → loss: {loss.item():.4f}")

    if batch_idx == 19:
        break

print("-" * 50)
print("DAY 2 SUCCESS — real U-Net is training!")