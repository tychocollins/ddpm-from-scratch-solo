# train_mnist.py — FIXED FOR MPS
from diffusion import GaussianDiffusion
from datasets import get_mnist_loader
from unet import UNet
import torch

# Real U-Net (the one you just typed)
model = UNet(in_channels=1, out_channels=1)

diffusion = GaussianDiffusion(model, timesteps=1000)
# This line is correct and should move ALL parameters/buffers of 
# GaussianDiffusion and its submodules (UNet, time_mlp)
diffusion.to('mps')  

loader = get_mnist_loader(batch_size=128)

# Set the model to training mode
diffusion.train()
# You also need an optimizer
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4) 

print("Starting Day 2 training test — REAL U-NET")
print("-" * 50)

for batch_idx, (x, _) in enumerate(loader):
    # Zero gradients
    optimizer.zero_grad() 
    
    # Move data to MPS
    x = x.to('mps')
    t = torch.randint(0, 1000, (x.shape[0],), device='mps').long()

    loss = diffusion.p_losses(x, t)
    
    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    print(f"Batch {batch_idx:2d} → loss: {loss.item():.4f}")

    if batch_idx == 19:
        break

print("-" * 50)
# You need to run this loop for many epochs to see meaningful results, 
# but for a quick test, this works!
print("DAY 2 SUCCESS — real U-Net is training!")