# train_mnist.py — FULL VERSION WITH EPOCHS AND SAVING
from diffusion import GaussianDiffusion
from datasets import get_mnist_loader
from unet import UNet
import torch
import os # Import the os library for path handling

# --- CONFIGURATION ---
NUM_EPOCHS = 20 # Set this higher (e.g., 50-100) for good results
SAVE_PATH = "trained_mnist_weights.pt" # File name for generate.py
DEVICE = 'mps'

# Real U-Net
model = UNet(in_channels=1, out_channels=1)
diffusion = GaussianDiffusion(model, timesteps=1000)
diffusion.to(DEVICE)  

loader = get_mnist_loader(batch_size=128)
diffusion.train()
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4) 

print(f"Starting Training for {NUM_EPOCHS} Epochs on {DEVICE}...")
print("-" * 50)

# --- TRAINING LOOP ---
for epoch in range(1, NUM_EPOCHS + 1):
    for batch_idx, (x, _) in enumerate(loader):
        optimizer.zero_grad() 
        
        # Move data to the correct device
        x = x.to(DEVICE)
        t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=DEVICE).long()

        # Calculate the loss (assumes diffusion.p_losses implements the DDPM objective)
        loss = diffusion.p_losses(x, t) 
        
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch:02d} | Batch {batch_idx:3d} | Loss: {loss.item():.4f}")

    # --- CHECKPOINT SAVING ---
    # Save the model's parameters (state_dict) to a file.
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n[INFO] Saved checkpoint to {SAVE_PATH} after Epoch {epoch}\n")
   

print("-" * 50)
print("TRAINING COMPLETE.")