# train_cifar.py — Day 7 (Optimized Color Training with EMA)

from diffusion import GaussianDiffusion
from datasets import get_cifar10_loader
from unet import UNet
import torch
import torch.optim as optim
import os
import copy
from collections import OrderedDict

# --- EMA Utility Class (CRUCIAL FOR STABILITY) ---
class EMA:
    """
    Exponential Moving Average helper class. 
    Stores a copy of the model parameters and updates them as a smoothed average.
    """
    def __init__(self, model, decay):
        # A deep copy of the model for holding the EMA weights
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        self.model_state_dict = model.state_dict()
        self.ema_state_dict = self.ema_model.state_dict()
        
        # Initialize EMA buffers to match training model buffers
        for name, param in self.model_state_dict.items():
            if param.dtype.is_floating_point:
                self.ema_state_dict[name].data.copy_(param.data)

    def update(self, model):
        # Update the EMA parameters based on the current model weights
        with torch.no_grad():
            new_state_dict = model.state_dict()
            
            for name, param in new_state_dict.items():
                if param.dtype.is_floating_point:
                    # EMA formula: ema_param = decay * ema_param + (1 - decay) * new_param
                    self.ema_state_dict[name].data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                else:
                    # Non-floating point tensors (e.g., batch norm stats) are copied directly
                    self.ema_state_dict[name].data.copy_(param.data)

    def get_ema_model(self):
        return self.ema_model
# -------------------------------------------------


# --- CONFIGURATION ---
NUM_EPOCHS = 100 # Recommended to increase for CIFAR-10
SAVE_PATH = "trained_cifar_weights_ema.pt" # Name reflects saving EMA weights
DEVICE = 'mps'
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
SAVE_INTERVAL = 5 # Save every 5 epochs
EMA_DECAY = 0.9999 

# U-Net for color (3 channels) - MATCHES TRAINED CHECKPOINT
model = UNet(in_channels=3, out_channels=3)
model.to(DEVICE)

# Note: Assumes GaussianDiffusion in diffusion.py now uses the Cosine Schedule
diffusion = GaussianDiffusion(model, timesteps=1000)
diffusion.to(DEVICE)

loader = get_cifar10_loader(batch_size=BATCH_SIZE)
diffusion.train()

# UPDATED Optimizer: AdamW is standard practice for DDPM/Transformers
optimizer = optim.AdamW(diffusion.parameters(), lr=LEARNING_RATE)

# NEW: EMA Initialization
ema_handler = EMA(model, decay=EMA_DECAY)
print(f"EMA initialized with decay: {EMA_DECAY}")


print(f"Starting Day 7 optimized training — COLOR IMAGES (CIFAR-10) for {NUM_EPOCHS} epochs")
print("-" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    for batch_idx, (x, _) in enumerate(loader):
        optimizer.zero_grad()
        
        x = x.to(DEVICE)
        t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=DEVICE).long()

        loss = diffusion.p_losses(x, t)
        loss.backward()
        optimizer.step()
        
        # NEW: EMA Update after every optimizer step
        ema_handler.update(model)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch:02d} | Batch {batch_idx:4d} | Loss: {loss.item():.6f}")

    # Save checkpoint using EMA weights
    if epoch % SAVE_INTERVAL == 0:
        # CRITICAL CHANGE: Save EMA weights instead of model weights
        torch.save(ema_handler.get_ema_model().state_dict(), SAVE_PATH)
        print(f"\n[INFO] Saved EMA checkpoint to {SAVE_PATH} after Epoch {epoch}\n")

print("-" * 60)
print("DAY 7 OPTIMIZED TRAINING COMPLETE — ready for high-quality color generation!")