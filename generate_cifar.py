# generate_cifar.py — Day 8
# Generate your first real color images from pure static

from diffusion import GaussianDiffusion
from unet import UNet
import torch
import matplotlib.pyplot as plt
import os # Added for path checking

# --- Configuration ---
WEIGHTS_PATH = "trained_cifar_weights_ema.pt" # <<< CORRECTED TO EMA WEIGHTS
DEVICE = 'mps' # Set your device

# --- Load Model and Weights ---
model = UNet(in_channels=3, out_channels=3)

# CRITICAL CHECK: Ensure the EMA weights file exists
if not os.path.exists(WEIGHTS_PATH):
    print(f"ERROR: EMA weights file not found at {WEIGHTS_PATH}")
    print("Please ensure your training run finished and saved the EMA model.")
    exit()

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Initialize Diffusion Wrapper (with Cosine Schedule baked in) ---
diffusion = GaussianDiffusion(model, timesteps=1000)
diffusion.to(DEVICE)

print("Generating your first real color images from pure static...")
print("-" * 60)

# --- Sampling ---
with torch.no_grad():
    samples = diffusion.sample(batch_size=16, img_size=32)

# --- Post-processing and Plotting ---
samples = samples.cpu()
samples = (samples + 1) / 2          # Rescale from [-1, 1] to [0, 1]
samples = samples.clamp(0, 1)

# Plot 4×4 grid
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i in range(16):
    ax = axes[i // 4, i % 4]
    # IMPORTANT: The permute is necessary to change PyTorch's (C, H, W) to Matplotlib's (H, W, C) for RGB display.
    ax.imshow(samples[i].permute(1, 2, 0))
    ax.axis("off")

plt.tight_layout()
OUTPUT_FILE = "day8_first_color_images.png"
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"DAY 8 COMPLETE — first real color images generated!")
print(f"Saved as {OUTPUT_FILE}")

