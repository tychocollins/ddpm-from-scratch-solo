# generate_cifar.py — Updated for Day 10 Fixed Architecture
import torch
import matplotlib.pyplot as plt
import os
from diffusion import GaussianDiffusion
from unet import UNet

# --- Configuration ---
# Ensure this matches the SAVE_PATH in your training script
WEIGHTS_PATH = "trained_cifar_weights_ema.pt" 
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 1. LOAD MODEL (MUST match training architecture)
# We add time_embed_dim=256 to fix the 'Size Mismatch' error
model = UNet(in_channels=3, out_channels=3, time_embed_dim=256)

if not os.path.exists(WEIGHTS_PATH):
    print(f"ERROR: Weights file not found at {WEIGHTS_PATH}")
    exit()

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval() # CRITICAL for GroupNorm stability

# 2. INITIALIZE DIFFUSION
# Ensure timesteps and time_dim match the training setup
diffusion = GaussianDiffusion(model, timesteps=1000, time_emb_dim=256)
diffusion.to(DEVICE)

print(f"Generating CIFAR-10 images on {DEVICE}...")

# 3. SAMPLING
with torch.no_grad():
    # CIFAR-10 images are 32x32
    samples = diffusion.sample(batch_size=16, img_size=32)

# 4. ENHANCED POST-PROCESSING
samples = samples.cpu()

# Min-Max Scaling per image to fix the 'Blue/Dark' tint
for i in range(samples.shape[0]):
    s_min = samples[i].min()
    s_max = samples[i].max()
    # Stretch values so the darkest pixel is 0 and brightest is 1
    samples[i] = (samples[i] - s_min) / (s_max - s_min + 1e-5)

# 5. PLOTTING
fig, axes = plt.subplots(4, 4, figsize=(8, 8), facecolor='black')
for i in range(16):
    ax = axes[i // 4, i % 4]
    # Convert (C, H, W) -> (H, W, C)
    img_to_show = samples[i].permute(1, 2, 0).numpy()
    ax.imshow(img_to_show)
    ax.axis("off")

plt.tight_layout()
plt.savefig("cifar_generation_fixed.png", dpi=300)
print("Saved result to 'cifar_generation_fixed.png'")
plt.show()