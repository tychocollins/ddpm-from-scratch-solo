# generate.py — FINAL WORKING VERSION
from diffusion import GaussianDiffusion
from unet import UNet
import torch
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# You need to specify the path where your trained model weights are saved.
CHECKPOINT_PATH = "trained_mnist_weights.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
IMG_SIZE = 28
BATCH_SIZE = 16

# Load the model structure (MNIST: 1 channel in/out)
model = UNet(in_channels=1, out_channels=1)
# Initialize the diffusion object with the model
diffusion = GaussianDiffusion(model, timesteps=1000)

# --- Load Weights (IMPORTANT) ---
try:
    # Check if a checkpoint path was specified and exists
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        
    # Load the state dictionary (only the model's weights)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
except FileNotFoundError as e:
    print(f"Error: {e}. You need to train the model and save its state_dict first!")
    print("Exiting generation script.")
    exit()

# Set up for generation
diffusion.to(DEVICE)
diffusion.eval() # VERY IMPORTANT for generation

print("Generating your first real handwritten digits from pure static...")
print("-" * 60)

with torch.no_grad():
    # Call the fixed sample function
    samples = diffusion.sample(batch_size=BATCH_SIZE, img_size=IMG_SIZE) 

# Convert to numpy for plotting
samples = samples.cpu()
samples = (samples + 1) / 2  # [-1, 1] → [0, 1]
samples = samples.clamp(0, 1)

# Plot 4×4 grid
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(BATCH_SIZE):
    ax = axes[i // 4, i % 4]
    ax.imshow(samples[i].squeeze(0), cmap="gray")
    ax.axis("off")

plt.tight_layout()
save_path = "day4_first_real_digits.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"GENERATION COMPLETE — your first {BATCH_SIZE} real handwritten digits generated!")
print(f"Saved as {save_path}")