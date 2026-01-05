import torch
import os
from diffusion import GaussianDiffusion
from unet import UNet
from torchvision.utils import save_image

# --- CONFIG ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "high_cap_celeba.pt"
OUTPUT_DIR = "stock_samples"
NUM_IMAGES = 100
BATCH_SIZE = 10  # Generates 10 at a time to save memory
STEPS = 1000     # Higher steps = better quality

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Model
model = UNet().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Handle both 'full' and 'weight-only' checkpoints
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
diffusion = GaussianDiffusion(model).to(DEVICE)

print(f"🎨 Generating {NUM_IMAGES} faces to '{OUTPUT_DIR}'...")

# 2. Generation Loop
with torch.no_grad():
    for i in range(0, NUM_IMAGES, BATCH_SIZE):
        # Generate a batch of faces
        samples = diffusion.sample(batch_size=BATCH_SIZE)
        
        # Scale back to 0-1 for saving
        samples = (samples + 1.0) / 2.0
        
        # Save individual images
        for j in range(samples.shape[0]):
            img_id = i + j + 1
            save_image(samples[j], f"{OUTPUT_DIR}/face_{img_id:03d}.png")
            
        print(f"✅ Saved images {i+1} through {min(i+BATCH_SIZE, NUM_IMAGES)}")

print(f"\n✨ Done! Check the '{OUTPUT_DIR}' folder for your assets.")