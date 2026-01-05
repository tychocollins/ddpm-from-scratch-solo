import torch
from torchvision.utils import save_image
import os
import sys

# --- FOLDER STRUCTURE FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.diffusion import Diffusion
from core.unet import UNet

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHTS_PATH = "../high_cap_celeba.pt"
OUTPUT_DIR = "../assets/progress"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model
model = UNet().to(DEVICE)
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

# Handle both full checkpoints and weight-only files
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
diffusion = Diffusion(model).to(DEVICE)

print("🎨 Generating faces...")
with torch.no_grad():
    # Generate 5 faces
    samples = diffusion.sample(model, n=5) 
    save_image(samples, f"{OUTPUT_DIR}/latest_generation.png", nrow=5, normalize=True)

print(f"✅ Generation complete. Check {OUTPUT_DIR}/latest_generation.png")