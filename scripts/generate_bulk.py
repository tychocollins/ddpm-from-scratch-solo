import torch
from torchvision.utils import save_image
import os
import sys

# --- FOLDER STRUCTURE FIX ---
# This tells Python to look one directory up (..) to find the 'core' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.diffusion import Diffusion
from core.unet import UNet

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHTS_PATH = "../high_cap_celeba.pt" # Look one folder up for weights
OUTPUT_DIR = "../assets/bulk_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- INITIALIZE MODEL ---
model = UNet().to(DEVICE)

# --- LOADING LOGIC ---
try:
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Weights loaded successfully from {WEIGHTS_PATH}")
    else:
        print(f"❌ Error: Could not find {WEIGHTS_PATH}")
        sys.exit()
except Exception as e:
    print(f"❌ Load error: {e}")
    sys.exit()

model.eval()
diffusion = Diffusion(model).to(DEVICE)

# --- BULK GENERATION ---
NUM_IMAGES = 20 # Change this to how many you want to generate
print(f"🎨 Generating {NUM_IMAGES} faces in bulk...")

with torch.no_grad():
    # We generate in small batches to avoid memory issues on Mac
    for i in range(NUM_IMAGES // 5):
        samples = diffusion.sample(model, n=5)
        save_image(samples, f"{OUTPUT_DIR}/batch_{i}.png", nrow=5, normalize=True)
        print(f"💾 Saved batch {i+1}/{(NUM_IMAGES//5)}")

print(f"🚀 All samples saved to {OUTPUT_DIR}")