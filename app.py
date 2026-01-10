import torch
import gradio as gr
import os
import sys
import numpy as np
from huggingface_hub import hf_hub_download


# No sys.path.append needed here because app.py is in the root 
# and can see the 'core' folder directly.
from core.diffusion import GaussianDiffusion
from core.unet import UNet

DEVICE = "cpu" # Stay on CPU for app stability to keep GPU free for training
WEIGHTS_PATH = "high_cap_celeba.pt"
REPO_ID = "tychocollins7/ddpm-face-generator"

# --- AUTO-DOWNLOAD SECTION (Place it here!) ---
if not os.path.exists(WEIGHTS_PATH):
    print(f"📥 {WEIGHTS_PATH} not found locally. Downloading from Hugging Face...")
    try:
        hf_hub_download(
            repo_id=REPO_ID, 
            filename=WEIGHTS_PATH, 
            local_dir="."
        )
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Download failed: {e}")





# --- INITIALIZE MODEL ---
model = UNet().to(DEVICE)

# --- ROBUST LOADING LOGIC ---
try:
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        
        # Check if we saved a full checkpoint dictionary or just the weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'Unknown')
            print(f"✅ High-Capacity Weights Loaded! (From Epoch {epoch})")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Weights Loaded (Standard format)")
            
        model.eval()
    else:
        print(f"⚠️ Warning: {WEIGHTS_PATH} not found. App will run with random weights.")
except Exception as e:
    print(f"❌ Could not load weights: {e}")

# Initialize the diffusion engine with the model
diffusion = GaussianDiffusion(model).to(DEVICE)

def generate(num_images, steps):
    """
    Gradio wrapper to generate images and format them for the web gallery.
    """
    # Ensure the diffusion object uses the requested number of steps
    # Note: Your Diffusion class must support setting timesteps dynamically
    try:
        diffusion.num_steps = int(steps)
    except AttributeError:
        pass # Fallback if steps are hardcoded in your class
    
    # Generate samples using your Diffusion.sample method
    with torch.no_grad():
        # Using n=num_images to match the logic used in your scripts
        samples = diffusion.sample(model, int(num_images))
    
    processed = []
    for img in samples:
        # CELEBA SYNC: Map [-1, 1] back to [0, 1] for display
        img = (img + 1.0) / 2.0 
        img = torch.clamp(img, 0, 1)
        
        # Move to CPU and permute from (C, H, W) to (H, W, C) for Gradio/NumPy
        img_np = img.cpu().permute(1, 2, 0).numpy()
        processed.append(img_np)
        
    return processed

# --- GRADIO INTERFACE ---
inputs = [
    gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Number of Images"),
    gr.Slider(minimum=100, maximum=1000, value=1000, step=100, label="Diffusion Steps")
]

demo = gr.Interface(
    fn=generate, 
    inputs=inputs, 
    outputs=gr.Gallery(label="Generated Faces", columns=2, height="auto"),
    title="CelebA High-Cap Face Generator",
    description="A 100% scratch-built DDPM. Higher steps yield clearer facial features. Trained on M4."
)

if __name__ == "__main__":
    demo.launch()
