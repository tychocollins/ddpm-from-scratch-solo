import torch
import gradio as gr
import os
import numpy as np
from huggingface_hub import hf_hub_download
from core.diffusion import GaussianDiffusion
from core.unet import UNet

# --- CONFIGURATION ---
DEVICE = "cpu" 
WEIGHTS_PATH = "high_cap_celeba.pt"
REPO_ID = "tychocollins7/ddpm-face-generator" 

# --- AUTO-DOWNLOAD WEIGHTS ---
# This pulls the weights from Hugging Face if they are missing locally
if not os.path.exists(WEIGHTS_PATH):
    print(f"📥 {WEIGHTS_PATH} not found. Downloading from Hugging Face...")
    try:
        hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_PATH, local_dir=".")
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Download failed: {e}")

# --- INITIALIZE MODEL ---
model = UNet().to(DEVICE)

try:
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Weights Loaded!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")

diffusion = GaussianDiffusion(model).to(DEVICE)

def generate(num_images, steps):
    try:
        diffusion.num_steps = int(steps)
    except AttributeError:
        pass 

    with torch.no_grad():
        samples = diffusion.sample(model, int(num_images))
    
    processed = []
    for img in samples:
        img = (img + 1.0) / 2.0 
        img = torch.clamp(img, 0, 1)
        img_np = img.cpu().permute(1, 2, 0).numpy()
        processed.append(img_np)
    return processed

# --- THE STACKED LAYOUT (NO GOLDEN 5) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👤 CelebA High-Cap Face Generator")
    gr.Markdown("A 100% scratch-built DDPM. Trained for 200 epochs on Apple M4.")
    
    # Inputs at the top
    num_img = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Number of Images")
    steps = gr.Slider(minimum=100, maximum=1000, value=1000, step=100, label="Diffusion Steps")
    btn = gr.Button("Generate New Faces", variant="primary")

    gr.Markdown("---")

    # Outputs at the bottom
    gallery = gr.Gallery(label="Generated Result", columns=2, height="auto")

    btn.click(fn=generate, inputs=[num_img, steps], outputs=gallery)

    gr.Markdown("---")
    gr.Markdown("🛠️ **Final Loss**: 0.0057 | **Optimizer**: AdamW | **Hardware**: Apple M4 (MPS)")

if __name__ == "__main__":
    demo.launch()
