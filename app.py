import torch
import gradio as gr
from diffusion import GaussianDiffusion
from unet import UNet

DEVICE = "cpu" # Stay on CPU for app stability while training on GPU
WEIGHTS_PATH = "high_cap_celeba.pt"

# Initialize Model
model = UNet().to(DEVICE)

# --- NEW LOADING LOGIC ---
try:
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
except Exception as e:
    print(f"⚠️ Could not load weights: {e}")
# -------------------------

diffusion = GaussianDiffusion(model).to(DEVICE)

def generate(num_images, steps):
    # Ensure the diffusion object uses the requested number of steps
    # Note: Ensure your GaussianDiffusion class handles this dynamic change
    diffusion.timesteps = int(steps)
    
    # Generate samples
    with torch.no_grad():
        samples = diffusion.sample(batch_size=int(num_images)).cpu()
    
    processed = []
    for img in samples:
        # CELEBA SYNC: Map [-1, 1] back to [0, 1]
        img = (img + 1.0) / 2.0 
        img = torch.clamp(img, 0, 1)
        # Permute from (C, H, W) to (H, W, C) for Gradio
        processed.append(img.permute(1, 2, 0).numpy())
    return processed

# Define sliders
inputs = [
    gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Number of Images"),
    gr.Slider(minimum=100, maximum=1000, value=1000, step=100, label="Diffusion Steps")
]

demo = gr.Interface(
    fn=generate, 
    inputs=inputs, 
    outputs=gr.Gallery(label="Generated Faces"),
    title="CelebA High-Cap Generator",
    description="Optimized for M4 training. Higher steps yield clearer results."
)

demo.launch()