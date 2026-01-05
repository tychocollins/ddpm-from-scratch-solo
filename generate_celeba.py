import torch
import matplotlib.pyplot as plt
from diffusion import GaussianDiffusion
from unet import UNet
import os

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHTS_PATH = "trained_celeba_weights_ema.pt"

def generate():
    # 1. INITIALIZE MODEL
    # time_embed_dim must match your training script (256)
    model = UNet(in_channels=3, out_channels=3, time_embed_dim=256).to(DEVICE)
    
    # 2. LOAD WEIGHTS WITH STABILITY
    try:
        # Load and force to float32 to prevent M4 rounding errors
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.float() 
        print(f"✅ Successfully loaded weights from: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: {WEIGHTS_PATH} not found.")
        return
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}")
        return

    model.eval()

    # 3. SETUP DIFFUSION
    # If 1500 steps turns black, this code will handle it better, 
    # but 1000-1200 is the 'Stability Sweet Spot' for M4.
    steps = 1200 
    diffusion = GaussianDiffusion(model, timesteps=steps, time_emb_dim=256).to(DEVICE)

    print(f"🎨 Sampling 16 faces at {steps} steps (Stability Mode)...")
    
    with torch.no_grad():
        # Generate samples from noise
        samples = diffusion.sample(batch_size=16, img_size=64)

    # 4. ROBUST POST-PROCESSING (Prevents Black/Blank Images)
    # Move to CPU and replace any 'Infinity' or 'NaN' with 0
    samples = samples.cpu().nan_to_num(0.0)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), facecolor='black')

    for i in range(16):
        ax = axes[i // 4, i % 4]
        img = samples[i]
        
        # Stability Clamp: Prevents extreme values from ruining the normalization
        img = torch.clamp(img, -2.0, 2.0)
        
        # AGGRESSIVE CONTRAST: Reveal faces hidden in the 0.06 loss 'fog'
        # We look at the 5th and 95th percentile to 'stretch' the features out
        low = torch.quantile(img, 0.05)
        high = torch.quantile(img, 0.95)
        
        if high > low:
            img = torch.clamp((img - low) / (high - low + 1e-5), 0, 1)
        else:
            # If the model failed this image, show a gray box instead of black
            img = torch.ones_like(img) * 0.5 
        
        # Convert (C, H, W) to (H, W, C) for plotting
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")

    plt.tight_layout()
    output_name = "stable_generation_epoch174.png"
    plt.savefig(output_name)
    print(f"🚀 Success! Image saved as {output_name}")
    plt.show()

if __name__ == "__main__":
    generate()