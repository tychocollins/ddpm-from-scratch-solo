import torch
import matplotlib.pyplot as plt
from diffusion import GaussianDiffusion
from unet import UNet

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHTS_PATH = "trained_celeba_weights_ema.pt"

def generate():
    # 1. Initialize Model (Must match training architecture)
    model = UNet(in_channels=3, out_channels=3, time_embed_dim=256).to(DEVICE)
    
    # 2. Load Weights
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"✅ Loaded weights from: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: {WEIGHTS_PATH} not found in this folder.")
        return

    model.eval()

    # 3. Setup Diffusion (1000 steps to match your training)
    diffusion = GaussianDiffusion(model, timesteps=1000, time_emb_dim=256).to(DEVICE)

    print("🎨 Sampling 16 faces... (Approx. 45-60 seconds on M4)")
    with torch.no_grad():
        samples = diffusion.sample(batch_size=16, img_size=64)

    # 4. ROBUST POST-PROCESSING FIX
    # This ignores the top and bottom 2% of pixels to prevent solid color blocks.
    samples = samples.cpu()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), facecolor='black')

    for i in range(16):
        ax = axes[i // 4, i % 4]
        img = samples[i]
        
        # Calculate 2nd and 98th percentile for this specific image
        low = torch.quantile(img, 0.02)
        high = torch.quantile(img, 0.98)
        
        # Scale the image between these two points and clamp to [0, 1]
        img = torch.clamp((img - low) / (high - low + 1e-5), 0, 1)
        
        # Convert Torch CHW to Matplotlib HWC
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")

    plt.tight_layout()
    output_name = "latest_robust_generation.png"
    plt.savefig(output_name)
    print(f"🚀 Success! Image saved as {output_name}")
    plt.show()

if __name__ == "__main__":
    generate()