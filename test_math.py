import torch
from unet import UNet
from diffusion import GaussianDiffusion

DEVICE = "cpu"
print(f"--- 🧪 Starting Math Check on {DEVICE} ---")

# 1. Initialize
model = UNet(in_channels=3, out_channels=3, time_embed_dim=256).to(DEVICE)
diffusion = GaussianDiffusion(model, timesteps=1000).to(DEVICE)

# 2. Create a fake noise batch
noise = torch.randn((1, 3, 64, 64)).to(DEVICE)
t = torch.tensor([500]).to(DEVICE) # Check midpoint of diffusion

# 3. Test the Prediction
with torch.no_grad():
    # Sync check: Does the model accept the sinusoidal embedding?
    t_emb = model._sinusoidal_embedding(t, 256)
    prediction = model(noise, t_emb)
    
    print(f"Model Output Shape: {prediction.shape}")
    print(f"Output Min/Max: {prediction.min().item():.4f} / {prediction.max().item():.4f}")

    if torch.all(prediction == 0):
        print("❌ ERROR: Model is outputting pure zeros (Dead Weights).")
    elif prediction.abs().mean() < 1e-5:
        print("⚠️ WARNING: Output is nearly zero. Weights might not be loaded.")
    else:
        print("✅ SUCCESS: Model is actively processing noise.")

# 4. Single Step Check
sample_step = diffusion.p_sample(noise, t, 500)
diff = (sample_step - noise).abs().mean().item()
print(f"Average change per pixel: {diff:.6f}")

if diff > 0:
    print("✅ SUCCESS: The diffusion math is modifying the image.")
else:
    print("❌ ERROR: The image is not changing at all (Logic Break).")