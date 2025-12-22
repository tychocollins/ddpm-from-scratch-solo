import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusion import GaussianDiffusion
from unet import UNet

# --- CONFIGURATION (ROCK-SOLID STABILITY) ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16        # Keeping it low to manage heat on M4 Air
LR = 1e-4              # SGD requires a slightly higher LR than AdamW
MOMENTUM = 0.9         # Standard momentum for SGD stability
WEIGHT_DECAY = 1e-2    
MAX_NORM = 0.1         # Strict safety fuse
EPOCHS = 200
SAVE_PATH = "trained_celeba_weights_ema.pt"
DATA_ROOT = "../data"

# 1. DATASET
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

try:
    dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Dataset loaded. Found {len(dataset)} images.")
except Exception as e:
    print(f"❌ Dataset Error: {e}")
    exit()

# 2. MODEL & DIFFUSION
model = UNet(in_channels=3, out_channels=3, time_embed_dim=256).to(DEVICE)
diffusion = GaussianDiffusion(model, timesteps=1000, time_emb_dim=256).to(DEVICE)

# --- THE BIG CHANGE: SWITCHING TO SGD ---
# SGD is more mathematically "stable" for hot hardware
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# EMA Model
ema_model = UNet(in_channels=3, out_channels=3, time_embed_dim=256).to(DEVICE)
ema_model.load_state_dict(model.state_dict())

# 3. START LOGIC
print("🆕 Starting fresh rebuild. No old weights loaded.")

# 4. TRAINING LOOP
print(f"🚀 Training started on {DEVICE} with SGD stability.")

for epoch in range(0, EPOCHS + 1):
    model.train()
    for i, (x, _) in enumerate(loader):
        x = x.to(DEVICE)
        t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
        
        loss = diffusion.p_losses(x, t)
        
        if torch.isnan(loss):
            print(f"❌ NaN detected at Epoch {epoch}. Cooling required.")
            exit()

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
        
        optimizer.step()
        
        with torch.no_grad():
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.mul_(0.999).add_(p, alpha=0.001)
        
        if i % 100 == 0:
            print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")

    # SAVE LOGIC
    torch.save(ema_model.state_dict(), SAVE_PATH)
    if epoch % 5 == 0:
        torch.save(ema_model.state_dict(), f"celeba_checkpoint_e{epoch}.pt")

print("✅ Training Complete!")