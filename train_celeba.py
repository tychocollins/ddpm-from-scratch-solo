import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusion import GaussianDiffusion
from unet import UNet
import os

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
LR = 2e-4
START_EPOCH = 1      # Default start
MAX_EPOCHS = 200     # New target
SAVE_PATH = "high_cap_celeba.pt"

# --- DATA SETUP ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root="../data", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- MODEL & OPTIMIZER ---
model = UNet().to(DEVICE)
diffusion = GaussianDiffusion(model).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# --- RESUME LOGIC ---
if os.path.exists(SAVE_PATH):
    print(f"📦 Found existing checkpoint: {SAVE_PATH}")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    
    # Check if this is a "full" checkpoint or just weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
        print(f"✅ Full state loaded. Resuming from Epoch {START_EPOCH}")
    else:
        # Fallback for your old file which only saved weights
        model.load_state_dict(checkpoint)
        START_EPOCH = 101 # Since you know you finished 100
        print(f"⚠️ Only weights found. Resuming from Epoch {START_EPOCH} (Optimizer reset)")

# --- TRAINING LOOP ---
print(f"🚀 Training on {DEVICE}. Target: {MAX_EPOCHS} Epochs...")

for epoch in range(START_EPOCH, MAX_EPOCHS + 1):
    model.train()
    for i, (x, _) in enumerate(loader):
        x = x.to(DEVICE)
        t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
        
        loss = diffusion.p_losses(x, t)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if i % 50 == 0: 
            print(f"Epoch {epoch}/{MAX_EPOCHS} | Batch {i} | Loss: {loss.item():.4f}")
    
    # Save a 'Full' checkpoint after every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, SAVE_PATH)
    print(f"💾 Checkpoint saved at end of Epoch {epoch}")