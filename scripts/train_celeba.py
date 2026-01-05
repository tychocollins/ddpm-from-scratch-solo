import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import sys

# --- FOLDER STRUCTURE FIX ---
# This tells Python to look one directory up to find the 'core' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.diffusion import Diffusion
from core.unet import UNet

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
LR = 2e-4
START_EPOCH = 1      
MAX_EPOCHS = 200     
SAVE_PATH = "../high_cap_celeba.pt" # Saves in root folder

# --- DATA SETUP ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Ensure root points to where your CelebA data actually is
dataset = datasets.ImageFolder(root="../data", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- MODEL & OPTIMIZER ---
model = UNet().to(DEVICE)
diffusion = Diffusion(model).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# --- RESUME LOGIC ---
if os.path.exists(SAVE_PATH):
    print(f"📦 Found existing checkpoint: {SAVE_PATH}")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
        print(f"✅ Full state loaded. Resuming from Epoch {START_EPOCH}")
    else:
        model.load_state_dict(checkpoint)
        START_EPOCH = 101 
        print(f"⚠️ Only weights found. Resuming from Epoch {START_EPOCH}")

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
        
        # Critical for CelebA stability
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