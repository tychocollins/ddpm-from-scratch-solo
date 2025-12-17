import torch
import torch.optim as optim
from diffusion import GaussianDiffusion
from datasets import get_celeba64_loader
from unet import UNet
from ema import EMA
import math

def get_time_embedding(timesteps, embed_dim=256):
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb

def main():
    DEVICE = "mps"
    # 1. Setup Data
    loader = get_celeba64_loader(batch_size=16)
    
    # 2. Setup Model
    model = UNet().to(DEVICE)
    diffusion = GaussianDiffusion(model, timesteps=1000).to(DEVICE)
    ema = EMA(model)
    optimizer = optim.AdamW(diffusion.parameters(), lr=1e-4)

    print("--- Day 9 Training Started ---")
    for epoch in range(1, 51):
        for i, (x, _) in enumerate(loader):
            x = x.to(DEVICE)
            t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
            
            # Project time to 256-dim
            t_emb = get_time_embedding(t, 256)
            
            loss = diffusion.p_losses(x, t, t_emb) # Ensure diffusion.py accepts t_emb
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")

        torch.save(ema.get_ema_model().state_dict(), "trained_celeba_weights_ema.pt")

if __name__ == "__main__":
    main()