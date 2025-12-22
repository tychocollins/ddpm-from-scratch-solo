# train_cifar.py — Full Fixed Version for Day 10
import torch
import torch.optim as optim
import copy
from diffusion import GaussianDiffusion
from unet import UNet
from datasets import get_cifar10_loader  # Ensure datasets.py has this function!

# --- EMA Utility Class ---
class EMA:
    def __init__(self, model, decay):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        self.ema_state_dict = self.ema_model.state_dict()
        
    def update(self, model):
        with torch.no_grad():
            new_state_dict = model.state_dict()
            for name, param in new_state_dict.items():
                if param.dtype.is_floating_point:
                    self.ema_state_dict[name].copy_(
                        self.ema_state_dict[name] * self.decay + param.data * (1 - self.decay)
                    )
                else:
                    self.ema_state_dict[name].copy_(param.data)

    def get_ema_model(self):
        return self.ema_model

# --- MAIN TRAINING FUNCTION ---
def train():
    # 1. Configuration
    NUM_EPOCHS = 50           # 50 epochs is a good "proof of concept"
    SAVE_PATH = "trained_cifar_weights_ema.pt"
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128          
    EMA_DECAY = 0.9999 

    # 2. Model Initialization (Matching Day 10 Architecture)
    # We use 256 for time_embed_dim to match the new UNet and generate scripts
    model = UNet(in_channels=3, out_channels=3, time_embed_dim=256).to(DEVICE)
    
    # 3. Diffusion Setup
    diffusion = GaussianDiffusion(model, timesteps=1000, time_emb_dim=256).to(DEVICE)

    # 4. Data Loader
    # Note: If this still crashes, go to datasets.py and set num_workers=0
    loader = get_cifar10_loader(batch_size=BATCH_SIZE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    ema_handler = EMA(model, decay=EMA_DECAY)

    print(f"🚀 Starting CIFAR-10 Training on {DEVICE}...")
    print(f"Parameters: Epochs={NUM_EPOCHS}, Batch Size={BATCH_SIZE}, Time Dim=256")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x, _) in enumerate(loader):
            optimizer.zero_grad()
            x = x.to(DEVICE)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=DEVICE).long()

            # Calculate loss
            loss = diffusion.p_losses(x, t)
            loss.backward()
            
            # Gradient clipping to prevent the "Blue Square" saturation issues
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            ema_handler.update(model)
            
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch:02d} | Batch {batch_idx:3d} | Loss: {loss.item():.6f}")

        # Save EMA weights after every epoch
        torch.save(ema_handler.get_ema_model().state_dict(), SAVE_PATH)
        print(f"✅ Epoch {epoch} complete. Avg Loss: {epoch_loss / len(loader):.6f}")

    print("-" * 30)
    print("TRAINING FINISHED. Weights saved to:", SAVE_PATH)

# --- THE CRITICAL GUARD ---
# This prevents the 'bootstrapping' error on macOS
if __name__ == '__main__':
    train()