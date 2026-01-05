import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image

# --- CIFAR-10 Loader (The missing piece!) ---
def get_cifar10_loader(batch_size=128):
    # Same normalization as CelebA to keep the UNet happy
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1] range
    ])
    
    # This will download the 160MB dataset automatically to a folder named './data'
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# --- CelebA Loader (Keep your existing code) ---
class CelebA64(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        search_path = Path("/Users/tychocollins/Desktop/DDPM Solo")
        print(f"Deep searching for images in {search_path}...")
        
        self.paths = []
        for ext in ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.JPG"]:
            self.paths.extend(list(search_path.rglob(ext)))

        if len(self.paths) == 0:
            raise RuntimeError(f"ZERO images found in {search_path}. Ensure your dataset is unzipped!")
        
        print(f"🚀 Success! Found {len(self.paths)} images.")
        self.paths = sorted(self.paths)[:100000]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0

def get_celeba64_loader(batch_size=16):
    dataset = CelebA64()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)