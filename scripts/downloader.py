import torchvision
from pathlib import Path

# Path to your project data folder
data_path = "/Users/tychocollins/Desktop/DDPM Solo/data"

print("Starting download... This may take 10-20 minutes.")
dataset = torchvision.datasets.CelebA(
    root=data_path,
    split='all',
    download=True
)
print(f"Done! Data is located at: {data_path}")