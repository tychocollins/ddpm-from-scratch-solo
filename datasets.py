# datasets.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import os

def get_mnist_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

def get_cifar10_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

class CelebA64(torch.utils.data.Dataset):
    def __init__(self, root="./data/celeba"):
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        self.paths = sorted((Path(root) / "img_align_celeba").glob("*.jpg"))[:100000]

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

def get_celeba64_loader(batch_size=64):
    dataset = CelebA64()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
