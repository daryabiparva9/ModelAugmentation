"""
DermaCNN - Skin Lesion Classifier
Convolutional Neural Network for 64x64x3 dermatology images.

Usage:
    python derma_cnn.py

Requirements:
    pip install torch torchvision numpy tqdm
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



class Derma64Dataset(Dataset):
    def __init__(self, images_path, labels_path=None, augment=False):
        self.images  = np.load(images_path)          # [N, 64, 64, 3]
        self.labels  = np.load(labels_path) if labels_path else None
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # HWC → CHW

        if self.augment:
            # Simple augmentations without torchvision dependency
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])          # horizontal flip
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[1])          # vertical flip

        if self.labels is not None:
            return img, int(self.labels[idx])
        return img


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class DermaCNN(nn.Module):
    """
    Simple but effective CNN for 64×64 skin lesion images.

    Architecture:
        3× Conv blocks (Conv → BN → ReLU → MaxPool)
        Global Average Pooling
        2× FC layers with dropout
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1  →  32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 64 → 32

            # Block 2  →  16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 32 → 16

            # Block 3  →  8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 16 → 8
        )

        # Global Average Pooling → removes spatial dims entirely
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)







def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


