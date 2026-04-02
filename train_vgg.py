import os
import time
import random
from typing import List, Tuple

from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

AI_DIRS = [
    r"D:\DATASET\AI\train",
    r"D:\DATASET\AI\train_2",
]

REAL_DIRS = [
    r"D:\DATASET\real",
    r"D:\DATASET\real_2\train\real",
    r"D:\DATASET\real_2\train_2\real",
    r"D:\DATASET\real_4",
]

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
MODEL_SAVE_PATH = "trained_resnet18_binary.pth"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

AI_LABEL = 1
REAL_LABEL = 0
MAX_PIXELS = 60_000_000


def list_image_files(folder: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(VALID_EXTS):
                files.append(os.path.join(root, name))
    return files


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            w, h = img.size
            return (w * h) <= MAX_PIXELS
    except Exception:
        return False


def open_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class BinaryImageDataset(Dataset):
    def __init__(self, ai_dirs: List[str], real_dirs: List[str], transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        ai_images = []
        for folder in ai_dirs:
            ai_images.extend(list_image_files(folder))

        real_images = []
        for folder in real_dirs:
            real_images.extend(list_image_files(folder))

        raw_samples = [(path, AI_LABEL) for path in ai_images] + [
            (path, REAL_LABEL) for path in real_images
        ]

        print("\nChecking images and removing bad ones...")

        skipped_bad = 0
        for path, label in tqdm(raw_samples, desc="Filtering dataset"):
            if is_valid_image(path):
                self.samples.append((path, label))
            else:
                skipped_bad += 1

        if len(self.samples) == 0:
            raise ValueError("No valid samples found. Check your paths.")

        random.shuffle(self.samples)

        ai_count = sum(1 for _, label in self.samples if label == AI_LABEL)
        real_count = sum(1 for _, label in self.samples if label == REAL_LABEL)

        print("\n===== DATASET SUMMARY =====")
        print(f"AI samples        : {ai_count}")
        print(f"REAL samples      : {real_count}")
        print(f"TOTAL             : {len(self.samples)}")
        print(f"Skipped bad files : {skipped_bad}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = open_rgb_image(path)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor([label], dtype=torch.float32)
        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=True)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = BinaryImageDataset(
        ai_dirs=AI_DIRS,
        real_dirs=REAL_DIRS,
        transform=train_transform
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    model = model.to(device)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch [{epoch}/{EPOCHS}] =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, EPOCHS
        )

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nFinal trained model saved to {MODEL_SAVE_PATH}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()