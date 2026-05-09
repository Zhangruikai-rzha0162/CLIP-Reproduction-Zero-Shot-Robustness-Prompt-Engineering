"""
Standalone script to train ResNet-18 baselines on CIFAR-10/100 or STL10.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from utils import set_seed
from datasets import load_dataset
from models import ResNetClassifier


def train_resnet(
    dataset_name: str = "CIFAR10",
    epochs: int = 30,
    lr: float = 0.001,
) -> Tuple[ResNetClassifier, float]:
    """
    Train a ResNet-18 baseline from scratch.

    Returns:
        model: Trained model instance.
        best_acc: Best test accuracy achieved (percentage).
    """
    set_seed(config.SEED)

    # Data loading
    train_dataset = load_dataset(dataset_name, train=True, model_type="resnet")
    test_dataset = load_dataset(dataset_name, train=False, model_type="resnet")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Model setup
    num_classes_map = {"CIFAR10": 10, "CIFAR100": 100, "STL10": 10}
    num_classes = num_classes_map[dataset_name]

    model = ResNetClassifier(num_classes=num_classes).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = f"{config.SAVE_DIR}/resnet_{dataset_name.lower()}.pth"

    print(f"Training ResNet on {dataset_name} ...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Validation every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = 100.0 * correct / total
            print(f"Epoch {epoch + 1}: Test Acc = {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved (Acc: {best_acc:.2f}%)")

    print(f"Training complete! Best accuracy: {best_acc:.2f}%")
    return model, best_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ResNet-18 baseline")
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "STL10"]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_resnet(args.dataset, epochs=args.epochs, lr=args.lr)