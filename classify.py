"""
classify.py
-----------
Trains a CNN classifier twice and compares accuracy:
  Run 1: original imbalanced dataset (all plants, few diseased images)
  Run 2: WGAN-augmented balanced dataset (synthetic diseased images added)

This is your "proof" — the accuracy improvement from Run 1 to Run 2
directly shows that WGAN augmentation works on a real, large-scale dataset.

The classifier is binary: healthy vs diseased (regardless of plant type).
This is the correct framing — our WGAN generates general diseased leaf
images to fix the class imbalance, not to classify specific diseases.

Run:
  python classify.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 15
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 2)
    ).to(DEVICE)


def train_and_evaluate(train_dir, run_label):
    train_data = ImageFolder(train_dir, transform=transform)
    test_data  = ImageFolder("data/processed/test", transform=transform)

    # Print class mapping so we know which label is which
    print(f"  Class map: {train_data.class_to_idx}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

    model     = build_cnn()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = total = 0
    class_correct = {0: 0, 1: 0}
    class_total   = {0: 0, 1: 0}

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds        = model(imgs).argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
            for cls in [0, 1]:
                mask                = labels == cls
                class_correct[cls] += (preds[mask] == labels[mask]).sum().item()
                class_total[cls]   += mask.sum().item()

    acc         = correct / total * 100
    class_names = test_data.classes

    print(f"\n  [{run_label}] Overall accuracy: {acc:.2f}%")
    for idx, name in enumerate(class_names):
        c_acc = (class_correct[idx] / class_total[idx] * 100) if class_total[idx] > 0 else 0
        print(f"    {name}: {c_acc:.2f}%  ({class_correct[idx]}/{class_total[idx]} correct)")

    return acc, {class_names[i]: class_correct[i]/max(class_total[i],1)*100
                 for i in [0,1]}, class_names


if __name__ == "__main__":
    print("="*60)
    print("Run 1: Without WGAN augmentation (imbalanced)")
    print("="*60)
    acc_before, per_class_before, class_names = train_and_evaluate(
        "data/processed/train",
        "Without WGAN augmentation"
    )

    print("\n" + "="*60)
    print("Run 2: With WGAN augmentation (balanced)")
    print("="*60)
    acc_after, per_class_after, _ = train_and_evaluate(
        "data/processed/augmented_train",
        "With WGAN augmentation"
    )

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"  Overall — before: {acc_before:.2f}%  after: {acc_after:.2f}%  "
          f"(+{acc_after - acc_before:.2f}%)")
    for name in class_names:
        b = per_class_before.get(name, 0)
        a = per_class_after.get(name, 0)
        print(f"  {name:10s} — before: {b:.1f}%  after: {a:.1f}%  ({'+' if a>=b else ''}{a-b:.1f}%)")
    print("="*60)

    np.save("outputs/classification_results.npy", {
        'before':           acc_before,
        'after':            acc_after,
        'per_class_before': per_class_before,
        'per_class_after':  per_class_after,
        'class_names':      class_names,
    })
    print("\nSaved to outputs/classification_results.npy")
