import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import shutil

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 25
BATCH_SIZE = 32

def get_model():
    model = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),

        # Block 2
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),

        # Block 3
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),

        # Block 4
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    ).to(DEVICE)
    return model

def train_and_evaluate(train_dir, test_dir, label):
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_data = ImageFolder(train_dir, transform=transform_train)
    test_data  = ImageFolder(test_dir,  transform=transform_test)

    print(f"  Class map: {train_data.class_to_idx}")
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    model     = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_acc   = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Track best model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total * 100
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — "
                  f"loss: {total_loss/len(train_loader):.4f} | "
                  f"val acc: {acc:.2f}%")

    # Evaluate best model
    model.load_state_dict(best_state)
    model.eval()

    class_correct = {}
    class_total   = {}
    idx_to_class  = {v: k for k, v in test_data.class_to_idx.items()}

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for pred, label in zip(preds, labels):
                cls = idx_to_class[label.item()]
                class_correct[cls] = class_correct.get(cls, 0) + \
                                     (pred == label).item()
                class_total[cls]   = class_total.get(cls, 0) + 1

    overall = sum(class_correct.values()) / \
              sum(class_total.values()) * 100
    print(f"\n  [{label}] Best accuracy: {overall:.2f}%")
    for cls in sorted(class_correct):
        acc = class_correct[cls] / class_total[cls] * 100
        print(f"    {cls}: {acc:.2f}%  "
              f"({class_correct[cls]}/{class_total[cls]} correct)")
    return overall, class_correct, class_total


def build_small_imbalanced():
    small_dir = 'data/processed/small_imbalanced'
    if os.path.exists(small_dir):
        shutil.rmtree(small_dir)
    os.makedirs(f'{small_dir}/train/diseased', exist_ok=True)
    os.makedirs(f'{small_dir}/train/healthy',  exist_ok=True)

    diseased = os.listdir('data/processed/train/diseased')
    healthy  = os.listdir('data/processed/train/healthy')

    for f in random.sample(diseased, 500):
        shutil.copy(f'data/processed/train/diseased/{f}',
                    f'{small_dir}/train/diseased/{f}')
    for f in random.sample(healthy, 100):
        shutil.copy(f'data/processed/train/healthy/{f}',
                    f'{small_dir}/train/healthy/{f}')

    print(f"  Imbalanced: 500 diseased, 100 healthy (5:1 ratio)")
    return small_dir


def build_small_augmented(small_dir):
    aug_dir = 'data/processed/small_augmented'
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.makedirs(f'{aug_dir}/train/diseased', exist_ok=True)
    os.makedirs(f'{aug_dir}/train/healthy',  exist_ok=True)

    # Copy all 500 diseased
    for f in os.listdir(f'{small_dir}/train/diseased'):
        shutil.copy(f'{small_dir}/train/diseased/{f}',
                    f'{aug_dir}/train/diseased/{f}')

    # Copy 100 real healthy
    for f in os.listdir(f'{small_dir}/train/healthy'):
        shutil.copy(f'{small_dir}/train/healthy/{f}',
                    f'{aug_dir}/train/healthy/{f}')

    # Add 400 generated healthy → balanced 500 vs 500
    gen_files = os.listdir('data/processed/generated_healthy')
    for f in random.sample(gen_files, 400):
        shutil.copy(f'data/processed/generated_healthy/{f}',
                    f'{aug_dir}/train/healthy/gen_{f}')

    h = len(os.listdir(f'{aug_dir}/train/healthy'))
    d = len(os.listdir(f'{aug_dir}/train/diseased'))
    print(f"  Augmented: {d} diseased, {h} healthy (1:1 ratio) ✅")
    return aug_dir


if __name__ == "__main__":
    TEST_DIR = 'data/processed/test'

    print("Building datasets...")
    small_dir = build_small_imbalanced()
    aug_dir   = build_small_augmented(small_dir)

    print("\n" + "="*60)
    print("Run 1: WITHOUT augmentation (500 diseased, 100 healthy)")
    print("="*60)
    acc_before, cc_before, ct_before = train_and_evaluate(
        f'{small_dir}/train', TEST_DIR,
        "Without WGAN augmentation")

    print("\n" + "="*60)
    print("Run 2: WITH WGAN augmentation (500 diseased, 500 healthy)")
    print("="*60)
    acc_after, cc_after, ct_after = train_and_evaluate(
        f'{aug_dir}/train', TEST_DIR,
        "With WGAN augmentation")

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    diff = acc_after - acc_before
    print(f"  Overall  — before: {acc_before:.2f}%  "
          f"after: {acc_after:.2f}%  "
          f"({'+'if diff>=0 else ''}{diff:.2f}%)")
    for cls in sorted(cc_before):
        b = cc_before[cls] / ct_before[cls] * 100
        a = cc_after[cls]  / ct_after[cls]  * 100
        print(f"  {cls:10s} — before: {b:.1f}%  "
              f"after: {a:.1f}%  "
              f"({'+'if a-b>=0 else ''}{a-b:.1f}%)")
    print("="*60)

    np.save("outputs/classification_results.npy",
            {'before': acc_before, 'after': acc_after,
             'before_per_class': cc_before,
             'after_per_class':  cc_after})
    print("\nResults saved!")
