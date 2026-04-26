"""
preprocess.py
-------------
Reads the ENTIRE PlantVillage dataset — all plants, all diseases.
Automatically detects which folders are "healthy" vs "diseased" by checking
if the folder name contains the word "healthy".

What this file does:
  - Scans ALL subfolders inside data/plantvillage dataset/color/
  - Splits each into healthy vs diseased automatically
  - Resizes every image to 64x64
  - Creates:
      data/processed/train/healthy/   <- all healthy images from all plants
      data/processed/train/diseased/  <- all diseased images from all plants
      data/processed/test/healthy/
      data/processed/test/diseased/
      data/processed/wgan_train/diseased/  <- WGAN trains only on diseased

  The WGAN's job: learn what a "diseased leaf" looks like across ALL plant types,
  then generate new synthetic diseased images to balance the dataset.

  We cap per-class counts to avoid memory issues and keep training time
  manageable on a Mac CPU. You can raise these caps if you have more time.

Run:
  python preprocess.py
"""

import os
import shutil
import random
from PIL import Image
from collections import defaultdict

IMG_SIZE   = 64
DATA_DIR   = "data/plantvillage dataset/color"
OUTPUT_DIR = "data/processed"

# Max images to sample per class to keep things manageable on CPU
# Total dataset has 54k+ images — we cap to keep training feasible
MAX_PER_DISEASED_CLASS = 200   # per disease type
MAX_PER_HEALTHY_CLASS  = 200   # per healthy plant type
TRAIN_SPLIT = 0.8

random.seed(42)


def is_healthy(folder_name):
    return "healthy" in folder_name.lower()


def preprocess():
    # Create output directories
    for split in ["train", "test"]:
        for label in ["healthy", "diseased"]:
            os.makedirs(f"{OUTPUT_DIR}/{split}/{label}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/wgan_train/diseased", exist_ok=True)

    all_classes = sorted(os.listdir(DATA_DIR))
    all_classes = [c for c in all_classes
                   if os.path.isdir(os.path.join(DATA_DIR, c))
                   and not c.startswith('.')]

    print(f"Found {len(all_classes)} class folders in dataset")

    healthy_classes  = [c for c in all_classes if is_healthy(c)]
    diseased_classes = [c for c in all_classes if not is_healthy(c)]

    print(f"  Healthy classes:  {len(healthy_classes)}")
    print(f"  Diseased classes: {len(diseased_classes)}")
    print()

    stats = defaultdict(int)

    def process_class(folder_name, label, max_count):
        src = os.path.join(DATA_DIR, folder_name)
        all_files = [f for f in os.listdir(src)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files = random.sample(all_files, min(max_count, len(all_files)))

        split_idx   = int(TRAIN_SPLIT * len(files))
        train_files = files[:split_idx]
        test_files  = files[split_idx:]

        # Use folder name as prefix to avoid filename collisions across classes
        safe_prefix = folder_name.replace(" ", "_").replace("(", "").replace(")", "")

        for fname in train_files:
            try:
                img = Image.open(os.path.join(src, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                out_name = f"{safe_prefix}__{fname}"
                img.save(f"{OUTPUT_DIR}/train/{label}/{out_name}")
                if label == "diseased":
                    img.save(f"{OUTPUT_DIR}/wgan_train/diseased/{out_name}")
                stats[f"train_{label}"] += 1
            except Exception as e:
                print(f"  Warning: skipped {fname}: {e}")

        for fname in test_files:
            try:
                img = Image.open(os.path.join(src, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                out_name = f"{safe_prefix}__{fname}"
                img.save(f"{OUTPUT_DIR}/test/{label}/{out_name}")
                stats[f"test_{label}"] += 1
            except Exception as e:
                print(f"  Warning: skipped {fname}: {e}")

    # Process all healthy classes
    print("Processing healthy classes...")
    for folder in healthy_classes:
        print(f"  {folder}")
        process_class(folder, "healthy", MAX_PER_HEALTHY_CLASS)

    # Process all diseased classes
    print("\nProcessing diseased classes...")
    for folder in diseased_classes:
        print(f"  {folder}")
        process_class(folder, "diseased", MAX_PER_DISEASED_CLASS)

    print("\n" + "="*55)
    print("PREPROCESSING COMPLETE")
    print("="*55)
    print(f"  Train — healthy:  {stats['train_healthy']:>5} images")
    print(f"  Train — diseased: {stats['train_diseased']:>5} images")
    print(f"  Test  — healthy:  {stats['test_healthy']:>5} images")
    print(f"  Test  — diseased: {stats['test_diseased']:>5} images")
    print(f"\n  WGAN will train on {len(os.listdir(OUTPUT_DIR+'/wgan_train/diseased'))} diseased images")
    imbalance = stats['train_healthy'] / max(stats['train_diseased'], 1)
    print(f"  Class imbalance ratio: {imbalance:.1f}:1 (healthy:diseased)")
    print(f"\n  Diseased classes covered: {len(diseased_classes)}")
    print(f"  Healthy classes covered:  {len(healthy_classes)}")
    print("="*55)

    # Save class info for visualizations later
    import json
    info = {
        "healthy_classes":   healthy_classes,
        "diseased_classes":  diseased_classes,
        "train_healthy":     stats["train_healthy"],
        "train_diseased":    stats["train_diseased"],
        "test_healthy":      stats["test_healthy"],
        "test_diseased":     stats["test_diseased"],
        "max_per_diseased":  MAX_PER_DISEASED_CLASS,
        "max_per_healthy":   MAX_PER_HEALTHY_CLASS,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  Dataset info saved to {OUTPUT_DIR}/dataset_info.json")


if __name__ == "__main__":
    preprocess()
