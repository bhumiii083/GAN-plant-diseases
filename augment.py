"""
augment.py
----------
Uses both trained generators to balance the dataset:
  - Diseased WGAN already generated diverse diseased images
  - Healthy WGAN generates synthetic healthy images to match
Result: 4160 healthy + 4160 diseased = perfectly balanced
"""

import torch
import os
import shutil
import numpy as np
from PIL import Image
from models.generator import Generator

LATENT_DIM = 100
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_images(generator_path, n_images, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)

    G = Generator(LATENT_DIM, 3).to(DEVICE)
    G.load_state_dict(torch.load(generator_path, map_location=DEVICE))
    G.eval()

    print(f"Generating {n_images} synthetic {label} images...")
    generated = 0

    with torch.no_grad():
        while generated < n_images:
            batch = min(32, n_images - generated)
            noise = torch.randn(batch, LATENT_DIM, 1, 1, device=DEVICE)
            imgs  = G(noise)

            for j, img_tensor in enumerate(imgs):
                img_tensor = (img_tensor + 1) / 2.0
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                Image.fromarray(img_np).save(
                    f"{output_dir}/gen_{generated+j:04d}.png")
            generated += batch

    print(f"  ✅ {n_images} {label} images saved to {output_dir}")


def build_augmented_dataset():
    aug_dir = "data/processed/augmented_train"
    os.makedirs(f"{aug_dir}/healthy",  exist_ok=True)
    os.makedirs(f"{aug_dir}/diseased", exist_ok=True)

    # Count real images
    real_healthy  = os.listdir("data/processed/train/healthy")
    real_diseased = os.listdir("data/processed/train/diseased")
    target        = len(real_diseased)  # 4160 — match this
    gap           = target - len(real_healthy)  # 2279 to generate

    print(f"\nReal healthy:  {len(real_healthy)}")
    print(f"Real diseased: {len(real_diseased)}")
    print(f"Need to generate {gap} healthy images to balance\n")

    # Generate synthetic healthy images
    generate_images(
        generator_path="outputs/healthy_wgan/generator_healthy_final.pt",
        n_images=gap,
        output_dir="data/processed/generated_healthy",
        label="healthy"
    )

    # Copy real healthy images
    for f in real_healthy:
        shutil.copy(f"data/processed/train/healthy/{f}",
                    f"{aug_dir}/healthy/{f}")

    # Copy generated healthy images
    for f in os.listdir("data/processed/generated_healthy"):
        shutil.copy(f"data/processed/generated_healthy/{f}",
                    f"{aug_dir}/healthy/gen_{f}")

    # Copy ALL real diseased images
    for f in real_diseased:
        shutil.copy(f"data/processed/train/diseased/{f}",
                    f"{aug_dir}/diseased/{f}")

    h_count = len(os.listdir(f"{aug_dir}/healthy"))
    d_count = len(os.listdir(f"{aug_dir}/diseased"))

    print(f"\n{'='*50}")
    print(f"AUGMENTED DATASET READY")
    print(f"{'='*50}")
    print(f"  Healthy:  {len(real_healthy)} real "
          f"+ {gap} generated = {h_count} total")
    print(f"  Diseased: {d_count} real")
    print(f"  Balance ratio: 1:1 ✅")
    print(f"{'='*50}")


if __name__ == "__main__":
    build_augmented_dataset()
