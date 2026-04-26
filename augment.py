"""
augment.py
----------
Uses the trained generator to produce synthetic diseased leaf images,
then merges them with real images to create a balanced dataset.

Because the WGAN trained on diseased images from ALL plant types,
the generated images capture general disease patterns across the
whole PlantVillage dataset.

How many to generate:
  We look at how many healthy vs diseased images exist in train/
  and generate enough synthetic diseased images to match the healthy count.
  This ensures a perfectly balanced 50:50 dataset.

Run:
  python augment.py   (after train_wgan.py completes)
"""

import torch
import os
import shutil
import numpy as np
from PIL import Image

from models.generator import Generator

LATENT_DIM = 100
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_images(n_images, output_dir="data/processed/generated_diseased"):
    os.makedirs(output_dir, exist_ok=True)

    G = Generator(LATENT_DIM, 3).to(DEVICE)
    G.load_state_dict(torch.load("outputs/generator_final.pt", map_location=DEVICE))
    G.eval()

    print(f"Generating {n_images} synthetic diseased leaf images...")
    generated  = 0
    batch_size = 32

    with torch.no_grad():
        while generated < n_images:
            batch = min(batch_size, n_images - generated)
            noise     = torch.randn(batch, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = G(noise)

            for j, img_tensor in enumerate(fake_imgs):
                img_tensor = (img_tensor + 1) / 2.0
                img_np     = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np     = (img_np * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_np).save(f"{output_dir}/gen_{generated+j:05d}.png")

            generated += batch

    print(f"  Saved {n_images} generated images to {output_dir}/")


def build_augmented_dataset():
    aug_dir = "data/processed/augmented_train"
    os.makedirs(f"{aug_dir}/healthy",  exist_ok=True)
    os.makedirs(f"{aug_dir}/diseased", exist_ok=True)

    # Count existing real images
    real_healthy  = os.listdir("data/processed/train/healthy")
    real_diseased = os.listdir("data/processed/train/diseased")
    n_healthy     = len(real_healthy)
    n_diseased    = len(real_diseased)

    print(f"\nReal training images:")
    print(f"  Healthy:  {n_healthy}")
    print(f"  Diseased: {n_diseased}")
    print(f"  Gap to fill: {n_healthy - n_diseased} synthetic diseased images needed")

    # Generate exactly enough to balance
    n_to_generate = max(0, n_healthy - n_diseased)
    if n_to_generate > 0:
        generate_images(n_to_generate)
    else:
        print("Dataset already balanced — no generation needed.")

    # Copy real healthy
    print("\nBuilding augmented dataset...")
    for f in real_healthy:
        shutil.copy(f"data/processed/train/healthy/{f}", f"{aug_dir}/healthy/{f}")

    # Copy real diseased
    for f in real_diseased:
        shutil.copy(f"data/processed/train/diseased/{f}", f"{aug_dir}/diseased/{f}")

    # Copy generated diseased
    gen_dir   = "data/processed/generated_diseased"
    gen_count = 0
    if os.path.exists(gen_dir):
        for f in os.listdir(gen_dir):
            shutil.copy(f"{gen_dir}/{f}", f"{aug_dir}/diseased/gen_{f}")
            gen_count += 1

    final_healthy  = len(os.listdir(f"{aug_dir}/healthy"))
    final_diseased = len(os.listdir(f"{aug_dir}/diseased"))

    print("\nAugmented dataset ready:")
    print(f"  Healthy:  {final_healthy} images (all real)")
    print(f"  Diseased: {n_diseased} real + {gen_count} generated = {final_diseased} total")
    print(f"  Balance ratio: {final_healthy}:{final_diseased}")


if __name__ == "__main__":
    build_augmented_dataset()
