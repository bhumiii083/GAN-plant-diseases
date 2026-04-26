"""
train_wgan_healthy.py
...
"""
"""
train_wgan_healthy.py
---------------------
Trains a second WGAN on healthy leaf images.
Generates synthetic healthy images to balance the dataset.
Same architecture as train_wgan.py — different training data.
"""

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import os
from tqdm import tqdm

from models.generator import Generator, weights_init
from models.critic import Critic

# ── Hyperparameters ───────────────────────────────────────────────
LATENT_DIM     = 100
IMG_SIZE       = 64
CHANNELS       = 3
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 0.00005
N_CRITIC       = 5
CLIP_VALUE     = 0.01
SAVE_INTERVAL  = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DATA_DIR       = "data/processed/wgan_train_healthy"
OUTPUT_DIR     = "outputs/healthy_wgan"
CHECKPOINT_DIR = "outputs/healthy_wgan/checkpoints"
IMG_DIR        = "outputs/healthy_wgan/generated_images"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(IMG_DIR,        exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)

# ── Prepare healthy training folder ──────────────────────────────
# ImageFolder needs a subfolder — wrap healthy images
WRAPPED_DIR = "data/processed/wgan_train_healthy_wrapped/healthy"
os.makedirs(WRAPPED_DIR, exist_ok=True)

import shutil
healthy_src = "data/processed/train/healthy"
count = 0
for f in os.listdir(healthy_src):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        dst = os.path.join(WRAPPED_DIR, f)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(healthy_src, f), dst)
        count += 1

print(f"Healthy training images: {count}")

# ── Dataset ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset    = ImageFolder("data/processed/wgan_train_healthy_wrapped",
                         transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)

print(f"Training Healthy WGAN on {len(dataset)} healthy leaf images")

# ── Models ────────────────────────────────────────────────────────
G = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
C = Critic(CHANNELS).to(DEVICE)
G.apply(weights_init)
C.apply(weights_init)

optimizer_G = optim.RMSprop(G.parameters(), lr=LR)
optimizer_C = optim.RMSprop(C.parameters(), lr=LR)

fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)

# ── Training loop ─────────────────────────────────────────────────
g_losses, c_losses, w_distances = [], [], []

print(f"\nStarting Healthy WGAN training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    g_loss_epoch = 0.0
    c_loss_epoch = 0.0
    loss_C = None

    for real_imgs, _ in tqdm(dataloader,
                             desc=f"Epoch {epoch+1}/{EPOCHS}",
                             leave=False):
        real_imgs  = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Train Critic
        for _ in range(N_CRITIC):
            optimizer_C.zero_grad()
            noise     = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = G(noise).detach()
            loss_C    = (-torch.mean(C(real_imgs))
                         + torch.mean(C(fake_imgs)))
            loss_C.backward()
            optimizer_C.step()
            for p in C.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        c_loss_epoch += loss_C.item()

        # Train Generator
        optimizer_G.zero_grad()
        noise     = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_imgs = G(noise)
        loss_G    = -torch.mean(C(fake_imgs))
        loss_G.backward()
        optimizer_G.step()
        g_loss_epoch += loss_G.item()

    w_dist = -loss_C.item() if loss_C is not None else 0.0
    g_losses.append(g_loss_epoch / len(dataloader))
    c_losses.append(c_loss_epoch / len(dataloader))
    w_distances.append(w_dist)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"G: {g_losses[-1]:+.4f} | "
          f"C: {c_losses[-1]:+.4f} | "
          f"W-dist: {w_dist:.4f}")

    if (epoch + 1) % SAVE_INTERVAL == 0:
        G.eval()
        with torch.no_grad():
            fake = G(fixed_noise)
        G.train()
        grid = torchvision.utils.make_grid(fake, nrow=4, normalize=True)
        torchvision.utils.save_image(
            grid, f"{IMG_DIR}/epoch_{epoch+1:03d}.png")
        torch.save({
            'epoch':   epoch,
            'G_state': G.state_dict(),
            'C_state': C.state_dict(),
        }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pt")
        print(f"  -> Saved grid + checkpoint at epoch {epoch+1}")

# Save final model
torch.save(G.state_dict(), f"{OUTPUT_DIR}/generator_healthy_final.pt")
torch.save(C.state_dict(), f"{OUTPUT_DIR}/critic_healthy_final.pt")
np.save(f"{OUTPUT_DIR}/g_losses.npy",    np.array(g_losses))
np.save(f"{OUTPUT_DIR}/c_losses.npy",    np.array(c_losses))
np.save(f"{OUTPUT_DIR}/w_distances.npy", np.array(w_distances))

print(f"\nHealthy WGAN training complete!")
print(f"Generator saved to {OUTPUT_DIR}/generator_healthy_final.pt")
