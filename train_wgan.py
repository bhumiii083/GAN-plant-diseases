"""
train_wgan.py
-------------
Vanilla WGAN training script — trains on ALL diseased leaf images
from the entire PlantVillage dataset (all plant types, all diseases).

What changes vs the original 2-class version:
  - DATA_DIR now points to wgan_train/diseased/ which contains images
    from ALL disease classes (corn, grapes, tomatoes, peppers, etc.)
  - Everything else is identical — same WGAN loss, same weight clipping,
    same architecture

The Generator learns a general "diseased leaf" visual distribution
across all plant types. The generated images may look like any plant's
disease pattern — which is fine, because our classifier just needs
to learn "diseased vs healthy", not "which specific disease".

Run:
  python train_wgan.py
  (On Mac CPU: ~2-4 hrs for 100 epochs depending on dataset size)
  Tip: let it run overnight. Check outputs/generated_images/ in the morning.
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

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM     = 100
IMG_SIZE       = 64
CHANNELS       = 3
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 0.00005   # WGAN standard — do not increase
N_CRITIC       = 5         # critic trains 5x per generator step
CLIP_VALUE     = 0.01      # weight clipping — THE key WGAN trick
SAVE_INTERVAL  = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DATA_DIR       = "data/processed/wgan_train"
OUTPUT_DIR     = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
IMG_DIR        = "outputs/generated_images"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(IMG_DIR,        exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
# wgan_train/ has one subfolder: diseased/
# ImageFolder picks it up automatically as class 0
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
])

dataset    = ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)

print(f"Training WGAN on {len(dataset)} diseased leaf images")
print(f"  (Covers all plant types in PlantVillage: corn, grapes, tomatoes, peppers, etc.)")

# ── Models ────────────────────────────────────────────────────────────────────
G = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
C = Critic(CHANNELS).to(DEVICE)
G.apply(weights_init)
C.apply(weights_init)

optimizer_G = optim.RMSprop(G.parameters(), lr=LR)
optimizer_C = optim.RMSprop(C.parameters(), lr=LR)

# Fixed noise for consistent visualisation grid across epochs
fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)

# ── Training loop ──────────────────────────────────────────────────────────────
g_losses, c_losses, w_distances = [], [], []

print(f"\nStarting WGAN training for {EPOCHS} epochs...")
print("Generated image grids saved every 10 epochs to outputs/generated_images/\n")

for epoch in range(EPOCHS):
    g_loss_epoch = 0.0
    c_loss_epoch = 0.0
    loss_C = None

    for real_imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        real_imgs  = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # ── Train Critic N_CRITIC times ───────────────────────────────────────
        for _ in range(N_CRITIC):
            optimizer_C.zero_grad()

            noise     = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = G(noise).detach()

            # WGAN critic loss: minimise E[C(fake)] - E[C(real)]
            loss_C = -torch.mean(C(real_imgs)) + torch.mean(C(fake_imgs))
            loss_C.backward()
            optimizer_C.step()

            # Weight clipping — enforces Lipschitz constraint
            for p in C.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        c_loss_epoch += loss_C.item()

        # ── Train Generator once ──────────────────────────────────────────────
        optimizer_G.zero_grad()

        noise     = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_imgs = G(noise)

        # Generator loss: minimise -E[C(fake)]  (fool the critic)
        loss_G = -torch.mean(C(fake_imgs))
        loss_G.backward()
        optimizer_G.step()

        g_loss_epoch += loss_G.item()

    # Wasserstein distance estimate (should decrease toward 0)
    w_dist = -loss_C.item() if loss_C is not None else 0.0
    g_losses.append(g_loss_epoch / len(dataloader))
    c_losses.append(c_loss_epoch / len(dataloader))
    w_distances.append(w_dist)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"G: {g_losses[-1]:+.4f} | "
          f"C: {c_losses[-1]:+.4f} | "
          f"W-dist: {w_dist:.4f}")

    # Save generated image grid + checkpoint
    if (epoch + 1) % SAVE_INTERVAL == 0:
        G.eval()
        with torch.no_grad():
            fake = G(fixed_noise)
        G.train()
        grid = torchvision.utils.make_grid(fake, nrow=4, normalize=True)
        torchvision.utils.save_image(grid, f"{IMG_DIR}/epoch_{epoch+1:03d}.png")
        torch.save({
            'epoch':   epoch,
            'G_state': G.state_dict(),
            'C_state': C.state_dict(),
        }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pt")
        print(f"  -> Saved grid + checkpoint at epoch {epoch+1}")

# ── Save final model and curves ───────────────────────────────────────────────
torch.save(G.state_dict(), f"{OUTPUT_DIR}/generator_final.pt")
torch.save(C.state_dict(), f"{OUTPUT_DIR}/critic_final.pt")
np.save(f"{OUTPUT_DIR}/g_losses.npy",    np.array(g_losses))
np.save(f"{OUTPUT_DIR}/c_losses.npy",    np.array(c_losses))
np.save(f"{OUTPUT_DIR}/w_distances.npy", np.array(w_distances))

print(f"\nTraining complete! Generator saved to {OUTPUT_DIR}/generator_final.pt")
