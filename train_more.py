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

LATENT_DIM    = 100
IMG_SIZE      = 64
CHANNELS      = 3
BATCH_SIZE    = 32
EXTRA_EPOCHS  = 50        # train 50 more on top of existing 100
LR            = 0.00005
N_CRITIC      = 5
CLIP_VALUE    = 0.01
SAVE_INTERVAL = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DATA_DIR       = "data/processed/wgan_train"
OUTPUT_DIR     = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
IMG_DIR        = "outputs/generated_images"

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset    = ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)

# Load existing trained models — continue from epoch 100
G = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
C = Critic(CHANNELS).to(DEVICE)
G.load_state_dict(torch.load("outputs/generator_final.pt", map_location=DEVICE))
C.load_state_dict(torch.load("outputs/critic_final.pt",    map_location=DEVICE))
print("Loaded existing models — continuing training from epoch 100")

optimizer_G = optim.RMSprop(G.parameters(), lr=LR)
optimizer_C = optim.RMSprop(C.parameters(), lr=LR)

fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)

# Load existing loss curves to append to
g_losses    = list(np.load("outputs/g_losses.npy"))
c_losses    = list(np.load("outputs/c_losses.npy"))
w_distances = list(np.load("outputs/w_distances.npy"))

START_EPOCH = len(g_losses)  # 100
print(f"Continuing from epoch {START_EPOCH}...")

for epoch in range(EXTRA_EPOCHS):
    g_loss_epoch = 0.0
    c_loss_epoch = 0.0
    loss_C       = None

    for real_imgs, _ in tqdm(dataloader,
                             desc=f"Epoch {START_EPOCH+epoch+1}/{START_EPOCH+EXTRA_EPOCHS}",
                             leave=False):
        real_imgs  = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        for _ in range(N_CRITIC):
            optimizer_C.zero_grad()
            noise     = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = G(noise).detach()
            loss_C    = -torch.mean(C(real_imgs)) + torch.mean(C(fake_imgs))
            loss_C.backward()
            optimizer_C.step()
            for p in C.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
        c_loss_epoch += loss_C.item()

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

    print(f"Epoch {START_EPOCH+epoch+1} | "
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
            grid, f"{IMG_DIR}/epoch_{START_EPOCH+epoch+1:03d}.png")
        torch.save({'epoch': START_EPOCH+epoch,
                    'G_state': G.state_dict(),
                    'C_state': C.state_dict()},
                   f"{CHECKPOINT_DIR}/checkpoint_epoch_{START_EPOCH+epoch+1}.pt")

# Overwrite final model with improved version
torch.save(G.state_dict(), f"{OUTPUT_DIR}/generator_final.pt")
torch.save(C.state_dict(), f"{OUTPUT_DIR}/critic_final.pt")
np.save(f"{OUTPUT_DIR}/g_losses.npy",    np.array(g_losses))
np.save(f"{OUTPUT_DIR}/c_losses.npy",    np.array(c_losses))
np.save(f"{OUTPUT_DIR}/w_distances.npy", np.array(w_distances))
print(f"\nDone! Model improved from epoch {START_EPOCH} to {START_EPOCH+EXTRA_EPOCHS}")
print("Now re-run evaluate.py to see improved FID")
