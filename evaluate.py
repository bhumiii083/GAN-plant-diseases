import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
import os

from models.generator import Generator

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
N_EVAL     = 500


def build_inception_fid():
    m = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.to(DEVICE).eval()


def build_inception_is():
    m = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    return m.to(DEVICE).eval()


def extract_features(imgs_tensor, model):
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs_tensor), 32):
            batch = imgs_tensor[i:i+32].to(DEVICE)
            batch = torch.nn.functional.interpolate(
                batch, size=299, mode='bilinear', align_corners=False)
            out = model(batch)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(real_feats, fake_feats):
    mu_r, sig_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sig_f = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    diff       = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2 * covmean))


def calculate_inception_score(imgs_tensor, model, n_splits=5):
    softmax = torch.nn.Softmax(dim=1)
    preds   = []
    with torch.no_grad():
        for i in range(0, len(imgs_tensor), 32):
            batch = imgs_tensor[i:i+32].to(DEVICE)
            batch = torch.nn.functional.interpolate(
                batch, size=299, mode='bilinear', align_corners=False)
            out = model(batch)
            if hasattr(out, 'logits'):
                out = out.logits
            preds.append(softmax(out).cpu().numpy())
    preds  = np.concatenate(preds, axis=0)
    scores = []
    n      = len(preds)
    for k in range(n_splits):
        part = preds[k * (n // n_splits) : (k + 1) * (n // n_splits)]
        py   = np.mean(part, axis=0)
        kl   = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate():
    print("Loading InceptionV3 models...")
    inc_fid = build_inception_fid()
    inc_is  = build_inception_is()

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    print(f"\nLoading real diseased images...")
    real_ds     = ImageFolder("data/processed/wgan_train", transform=transform)
    real_loader = DataLoader(real_ds, batch_size=32, shuffle=True)
    real_imgs   = []
    for imgs, _ in real_loader:
        real_imgs.append(imgs)
        if sum(x.size(0) for x in real_imgs) >= N_EVAL:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:N_EVAL]
    print(f"  Using {len(real_imgs)} real diseased images")

    print(f"\nGenerating {N_EVAL} synthetic diseased images...")
    G = Generator(LATENT_DIM, 3).to(DEVICE)
    G.load_state_dict(torch.load(
        "outputs/generator_final.pt", map_location=DEVICE))
    G.eval()

    fake_imgs = []
    with torch.no_grad():
        while sum(x.size(0) for x in fake_imgs) < N_EVAL:
            noise = torch.randn(32, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs.append(G(noise).cpu())
    fake_imgs = torch.cat(fake_imgs, dim=0)[:N_EVAL]

    print("\nExtracting features for FID...")
    real_feats = extract_features(real_imgs, inc_fid)
    fake_feats = extract_features(fake_imgs, inc_fid)

    print("Computing FID...")
    fid = calculate_fid(real_feats, fake_feats)

    print("Computing Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_imgs, inc_is)

    w_distances  = np.load("outputs/w_distances.npy")
    final_w_dist = float(np.mean(w_distances[-10:]))

    print("\n" + "="*55)
    print("EVALUATION RESULTS")
    print("="*55)
    print(f"  FID Score:          {fid:.2f}   (lower is better, <100 acceptable)")
    print(f"  Inception Score:    {is_mean:.2f} ± {is_std:.2f}  (higher is better)")
    print(f"  Wasserstein Dist:   {final_w_dist:.4f}  (tracked during training)")
    print("="*55)

    results = {
        'FID':                  fid,
        'IS_mean':              is_mean,
        'IS_std':               is_std,
        'Wasserstein_distance': final_w_dist
    }
    np.save("outputs/evaluation_results.npy", results)
    print("\nSaved to outputs/evaluation_results.npy")
    return results


if __name__ == "__main__":
    evaluate()
