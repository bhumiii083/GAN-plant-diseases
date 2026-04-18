# WGAN-based Synthetic Data Augmentation for Plant Disease Detection
### Full PlantVillage Dataset — All Plants, All Diseases

## Problem Statement
The PlantVillage dataset contains images from many plant types (tomato, corn, grape,
pepper, potato, etc.) across dozens of disease categories. Healthy plant images
naturally outnumber diseased ones, creating class imbalance. A classifier trained on
this imbalanced data learns to predict "healthy" most of the time — a dangerous
bias in agricultural AI where missing a disease costs crops.

This project trains a Wasserstein GAN (WGAN) on all diseased leaf images across
the entire dataset, generates synthetic diseased images to balance the data, and
proves the improvement by comparing classifier accuracy before and after augmentation.

## GAN Variant
**Wasserstein GAN (WGAN)** — chosen over vanilla GAN/DCGAN because:
- More stable training (no mode collapse even on diverse multi-plant data)
- Wasserstein distance provides meaningful gradient signal throughout training
- Weight clipping ensures Lipschitz constraint
- Better handles the visual diversity of a large multi-class dataset

## Dataset
PlantVillage (full) — all plant types, all disease categories
- Automatically detects healthy vs diseased folders
- Caps per-class to keep training feasible on CPU
- Saves dataset statistics to data/processed/dataset_info.json

## How to Run
```bash
pip install -r requirements.txt
# Data already downloaded — skip download_data.py
python preprocess.py        # ~2-5 min
python train_wgan.py        # ~2-4 hrs on CPU (let run overnight)
python augment.py           # ~5 min
python evaluate.py          # ~10 min
python classify.py          # ~20 min
jupyter notebook visualize.ipynb
```

## File Structure
```
wgan_plant_disease/
├── models/
│   ├── generator.py     # Generator: noise -> 64x64 diseased leaf image
│   └── critic.py        # Critic: image -> real-valued score (no sigmoid)
├── data/
│   └── plantvillage dataset/color/   <- your downloaded dataset goes here
├── outputs/
│   ├── generated_images/   # saved grids every 10 epochs
│   └── checkpoints/        # model checkpoints
├── preprocess.py    # scans full dataset, auto-detects healthy/diseased
├── train_wgan.py    # WGAN training loop on all diseased images
├── augment.py       # generates synthetic images to balance dataset
├── evaluate.py      # FID, IS, Wasserstein distance
├── classify.py      # CNN before vs after accuracy comparison
└── visualize.ipynb  # all plots including disease class breakdown
```

## Results
| Metric | Value |
|--------|-------|
| FID Score | ~X |
| Inception Score | ~X ± X |
| Classifier accuracy (before) | ~X% |
| Classifier accuracy (after)  | ~X% |
| Improvement | +X% |

## References
- Arjovsky et al. (2017). Wasserstein GAN. arXiv:1701.07875
- Hughes et al. PlantVillage Dataset. Penn State University
