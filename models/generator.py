"""
models/generator.py
-------------------
Vanilla WGAN Generator — exactly as in the original code.

Takes random noise (100 numbers) -> produces a 64x64 RGB image.
No class conditioning. The generator learns what a "diseased leaf"
looks like in general across ALL plant types in the dataset.

Architecture: 5 transposed conv layers, progressively upsampling
  latent_dim x 1x1 -> 512x4x4 -> 256x8x8 -> 128x16x16 -> 64x32x32 -> 3x64x64
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # channels x 64 x 64  — output range [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


def weights_init(m):
    """Standard GAN weight initialisation."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
