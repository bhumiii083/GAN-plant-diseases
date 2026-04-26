"""
models/critic.py
----------------
Vanilla WGAN Critic — exactly as in the original code.

Takes any 64x64 image -> outputs a single real-valued score (NOT 0/1).
No Sigmoid at the end. Uses InstanceNorm instead of BatchNorm.

The critic trains 5x per generator update and has its weights
clipped to [-0.01, 0.01] after every update (Lipschitz constraint).
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels=3):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            # Input: channels x 64 x 64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # Output: 1 x 1 x 1 — a single score, NOT a probability, NO Sigmoid
        )

    def forward(self, img):
        return self.model(img).view(-1)
