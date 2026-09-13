"""
DCGAN  —  the image PRIOR for the GAN reconstruction verification signal.

Owner: image-workflow / Person B (signal) + Person A (image models).

This is the piece the Review-2 panel asked for. On its own a DCGAN is just a
generator of plausible images; its JOB here is to provide a learned manifold of
realistic in-distribution images, G(z), that the reconstruction signal
(core/verify/gan_signal.py) optimises over. Projecting a query image onto that
manifold strips away off-manifold noise, so the target model's confidence on the
reconstruction is a cleaner membership read than raw pixels.

Parametric to the image shape so the SAME code covers Fashion-MNIST (1x28x28)
and CIFAR-10 (3x32x32): the generator starts at (H//4, W//4) and upsamples x4.
Small enough to train on a 4 GB GPU. Output is tanh in [-1, 1] — the exact range
the dataset loaders normalise to, so generator and classifier share a convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.config import set_seed
from core.train import _device


class Generator(nn.Module):
    def __init__(self, z_dim: int, out_channels: int, height: int, width: int):
        super().__init__()
        self.z_dim = z_dim
        self.h0, self.w0 = height // 4, width // 4
        self.fc = nn.Linear(z_dim, 128 * self.h0 * self.w0)
        self.net = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # x2
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),  # x2
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, self.h0, self.w0)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),   # /2
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # /2
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(128 * (height // 4) * (width // 4), 1)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h).squeeze(1)


def train_dcgan(X: np.ndarray, *, z_dim: int = 100, epochs: int = 25,
                batch_size: int = 128, lr: float = 2e-4, seed: int = 0,
                verbose: bool = True) -> Generator:
    """Train a DCGAN on images X [N, C, H, W] in [-1, 1]; return the generator
    (eval mode). Standard DCGAN recipe: BCE-with-logits, Adam(betas=0.5,0.999)."""
    set_seed(seed)
    device = _device()
    N, C, H, W = X.shape
    G = Generator(z_dim, C, H, W).to(device)
    D = Discriminator(C, H, W).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X).float()  # kept on CPU; move per batch to save VRAM
    for ep in range(epochs):
        perm = torch.randperm(N)
        d_running = g_running = 0.0
        for i in range(0, N, batch_size):
            real = Xt[perm[i:i + batch_size]].to(device)
            b = real.size(0)
            z = torch.randn(b, z_dim, device=device)
            fake = G(z)

            # --- discriminator: real -> 1, fake -> 0
            optD.zero_grad()
            loss_d = bce(D(real), torch.ones(b, device=device)) + \
                     bce(D(fake.detach()), torch.zeros(b, device=device))
            loss_d.backward(); optD.step()

            # --- generator: fool D (fake -> 1)
            optG.zero_grad()
            loss_g = bce(D(fake), torch.ones(b, device=device))
            loss_g.backward(); optG.step()
            d_running += loss_d.item() * b
            g_running += loss_g.item() * b
        if verbose:
            print(f"  DCGAN epoch {ep + 1}/{epochs}  D={d_running / N:.3f}  G={g_running / N:.3f}")
    G.eval()
    return G


@torch.no_grad()
def sample(G: Generator, n: int = 64, seed: int = 0) -> np.ndarray:
    """Draw n samples as a numpy array [n, C, H, W] in [-1, 1] (for a samples grid)."""
    set_seed(seed)
    device = _device()
    z = torch.randn(n, G.z_dim, device=device)
    return G(z).detach().cpu().numpy()


def save_generator(G: Generator, path: str) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": G.state_dict(), "z_dim": G.z_dim,
                "h0": G.h0, "w0": G.w0}, p)
    return p
