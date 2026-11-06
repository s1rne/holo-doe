"""Неявные представления фазы: сеть от координат пикселя (SIREN и MLP с Fourier features)."""

import math

import torch
from torch import nn

from .phase import to_phase


def grid(n: int) -> torch.Tensor:
    coord = torch.linspace(-1, 1, n)
    return torch.stack(torch.meshgrid(coord, coord, indexing="ij"), -1).reshape(-1, 2)


class Sine(nn.Module):
    def __init__(self, cin: int, cout: int, w0: float, first: bool):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(cin, cout)
        bound = 1 / cin if first else math.sqrt(6 / cin) / w0
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class Siren(nn.Module):
    def __init__(self, n: int = 128, width: int = 256, depth: int = 4, w0: float = 30.0):
        super().__init__()
        self.n = n
        self.register_buffer("coords", grid(n))
        layers = [Sine(2 if i == 0 else width, width, w0, i == 0) for i in range(depth)]
        self.net = nn.Sequential(*layers, nn.Linear(width, 2))

    def forward(self) -> torch.Tensor:
        return to_phase(self.net(self.coords).T.reshape(1, 2, self.n, self.n))


class FourierMLP(nn.Module):
    """Случайные гармоники с масштабом sigma (в циклах на апертуру) и ReLU-MLP."""

    def __init__(self, n: int = 128, features: int = 512, sigma: float = 32.0, width: int = 256, depth: int = 3, seed: int = 0):
        super().__init__()
        self.n = n
        freqs = torch.randn(2, features, generator=torch.Generator().manual_seed(seed)) * sigma * math.pi
        self.register_buffer("coords", grid(n))
        self.register_buffer("freqs", freqs)
        layers, cin = [], 2 * features
        for _ in range(depth):
            layers += [nn.Linear(cin, width), nn.ReLU()]
            cin = width
        self.net = nn.Sequential(*layers, nn.Linear(width, 2))

    def forward(self) -> torch.Tensor:
        proj = self.coords @ self.freqs
        out = self.net(torch.cat([proj.sin(), proj.cos()], -1))
        return to_phase(out.T.reshape(1, 2, self.n, self.n))
