"""U-Net и вспомогательные блоки, общие для прямой сети и начальной фазы обучаемого оптимизатора."""

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cin: int, cout: int, width: int = 32, depth: int = 4):
        super().__init__()
        widths = [width * 2 ** min(i, 3) for i in range(depth + 1)]
        self.inp = Block(cin, widths[0])
        self.down = nn.ModuleList(Block(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.up = nn.ModuleList(Block(b + a, a) for a, b in zip(widths[:-1], widths[1:]))
        self.out = nn.Conv2d(widths[0], cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = [self.inp(x)]
        for block in self.down:
            skips.append(block(nn.functional.avg_pool2d(skips[-1], 2)))
        x = skips.pop()
        for block in reversed(self.up):
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skips.pop()], 1))
        return self.out(x)


def coords(n: int, device) -> torch.Tensor:
    axis = torch.linspace(-1, 1, n, device=device)
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"))
