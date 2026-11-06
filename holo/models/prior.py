import torch
from torch import nn

from .phase import to_phase


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2),
                         nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


class DeepPrior(nn.Module):
    """Deep image prior: U-Net генерирует фазу из фиксированного входа. Вход: шум или сами цели
    (4 канала) с шумом. Свёртки циклические, как и FFT-модель."""

    def __init__(self, inputs: torch.Tensor, width: int = 64, depth: int = 3, noise: float = 0.0):
        super().__init__()
        self.register_buffer("inputs", inputs.float())
        self.noise = noise
        chans = [width * 2**i for i in range(depth)]
        self.down = nn.ModuleList([block(inputs.shape[1] if i == 0 else chans[i - 1], c) for i, c in enumerate(chans)])
        self.mid = block(chans[-1], chans[-1])
        self.up = nn.ModuleList([block(chans[i] * 2, chans[i - 1] if i else width) for i in reversed(range(depth))])
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self) -> torch.Tensor:
        x = self.inputs
        if self.noise and self.training:
            x = x + self.noise * torch.randn_like(x)
        skips = []
        for i, layer in enumerate(self.down):
            x = layer(x)
            skips.append(x)
            x = nn.functional.avg_pool2d(x, 2)
        x = self.mid(x)
        for layer, skip in zip(self.up, reversed(skips)):
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = layer(torch.cat([x, skip], 1))
        return to_phase(self.head(x))
