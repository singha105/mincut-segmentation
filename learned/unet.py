"""A small U-Net that predicts how likely each pixel is to be foreground."""

import torch
from torch import nn


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Four downsampling steps, `base` channels at the top level.

    Takes ``(B, 3, H, W)`` with H and W divisible by 16, and returns one logit
    per pixel as ``(B, 1, H, W)``.
    """

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        channels = [base * 2**level for level in range(5)]
        self.encoders = nn.ModuleList(
            [_double_conv(3, channels[0])]
            + [_double_conv(a, b) for a, b in zip(channels, channels[1:])]
        )
        self.pool = nn.MaxPool2d(2)
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in zip(channels, channels[1:])
        )
        self.decoders = nn.ModuleList(_double_conv(a * 2, a) for a in channels[:-1])
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x if level == 0 else self.pool(x))
            skips.append(x)

        x = skips.pop()
        for level in reversed(range(len(self.decoders))):
            x = self.decoders[level](torch.cat([self.upsamples[level](x), skips[level]], dim=1))
        return self.head(x)
