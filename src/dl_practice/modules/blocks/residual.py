import torch.nn as nn
from typing import Literal


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample: Literal["none", "stride_conv", "pool_proj"] | None = None,
        act_layer=nn.ReLU,
    ):
        super(ResidualBlock, self).__init__()

        allowed_downsample = ("none", "stride_conv", "pool_proj", None)
        if downsample not in allowed_downsample:
            raise ValueError(
                f"Invalid downsample option: {downsample}. "
                "Must be one of 'none', 'stride_conv', 'pool_proj', or None."
            )

        # Keep backward compatibility for string-based "none".
        if downsample == "none":
            downsample = None

        if downsample is None:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = act_layer()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample == "stride_conv":
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        elif downsample == "pool_proj":
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.act(out)

        return out
