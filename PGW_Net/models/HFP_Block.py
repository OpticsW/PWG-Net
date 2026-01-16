import torch
import torch.nn as nn
from PU.PGW_Net.models import attention_pixel


class HFP(nn.Module):
    """High Frequency Processing Module: Extract and enhance high-frequency features of images"""

    def __init__(self, in_channels, out_channels=None, k_size=3, bias=False):
        super(HFP, self).__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels

        assert in_channels > 0, f"Input channels must be positive integer, got {in_channels}"
        assert self.out_channels > 0, f"Output channels must be positive integer, got {self.out_channels}"

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, bias=bias)

        self.pad2 = nn.ReflectionPad2d(2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=0, bias=bias)

        self.fuse_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=bias)

        self.insnorm = nn.InstanceNorm2d(in_channels, affine=True, eps=1e-6)

        self.act = nn.SiLU(inplace=True)

        self.attention = attention_pixel.Efficient_Pixel_Attention(in_channels, k_size=k_size)

        self.scale = nn.Parameter(torch.tensor(0.2), requires_grad=True)

    def forward(self, x):
        """
        Input: x (B, in_channels, H, W) - Input feature map
        Output: x_out (B, out_channels, H, W) - High-frequency enhanced feature map
        """
        residual = x

        x1 = self.conv1(self.pad1(x))
        x2 = self.conv2(self.pad2(x))

        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.fuse_conv(x_fused)

        x_norm = self.insnorm(x_fused)
        x_act = self.act(x_norm)

        x_attn = self.attention(x_act)

        x_out = x_attn * self.scale + residual

        return x_out