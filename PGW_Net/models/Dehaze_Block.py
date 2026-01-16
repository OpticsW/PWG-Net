import torch
import torch.nn as nn
import torch.nn.functional as F


class Dehaze(nn.Module):
    """Optimized Dehaze Module: Use reflection padding to avoid edge darkening, seamless match with LFP Module"""
    def __init__(self, in_channels, out_channels):
        super(Dehaze, self).__init__()
        self.in_channels = in_channels  # Single channel number of input features (actual input is 2*in_channels)
        self.out_channels = out_channels

        # Atmospheric light A estimation branch (Reflection padding instead of zero padding)
        self.A_branch = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels // 8, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        # Transmission T estimation branch (Reflection padding applied)
        self.T_branch = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels // 8, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels, eps=1e-6, affine=True),
            nn.SiLU(inplace=True)
        )

    def forward(self, x, haze):
        """
        Input:
            x: Feature map with 2*in_channels (conv output of LFP Module), shape (B, 2*C, H, W)
            haze: Original haze input image, shape (B, C, H, W)
        Output:
            recover: Dehazed feature map, shape (B, C, H, W)
        """
        a, t = torch.split(x, split_size_or_sections=self.in_channels, dim=1)
        assert a.shape[1] == self.in_channels and t.shape[1] == self.in_channels, \
            f"Dehaze input split error: a={a.shape[1]}ch, t={t.shape[1]}ch, required {self.in_channels}ch for both"

        B, C, H, W = haze.shape
        A = self.A_branch(a)
        A = A.repeat(1, 1, H, W)

        T = self.T_branch(t)

        recover = (haze - A) * T + A
        recover = torch.clamp(recover, min=-1.0 if (recover.min() < 0) else 0.0, max=1.0)

        return recover