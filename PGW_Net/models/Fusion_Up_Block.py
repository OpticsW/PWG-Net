import torch
import torch.nn as nn
from PU.PGW_Net.models.attention_pixel import Efficient_Pixel_Attention


class ResidualRefineBlock(nn.Module):
    """Residual Refinement Block: Enhance feature representation, reflection padding for edge optimization"""
    def __init__(self, channels, bias=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.BatchNorm2d(channels, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.BatchNorm2d(channels, affine=True)
        )
        self.mish = nn.Mish(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        return self.mish(x + residual)


class Fusion_Up(nn.Module):
    """Upsampling Module with Downsampled Feature + Low-Freq + High-Freq Fusion, optimized edge processing"""
    def __init__(self, in_channels, out_channels, scale_factor=2, bias=False):
        super(Fusion_Up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.shuffle_in_channels = out_channels * (scale_factor ** 2)

        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.InstanceNorm2d(in_channels//2, eps=1e-6, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(in_channels, eps=1e-6, affine=True),
            nn.Mish(inplace=True),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.BatchNorm2d(in_channels, eps=1e-6),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels, self.shuffle_in_channels, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.BatchNorm2d(self.shuffle_in_channels, eps=1e-6),
            nn.Mish(inplace=True),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.shuffle_in_channels, self.shuffle_in_channels//4, kernel_size=1, bias=bias),
            nn.Mish(inplace=True),
            nn.Conv2d(self.shuffle_in_channels//4, self.shuffle_in_channels, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.pixel_attn = Efficient_Pixel_Attention(self.shuffle_in_channels, k_size=3)

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.refine_block = ResidualRefineBlock(out_channels, bias=bias)

    def forward(self, x_down, x_prev_lf, x_pre_hf):
        assert x_down.ndim == 4 and x_prev_lf.ndim == 4 and x_pre_hf.ndim == 4, "Inputs must be 4D tensors (B,C,H,W)"
        assert x_down.shape[1] == self.in_channels, f"x_down channel error: expected {self.in_channels}, got {x_down.shape[1]}"
        assert x_prev_lf.shape[1] == self.in_channels, f"x_prev_lf channel error: expected {self.in_channels}, got {x_prev_lf.shape[1]}"
        assert x_pre_hf.shape[1] == self.in_channels//2, f"x_pre_hf channel error: expected {self.in_channels//2}, got {x_pre_hf.shape[1]}"

        x_pre_hf_adjusted = self.channel_adjust(x_pre_hf)
        x_fused = x_prev_lf + x_pre_hf_adjusted
        x_concat = torch.cat([x_down, x_fused], dim=1)
        x_strengthened = self.fusion_conv(x_concat)

        channel_weights = self.channel_attn(x_strengthened)
        x_attended = x_strengthened * channel_weights
        x_attended = self.pixel_attn(x_attended)

        x_up = self.pixel_shuffle(x_attended)
        x_out = self.refine_block(x_up)
        return x_out


class Fusion_Up_L(nn.Module):
    """Upsampling Module with Only Low-Freq + High-Freq Fusion (fixed channel mismatch, optimized edge processing)"""
    def __init__(self, in_channels, out_channels, scale_factor=2, bias=False):
        super(Fusion_Up_L, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.shuffle_in_channels = out_channels * (scale_factor ** 2)

        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.InstanceNorm2d(in_channels//2, eps=1e-6, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(in_channels, eps=1e-6, affine=True),
            nn.Mish(inplace=True),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.BatchNorm2d(in_channels, eps=1e-6),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels, self.shuffle_in_channels, kernel_size=3, padding=1,
                      padding_mode='reflect', bias=bias),
            nn.BatchNorm2d(self.shuffle_in_channels, eps=1e-6),
            nn.Mish(inplace=True),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.shuffle_in_channels, self.shuffle_in_channels//4, kernel_size=1, bias=bias),
            nn.Mish(inplace=True),
            nn.Conv2d(self.shuffle_in_channels//4, self.shuffle_in_channels, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.pixel_attn = Efficient_Pixel_Attention(self.shuffle_in_channels, k_size=3)

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.refine_block = ResidualRefineBlock(out_channels, bias=bias)

    def forward(self, x_prev_lf, x_pre_hf):
        assert x_prev_lf.ndim == 4 and x_pre_hf.ndim == 4, "Inputs must be 4D tensors (B,C,H,W)"

        x_pre_hf_adjusted = self.channel_adjust(x_pre_hf)
        x_fused = x_prev_lf + x_pre_hf_adjusted
        x_strengthened = self.fusion_conv(x_fused)

        channel_weights = self.channel_attn(x_strengthened)
        x_attended = x_strengthened * channel_weights
        x_attended = self.pixel_attn(x_attended)

        x_up = self.pixel_shuffle(x_attended)
        x_out = self.refine_block(x_up)
        return x_out