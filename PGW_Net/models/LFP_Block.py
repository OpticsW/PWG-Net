import torch
from PU.PGW_Net.models.Dehaze_Block import Dehaze
from PU.PGW_Net.models.DoubleConv_Block import *
from PU.PGW_Net.models import Attention_Block


class FEAB(nn.Module):
    """
    Feature Enhancement Attention Block
    Optimizations: 1. Use native PyTorch SiLU instead of Swish (equivalent); 2. Reflection padding to avoid edge darkening; 3. Pretrained weight compatible
    Input: (B, in_channels, H, W)
    Output: (B, out_channels, H, W)
    """
    def __init__(self, in_channels=32, out_channels=32, k_size=3, bias=False):
        super(FEAB, self).__init__()
        assert k_size % 2 == 1, f"Kernel size must be odd for symmetric padding, got {k_size}"
        assert in_channels > 0 and out_channels > 0, f"Input/Output channels must be positive integers, got in={in_channels}, out={out_channels}"

        self.pad_base = nn.ReflectionPad2d(padding=k_size // 2)
        self.conv_base = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
            stride=1,
            padding=0,
            bias=bias
        )
        self.act = nn.SiLU(inplace=True)

        self.attn = Attention_Block.Attn_Block(
            channel=out_channels,
            k_size=k_size
        )

        self.conv_fuse = nn.Conv2d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        self.norm_residual = nn.InstanceNorm2d(
            num_features=in_channels,
            eps=1e-6,
            affine=True
        )
        self.conv_residual_adjust = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.norm_residual(x)
        residual = self.conv_residual_adjust(residual)

        x_main = self.pad_base(x)
        x_main = self.conv_base(x_main)
        x_main = self.act(x_main)
        x_attn = self.attn(x_main)
        x_main = self.conv_fuse(x_attn)

        x_out = x_main + residual
        return x_out


class LFP(nn.Module):
    """
    Low Frequency Processing Block
    Optimizations: 1. Dependent FEAB blocks use SiLU activation; 2. Reflection padding to avoid edge darkening; 3. Pretrained weight compatible
    Input: (B, in_channels, H, W)
    Output: (B, out_channels, H, W)
    """
    def __init__(self, in_channels=32, out_channels=32, k_size=3, bias=False, feab_num=1, use_global_residual=False):
        super(LFP, self).__init__()
        assert feab_num >= 1, f"Number of FEAB blocks must be ≥ 1 for sufficient low-frequency enhancement, got {feab_num}"
        assert k_size % 2 == 1, f"Kernel size must be odd, got {k_size}"
        assert in_channels > 0 and out_channels > 0, f"Input/Output channels must be positive integers, got in={in_channels}, out={out_channels}"

        self.feab_blocks = nn.ModuleList([
            FEAB(
                in_channels=in_channels,
                out_channels=in_channels,
                k_size=k_size,
                bias=bias
            ) for _ in range(feab_num)
        ])

        self.pad_expand = nn.ReflectionPad2d(padding=k_size // 2)
        self.conv_expand = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * in_channels,
            kernel_size=k_size,
            stride=1,
            padding=0,
            bias=bias
        )

        self.dehaze = Dehaze(
            in_channels=in_channels,
            out_channels=in_channels
        )

        self.conv_final = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.use_global_residual = use_global_residual
        self.conv_global_residual = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        ) if (self.use_global_residual and in_channels != out_channels) else nn.Identity()

    def forward(self, x):
        global_residual = self.conv_global_residual(x) if self.use_global_residual else 0.0

        x_feab = x
        for feab in self.feab_blocks:
            x_feab = feab(x_feab)

        x_expand = self.pad_expand(x_feab)
        x_expand = self.conv_expand(x_expand)

        x_dehaze = self.dehaze(x_expand, x)

        x_out = self.conv_final(x_dehaze)
        if self.use_global_residual:
            x_out = x_out + global_residual

        return x_out


# Pretrained Weight Loading Guide
def load_pretrained_lfp(pretrained_path, device, **lfp_kwargs):
    """Load pretrained weights (compatible with SiLU replacement, no weight mismatch for activation functions without params)"""
    model = LFP(**lfp_kwargs).to(device)
    pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
    model.load_state_dict(pretrained_dict, strict=False)

    core_params = ["conv_expand.weight", "feab_blocks.0.conv_base.weight", "dehaze.A_branch.0.1.weight"]
    for param_name in core_params:
        if param_name in pretrained_dict and param_name in model.state_dict():
            assert torch.equal(pretrained_dict[param_name], model.state_dict()[param_name]), \
                f"Core parameter {param_name} loading mismatch!"

    print(f"✅ Pretrained LFP Module Loaded (SiLU Activated), Device: {device}")
    return model