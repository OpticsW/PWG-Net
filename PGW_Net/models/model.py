from PU.PGW_Net.models.LFP_Block import LFP
from PU.PGW_Net.models.HFP_Block import HFP
from PU.PGW_Net.models.DoubleConv_Block import *
from PU.PGW_Net.models.DWT_Block import *
from PU.PGW_Net.models import attention_channel, attention_pixel, Fusion_Up_Block


class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bilinear=False, base_channel=32, dropout_rate=0.1, sampling=2):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.incov = doubleConv(in_channels=in_channels, out_channels=base_channel)

        self.dwt1 = DWT_Block(in_channels=base_channel, out_channels=base_channel, sampling=sampling)
        self.dwt2 = DWT_Block(in_channels=base_channel * 2, out_channels=base_channel * 2, sampling=sampling)
        self.dwt3 = DWT_Block(in_channels=base_channel * 4, out_channels=base_channel * 4, sampling=sampling)
        self.dwt4 = DWT_Block(in_channels=base_channel * 8, out_channels=base_channel * 8, sampling=sampling)

        self.lfp1 = LFP(in_channels=base_channel, out_channels=base_channel * 2, )
        self.lfp2 = LFP(in_channels=base_channel * 2, out_channels=base_channel * 4)
        self.lfp3 = LFP(in_channels=base_channel * 4, out_channels=base_channel * 8)
        self.lfp4 = LFP(in_channels=base_channel * 8, out_channels=base_channel * 16 // factor)

        self.hfp1 = HFP(in_channels=base_channel, out_channels=base_channel * 2)
        self.hfp2 = HFP(in_channels=base_channel * 2, out_channels=base_channel * 4)
        self.hfp3 = HFP(in_channels=base_channel * 4, out_channels=base_channel * 8)
        self.hfp4 = HFP(in_channels=base_channel * 8, out_channels=base_channel * 16 // factor)

        self.hf_d1 = nn.Conv2d(in_channels=base_channel, out_channels=base_channel * 2, kernel_size=(3, 3), stride=2,
                               padding=1, bias=False)
        self.hf_d2 = nn.Conv2d(in_channels=base_channel * 2, out_channels=base_channel * 4, kernel_size=(3, 3),
                               stride=2, padding=1, bias=False)
        self.hf_d3 = nn.Conv2d(in_channels=base_channel * 4, out_channels=base_channel * 8, kernel_size=(3, 3),
                               stride=2, padding=1, bias=False)
        self.hf_d4 = nn.Conv2d(in_channels=base_channel * 8, out_channels=base_channel * 16 // factor,
                               kernel_size=(3, 3), stride=2, padding=1, bias=False)

        self.up1 = Fusion_Up_Block.Fusion_Up_L(base_channel * 16 // factor, base_channel * 8)
        self.up2 = Fusion_Up_Block.Fusion_Up(base_channel * 8, base_channel * 4)
        self.up3 = Fusion_Up_Block.Fusion_Up(base_channel * 4, base_channel * 2)
        self.up4 = Fusion_Up_Block.Fusion_Up(base_channel * 2, base_channel)

        self.out = nn.Sequential(
            attention_channel.Efficient_Channel_Attention(base_channel),
            attention_pixel.Efficient_Pixel_Attention(base_channel),
            nn.Conv2d(base_channel, base_channel // 2, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv2d(base_channel // 2, out_channels, kernel_size=3, padding=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.InstanceNorm2d(out_channels, affine=True, eps=1e-6)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = x
        x1 = self.incov(x)

        x1_l, x1_h = self.dwt1(x1)
        x1_l = self.lfp1(x1_l)
        x1_h = self.hfp1(x1_h)
        x1_h_down = self.hf_d1(x1_h)

        x2_l, x2_h = self.dwt2(x1_l)
        x2_l = self.lfp2(x2_l)
        x2_h = self.hfp2(x2_h + x1_h_down)
        x2_h_down = self.hf_d2(x2_h)

        x3_l, x3_h = self.dwt3(x2_l)
        x3_l = self.lfp3(x3_l)
        x3_h = self.hfp3(x3_h + x2_h_down)
        x3_h_down = self.hf_d3(x3_h)

        x4_l, x4_h = self.dwt4(x3_l)
        x4_l = self.lfp4(x4_l)
        x4_h = self.hfp4(x4_h + x3_h_down)
        x4_h_down = self.hf_d4(x4_h)

        x = self.up1(x4_l, x4_h)
        x = self.up2(x, x3_l, x3_h)
        x = self.up3(x, x2_l, x2_h)
        x = self.up4(x, x1_l, x1_h)

        out = self.out(x)
        out = out + y
        out = self.tanh(out)

        return out


if __name__ == "__main__":
    model = Model(in_channels=3, out_channels=3, bilinear=True, base_channel=32)
    x = torch.randn(8, 3, 512, 512)

    try:
        output = model(x)
        print("\nModel forward propagation succeeded!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"\nForward propagation failed: {e}")