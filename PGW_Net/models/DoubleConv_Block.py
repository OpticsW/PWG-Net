from torch import nn


# Double Convolution Block
def doubleConv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    # First convolution layer
    layer.append(nn.ReflectionPad2d(1))
    layer.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=0, bias=False))
    layer.append(nn.BatchNorm2d(mid_channels, affine=True))
    layer.append(nn.SiLU(inplace=True))
    # Second convolution layer
    layer.append(nn.ReflectionPad2d(1))
    layer.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False))
    layer.append(nn.BatchNorm2d(out_channels, affine=True))
    layer.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layer)

# Single Convolution Block
def SimpleConv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    # Single convolution layer
    layer.append(nn.ReflectionPad2d(1))
    layer.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=0, bias=False))
    #layer.append(nn.InstanceNorm2d(mid_channels))
    #layer.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layer)