import torch
import torch.nn.functional as F
from math import exp


# Calculate 1-dimensional Gaussian distribution vector (unchanged)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# Create Gaussian kernel (Added device and dtype adaptation)
def create_window(window_size, channel=1, device=None, dtype=torch.float32):
    """Create Gaussian kernel, support specified device and data type (Mixed Precision compatible)"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device=device, dtype=dtype)


# SSIM calculation function (Enhanced numerical stability)
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, eps=1e-8):
    """
    Calculate SSIM index (Structural Similarity Index)
    Optimizations:
    1. Add eps to prevent division by zero
    2. Constrain sigma to non-negative values
    3. Enhanced compatibility with mixed precision training
    """
    # Determine value range (Adapt to different input ranges such as [-1,1] and [0,1])
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255.0
        else:
            max_val = 1.0 if torch.min(img1) >= 0 else 2.0  # Distinguish [0,1] and [-1,1]
        if torch.min(img1) < -0.5:
            min_val = -1.0
        else:
            min_val = 0.0
        L = max_val - min_val
    else:
        L = val_range

    # Calculate padding to ensure output size matches input size
    padd = window_size // 2
    (_, channel, height, width) = img1.size()

    # Automatically create or adapt window (Ensure device and dtype match the input)
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel, device=img1.device, dtype=img1.dtype)
    else:
        # Ensure window is on the same device and dtype with input
        window = window.to(device=img1.device, dtype=img1.dtype)

    # Calculate mean value (Using Gaussian filtering)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate variance and covariance (Add eps to prevent negative values)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # Key optimization: Constrain variance to non-negative values (Avoid negative variance caused by numerical fluctuation)
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    # Calculate SSIM constants
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Calculate SSIM numerator and denominator (Add eps to prevent division by zero)
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2 + eps)  # Core optimization: add eps

    # Prevent numerical overflow (Especially important in mixed precision mode)
    ssim_map = torch.clamp(numerator / denominator, min=0.0, max=1.0)

    # Calculate average value or return complete SSIM map
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, sigma12.mean()  # Return mean of cs instead of original cs map
    return ret


# SSIM Class (Adapted for mixed precision training)
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None, eps=1e-8):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.eps = eps  # Parameter for division-by-zero protection
        self.channel = 1
        self.window = None  # Lazy initialization of window to avoid device mismatch

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # Window adaptation: Recreate window when channel number changes or device/dtype mismatch
        if (self.window is None) or (channel != self.channel) or \
                (self.window.device != img1.device) or (self.window.dtype != img1.dtype):
            self.window = create_window(
                self.window_size,
                channel=channel,
                device=img1.device,
                dtype=img1.dtype
            )
            self.channel = channel

        return ssim(
            img1, img2,
            window=self.window,
            window_size=self.window_size,
            size_average=self.size_average,
            val_range=self.val_range,
            eps=self.eps
        )