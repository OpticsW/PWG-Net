import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from SSIM_method import SSIM


# Wavelet Transform (Adapt to arbitrary input size, solve odd dimension issue)
def dwt_init(x):
    """Wavelet Transform: Use reflection padding for odd dimensions, preserve boundary information"""
    B, C, H, W = x.shape
    pad_h = (2 - H % 2) % 2
    pad_w = (2 - W % 2) % 2

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    x01 = x[:, :, 0::2, :] / 2.0
    x02 = x[:, :, 1::2, :] / 2.0
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


def wavelet_loss(pred, target, charbonnier_eps=1e-3):
    """Adaptive Wavelet Loss: Dynamically adjust high-frequency subband weights based on image edge density"""
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch.Tensor type")
    pred = pred.to(target.device)

    pred_LL, pred_HL, pred_LH, pred_HH = dwt_init(pred)
    target_LL, target_HL, target_LH, target_HH = dwt_init(target)

    target_gray = torch.mean(target, dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=target.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=target.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    edge_x = torch.nn.functional.conv2d(target_gray, sobel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(target_gray, sobel_y, padding=1)
    edge_density = torch.mean(torch.sqrt(edge_x ** 2 + edge_y ** 2))
    edge_weight = torch.clamp(edge_density * 2, 1.0, 1.5)

    def _charbonnier_loss(pred, target):
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + charbonnier_eps ** 2)
        return loss.mean()

    loss = ( _charbonnier_loss(pred_LL, target_LL) +
            edge_weight * _charbonnier_loss(pred_HL, target_HL) +
            edge_weight * _charbonnier_loss(pred_LH, target_LH) +
            edge_weight * _charbonnier_loss(pred_HH, target_HH))
    return loss


class PerceptualLoss(nn.Module):
    """Perceptual Loss: Calculate MSE based on high-level features extracted by VGG16"""
    def __init__(self, device='cuda', requires_grad=False):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:8].to(device)
        if not requires_grad:
            for param in vgg.parameters():
                param.requires_grad_(False)
        self.vgg = vgg.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.mse = nn.MSELoss()
        self.channel_conv = None

    def _adjust_channels(self, x):
        """Channel adaptation: Single/Multi-channel -> 3 channels (match VGG input)"""
        C = x.shape[1]
        if C == 1:
            return x.repeat(1, 3, 1, 1)
        elif C > 3:
            if self.channel_conv is None or self.channel_conv.in_channels != C:
                self.channel_conv = nn.Conv2d(C, 3, kernel_size=1, bias=False).to(x.device)
                self.channel_conv.weight.requires_grad = False
            return self.channel_conv(x)
        return x

    def forward(self, pred, target):
        pred = self._adjust_channels(pred)
        target = self._adjust_channels(target)

        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        return self.mse(self.vgg(pred_norm), self.vgg(target_norm))


class EdgeAwareLoss(nn.Module):
    """Edge-Aware Loss: Apply higher weights to edge regions"""
    def __init__(self, edge_weight=5.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.mse = nn.MSELoss(reduction='none')

        sobel_x = torch.tensor(
            [[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]],
            dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]],
            dtype=torch.float32
        )

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_edge_mask(self, x):
        """Generate edge mask: 1 for edge regions, 0 for non-edge regions"""
        assert x.dim() == 4, f"Edge detection input must be 4D tensor, got {x.dim()}D"
        B, C, H, W = x.shape

        x_gray = torch.mean(x, dim=1, keepdim=True)
        assert x_gray.shape[1] == 1, f"Grayscale image must be 1 channel, got {x_gray.shape[1]} channels"

        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1, stride=(1, 1))
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1, stride=(1, 1))

        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_mag = edge_mag / (edge_mag.max() + 1e-8)

        edge_mask = (edge_mag > 0.1).float()
        return edge_mask

    def forward(self, pred, target):
        assert target.dim() == 4, f"target must be 4D tensor, got {target.dim()}D"

        edge_mask = self.get_edge_mask(target)
        pixel_loss = self.mse(pred, target)

        if pixel_loss.shape[1] > 1:
            edge_mask = edge_mask.repeat(1, pixel_loss.shape[1], 1, 1)

        weighted_loss = pixel_loss * (1.0 + (self.edge_weight - 1.0) * edge_mask)
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Loss: Integrate multiple loss functions (support automatic weight normalization)"""
    def __init__(self, alpha=0.15, beta=0.2, gamma=0, delta=0, epsilon=0, zeta=0,
                 num_channels=3, input_range="[0,1]", loss_clip_value=1e4,
                 edge_weight=5.0, device='cuda', charbonnier_eps=1e-3,
                 print_at_epoch_start=True, auto_normalize=False):
        super().__init__()

        self.raw_weights = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta,
            'epsilon': epsilon,
            'zeta': zeta
        }

        weight_sum = sum(max(w, 0.0) for w in self.raw_weights.values())

        if weight_sum < 1e-12:
            raise ValueError("All loss weights cannot be zero! Please set positive weight for at least one loss")

        if auto_normalize:
            self.alpha = max(alpha, 0.0) / weight_sum
            self.beta = max(beta, 0.0) / weight_sum
            self.gamma = max(gamma, 0.0) / weight_sum
            self.delta = max(delta, 0.0) / weight_sum
            self.epsilon = max(epsilon, 0.0) / weight_sum
            self.zeta = max(zeta, 0.0) / weight_sum
            self.normalize_info = f"Auto-normalized (Original weight sum={weight_sum:.6f})"
        else:
            self.alpha = max(alpha, 0.0)
            self.beta = max(beta, 0.0)
            self.gamma = max(gamma, 0.0)
            self.delta = max(delta, 0.0)
            self.epsilon = max(epsilon, 0.0)
            self.zeta = max(zeta, 0.0)
            self.normalize_info = f"Not auto-normalized (Current weight sum={weight_sum:.6f})"

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.edge_aware_loss = EdgeAwareLoss(edge_weight=edge_weight)

        self.charbonnier_eps = charbonnier_eps
        self.device = device

        self.num_channels = num_channels
        self.input_range = input_range
        self.loss_clip_value = loss_clip_value

        self.print_at_epoch_start = print_at_epoch_start
        self._should_print = False

        print(f"📌 Loss Weight Initialization Complete | {self.normalize_info}")
        print(f"   Normalized Weights: alpha={self.alpha:.6f}, beta={self.beta:.6f}, gamma={self.gamma:.6f}, "
              f"delta={self.delta:.6f}, epsilon={self.epsilon:.6f}, zeta={self.zeta:.6f}")
        print(f"   Sum of Normalized Weights: {self.alpha+self.beta+self.gamma+self.delta+self.epsilon+self.zeta:.6f}")

    def _charbonnier_loss(self, pred, target):
        """Charbonnier Loss: Smooth version of L1 loss"""
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.charbonnier_eps ** 2)
        return loss.mean()

    def _normalize_to_01(self, x):
        """Normalize input to [0, 1] range"""
        if self.input_range == "[-1,1]":
            return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        elif self.input_range == "[0,1]":
            return x.clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported input range: {self.input_range}, optional [-1,1] or [0,1]")

    def trigger_epoch_start_print(self):
        """Trigger loss detail printing at the start of epoch"""
        if self.print_at_epoch_start:
            self._should_print = True

    def forward(self, pred, target):
        if isinstance(pred, dict):
            pred = pred.get('out', pred)
        if isinstance(target, dict):
            target = target.get('out', target)

        pred = pred.to(self.device).float()
        target = target.to(self.device).float()
        assert pred.shape == target.shape, f"Shape mismatch: pred={pred.shape}, target={target.shape}"
        assert pred.dim() == 4, f"pred must be 4D tensor, got {pred.dim()}D"
        assert pred.shape[1] == self.num_channels, f"Channel error: required {self.num_channels}, got {pred.shape[1]}"

        pred_01 = self._normalize_to_01(pred)
        target_01 = self._normalize_to_01(target)

        charbonnier_raw = self._charbonnier_loss(pred, target) if self.alpha > 0 else torch.tensor(0.0, device=self.device)
        charbonnier_weighted = self.alpha * charbonnier_raw

        ssim_raw = (1 - self.ssim_loss(pred_01, target_01)).clamp(min=0.0) if self.beta > 0 else torch.tensor(0.0, device=self.device)
        ssim_weighted = self.beta * ssim_raw

        if self.gamma > 0:
            pred_fft = torch.fft.fft2(pred_01, dim=(-2, -1))
            target_fft = torch.fft.fft2(target_01, dim=(-2, -1))
            fft_raw = self.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        else:
            fft_raw = torch.tensor(0.0, device=self.device)
        fft_weighted = self.gamma * fft_raw

        wavelet_raw = wavelet_loss(pred_01, target_01, self.charbonnier_eps) if self.delta > 0 else torch.tensor(0.0, device=self.device)
        wavelet_weighted = self.delta * wavelet_raw

        perceptual_raw = self.perceptual_loss(pred_01, target_01) if self.epsilon > 0 else torch.tensor(0.0, device=self.device)
        perceptual_weighted = self.epsilon * perceptual_raw

        edge_aware_raw = self.edge_aware_loss(pred_01, target_01) if self.zeta > 0 else torch.tensor(0.0, device=self.device)
        edge_aware_weighted = self.zeta * edge_aware_raw

        total_loss = (charbonnier_weighted + ssim_weighted + fft_weighted +
                      wavelet_weighted + perceptual_weighted + edge_aware_weighted)
        total_loss_clipped = total_loss.clamp(min=0.0, max=self.loss_clip_value)

        if self.print_at_epoch_start and self._should_print:
            print("=" * 80)
            print("                      All Loss Details (6 Decimal Places)")
            print(f"                      {self.normalize_info}")
            print("=" * 80)
            print(f"1. Charbonnier Loss (α={self.alpha:.6f}):")
            print(f"   - Raw Value: {charbonnier_raw.item():.6f} | Weighted Value: {charbonnier_weighted.item():.6f}")
            print(f"2. SSIM Loss (β={self.beta:.6f}):")
            print(f"   - Raw Value: {ssim_raw.item():.6f} | Weighted Value: {ssim_weighted.item():.6f}")
            print(f"3. FFT Loss (γ={self.gamma:.6f}):")
            print(f"   - Raw Value: {fft_raw.item():.6f} | Weighted Value: {fft_weighted.item():.6f}")
            print(f"4. Wavelet Loss (δ={self.delta:.6f}):")
            print(f"   - Raw Value: {wavelet_raw.item():.6f} | Weighted Value: {wavelet_weighted.item():.6f}")
            print(f"5. Perceptual Loss (ε={self.epsilon:.6f}):")
            print(f"   - Raw Value: {perceptual_raw.item():.6f} | Weighted Value: {perceptual_weighted.item():.6f}")
            print(f"6. Edge-Aware Loss (ζ={self.zeta:.6f}):")
            print(f"   - Raw Value: {edge_aware_raw.item():.6f} | Weighted Value: {edge_aware_weighted.item():.6f}")
            print("-" * 80)
            print(f"📊 Total Loss (Clipped): {total_loss_clipped.item():.6f}")
            print("=" * 80)
            self._should_print = False

        return total_loss_clipped