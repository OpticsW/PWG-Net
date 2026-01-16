import torch
from torch import nn
import numpy as np


class ForceFloat32BatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm2d implementation with forced float32 precision for all internal operations and parameters:
    - weight
    - bias
    - running_mean
    - running_var
    - num_batches_tracked
    All calculations are performed strictly in float32 precision
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=torch.float32
        )

        if self.affine:
            if self.weight is not None:
                self.weight.data = self.weight.data.to(torch.float32)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(torch.float32)

        if self.track_running_stats:
            if self.running_mean is not None:
                self.running_mean.data = self.running_mean.data.to(torch.float32)
            if self.running_var is not None:
                self.running_var.data = self.running_var.data.to(torch.float32)
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.data = self.num_batches_tracked.data.to(torch.float32)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        input_float32 = input.to(torch.float32)
        output_float32 = super().forward(input_float32)
        output = output_float32.to(input_dtype)
        return output


if __name__ == "__main__":
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.incov = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                ForceFloat32BatchNorm2d(32)
            )

        def forward(self, x):
            return self.incov(x)

    model = ExampleModel()

    print("Model Structure and Parameter Info：")
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(
            f"[{idx:4d}] Layer Name：{name:<30} Shape：{list(param.shape):<20} Device：{param.device} Dtype：{param.dtype} Param Num：{param.numel()}")

    print("\nBuffer Info (Including num_batches_tracked)：")
    for idx, (name, buf) in enumerate(model.named_buffers()):
        print(f"[{idx:4d}] Buffer Name：{name:<30} Shape：{list(buf.shape):<20} Device：{buf.device} Dtype：{buf.dtype}")

    print("\nTesting Forward Propagation...")
    input_tensor = torch.randn(2, 3, 64, 64)
    with torch.autocast(device_type='cpu', dtype=torch.float16):
        output = model(input_tensor)
        print(f"Input Dtype：{input_tensor.dtype}")
        print(f"Output Dtype：{output.dtype}")
        print(f"Forward Propagation Completed Without NaN or Exception")