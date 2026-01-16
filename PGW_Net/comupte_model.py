import torch
import os
import warnings
import time
import numpy as np
from thop import profile
from thop import clever_format

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- Import Model --------------------------
from models.model import Model

# -------------------------- Hyperparameter Configuration (Only Model & Input Related) --------------------------
HYPERPARAMS = {
    "DATA": {
        "IMAGE_SIZE": (512, 512),  # Input image size
    },
    "MODEL": {
        "IN_CHANNELS": 3,
        "OUT_CHANNELS": 4,
    },
    "TRAIN": {
        "BATCH_SIZE": 1,  # Inference batch size
    }
}

# Path Configuration
weight_root = "/20251105"
model_weight_name = "model_best_avg_ssim.pth"

# -------------------------- Extract Hyperparameters --------------------------
IMAGE_SIZE = HYPERPARAMS["DATA"]["IMAGE_SIZE"]
IN_CHANNELS = HYPERPARAMS["MODEL"]["IN_CHANNELS"]
OUT_CHANNELS = HYPERPARAMS["MODEL"]["OUT_CHANNELS"]
BATCH_SIZE = HYPERPARAMS["TRAIN"]["BATCH_SIZE"]

# -------------------------- Device Configuration --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using computing device: {device}")

# -------------------------- Initialize and Load Model --------------------------
model = Model(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device, dtype=torch.float32)

# -------------------------- Calculate Basic Model Metrics (Params, Size, FLOPs) --------------------------
# Total number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Model size (32-bit float, 4 bytes per parameter)
model_size = total_params * 4 / (1024 ** 2)  # MB

# FLOPs calculation with dummy input
input_shape = (BATCH_SIZE, IN_CHANNELS) + IMAGE_SIZE  # (batch, channel, h, w)
dummy_input = torch.randn(input_shape).to(device)
flops, params = profile(model, inputs=(dummy_input,))
flops, params = clever_format([flops, params], "%.3f")

# -------------------------- Calculate Latency and FPS --------------------------
model.eval()
latency_list = []
num_warmup = 10  # Warmup iterations to exclude initial GPU loading time
num_test = 100   # Test iterations for average calculation

with torch.no_grad():
    # Warmup inference (no timing)
    for _ in range(num_warmup):
        model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()  # Ensure all GPU operations are completed

    # Formal test with timing
    start_total = time.time()
    for _ in range(num_test):
        start = time.time()
        model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()  # Synchronize GPU for accurate timing
        end = time.time()
        latency_list.append(end - start)
    total_time = time.time() - start_total

# Compute latency and FPS
avg_latency = np.mean(latency_list)
p50_latency = np.percentile(latency_list, 50)
p90_latency = np.percentile(latency_list, 90)
fps = (num_test * BATCH_SIZE) / total_time  # Total frames / total time

# -------------------------- Print Results --------------------------
print("\n" + "=" * 80)
print(f"Model Performance Metrics:")
print(f"  Total Parameters: {total_params:,}")
print(f"  Model Size: {model_size:.2f} MB")
print(f"  FLOPs: {flops}")
print(f"  Average Latency: {avg_latency:.6f} s")
print(f"  P50 Latency: {p50_latency:.6f} s")
print(f"  P90 Latency: {p90_latency:.6f} s")
print(f"  FPS (Frames Per Second): {fps:.2f}")
print("=" * 80)