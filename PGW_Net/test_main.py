import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from torchvision.transforms import v2
from PU.PGW_Net.models.SSIM_method import SSIM
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from models.model import Model
from My_Dataset import DehazeDataset

HYPERPARAMS = {
    "DATA": {
        "DATA_ROOT": r"D:\data\data_new_1010",
        "IMAGE_SIZE": (512, 512),
        "USE_RANDOM_CROP": True,
        "CROP_SIZE": (512, 512),
        "PADDING_SIZE": 0,
        "SPLITS": {"train": "train", "val": "test", "test": "test"},
        "PRELOAD_TO_MEMORY": False,
        "CACHE_SIZE": {"train": 128, "val": 64, "test": 64},
        "USE_NORMALIZE": True,
        "CHECK_NAN_IN_DATA": True,
    },
    "MODEL": {
        "IN_CHANNELS": 3,
        "OUT_CHANNELS": 3,
    },
    "TRAIN": {
        "BATCH_SIZE": 1,
    }
}

weight_root = "./20251113"
model_weight_name = "model_best_avg_ssim.pth"
save_dir = r"./data"

DATA_ROOT = HYPERPARAMS["DATA"]["DATA_ROOT"]
PADDING_SIZE = HYPERPARAMS["DATA"]["PADDING_SIZE"]
CROP_SIZE = HYPERPARAMS["DATA"]["CROP_SIZE"]
USE_NORMALIZE = HYPERPARAMS["DATA"]["USE_NORMALIZE"]
CACHE_SIZE_TEST = HYPERPARAMS["DATA"]["CACHE_SIZE"]["test"]
DATASET_TYPE = HYPERPARAMS["DATA"]["SPLITS"]["test"]
IMAGE_SIZE = HYPERPARAMS["DATA"]["IMAGE_SIZE"]

IN_CHANNELS = HYPERPARAMS["MODEL"]["IN_CHANNELS"]
OUT_CHANNELS = HYPERPARAMS["MODEL"]["OUT_CHANNELS"]

# -------------------------- Basic Configuration --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using computing device: {device}")

os.makedirs(save_dir, exist_ok=True)
print(f"Restoration results will be saved to: {save_dir}")

# -------------------------- Data Preprocessing --------------------------
val_test_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if USE_NORMALIZE else v2.Lambda(lambda x: x),
])

# -------------------------- Denormalization Function --------------------------
def denormalize(tensor):
    if not USE_NORMALIZE:
        return tensor.clamp(0.0, 1.0)
    return (tensor / 2 + 0.5).clamp(0.0, 1.0)

# -------------------------- Load Test Dataset --------------------------
print("Loading test dataset...")
test_dataset = DehazeDataset(
    data_root=DATA_ROOT,
    dataset_type=DATASET_TYPE,
    transform=val_test_transform,
    preload_to_memory=HYPERPARAMS["DATA"]["PRELOAD_TO_MEMORY"],
    cache_size=CACHE_SIZE_TEST
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=HYPERPARAMS["TRAIN"]["BATCH_SIZE"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    pin_memory_device=str(device) if torch.cuda.is_available() else ""
)
print(f"Number of test samples: {len(test_dataset)}")
sample_haze, sample_clear = test_dataset[0]
print(f"Tensor range returned by Dataset: haze={sample_haze.min():.2f}~{sample_haze.max():.2f}")

# -------------------------- Initialize Model --------------------------
model = Model(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device, dtype=torch.float32)
model_path = os.path.join(weight_root, model_weight_name)

if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded model weights (including training state) from {model_path}!")
    except:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Successfully loaded model weights (only model params) from {model_path}!")
else:
    print(f"Model file {model_path} not found")
    exit()

# -------------------------- Initialize Metric Calculators --------------------------
ssim_calculator = SSIM(window_size=11, size_average=True).to(device)
total_psnr = 0.0
total_ssim = 0.0
num_samples = len(test_dataset)

# -------------------------- Model Inference & Metric Calculation --------------------------
model.eval()
with torch.no_grad():
    for batch_idx, (hazy_img, clear_img) in enumerate(test_dataloader):
        hazy_img = hazy_img.to(device)
        clear_img = clear_img.to(device)

        output = model(hazy_img)[:, :3, :, :]

        output_denorm = denormalize(output)
        clear_img_denorm = denormalize(clear_img)

        print(
            f"Batch {batch_idx+1} output channel mean: R={output_denorm[0, 0].mean():.3f}, "
            f"G={output_denorm[0, 1].mean():.3f}, B={output_denorm[0, 2].mean():.3f}"
        )

        mse = torch.mean((output_denorm - clear_img_denorm) ** 2, dim=[1, 2, 3])
        psnr = 10 * torch.log10(1 / (mse + 1e-8))
        total_psnr += psnr.sum().item()

        ssim = ssim_calculator(output_denorm, clear_img_denorm)
        total_ssim += ssim.item() * hazy_img.shape[0]

        for idx_in_batch in range(hazy_img.shape[0]):
            global_idx = batch_idx * hazy_img.shape[0] + idx_in_batch
            if global_idx >= num_samples:
                break

            single_output = output_denorm[idx_in_batch].squeeze(0).permute(1, 2, 0).cpu().numpy()
            single_output = (single_output * 255).astype(np.uint8)
            single_output_img = Image.fromarray(single_output)

            save_path = os.path.join(save_dir, f"{global_idx:04d}.png")
            single_output_img.save(save_path)

        print(
            f"Batch {batch_idx+1}/{len(test_dataloader)} | "
            f"Current batch average PSNR: {psnr.mean().item():.2f} dB | "
            f"Current batch SSIM: {ssim.item():.4f}"
        )

# -------------------------- Print Average Metrics --------------------------
avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples
print("\n" + "="*60)
print(f"Test set average PSNR: {avg_psnr:.4f} dB")
print(f"Test set average SSIM: {avg_ssim:.4f}")
print("="*60)
print(f"All test images processed, results saved to {save_dir}")