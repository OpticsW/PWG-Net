import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import os
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
import datetime
from My_Dataset import DehazeDataset
from models.model import Model
from PU.PGW_Net.models.Loss import CombinedLoss
from Train_Block import train_model
from torch.amp import GradScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- Hyperparameter Configuration --------------------------
HYPERPARAMS = {
    "DATA": {
        "DATA_ROOT": r"D:\data_new_1010",
        "IMAGE_SIZE": (512, 512),
        "USE_RANDOM_CROP": True,
        "CROP_SIZE": (256, 256),
        "PADDING_SIZE": 0,
        "SPLITS": {"train": "train", "val": "test", "test": "test"},
        "PRELOAD_TO_MEMORY": False,
        "CACHE_SIZE": {"train": 128, "val": 64, "test": 64},
        "NUM_WORKERS_MAX": 4,
        "USE_NORMALIZE": True,
        "CHECK_NAN_IN_DATA": True,
        "TRAIN_SAMPLE_NUM": 2048,
        "TRAIN_SAMPLE_SEED": 42,
        "USE_SAMPLE_SEED": False
    },
    "MODEL": {
        "IN_CHANNELS": 3,
        "OUT_CHANNELS": 3,
        "FORCE_INIT_LR": True,
        "INIT_LR": 1e-5,
        "PRETRAINED_DEFAULT_PATH": "unet_model_best_avg_ssim.pth",
        "GRAD_CLIP_MAX_NORM": 5.0,
        "FREEZE_BATCH_NORM": False,
        "FREEZE_BATCH_NORM_AFFINE": True
    },
    "TRAIN": {
        "BATCH_SIZE": 4,
        "ACCUMULATE_GRAD": 8,
        "NUM_EPOCHS": 400,
        "WARMUP_EPOCHS": 0,
        "FROZEN_BN_EPOCHS": 0,
        "MIXED_PRECISION": False,
        "SAVE_INTERVAL": 50,
        "MAX_VAL_BATCHES": None,
        "MAX_TEST_BATCHES": None,
        "USE_TEST": False,
        "PRINT_AT_EPOCH_START": False,
    },
    "LOSS": {
        "ALPHA": 3,
        "BETA": 7,
        "GAMMA": 0,
        "DELTA": 0,
        "EPSILON": 0.0,
        "ZETA": 0,
        "CLIP_VALUE": 10.0
    },
    "OPTIMIZER": {
        "TYPE": "AdamW",
        "WEIGHT_DECAY": 1e-5,
        "BETAS": (0.9, 0.999),
        "EPS": 1e-9,
    },
    "SCHEDULER": {
        "TYPE": "ReduceLROnPlateau",
        "MODE": "min",
        "FACTOR": 0.5,
        "PATIENCE": 5,
        "MIN_LR": 1e-6,
        "THRESHOLD": 1e-4,
        "THRESHOLD_MODE": "abs",
        "verbose": True
    },
    "LOG": {
        "SAVE_ROOT": "DFP_Net_weights",
        "LOG_ROOT": "DFP_Net_runs",
        "FETCH_LR": False,
        "PRINT_PER_EPOCH": True
    }
}


# -------------------------- Auxiliary Functions --------------------------
def identity(x):
    return x


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def calculate_model_params(model, device):
    def format_params(count):
        if count >= 1e9:
            return f"{count / 1e9:.2f}B"
        elif count >= 1e6:
            return f"{count / 1e6:.2f}M"
        elif count >= 1e3:
            return f"{count / 1e3:.2f}K"
        else:
            return f"{count:.0f}"

    total_count = 0
    trainable_count = 0
    for param in model.parameters():
        param_count = param.numel()
        total_count += param_count
        if param.requires_grad:
            trainable_count += param_count

    total_params = format_params(total_count)
    trainable_params = format_params(trainable_count)
    in_ch = HYPERPARAMS["MODEL"]["IN_CHANNELS"]
    out_ch = HYPERPARAMS["MODEL"]["OUT_CHANNELS"]
    print(f"\n📊 Model Parameter Statistics (Device: {device} | Input Channels: {in_ch} | Output Channels: {out_ch}):")
    print(f"  - Total Params: {total_params} ({total_count:,} params)")
    print(f"  - Trainable Params: {trainable_params} ({trainable_count:,} params)")
    print(f"  - Trainable Ratio: {trainable_count / total_count * 100:.2f}%")
    print(f"⚠️  Please confirm that the last layer of U_net in model1.py has added nn.Tanh() or (Normalization + torch.clamp)")
    return total_params, trainable_params, total_count, trainable_count


def validate_data_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    print(f"✅ Dataset root path validation passed: {path}")


def validate_dataset(dataset, dataset_name):
    if len(dataset) == 0:
        raise ValueError(f"{dataset_name} has 0 samples!")
    print(f"✅ {dataset_name} sample count validation passed: {len(dataset)} samples (All samples will be used for metric calculation)")


def check_data_nan_inf(tensor, name, dataset_type):
    if torch.isnan(tensor).any():
        raise ValueError(f"❌ {dataset_type} {name} contains NaN values!")
    if torch.isinf(tensor).any():
        raise ValueError(f"❌ {dataset_type} {name} contains Inf values!")


def validate_dataloader(dataloader, dataset_name, expected_shape, expected_dtype=torch.float32):
    try:
        batch_iter = iter(dataloader)
        hazy_imgs, clean_imgs = next(batch_iter)

        use_norm = HYPERPARAMS["DATA"]["USE_NORMALIZE"]
        if use_norm:
            assert (hazy_imgs >= -1.01).all() and (hazy_imgs <= 1.01).all(), f"{dataset_name} data range error (should be [-1,1])"
            assert (clean_imgs >= -1.01).all() and (clean_imgs <= 1.01).all(), f"{dataset_name} label range error"
            range_info = "[-1.0, 1.0]"
        else:
            assert (hazy_imgs >= -0.01).all() and (hazy_imgs <= 1.01).all(), f"{dataset_name} data range error (should be [0,1])"
            assert (clean_imgs >= -0.01).all() and (clean_imgs <= 1.01).all(), f"{dataset_name} label range error"
            range_info = "[0.0, 1.0]"

        if HYPERPARAMS["DATA"]["CHECK_NAN_IN_DATA"]:
            check_data_nan_inf(hazy_imgs, "hazy images", dataset_name)
            check_data_nan_inf(clean_imgs, "clean images", dataset_name)

        print(f"✅ {dataset_name} DataLoader validation passed:")
        print(f"   - Shape: {hazy_imgs.shape} | Dtype: {hazy_imgs.dtype} | Range: {range_info}")
        print(f"   - Hazy Img Stats: min={hazy_imgs.min():.4f}, max={hazy_imgs.max():.4f}, mean={hazy_imgs.mean():.4f}")
        print(f"   - GT Img Stats: min={clean_imgs.min():.4f}, max={clean_imgs.max():.4f}, mean={clean_imgs.mean():.4f}")
        print(f"   - Tip: All {dataset_name} samples ({len(dataloader.dataset)} total) will be used for metric calculation")
    except Exception as e:
        raise RuntimeError(f"{dataset_name} DataLoader validation failed: {str(e)}") from e


# -------------------------- Main Training Process --------------------------
if __name__ == "__main__":
    # 1. Parse Hyperparameters
    DATA_ROOT = HYPERPARAMS["DATA"]["DATA_ROOT"]
    IMAGE_SIZE = HYPERPARAMS["DATA"]["IMAGE_SIZE"]
    USE_RANDOM_CROP = HYPERPARAMS["DATA"]["USE_RANDOM_CROP"]
    CROP_SIZE = HYPERPARAMS["DATA"]["CROP_SIZE"]
    PADDING_SIZE = HYPERPARAMS["DATA"]["PADDING_SIZE"]
    DATA_SPLITS = HYPERPARAMS["DATA"]["SPLITS"]
    PRELOAD_MEM = HYPERPARAMS["DATA"]["PRELOAD_TO_MEMORY"]
    CACHE_SIZES = HYPERPARAMS["DATA"]["CACHE_SIZE"]
    MAX_WORKERS = HYPERPARAMS["DATA"]["NUM_WORKERS_MAX"]
    USE_NORMALIZE = HYPERPARAMS["DATA"]["USE_NORMALIZE"]
    CHECK_NAN_IN_DATA = HYPERPARAMS["DATA"]["CHECK_NAN_IN_DATA"]
    TRAIN_SAMPLE_NUM = HYPERPARAMS["DATA"]["TRAIN_SAMPLE_NUM"]
    TRAIN_SAMPLE_SEED = HYPERPARAMS["DATA"]["TRAIN_SAMPLE_SEED"]
    USE_SAMPLE_SEED = HYPERPARAMS["DATA"]["USE_SAMPLE_SEED"]

    IN_CH = HYPERPARAMS["MODEL"]["IN_CHANNELS"]
    OUT_CH = HYPERPARAMS["MODEL"]["OUT_CHANNELS"]
    INIT_LR = HYPERPARAMS["MODEL"]["INIT_LR"]
    FORCE_INIT_LR = HYPERPARAMS["MODEL"]["FORCE_INIT_LR"]
    PRETRAINED_FNAME = HYPERPARAMS["MODEL"]["PRETRAINED_DEFAULT_PATH"]
    GRAD_CLIP_MAX_NORM = HYPERPARAMS["MODEL"]["GRAD_CLIP_MAX_NORM"]
    USE_FREEZE_BN = HYPERPARAMS["MODEL"]["FREEZE_BATCH_NORM"]
    USE_FREEZE_BN_AFFINE = HYPERPARAMS["MODEL"]["FREEZE_BATCH_NORM_AFFINE"]

    train_config = HYPERPARAMS["TRAIN"]
    TOTAL_EPOCHS = train_config["NUM_EPOCHS"]
    WARMUP_EPOCHS = train_config["WARMUP_EPOCHS"]
    FROZEN_BN_EPOCHS = train_config["FROZEN_BN_EPOCHS"]
    BATCH_SIZE = train_config["BATCH_SIZE"]
    ACCUMULATE_GRAD = train_config["ACCUMULATE_GRAD"]
    MIXED_PREC = train_config["MIXED_PRECISION"]
    SAVE_INTERVAL = train_config["SAVE_INTERVAL"]
    MAX_VAL_BATCHES = train_config["MAX_VAL_BATCHES"]
    MAX_TEST_BATCHES = train_config["MAX_TEST_BATCHES"]
    USE_TEST = train_config["USE_TEST"]
    PRINT_AT_EPOCH_START = train_config["PRINT_AT_EPOCH_START"]

    LOSS_ALPHA = HYPERPARAMS["LOSS"]["ALPHA"]
    LOSS_BETA = HYPERPARAMS["LOSS"]["BETA"]
    LOSS_GAMMA = HYPERPARAMS["LOSS"]["GAMMA"]
    LOSS_DELTA = HYPERPARAMS["LOSS"]["DELTA"]
    LOSS_EPSILON = HYPERPARAMS["LOSS"]["EPSILON"]
    LOSS_ZETA = HYPERPARAMS["LOSS"]["ZETA"]
    LOSS_CLIP_VALUE = HYPERPARAMS["LOSS"]["CLIP_VALUE"]
    SAVE_ROOT = HYPERPARAMS["LOG"]["SAVE_ROOT"]
    LOG_ROOT = HYPERPARAMS["LOG"]["LOG_ROOT"]
    PRINT_PER_EPOCH = HYPERPARAMS["LOG"]["PRINT_PER_EPOCH"]

    # 2. Environment Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"✅ GPU Acceleration Config Completed: Device: {device} | Model: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Training with CPU (GPU is recommended for faster training)")

    # 3. Path & Log Initialization
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    current_time = datetime.datetime.now().strftime('%H%M')
    SAVE_DIR = os.path.join(SAVE_ROOT, current_date)
    LOG_DIR = os.path.join(LOG_ROOT, current_date, f"{current_time}_experiment")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"✅ TensorBoard Log Initialized: Path: {LOG_DIR}")
    print(f"📌 Dataset Calculation Config:")
    print(f"   - Validation Set: Use all samples (MAX_VAL_BATCHES={MAX_VAL_BATCHES})")
    print(f"   - Test Set: Use all samples (MAX_TEST_BATCHES={MAX_TEST_BATCHES})")

    # 4. Dataset Loading & Sampling
    validate_data_path(DATA_ROOT)
    train_transform_list = []
    train_transform_list.extend([v2.Resize(CROP_SIZE)])
    if USE_RANDOM_CROP:
        train_transform_list.extend([
            v2.Pad(padding=PADDING_SIZE, padding_mode="reflect"),
            v2.RandomCrop(size=CROP_SIZE)
        ])
        print(f"📌 Random crop augmentation with Padding enabled: Padding={PADDING_SIZE}px | Crop Size={CROP_SIZE}")
    else:
        train_transform_list.append(v2.Resize(CROP_SIZE, antialias=True))
        print(f"📌 Random crop augmentation disabled: Resize to {IMAGE_SIZE}")

    train_transform_list.extend([
        v2.Resize(CROP_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if USE_NORMALIZE else v2.Lambda(identity),
    ])
    train_transform = v2.Compose(train_transform_list)

    val_test_transform = v2.Compose([
        v2.Resize(CROP_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if USE_NORMALIZE else v2.Lambda(identity),
    ])

    def denormalize(tensor):
        """Denormalization (Ensure range is [0,1])"""
        if not USE_NORMALIZE:
            return tensor.clamp(0.0, 1.0)
        return (tensor / 2 + 0.5).clamp(0.0, 1.0)

    print("\n📥 Start Loading Datasets...")
    train_dataset = DehazeDataset(
        data_root=DATA_ROOT,
        dataset_type=DATA_SPLITS["train"],
        transform=train_transform,
        preload_to_memory=PRELOAD_MEM,
        cache_size=CACHE_SIZES["train"]
    )
    val_dataset = DehazeDataset(
        data_root=DATA_ROOT,
        dataset_type=DATA_SPLITS["val"],
        transform=val_test_transform,
        preload_to_memory=PRELOAD_MEM,
        cache_size=CACHE_SIZES["val"]
    )
    test_dataset = DehazeDataset(
        data_root=DATA_ROOT,
        dataset_type=DATA_SPLITS["test"],
        transform=val_test_transform,
        preload_to_memory=PRELOAD_MEM,
        cache_size=CACHE_SIZES["test"]
    ) if USE_TEST else None

    train_total = len(train_dataset)
    if TRAIN_SAMPLE_NUM is None or TRAIN_SAMPLE_NUM <= 0 or TRAIN_SAMPLE_NUM >= train_total:
        print(f"⚠️ Invalid train set sampling parameter (Target: {TRAIN_SAMPLE_NUM}, Original: {train_total}), use all samples")
        sampled_train_dataset = train_dataset
        sampled_num = train_total
        sample_seed_log = f"Seed={TRAIN_SAMPLE_SEED} (Not activated)" if USE_SAMPLE_SEED else "Fixed seed not used"
    else:
        if USE_SAMPLE_SEED:
            torch.manual_seed(TRAIN_SAMPLE_SEED)
            sample_seed_log = f"Seed={TRAIN_SAMPLE_SEED} (Activated)"
        else:
            sample_seed_log = "Fixed seed not used"
        random_indices = torch.randperm(train_total)[:TRAIN_SAMPLE_NUM]
        sampled_train_dataset = Subset(train_dataset, random_indices)
        sampled_num = TRAIN_SAMPLE_NUM
    print(f"✅ Train Set Sampling Completed: Original Samples={train_total} | Sampled Samples={sampled_num} | {sample_seed_log}")

    validate_dataset(sampled_train_dataset, "Sampled Train Set")
    validate_dataset(val_dataset, "Validation Set")
    if USE_TEST and test_dataset is not None:
        validate_dataset(test_dataset, "Test Set")

    # 5. DataLoader Configuration
    cpu_cores = os.cpu_count() or 6
    num_workers_train = min(cpu_cores, MAX_WORKERS)
    num_workers_val_test = min(num_workers_train // 2, 2)
    input_size = CROP_SIZE if USE_RANDOM_CROP else IMAGE_SIZE

    train_dataloader = DataLoader(
        sampled_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=True,
        pin_memory_device=str(device),
        drop_last=True,
        persistent_workers=True if num_workers_train > 0 else False,
        prefetch_factor=2 if num_workers_train > 0 else None
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers_val_test,
        pin_memory=True,
        pin_memory_device=str(device),
        persistent_workers=True if num_workers_val_test > 0 else False,
        prefetch_factor=1 if num_workers_val_test > 0 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers_val_test,
        pin_memory=True,
        pin_memory_device=str(device),
        persistent_workers=True if num_workers_val_test > 0 else False,
        prefetch_factor=1 if num_workers_val_test > 0 else None
    ) if USE_TEST and test_dataset is not None else None

    train_expected_shape = (BATCH_SIZE, IN_CH, input_size[0], input_size[1])
    val_test_expected_shape = (BATCH_SIZE * 2, IN_CH, IMAGE_SIZE[0], IMAGE_SIZE[1])
    validate_dataloader(train_dataloader, "Sampled Train Set", train_expected_shape)
    validate_dataloader(val_dataloader, "Validation Set", val_test_expected_shape)
    if USE_TEST and test_dataloader is not None:
        validate_dataloader(test_dataloader, "Test Set", val_test_expected_shape)

    # 6. Model/Optimizer/Scheduler Initialization
    model = Model(in_channels=IN_CH, out_channels=OUT_CH).to(device, dtype=torch.float32)
    calculate_model_params(model, device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=INIT_LR,
        weight_decay=HYPERPARAMS["OPTIMIZER"]["WEIGHT_DECAY"],
        betas=HYPERPARAMS["OPTIMIZER"]["BETAS"],
        eps=HYPERPARAMS["OPTIMIZER"]["EPS"]
    )

    if FORCE_INIT_LR:
        current_lr_after_init = get_current_lr(optimizer)
        if abs(current_lr_after_init - INIT_LR) > 1e-10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = INIT_LR
            print(f"🔒 FORCE_INIT_LR=True: Optimizer LR calibrated to {INIT_LR:.8f} (Original LR: {current_lr_after_init:.8f})")
        else:
            print(f"🔒 FORCE_INIT_LR=True: Optimizer initial LR is already {INIT_LR:.8f}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=HYPERPARAMS["SCHEDULER"]["MODE"],
        factor=HYPERPARAMS["SCHEDULER"]["FACTOR"],
        patience=HYPERPARAMS["SCHEDULER"]["PATIENCE"],
        min_lr=HYPERPARAMS["SCHEDULER"]["MIN_LR"],
        threshold=HYPERPARAMS["SCHEDULER"]["THRESHOLD"],
        threshold_mode=HYPERPARAMS["SCHEDULER"]["THRESHOLD_MODE"],
        verbose=True
    )

    current_lr = get_current_lr(optimizer)
    pretrained_path = os.path.join(SAVE_DIR, PRETRAINED_FNAME)
    if os.path.exists(pretrained_path):
        try:
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_after_pretrain = get_current_lr(optimizer)
            print(f"\n✅ Loaded pretrained weights (including optimizer state): {pretrained_path} | Pretrained LR: {lr_after_pretrain:.8f}")

            if FORCE_INIT_LR:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = INIT_LR
                current_lr = INIT_LR
                print(f"🔒 FORCE_INIT_LR=True: Override pretrained LR, reset to {INIT_LR:.8f}")
            else:
                current_lr = lr_after_pretrain

            if MIXED_PREC:
                scaler = GradScaler(init_scale=2048, growth_interval=100)
                print(f"🔄 Mixed Precision Scaler reset (after loading pretrained weights)")

        except:
            model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
            current_lr = INIT_LR
            print(f"\n✅ Loaded pretrained weights (model only): {pretrained_path} | Current LR: {current_lr:.8f}")
    else:
        current_lr = INIT_LR
        print(f"\n⚠️ Pretrained weights not found: {pretrained_path}, training from scratch | Current LR: {current_lr:.8f}")

    criterion = CombinedLoss(
        alpha=LOSS_ALPHA,
        beta=LOSS_BETA,
        gamma=LOSS_GAMMA,
        delta=LOSS_DELTA,
        epsilon=LOSS_EPSILON,
        zeta=LOSS_ZETA,
        num_channels=IN_CH,
        device=device,
        input_range="[-1,1]" if USE_NORMALIZE else "[0,1]",
        loss_clip_value=LOSS_CLIP_VALUE,
        print_at_epoch_start=PRINT_AT_EPOCH_START
    ).to(device)

    if MIXED_PREC and not (os.path.exists(pretrained_path) and "scaler" in locals()):
        scaler = GradScaler(init_scale=2048, growth_interval=100)
        print(f"✅ Mixed Precision Scaler Initialized: init_scale=2048, growth_interval=100")
    elif not MIXED_PREC:
        scaler = None
        print(f"⚠️ Mixed Precision Training not enabled")

    # 7. Start Training
    print(f"\n📌 Training Plan Confirmed: Total Epochs={TOTAL_EPOCHS} | Warmup={WARMUP_EPOCHS}epoch | FrozenBN={FROZEN_BN_EPOCHS}epoch")
    print(f"📌 Learning Rate Config: Initial LR={current_lr:.8f} (Forced Initialization), dynamically adjusted by scheduler later")
    print(f"📌 Start Training...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        denormalize_fn=denormalize,
        total_epochs=TOTAL_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        frozen_bn_epochs=FROZEN_BN_EPOCHS,
        init_lr=INIT_LR,
        in_channels=IN_CH,
        use_normalize=USE_NORMALIZE,
        force_init_lr=FORCE_INIT_LR,
        device=device,
        save_dir=SAVE_DIR,
        base_model_name='unet_model',
        train_config=train_config,
        grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
        loss_clip_value=LOSS_CLIP_VALUE,
        print_per_epoch=PRINT_PER_EPOCH,
        use_test=USE_TEST,
        use_freeze_bn_affine=USE_FREEZE_BN_AFFINE
    )

    writer.close()
    print(f"\n🎉 All Training Tasks Completed!")