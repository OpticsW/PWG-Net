import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms.v2 as transforms


class DehazeDataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset_type='train',
                 transform=None,
                 preload_to_memory=False,
                 cache_size=32):
        """
        Haze Removal Dataset Class (Compatible with training set sampling function)
        :param data_root: Root path of the dataset
        :param dataset_type: Dataset type (train/val/test)
        :param transform: Data augmentation/transform function
        :param preload_to_memory: Whether to preload all data into memory
        :param cache_size: Size of the LRU cache for recently loaded samples
        """
        self.dataset_type = dataset_type
        self.transform = transform
        self.preload_to_memory = preload_to_memory
        self.cache_size = cache_size
        self._cache = {}  # Cache for loaded samples (index: (haze_tensor, gt_tensor))
        self.data_root = data_root

        # Assign dataset paths (separate hazy and GT image folders)
        self.haze_dir = os.path.join(data_root, dataset_type, 'hazy')
        self.gt_dir = os.path.join(data_root, dataset_type, 'GT')

        # Validate dataset integrity
        self._validate_dataset_integrity()

        # Load sorted image path lists (Ensure one-to-one matching of haze and GT images)
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        self.haze_paths = self._get_image_paths(self.haze_dir, img_extensions)
        self.gt_paths = self._get_image_paths(self.gt_dir, img_extensions)

        # Preload data (Recommended to disable for training set to avoid high memory usage)
        if self.preload_to_memory:
            self._preload_data()
        else:
            self._cache = {}  # Dynamic cache enabled

        self.total_samples = len(self.haze_paths)
        # Dataset status log
        if self.dataset_type == 'train':
            print(
                f"📊 {dataset_type} Full Dataset Loaded: {self.total_samples} samples (Sampling supported) | Cache Size: {cache_size} | Root Path: {data_root}")
        else:
            print(
                f"📊 {dataset_type} Dataset Loaded: {self.total_samples} samples | Cache Size: {cache_size} | Root Path: {data_root}")

    def _validate_dataset_integrity(self):
        """Validate the validity of dataset folders and files"""
        # Validate folder existence
        for dir_path in [self.haze_dir, self.gt_dir]:
            if not os.path.isdir(dir_path):
                raise NotADirectoryError(
                    f"Dataset folder not found: {dir_path}\n"
                    f"Please check if the root path '{self.data_root}' and subset '{self.dataset_type}' are correct"
                )

        # Validate non-empty folders
        haze_files = glob.glob(os.path.join(self.haze_dir, '*'))
        gt_files = glob.glob(os.path.join(self.gt_dir, '*'))
        if len(haze_files) == 0:
            raise FileNotFoundError(f"Hazy image folder is empty: {self.haze_dir}")
        if len(gt_files) == 0:
            raise FileNotFoundError(f"GT image folder is empty: {self.gt_dir}")

        # Validate sample count matching between haze and GT images
        if len(haze_files) != len(gt_files):
            raise ValueError(
                f"Mismatched sample count between haze and GT images!\n"
                f"Hazy images: {len(haze_files)} | GT images: {len(gt_files)}"
            )

    def _get_image_paths(self, dir_path, extensions):
        """Get sorted list of all supported image paths in the target folder"""
        paths = sorted([
            p for p in glob.glob(os.path.join(dir_path, '*'))
            if p.lower().endswith(extensions)
        ])
        if not paths:
            raise FileNotFoundError(
                f"No supported image files found in {dir_path}!\n"
                f"Supported formats: {extensions}"
            )
        return paths

    def _preload_data(self):
        """Preload all dataset samples into memory (suitable for small datasets)"""
        print(f"⚠️ Preloading {self.dataset_type} dataset into memory ({self.total_samples} samples)...")
        self._cache = {}
        for idx in range(self.total_samples):
            haze_tensor, gt_tensor = self._load_and_process(idx)
            self._cache[idx] = (haze_tensor, gt_tensor)
        # Estimate memory usage (512x512x3 float32 tensor ≈ 3MB per image)
        approx_memory = self.total_samples * 3 * 1024 ** -2  # Convert to MB
        print(f"✅ {self.dataset_type} dataset preloaded successfully, estimated memory usage: {approx_memory:.2f}MB")

    def _load_and_process(self, idx):
        """Load single sample and perform basic preprocessing (without transform)"""
        haze_path = self.haze_paths[idx]
        gt_path = self.gt_paths[idx]

        # Validate filename matching (Ensure haze image corresponds to GT image)
        haze_name = os.path.splitext(os.path.basename(haze_path))[0]
        gt_name = os.path.splitext(os.path.basename(gt_path))[0]
        if haze_name != gt_name:
            raise ValueError(
                f"Sample filename mismatch!\n"
                f"Hazy image: {haze_name} ({haze_path})\n"
                f"GT image: {gt_name} ({gt_path})"
            )

        # Read images and convert to RGB format uniformly
        try:
            with Image.open(haze_path) as img:
                haze_img = img.convert('RGB')  # Unified RGB format to avoid grayscale issues
            with Image.open(gt_path) as img:
                gt_img = img.convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to read image files: {haze_path} or {gt_path}\nError: {str(e)}") from e

        # Validate image size consistency
        if haze_img.size != gt_img.size:
            raise ValueError(
                f"Image size mismatch!\n"
                f"Hazy image {haze_name}: {haze_img.size} | GT image {gt_name}: {gt_img.size}"
            )

        # Normalize pixel values to [0,1] and convert to numpy array
        haze_np = np.array(haze_img, dtype=np.float32) / 255.0
        gt_np = np.array(gt_img, dtype=np.float32) / 255.0

        # Validate pixel value range (prevent abnormal values)
        if not (np.all(haze_np >= 0 - 1e-5) and np.all(haze_np <= 1 + 1e-5)):
            raise ValueError(f"Abnormal pixel values in hazy image {haze_name}! Should be in [0,255] range")
        if not (np.all(gt_np >= 0 - 1e-5) and np.all(gt_np <= 1 + 1e-5)):
            raise ValueError(f"Abnormal pixel values in GT image {gt_name}! Should be in [0,255] range")

        # Convert numpy array to tensor and adjust dimension (H,W,C) → (C,H,W)
        haze_tensor = torch.from_numpy(haze_np.transpose(2, 0, 1))
        gt_tensor = torch.from_numpy(gt_np.transpose(2, 0, 1))

        return haze_tensor, gt_tensor

    def __len__(self):
        """Return total number of samples in the dataset"""
        return self.total_samples

    def __getitem__(self, idx):
        """Get sample by index (supports cache and transform with consistent random params)"""
        # Handle negative index (compatible with Subset and other slicing scenarios)
        if idx < 0:
            idx = self.total_samples + idx
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Sample index {idx} out of bounds (Total samples: {self.total_samples})")

        # Load sample from cache if exists
        if idx in self._cache:
            return self._cache[idx]

        # Load and preprocess sample
        haze_tensor, gt_tensor = self._load_and_process(idx)

        # Apply transform (Ensure haze and GT use identical random parameters)
        if self.transform is not None:
            if isinstance(self.transform, transforms.Compose):
                # Save random state for consistent augmentation (same crop/flip for pair images)
                rng_state = torch.get_rng_state()
                haze_tensor = self.transform(haze_tensor)
                torch.set_rng_state(rng_state)  # Restore random state
                gt_tensor = self.transform(gt_tensor)
            else:
                haze_tensor = self.transform(haze_tensor)
                gt_tensor = self.transform(gt_tensor)

        # Dynamic cache update (only for non-preload mode)
        if not self.preload_to_memory:
            # Evict oldest sample when cache is full
            if len(self._cache) >= self.cache_size:
                oldest_idx = next(iter(self._cache.keys()))  # Get first inserted index
                del self._cache[oldest_idx]
            self._cache[idx] = (haze_tensor, gt_tensor)

        return haze_tensor, gt_tensor