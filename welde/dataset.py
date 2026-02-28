"""
dataset.py — Dataset loading for WELDE experiments.

Reads YOLO-format bounding box labels, extracts object patches from
full MRI images, and provides PyTorch Datasets with augmentation.
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from welde.config import (
    DATA_ROOT, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, SEED,
)

# Derive image/label directories from DATA_ROOT
IMAGE_DIRS = {
    "train": DATA_ROOT / "train",
    "val":   DATA_ROOT / "val",
    "test":  DATA_ROOT / "test",
}
LABEL_DIRS = {
    "train": DATA_ROOT / "labels" / "train",
    "val":   DATA_ROOT / "labels" / "val",
    "test":  DATA_ROOT / "labels" / "test",
}


def _parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file. Returns list of (class_id, cx, cy, w, h)."""
    entries = []
    if not label_path.exists():
        return entries
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            # YOLO bbox format — use first 4 coords after class_id
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            entries.append((cls_id, cx, cy, w, h))
    return entries


def _crop_patch(
    img: Image.Image,
    cx: float, cy: float, w: float, h: float,
    pad_ratio: float = 0.1
) -> Image.Image:
    """Crop a patch from an image using normalised YOLO coords.

    Adds a small padding (pad_ratio) around the bbox to capture context.
    """
    iw, ih = img.size
    # Convert normalised to pixel
    px_cx, px_cy = cx * iw, cy * ih
    px_w, px_h = w * iw, h * ih
    # Add contextual padding
    px_w *= (1.0 + pad_ratio)
    px_h *= (1.0 + pad_ratio)
    x1 = max(0, int(px_cx - px_w / 2))
    y1 = max(0, int(px_cy - px_h / 2))
    x2 = min(iw, int(px_cx + px_w / 2))
    y2 = min(ih, int(px_cy + px_h / 2))
    # Ensure minimum size
    if x2 - x1 < 4:
        x2 = min(iw, x1 + 4)
    if y2 - y1 < 4:
        y2 = min(ih, y1 + 4)
    return img.crop((x1, y1, x2, y2))


class PatchDataset(Dataset):
    """Dataset of object patches extracted from MRI images."""

    def __init__(
        self,
        split: str,
        transform: Optional[transforms.Compose] = None,
        return_meta: bool = False,
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: torchvision transforms; defaults applied if None
            return_meta: if True, returns (patch, label, metadata_dict)
        """
        self.split = split
        self.return_meta = return_meta
        self.transform = transform or self._default_transform(split)

        img_dir = IMAGE_DIRS[split]
        lbl_dir = LABEL_DIRS[split]

        self.samples: list[dict] = []
        image_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

        for img_path in image_files:
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            entries = _parse_yolo_label(lbl_path)
            for cls_id, cx, cy, w, h in entries:
                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    continue
                self.samples.append({
                    "img_path": str(img_path),
                    "cls_id": cls_id,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "img_stem": stem,
                })

        # Compute class counts
        labels = [s["cls_id"] for s in self.samples]
        self.class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for c in labels:
            self.class_counts[c] += 1

    @staticmethod
    def _default_transform(split: str) -> transforms.Compose:
        if split == "train":
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    # Cache images to speed up repeated patch extraction
    _img_cache: dict = {}

    def _load_image(self, path: str) -> Image.Image:
        if path not in self._img_cache:
            img = Image.open(path).convert("RGB")
            self._img_cache[path] = img
        return self._img_cache[path].copy()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load_image(s["img_path"])
        patch = _crop_patch(img, s["cx"], s["cy"], s["w"], s["h"])
        patch = self.transform(patch)
        label = s["cls_id"]
        if self.return_meta:
            return patch, label, s
        return patch, label


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Build train/val/test DataLoaders and return class counts from training set."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = PatchDataset("train")
    val_ds   = PatchDataset("val")
    test_ds  = PatchDataset("test", return_meta=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[DATA] Train: {len(train_ds)} patches")
    print(f"[DATA] Val:   {len(val_ds)} patches")
    print(f"[DATA] Test:  {len(test_ds)} patches")
    print(f"[DATA] Class counts (train): {train_ds.class_counts}")

    return train_loader, val_loader, test_loader, train_ds.class_counts


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader, test_loader, class_counts = build_dataloaders(batch_size=8)
    for batch in train_loader:
        patches, labels = batch
        print(f"Batch shape: {patches.shape}, labels: {labels}")
        break
    print(f"Class counts: {class_counts}")

