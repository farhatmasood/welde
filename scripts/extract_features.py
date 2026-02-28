"""
extract_features.py — Pre-extract ResNet-50 backbone features for all patches.

Uses the full ResNet-50 backbone (frozen, eval mode) to convert each
cropped patch into a 2048-dim feature vector. Augmented copies are
generated for training data to preserve data diversity.

Saves:
    results/features/train_features.npy  (N_train × n_aug, 2048)
    results/features/train_labels.npy    (N_train × n_aug,)
    results/features/val_features.npy    (N_val, 2048)
    results/features/val_labels.npy      (N_val,)
    results/features/test_features.npy   (N_test, 2048)
    results/features/test_labels.npy     (N_test,)
    results/features/test_meta.npy       (N_test,) dict per sample
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from pathlib import Path
import sys

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from welde.config import OUTPUT_ROOT, BATCH_SIZE, DEVICE, SEED
from welde.dataset import PatchDataset


FEATURE_DIR = OUTPUT_ROOT / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

NUM_AUG_COPIES = 5  # Number of augmented copies for training


def build_backbone(device: str = DEVICE) -> nn.Module:
    """Build frozen ResNet-50 backbone that extracts 2048-dim features."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove the final FC layer; keep up to avgpool
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False
    return feature_extractor.to(device)


@torch.no_grad()
def extract_split(
    backbone: nn.Module,
    split: str,
    n_copies: int = 1,
    device: str = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for a dataset split.

    Args:
        backbone: frozen ResNet-50 feature extractor
        split: 'train', 'val', or 'test'
        n_copies: number of augmented copies (>1 applies random augmentation)
        device: torch device

    Returns:
        features: (N * n_copies, 2048) array
        labels: (N * n_copies,) array
    """
    all_features = []
    all_labels = []

    for copy_idx in range(n_copies):
        ds = PatchDataset(split, return_meta=(split == "test"))
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        copy_feats = []
        copy_labels = []
        desc = f"Extracting {split} (copy {copy_idx + 1}/{n_copies})"

        for batch in tqdm(loader, desc=desc, leave=False):
            if split == "test":
                patches, labels, _ = batch
            else:
                patches, labels = batch
            patches = patches.to(device)
            feats = backbone(patches).flatten(1)  # (B, 2048)
            copy_feats.append(feats.cpu().numpy())
            copy_labels.append(labels.numpy())

        all_features.append(np.concatenate(copy_feats, axis=0))
        all_labels.append(np.concatenate(copy_labels, axis=0))

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def main():
    print("=" * 60)
    print("FEATURE EXTRACTION (ResNet-50 backbone)")
    print(f"Device: {DEVICE}")
    print(f"Augmentation copies for train: {NUM_AUG_COPIES}")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    backbone = build_backbone()

    # Training: extract with augmentation
    print("\n--- Training set ---")
    train_feats, train_labels = extract_split(
        backbone, "train", n_copies=NUM_AUG_COPIES
    )
    np.save(FEATURE_DIR / "train_features.npy", train_feats)
    np.save(FEATURE_DIR / "train_labels.npy", train_labels)
    print(f"  Train features: {train_feats.shape}")

    # Validation: single pass, no augmentation
    print("\n--- Validation set ---")
    val_feats, val_labels = extract_split(backbone, "val", n_copies=1)
    np.save(FEATURE_DIR / "val_features.npy", val_feats)
    np.save(FEATURE_DIR / "val_labels.npy", val_labels)
    print(f"  Val features: {val_feats.shape}")

    # Test: single pass, no augmentation
    print("\n--- Test set ---")
    test_feats, test_labels = extract_split(backbone, "test", n_copies=1)
    np.save(FEATURE_DIR / "test_features.npy", test_feats)
    np.save(FEATURE_DIR / "test_labels.npy", test_labels)
    print(f"  Test features: {test_feats.shape}")

    # Also save class counts from training data (before augmentation)
    ds_train = PatchDataset("train")
    np.save(FEATURE_DIR / "class_counts.npy", ds_train.class_counts)
    print(f"  Class counts: {ds_train.class_counts}")

    print("\n✓ Feature extraction complete.")
    print(f"  Saved to: {FEATURE_DIR}")


if __name__ == "__main__":
    main()

