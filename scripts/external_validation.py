"""
external_validation.py — Cross-domain validation of WELDE on ISIC 2018 (DermaMNIST).

Downloads the DermaMNIST (HAM10000 / ISIC 2018) skin lesion dataset,
extracts frozen ResNet-50 features, and runs stratified 5-fold CV comparing:
  - CE (single-head baseline)
  - LDAM (best single-loss baseline on spinal data)
  - CE_ensemble (architecture-matched 4-head control)
  - WELDE (full framework: CE + FL + CBL + LDAM, per-head adapters)

Matching spinal protocol:
  - Raw CE loss for all CE-based methods (no label smoothing)
  - LDAM_C auto-tuned per fold via 5-epoch validation scan
  - Early stopping on validation macro-F1 (patience=10)
  - Grad clipping (max_norm=5.0)
  - Dropout (0.3 adapter, 0.2 heads — unchanged from spinal)
  - Weight decay 1e-4 (AdamW)

Outputs:
  - results/external_validation_results.json
  - results/tables/external_cv_summary.csv
  - results/figures/fig_external_validation.pdf
"""
import os, sys, json, time, warnings
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    precision_score, recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from welde.model import MultiHeadAdapterOnly, HeadOnly
from welde.losses import (
    WELDELoss, LDAMLoss, get_loss_fn,
)
from welde.trainer import set_seed, GradientTracker

# ─────────────── Configuration ───────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = Path(os.environ.get("WELDE_OUTPUT_ROOT", PROJECT_ROOT / "results"))
FIGURE_DIR  = OUTPUT_ROOT / "figures"
TABLE_DIR   = OUTPUT_ROOT / "tables"
for d in [OUTPUT_ROOT, FIGURE_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NUM_FOLDS       = 5
NUM_EPOCHS      = 50
BATCH_SIZE      = 64
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
WARMUP_EPOCHS   = 3
PATIENCE        = 10        # early stopping patience
IMG_SIZE        = 224
SEED            = 42
NUM_CLASSES_EXT = 7         # DermaMNIST: 7 classes

# WELDE hyperparameters (same as spinal experiments)
WELDE_ALPHA     = 0.01
WELDE_ETA       = 0.1
WELDE_S         = 0.1
LDAM_C_GRID     = [0.1, 0.3, 0.5, 1.0]   # auto-tuned per fold
FOCAL_GAMMA     = 2.0
CBL_BETA        = 0.999

# DermaMNIST class info
CLASS_NAMES = [
    "AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"
]

def P(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════
# 1. Dataset: DermaMNIST (ISIC 2018 / HAM10000)
# ═══════════════════════════════════════════════════════════════

def load_dermamnist():
    """Download and load DermaMNIST. Returns images (N,3,28,28) and labels (N,)."""
    from medmnist import DermaMNIST

    data_dir = OUTPUT_ROOT / "dermamnist_data"
    data_dir.mkdir(exist_ok=True)

    all_images = []
    all_labels = []

    for split in ["train", "val", "test"]:
        ds = DermaMNIST(
            split=split,
            download=True,
            root=str(data_dir),
            size=224,       # 224×224 for ResNet-50 compatibility
        )
        # ds.imgs: (N, 224, 224, 3) uint8
        # ds.labels: (N, 1) int
        all_images.append(ds.imgs)
        all_labels.append(ds.labels.squeeze())

    images = np.concatenate(all_images, axis=0)   # (10015, 224, 224, 3)
    labels = np.concatenate(all_labels, axis=0)    # (10015,)

    P(f"[DermaMNIST] Loaded {len(images)} images, {NUM_CLASSES_EXT} classes")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        P(f"  Class {u} ({CLASS_NAMES[u]}): {c} ({100*c/len(labels):.1f}%)")
    rho = counts.max() / counts.min()
    P(f"  Imbalance ratio ρ = {rho:.1f}")

    return images, labels


# ═══════════════════════════════════════════════════════════════
# 2. Feature Extraction (frozen ResNet-50)
# ═══════════════════════════════════════════════════════════════

def extract_resnet50_features(images: np.ndarray) -> np.ndarray:
    """Extract 2048-d features from frozen ResNet-50 for all images."""
    cache_path = OUTPUT_ROOT / "dermamnist_features_224.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        P(f"[FEATURES] Loaded cached features: {data['features'].shape}")
        return data["features"]

    P("[FEATURES] Extracting ResNet-50 features (this may take a few minutes)...")
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
    backbone.eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    features = []
    bs = 128
    n = len(images)
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = images[i:i+bs]
            # (B, H, W, 3) uint8 → (B, 3, H, W) float [0,1]
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            tensor = transform(tensor).to(DEVICE)
            feat = backbone(tensor).flatten(1)  # (B, 2048)
            features.append(feat.cpu().numpy())
            if (i // bs) % 10 == 0:
                P(f"  ... {i+len(batch)}/{n}")

    features = np.concatenate(features, axis=0)
    np.savez_compressed(cache_path, features=features)
    P(f"[FEATURES] Extracted and cached: {features.shape}")
    return features


# ═══════════════════════════════════════════════════════════════
# 3. Training with Early Stopping
# ═══════════════════════════════════════════════════════════════

def _quick_search_ldam_c(
    train_loader, val_loader, class_counts,
    num_classes=NUM_CLASSES_EXT, seed=SEED,
):
    """Quick 5-epoch scan to pick best LDAM_C for this fold."""
    best_c, best_f1 = LDAM_C_GRID[0], -1
    for c in LDAM_C_GRID:
        set_seed(seed)
        model = HeadOnly(feat_dim=2048, num_classes=num_classes).to(DEVICE)
        loss_fn = LDAMLoss(class_counts, C=c).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(5):
            model.train()
            for feats, labels in train_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                logits, _ = model(feats)
                loss_fn(logits, labels).backward()
                opt.step()
        f1 = _eval_f1(model, val_loader, is_multi=False)
        if f1 > best_f1:
            best_f1, best_c = f1, c
        del model, loss_fn, opt
        torch.cuda.empty_cache()
    return best_c


def train_single_head_ext(
    train_loader, val_loader, class_counts, loss_name,
    num_classes=NUM_CLASSES_EXT, num_epochs=NUM_EPOCHS,
    patience=PATIENCE, seed=SEED, ldam_c=None,
):
    """Train single-head baseline with early stopping."""
    set_seed(seed)
    model = HeadOnly(feat_dim=2048, num_classes=num_classes).to(DEVICE)

    if loss_name == "CE":
        loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    elif loss_name == "LDAM":
        c = ldam_c if ldam_c is not None else LDAM_C_GRID[0]
        loss_fn = LDAMLoss(class_counts, C=c).to(DEVICE)
    else:
        loss_fn = get_loss_fn(loss_name, class_counts).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = LambdaLR(opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / max(WARMUP_EPOCHS, 1)))
    cosine = CosineAnnealingLR(opt, T_max=max(num_epochs - WARMUP_EPOCHS, 1))
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_f1 = -1
    best_state = None
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

        # Validate
        val_f1 = _eval_f1(model, val_loader, is_multi=False)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def train_ce_ensemble_ext(
    train_loader, val_loader, class_counts,
    num_classes=NUM_CLASSES_EXT, num_epochs=NUM_EPOCHS,
    patience=PATIENCE, seed=SEED, num_heads=4,
):
    """Train architecture-matched CE ensemble with early stopping.
    Uses raw CE (no label smoothing) to match the spinal protocol."""
    set_seed(seed)
    model = MultiHeadAdapterOnly(
        feat_dim=2048, num_classes=num_classes, num_heads=num_heads,
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = LambdaLR(opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / max(WARMUP_EPOCHS, 1)))
    cosine = CosineAnnealingLR(opt, T_max=max(num_epochs - WARMUP_EPOCHS, 1))
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_f1 = -1
    best_state = None
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits_list, _ = model(feats)
            loss = sum(loss_fn(logits, labels) for logits in logits_list) / num_heads
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

        val_f1 = _eval_f1(model, val_loader, is_multi=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def train_welde_ext(
    train_loader, val_loader, class_counts,
    num_classes=NUM_CLASSES_EXT, num_epochs=NUM_EPOCHS,
    patience=PATIENCE, seed=SEED, ldam_c=None,
):
    """Train WELDE with early stopping. Uses raw CE (no label smoothing)
    to match the spinal protocol."""
    set_seed(seed)
    num_heads = 4
    model = MultiHeadAdapterOnly(
        feat_dim=2048, num_classes=num_classes, num_heads=num_heads,
    ).to(DEVICE)

    c = ldam_c if ldam_c is not None else LDAM_C_GRID[0]
    criterion = WELDELoss(
        class_counts,
        loss_names=["CE", "FL", "CBL", "LDAM"],
        alpha=WELDE_ALPHA, eta=WELDE_ETA, lambda_div=0.0,
        s=WELDE_S, delta_thr=2.0,
        gamma=FOCAL_GAMMA, beta=CBL_BETA, ldam_C=c,
        use_ema=True, use_diversity=False,
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    warmup = LambdaLR(opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / max(WARMUP_EPOCHS, 1)))
    cosine = CosineAnnealingLR(opt, T_max=max(num_epochs - WARMUP_EPOCHS, 1))
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_f1 = -1
    best_state = None
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits_list, _ = model(feats)
            loss, _ = criterion(logits_list, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

        val_f1 = _eval_f1(model, val_loader, is_multi=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def _eval_f1(model, loader, is_multi=False):
    """Quick macro-F1 for validation."""
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(DEVICE)
            if is_multi:
                logits_list, _ = model(feats)
                probs = torch.stack([F.softmax(l, dim=1) for l in logits_list]).mean(0)
            else:
                logits, _ = model(feats)
                probs = F.softmax(logits, dim=1)
            preds.append(probs.argmax(1).cpu())
            labels_all.append(labels)
    preds = torch.cat(preds).numpy()
    labels_all = torch.cat(labels_all).numpy()
    return f1_score(labels_all, preds, average="macro", zero_division=0)


# ═══════════════════════════════════════════════════════════════
# 4. Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_on_fold(model, test_loader, is_multi=False, num_classes=NUM_CLASSES_EXT):
    """Full evaluation on one fold's held-out test set."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(DEVICE)
            if is_multi:
                logits_list, _ = model(feats)
                probs = torch.stack([F.softmax(l, dim=1) for l in logits_list]).mean(0)
            else:
                logits, _ = model(feats)
                probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(probs.argmax(1).cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Per-class AP
    one_hot = np.zeros((len(all_labels), num_classes))
    for i, c in enumerate(all_labels):
        one_hot[i, c] = 1

    per_class_ap = {}
    for k in range(num_classes):
        if one_hot[:, k].sum() > 0:
            per_class_ap[k] = float(average_precision_score(one_hot[:, k], all_probs[:, k]))
        else:
            per_class_ap[k] = 0.0

    mAP = float(np.mean(list(per_class_ap.values())))

    # Identify tail classes (bottom-3 by count in full dataset)
    # Will be set from outside; use all classes for now
    tail_ap = float(np.mean([per_class_ap.get(k, 0) for k in TAIL_CLASSES]))

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "mAP": mAP,
        "mAP_tail": tail_ap,
        "per_class_ap": {CLASS_NAMES[k]: v for k, v in per_class_ap.items()},
    }


# Tail classes for DermaMNIST: AKIEC (0), DF (3), VASC (6) — the 3 rarest
TAIL_CLASSES = [0, 3, 6]


# ═══════════════════════════════════════════════════════════════
# 5. Stratified K-Fold Cross-Validation
# ═══════════════════════════════════════════════════════════════

def run_cross_validation(features, labels):
    """Run stratified 5-fold CV for all methods."""
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_results = {name: [] for name in ["CE", "LDAM", "CE_ensemble", "WELDE"]}
    features_t = torch.from_numpy(features).float()
    labels_t = torch.from_numpy(labels).long()

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(features, labels)):
        P(f"\n{'='*60}")
        P(f"FOLD {fold_idx + 1}/{NUM_FOLDS}")
        P(f"{'='*60}")

        # Split train_val into train (80%) and val (20%)
        np.random.seed(SEED + fold_idx)
        n_tv = len(train_val_idx)
        perm = np.random.permutation(n_tv)
        n_val = max(1, int(0.2 * n_tv))
        val_sub_idx = train_val_idx[perm[:n_val]]
        train_sub_idx = train_val_idx[perm[n_val:]]

        # Build loaders
        train_loader = DataLoader(
            TensorDataset(features_t[train_sub_idx], labels_t[train_sub_idx]),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(features_t[val_sub_idx], labels_t[val_sub_idx]),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(features_t[test_idx], labels_t[test_idx]),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        # Class counts from training fold
        train_labels = labels[train_sub_idx]
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES_EXT).astype(np.float64)
        P(f"  Train: {len(train_sub_idx)}, Val: {len(val_sub_idx)}, Test: {len(test_idx)}")
        P(f"  Class counts: {class_counts.astype(int).tolist()}")
        P(f"  ρ = {class_counts.max()/class_counts.min():.1f}")

        # Auto-tune LDAM_C for this fold's class distribution
        P(f"  Auto-tuning LDAM_C from {LDAM_C_GRID} ...")
        best_c = _quick_search_ldam_c(train_loader, val_loader, class_counts, seed=SEED + fold_idx)
        P(f"  Best LDAM_C = {best_c}")

        fold_methods = {
            "CE":          lambda tl, vl, cc, s: train_single_head_ext(tl, vl, cc, "CE", seed=s),
            "LDAM":        lambda tl, vl, cc, s: train_single_head_ext(tl, vl, cc, "LDAM", seed=s, ldam_c=best_c),
            "CE_ensemble": lambda tl, vl, cc, s: train_ce_ensemble_ext(tl, vl, cc, seed=s),
            "WELDE":       lambda tl, vl, cc, s: train_welde_ext(tl, vl, cc, seed=s, ldam_c=best_c),
        }

        for name, train_fn in fold_methods.items():
            P(f"\n  --- Training {name} ---")
            t0 = time.time()
            is_multi = name in ("CE_ensemble", "WELDE")
            model = train_fn(train_loader, val_loader, class_counts, SEED + fold_idx)
            elapsed = time.time() - t0

            metrics = evaluate_on_fold(model, test_loader, is_multi=is_multi)
            metrics["train_time"] = elapsed
            metrics["fold"] = fold_idx + 1
            if name in ("LDAM", "WELDE"):
                metrics["ldam_c"] = best_c
            all_results[name].append(metrics)

            P(f"  {name}: mAP={metrics['mAP']:.3f}  mAP_tail={metrics['mAP_tail']:.3f}  "
              f"F1={metrics['macro_f1']:.3f}  time={elapsed:.1f}s")

        # Free GPU memory
        torch.cuda.empty_cache()

    return all_results


# ═══════════════════════════════════════════════════════════════
# 6. Summary & Figure Generation
# ═══════════════════════════════════════════════════════════════

def summarise_results(all_results):
    """Compute mean ± std across folds and save outputs."""
    P(f"\n{'='*60}")
    P("CROSS-VALIDATION SUMMARY (DermaMNIST / ISIC 2018)")
    P(f"{'='*60}")

    summary = {}
    metrics_keys = ["mAP", "mAP_tail", "macro_f1", "accuracy"]

    for name, fold_results in all_results.items():
        summary[name] = {}
        P(f"\n{name}:")
        for key in metrics_keys:
            vals = [r[key] for r in fold_results]
            mean = np.mean(vals)
            std = np.std(vals)
            summary[name][key] = {"mean": float(mean), "std": float(std), "values": vals}
            P(f"  {key}: {mean:.3f} ± {std:.3f}")

        # Per-class AP means
        per_class_means = {}
        for cls_name in CLASS_NAMES:
            vals = [r["per_class_ap"].get(cls_name, 0) for r in fold_results]
            per_class_means[cls_name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        summary[name]["per_class_ap"] = per_class_means

        # Training time
        times = [r["train_time"] for r in fold_results]
        summary[name]["train_time"] = {"mean": float(np.mean(times)), "std": float(np.std(times))}

    # Save raw results
    serialisable = {}
    for name, fold_results in all_results.items():
        serialisable[name] = []
        for r in fold_results:
            sr = {k: v for k, v in r.items() if k not in ("all_probs", "all_preds", "all_labels")}
            serialisable[name].append(sr)

    with open(OUTPUT_ROOT / "external_validation_results.json", "w") as f:
        json.dump({"raw": serialisable, "summary": summary}, f, indent=2)
    P(f"\nSaved: {OUTPUT_ROOT / 'external_validation_results.json'}")

    # Save CSV summary
    import csv
    csv_path = TABLE_DIR / "external_cv_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "mAP (mean±std)", "mAP_tail (mean±std)",
                         "F1 (mean±std)", "Accuracy (mean±std)"])
        for name in ["CE", "LDAM", "CE_ensemble", "WELDE"]:
            s = summary[name]
            writer.writerow([
                name,
                f"{s['mAP']['mean']:.3f}±{s['mAP']['std']:.3f}",
                f"{s['mAP_tail']['mean']:.3f}±{s['mAP_tail']['std']:.3f}",
                f"{s['macro_f1']['mean']:.3f}±{s['macro_f1']['std']:.3f}",
                f"{s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f}",
            ])
    P(f"Saved: {csv_path}")

    return summary


def generate_figure(summary):
    """Generate grouped bar chart for external validation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        P("[WARN] matplotlib not available — skipping figure generation")
        return

    methods = ["CE", "LDAM", "CE_ensemble", "WELDE"]
    display_names = ["CE", "LDAM", r"CE$_{\mathrm{ens}}$", "WELDE"]
    metrics = ["mAP", "mAP_tail", "macro_f1"]
    metric_labels = ["mAP", r"mAP$_{\mathrm{tail}}$", "Macro F1"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#8da0cb", "#e78ac3", "#66c2a5", "#fc8d62"]

    for i, (method, color) in enumerate(zip(methods, colors)):
        means = [summary[method][m]["mean"] for m in metrics]
        stds  = [summary[method][m]["std"] for m in metrics]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=display_names[i],
                      color=color, edgecolor="black", linewidth=0.5,
                      capsize=3, error_kw={"linewidth": 0.8})

    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_title("External Validation: DermaMNIST (ISIC 2018) — 5-Fold CV", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_path = FIGURE_DIR / "fig_external_validation.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    P(f"Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    P("=" * 60)
    P("WELDE External Validation — DermaMNIST (ISIC 2018)")
    P(f"Device: {DEVICE}")
    P(f"Folds: {NUM_FOLDS}, Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}")
    P(f"LDAM_C grid: {LDAM_C_GRID} (auto-tuned per fold)")
    P("=" * 60)

    # 1. Load dataset
    images, labels = load_dermamnist()

    # 2. Extract features
    features = extract_resnet50_features(images)

    # 3. Cross-validation
    all_results = run_cross_validation(features, labels)

    # 4. Summarise
    summary = summarise_results(all_results)

    # 5. Figure
    generate_figure(summary)

    P("\n" + "=" * 60)
    P("EXTERNAL VALIDATION COMPLETE")
    P("=" * 60)

    return summary


if __name__ == "__main__":
    main()

