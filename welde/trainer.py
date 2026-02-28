"""
trainer.py — Training and evaluation loops for WELDE experiments.

Supports feature-based training (HeadOnly / MultiHeadAdapterOnly on
pre-extracted ResNet-50 features). Logs gradient magnitudes, weight
evolution, and metrics.
"""
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score,
)

from welde.config import DEVICE, LR, WEIGHT_DECAY, NUM_EPOCHS, NUM_CLASSES, SEED, BATCH_SIZE, WARMUP_EPOCHS


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ═══════════════════════ Gradient Tracker ══════════════════════

class GradientTracker:
    """Records per-head gradient magnitudes."""

    def __init__(self):
        self.history: dict[str, list[float]] = defaultdict(list)

    def record_head_grads(self, model):
        """Record per-head gradient norms."""
        heads = model.heads if hasattr(model, 'heads') else [model.head]
        for j, head in enumerate(heads):
            total = 0.0
            count = 0
            for p in head.parameters():
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
                    count += 1
            if count > 0:
                self.history[f"head_{j}"].append(np.sqrt(total))


# ═══════════════════ Feature DataLoader ═══════════════════════

def build_feature_loaders(feature_dir, batch_size: int = BATCH_SIZE):
    """Build DataLoaders from pre-extracted features."""
    train_feats = torch.from_numpy(np.load(feature_dir / "train_features.npy"))
    train_labels = torch.from_numpy(np.load(feature_dir / "train_labels.npy")).long()
    val_feats = torch.from_numpy(np.load(feature_dir / "val_features.npy"))
    val_labels = torch.from_numpy(np.load(feature_dir / "val_labels.npy")).long()
    test_feats = torch.from_numpy(np.load(feature_dir / "test_features.npy"))
    test_labels = torch.from_numpy(np.load(feature_dir / "test_labels.npy")).long()
    class_counts = np.load(feature_dir / "class_counts.npy")

    train_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_feats, val_labels),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_feats, test_labels),
        batch_size=batch_size, shuffle=False,
    )

    print(f"[FEATURES] Train: {len(train_feats)} samples")
    print(f"[FEATURES] Val:   {len(val_feats)} samples")
    print(f"[FEATURES] Test:  {len(test_feats)} samples")
    print(f"[FEATURES] Class counts: {class_counts}")

    return train_loader, val_loader, test_loader, class_counts


# ══════════════════ Single-Head Training ══════════════════════

def train_single_head(
    train_loader,
    val_loader,
    class_counts: np.ndarray,
    loss_name: str,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    device: str = DEVICE,
    verbose: bool = True,
):
    """Train a single-head classifier on pre-extracted features."""
    from welde.model import HeadOnly
    from welde.losses import get_loss_fn

    set_seed()
    model = HeadOnly().to(device)
    loss_fn = get_loss_fn(loss_name, class_counts).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=num_epochs)

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": [], "epoch_time": [],
    }

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimiser.zero_grad()
            logits, _ = model(feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        val_loss, val_acc, val_f1 = _eval_single(model, val_loader, loss_fn, device)
        history["train_loss"].append(running_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["epoch_time"].append(time.time() - t0)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [{loss_name}] Epoch {epoch+1}/{num_epochs} — "
                  f"loss={running_loss/max(n_batches,1):.4f}  "
                  f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

    return model, history


def train_ce_ensemble(
    train_loader,
    val_loader,
    class_counts: np.ndarray,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    device: str = DEVICE,
    verbose: bool = True,
    num_heads: int = 4,
    seed: int = None,
):
    """Train a multi-head model with CE loss on all heads (architecture-matched baseline).

    This isolates the architecture contribution (multi-head + adapters) from
    the loss-diversity contribution of WELDE.
    """
    from welde.config import WARMUP_EPOCHS
    from welde.model import MultiHeadAdapterOnly

    set_seed(seed if seed is not None else SEED)
    model = MultiHeadAdapterOnly(num_heads=num_heads).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = LambdaLR(optimiser, lr_lambda=lambda ep: min(1.0, (ep + 1) / max(WARMUP_EPOCHS, 1)))
    cosine_scheduler = CosineAnnealingLR(optimiser, T_max=max(num_epochs - WARMUP_EPOCHS, 1))
    scheduler = SequentialLR(optimiser, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": [], "epoch_time": [],
    }

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimiser.zero_grad()
            logits_list, _ = model(feats)
            loss = sum(ce_loss(logits, labels) for logits in logits_list) / num_heads
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits_list, _ = model(feats)
                probs = torch.stack([F.softmax(l, dim=1) for l in logits_list]).mean(0)
                val_preds.append(probs.argmax(1).cpu())
                val_labels_list.append(labels.cpu())
        val_preds = torch.cat(val_preds).numpy()
        val_labels_arr = torch.cat(val_labels_list).numpy()
        val_acc = accuracy_score(val_labels_arr, val_preds)
        val_f1 = f1_score(val_labels_arr, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(running_loss / max(n_batches, 1))
        history["val_loss"].append(0.0)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["epoch_time"].append(time.time() - t0)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [CE_ensemble] Epoch {epoch+1}/{num_epochs} — "
                  f"loss={running_loss/max(n_batches,1):.4f}  "
                  f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

    return model, history


def _eval_single(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits, _ = model(feats)
            total_loss += loss_fn(logits, labels).item()
            n += 1
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / max(n, 1), acc, f1


# ══════════════════ WELDE Training ════════════════════════════

def train_welde(
    train_loader,
    val_loader,
    class_counts: np.ndarray,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    device: str = DEVICE,
    verbose: bool = True,
    alpha: float = None,
    eta: float = None,
    lambda_div: float = None,
    s: float = None,
    delta_thr: float = None,
    use_ema: bool = True,
    use_diversity: bool = False,
    active_heads: list = None,
    use_adapters: bool = True,
    loss_names: list[str] | None = None,
    ldam_c: float = None,
    focal_gamma: float = None,
    seed: int = None,
):
    """Train the multi-head WELDE model on pre-extracted features."""
    from welde.config import (
        WELDE_ALPHA, WELDE_ETA, WELDE_LAMBDA, WELDE_S, WELDE_DELTA_THR,
        FOCAL_GAMMA, CBL_BETA, LDAM_C, WELDE_LOSS_COMPONENTS, WARMUP_EPOCHS,
    )
    from welde.model import MultiHeadAdapterOnly, MultiHeadOnly
    from welde.losses import WELDELoss

    set_seed(seed if seed is not None else SEED)

    alpha     = alpha     if alpha     is not None else WELDE_ALPHA
    eta       = eta       if eta       is not None else WELDE_ETA
    lambda_div = lambda_div if lambda_div is not None else WELDE_LAMBDA
    s         = s         if s         is not None else WELDE_S
    delta_thr = delta_thr if delta_thr is not None else WELDE_DELTA_THR
    ldam_c_val = ldam_c if ldam_c is not None else LDAM_C
    focal_gamma_val = focal_gamma if focal_gamma is not None else FOCAL_GAMMA
    if loss_names is None:
        loss_names = WELDE_LOSS_COMPONENTS

    if use_adapters:
        model = MultiHeadAdapterOnly(num_heads=len(loss_names)).to(device)
    else:
        model = MultiHeadOnly(num_heads=len(loss_names)).to(device)
    criterion = WELDELoss(
        class_counts, loss_names=loss_names,
        alpha=alpha, eta=eta, lambda_div=lambda_div,
        s=s, delta_thr=delta_thr, gamma=focal_gamma_val, beta=CBL_BETA,
        ldam_C=ldam_c_val, use_ema=use_ema, use_diversity=use_diversity,
    ).to(device)

    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    # Warmup + cosine annealing schedule
    warmup_scheduler = LambdaLR(optimiser, lr_lambda=lambda ep: min(1.0, (ep + 1) / max(WARMUP_EPOCHS, 1)))
    cosine_scheduler = CosineAnnealingLR(optimiser, T_max=max(num_epochs - WARMUP_EPOCHS, 1))
    scheduler = SequentialLR(optimiser, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    grad_tracker = GradientTracker()

    num_heads = len(loss_names)

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": [], "epoch_time": [],
        "weights": [], "ema": [], "diversity": [],
        "raw_losses": [], "norm_losses": [],
    }

    if active_heads is None:
        active_heads = [True] * num_heads

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        epoch_weights, epoch_ema, epoch_div = [], [], []
        epoch_raw, epoch_norm = [], []

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimiser.zero_grad()
            logits_list, _ = model(feats)

            for j in range(num_heads):
                if not active_heads[j]:
                    logits_list[j] = logits_list[j].detach() * 0

            loss, info = criterion(logits_list, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            grad_tracker.record_head_grads(model)
            optimiser.step()

            running_loss += loss.item()
            n_batches += 1
            epoch_weights.append(info["weights"])
            epoch_ema.append(info["ema"])
            epoch_div.append(info["diversity"])
            epoch_raw.append(info["raw_losses"])
            epoch_norm.append(info["norm_losses"])

        scheduler.step()

        val_loss, val_acc, val_f1 = _eval_welde(model, val_loader, criterion, device)
        history["train_loss"].append(running_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["epoch_time"].append(time.time() - t0)
        history["weights"].append(np.mean(epoch_weights, axis=0).tolist())
        history["ema"].append(np.mean(epoch_ema, axis=0).tolist())
        history["diversity"].append(float(np.mean(epoch_div)))
        history["raw_losses"].append(np.mean(epoch_raw, axis=0).tolist())
        history["norm_losses"].append(np.mean(epoch_norm, axis=0).tolist())

        if verbose and (epoch + 1) % 5 == 0:
            w_str = ", ".join(f"{v:.3f}" for v in history["weights"][-1])
            print(f"  [WELDE] Epoch {epoch+1}/{num_epochs} — "
                  f"loss={running_loss/max(n_batches,1):.4f}  "
                  f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
                  f"w=[{w_str}]  div={history['diversity'][-1]:.3f}")

    history["grad_tracker"] = dict(grad_tracker.history)
    return model, history


def _eval_welde(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits_list, _ = model(feats)
            loss, _ = criterion(logits_list, labels)
            total_loss += loss.item()
            n += 1
            probs = torch.stack([F.softmax(l, dim=1) for l in logits_list]).mean(0)
            all_preds.append(probs.argmax(1).cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / max(n, 1), acc, f1


# ═════════════════ Test Evaluation ════════════════════════════

def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str = DEVICE,
    is_welde: bool = False,
) -> dict:
    """Full evaluation on test set. Returns comprehensive metrics."""
    model.eval()
    all_probs, all_preds, all_labels, all_features = [], [], [], []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            if is_welde:
                logits_list, features = model(feats)
                probs = torch.stack([F.softmax(l, dim=1) for l in logits_list]).mean(0)
            else:
                logits, features = model(feats)
                probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(probs.argmax(1).cpu())
            all_labels.append(labels.cpu())
            all_features.append(features.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_features = torch.cat(all_features).numpy()

    per_class_ap = {}
    one_hot = np.zeros((len(all_labels), NUM_CLASSES))
    for i, c in enumerate(all_labels):
        one_hot[i, c] = 1

    for k in range(NUM_CLASSES):
        if one_hot[:, k].sum() > 0:
            per_class_ap[k] = float(average_precision_score(one_hot[:, k], all_probs[:, k]))
        else:
            per_class_ap[k] = 0.0

    macro_ap = float(np.mean(list(per_class_ap.values())))
    tail_ap = float(np.mean([per_class_ap.get(k, 0) for k in [0, 4, 5]]))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "per_class_ap": per_class_ap,
        "mAP": macro_ap,
        "mAP_tail": tail_ap,
        "confusion_matrix": cm.tolist(),
        "per_class_acc": per_class_acc.tolist(),
        "all_probs": all_probs,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_features": all_features,
    }


def evaluate_welde_detailed(
    model,
    test_loader,
    device: str = DEVICE,
) -> dict:
    """Extended WELDE evaluation with per-head analysis."""
    base = evaluate_model(model, test_loader, device, is_welde=True)

    model.eval()
    # Infer number of heads from model
    num_heads = len(model.heads)
    per_head_probs = [[] for _ in range(num_heads)]
    per_head_preds = [[] for _ in range(num_heads)]
    all_labels = []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits_list, _ = model(feats)
            all_labels.append(labels.cpu())
            for j in range(num_heads):
                p = F.softmax(logits_list[j], dim=1)
                per_head_probs[j].append(p.cpu())
                per_head_preds[j].append(p.argmax(1).cpu())

    all_labels = torch.cat(all_labels).numpy()
    base["per_head_probs"] = [torch.cat(pp).numpy() for pp in per_head_probs]
    base["per_head_preds"] = [torch.cat(pp).numpy() for pp in per_head_preds]

    ensemble_preds = np.stack(base["per_head_preds"])
    disagreement = np.mean(ensemble_preds != ensemble_preds[0:1], axis=0)
    base["head_disagreement_rate"] = float(disagreement.mean())

    per_head_acc = []
    for j in range(num_heads):
        acc = float(accuracy_score(all_labels, base["per_head_preds"][j]))
        per_head_acc.append(acc)
    base["per_head_accuracy"] = per_head_acc

    return base

