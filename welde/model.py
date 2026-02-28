"""
model.py — Network architectures for WELDE experiments.

Provides:
  - SingleHeadClassifier:  backbone + 1 MLP head (for baselines)
  - MultiHeadClassifier:   backbone + 4 MLP heads (for WELDE)
  - HeadOnly:              1 MLP head operating on pre-extracted features
  - MultiHeadOnly:         4 MLP heads operating on pre-extracted features
  - MultiHeadAdapterOnly:  4 adapter+MLP heads with per-head projections
"""
import torch
import torch.nn as nn
from torchvision import models

from welde.config import NUM_CLASSES, HEAD_HIDDEN, PROJ_DIM


def _make_head(in_dim: int, num_classes: int, hidden: int = HEAD_HIDDEN) -> nn.Sequential:
    """Two-layer MLP classification head."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(hidden, num_classes),
    )


def _make_adapter(in_dim: int, proj_dim: int) -> nn.Sequential:
    """Per-head adapter: projects shared features into a head-specific subspace."""
    return nn.Sequential(
        nn.Linear(in_dim, proj_dim),
        nn.BatchNorm1d(proj_dim),
        nn.GELU(),
        nn.Dropout(0.3),
    )


# ═══════════════════ Head-Only Models (for pre-extracted features) ═══════

class HeadOnly(nn.Module):
    """Single classification head operating on 2048-dim features."""

    def __init__(self, feat_dim: int = 2048, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feat_dim = feat_dim
        self.head = _make_head(feat_dim, num_classes)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features_passthrough)."""
        logits = self.head(feat)
        return logits, feat


class MultiHeadOnly(nn.Module):
    """Multiple classification heads on 2048-dim features (legacy, no adapters)."""

    def __init__(
        self,
        feat_dim: int = 2048,
        num_classes: int = NUM_CLASSES,
        num_heads: int = 4,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.heads = nn.ModuleList([
            _make_head(feat_dim, num_classes) for _ in range(num_heads)
        ])

    def forward(self, feat: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Returns (list_of_logits, features_passthrough)."""
        logits = [head(feat) for head in self.heads]
        return logits, feat


class MultiHeadAdapterOnly(nn.Module):
    """4 independent adapter+classifier pipelines on frozen 2048-dim features.

    Each head has its own projection layer that maps the shared features
    into a head-specific subspace, allowing different loss functions to
    shape different feature representations.

    Architecture per head:
        adapter_j: Linear(2048, proj_dim) → BN → GELU → Dropout(0.3)
        head_j:    Linear(proj_dim, 256)  → ReLU → Dropout(0.2) → Linear(256, 6)
    """

    def __init__(
        self,
        feat_dim: int = 2048,
        proj_dim: int = PROJ_DIM,
        num_classes: int = NUM_CLASSES,
        num_heads: int = 4,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.adapters = nn.ModuleList([
            _make_adapter(feat_dim, proj_dim) for _ in range(num_heads)
        ])
        self.heads = nn.ModuleList([
            _make_head(proj_dim, num_classes) for _ in range(num_heads)
        ])

    def forward(self, feat: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Returns (list_of_logits, features_passthrough)."""
        logits = [head(adapter(feat)) for adapter, head in zip(self.adapters, self.heads)]
        return logits, feat


# ═══════════════════ Full Models (backbone + heads) ═══════════════════════

class SingleHeadClassifier(nn.Module):
    """ResNet-50 backbone (frozen early layers) + single classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        children = list(backbone.children())
        self.frozen = nn.Sequential(*children[:6])
        for p in self.frozen.parameters():
            p.requires_grad = False
        self.frozen.eval()
        self.trainable = nn.Sequential(*children[6:-1])
        self.feat_dim = 2048
        self.head = _make_head(self.feat_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = self.frozen(x)
        feat = self.trainable(x).flatten(1)
        logits = self.head(feat)
        return logits, feat


class MultiHeadClassifier(nn.Module):
    """ResNet-50 backbone + 4 classification heads (for WELDE)."""

    def __init__(self, num_classes: int = NUM_CLASSES, num_heads: int = 4, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        children = list(backbone.children())
        self.frozen = nn.Sequential(*children[:6])
        for p in self.frozen.parameters():
            p.requires_grad = False
        self.frozen.eval()
        self.trainable = nn.Sequential(*children[6:-1])
        self.feat_dim = 2048
        self.heads = nn.ModuleList([
            _make_head(self.feat_dim, num_classes) for _ in range(num_heads)
        ])

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            x = self.frozen(x)
        feat = self.trainable(x).flatten(1)
        logits = [head(feat) for head in self.heads]
        return logits, feat

