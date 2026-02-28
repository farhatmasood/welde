"""
WELDE — Weighted Ensemble Loss with Diversity Enhancement.

A framework for class-imbalanced medical image classification that
combines four complementary loss functions (CE, Focal, CBL, LDAM)
via per-head adapter projections, EMA normalisation, and learnable
adaptive weighting.
"""

from welde.losses import WELDELoss, get_loss_fn
from welde.model import (
    HeadOnly,
    MultiHeadAdapterOnly,
    MultiHeadOnly,
)
from welde.trainer import (
    train_welde,
    train_single_head,
    train_ce_ensemble,
    evaluate_model,
)

__version__ = "1.0.0"
__all__ = [
    "WELDELoss",
    "get_loss_fn",
    "HeadOnly",
    "MultiHeadAdapterOnly",
    "MultiHeadOnly",
    "train_welde",
    "train_single_head",
    "train_ce_ensemble",
    "evaluate_model",
]
