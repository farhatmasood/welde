"""
losses.py — All loss functions for the WELDE framework.

Implements:
  - WeightedCrossEntropy (wCE)
  - FocalLoss (FL)
  - ClassBalancedLoss (CBL)
  - LDAMLoss (LDAM)
  - DistributionBalancedLoss (DB Loss, simplified)
  - EqualizationLossV2 (EqL v2, simplified)
  - WELDELoss (full framework: 4 heads, EMA, adaptive weights, diversity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedCrossEntropy(nn.Module):
    """Weighted CE with w_k = n / (K * n_k)."""

    def __init__(self, class_counts: np.ndarray):
        super().__init__()
        n = class_counts.sum()
        K = len(class_counts)
        weights = n / (K * class_counts.astype(np.float64))
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """Focal Loss: -(1-p_t)^gamma * log(p_t), with optional per-class alpha."""

    def __init__(self, gamma: float = 2.0, class_counts: np.ndarray = None):
        super().__init__()
        self.gamma = gamma
        if class_counts is not None:
            n = class_counts.sum()
            K = len(class_counts)
            alpha = n / (K * class_counts.astype(np.float64))
            alpha = alpha / alpha.sum()  # Normalise so they sum to 1
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1.0 - p_t) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        return focal.mean()


class ClassBalancedLoss(nn.Module):
    """CBL with effective number weighting: (1-beta)/(1-beta^n_k)."""

    def __init__(self, class_counts: np.ndarray, beta: float = 0.999):
        super().__init__()
        effective_num = 1.0 - np.power(beta, class_counts.astype(np.float64))
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss.

    Applies class-dependent margin Delta_k = C * n_k^{-1/4} to the
    target-class logit, matching Eq. (6) in the manuscript.
    """

    def __init__(self, class_counts: np.ndarray, C: float = 0.5, max_margin: float = 0.5):
        super().__init__()
        # Δ_k = C · n_k^{−1/4} — C directly controls the overall margin scale
        margins = C * np.power(class_counts.astype(np.float64), -0.25)
        # Clip individual margins to max_margin for stability, but C still matters
        margins = np.minimum(margins, max_margin)
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Subtract margin from target-class logit
        margin_matrix = self.margins.unsqueeze(0).expand_as(logits)
        target_mask = F.one_hot(targets, num_classes=logits.size(1)).bool()
        adjusted_logits = logits.clone()
        adjusted_logits[target_mask] -= margin_matrix[target_mask]
        return F.cross_entropy(adjusted_logits, targets)


class DistributionBalancedLoss(nn.Module):
    """Simplified Distribution-Balanced Loss (Wu et al., ECCV 2020).

    Combines rebalanced negative sampling with class-balanced weighting.
    This is a simplified version using negative-tolerant regularisation
    and class-balanced weighting rather than the full formulation.
    """

    def __init__(self, class_counts: np.ndarray, beta: float = 0.999):
        super().__init__()
        # Class prior probabilities
        priors = class_counts.astype(np.float64) / class_counts.sum()
        self.register_buffer("log_prior", torch.tensor(np.log(priors + 1e-8), dtype=torch.float32))
        # CBL-style weights
        effective_num = 1.0 - np.power(beta, class_counts.astype(np.float64))
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits - self.log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted, targets, weight=self.weight)


class EqualizationLossV2(nn.Module):
    """Simplified Equalization Loss v2 (Tan et al., CVPR 2021).

    Protects rare-class gradients by down-weighting the negative gradient
    contributions for tail classes.
    """

    def __init__(self, class_counts: np.ndarray, threshold: float = 0.05):
        super().__init__()
        freqs = class_counts.astype(np.float64) / class_counts.sum()
        # Tail indicator: classes with frequency < threshold
        tail_mask = (freqs < threshold).astype(np.float64)
        # Gradient protection weight: reduce neg gradient for tail classes
        eq_weight = 1.0 - tail_mask * 0.5  # Halve neg gradient for tails
        self.register_buffer("eq_weight", torch.tensor(eq_weight, dtype=torch.float32))
        # Standard CE weights
        n = class_counts.sum()
        K = len(class_counts)
        w = n / (K * class_counts.astype(np.float64))
        self.register_buffer("weight", torch.tensor(w, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply equalization: modify logits for non-target classes of tail classes
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(targets, num_classes=logits.size(1)).float()
        # For tail classes, reduce the logit for non-target positions
        eq_logits = logits * (target_oh + (1 - target_oh) * self.eq_weight.unsqueeze(0))
        return F.cross_entropy(eq_logits, targets, weight=self.weight)


# ═══════════════════════════════════════════════════════════════
# WELDE — Weighted Ensemble Loss with Diversity Enhancement
# ═══════════════════════════════════════════════════════════════

class WELDELoss(nn.Module):
    """Complete WELDE loss framework.

    Combines complementary losses (configurable, default: CE, FL, CBL, LDAM):
      - EMA-based loss normalisation
      - Learnable adaptive weighting (squared coefficients + floor)
      - Quadratic penalty for sum-to-one constraint
      - Optional diversity regularisation (angular distance hinge)
    """

    def __init__(
        self,
        class_counts: np.ndarray,
        loss_names: list[str] | None = None,
        alpha: float = 0.01,
        eta: float = 1.0,
        lambda_div: float = 0.0,
        s: float = 0.1,
        delta_thr: float = 2.0,
        gamma: float = 2.0,
        beta: float = 0.999,
        ldam_C: float = 0.5,
        use_ema: bool = True,
        use_diversity: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.eta = eta
        self.lambda_div = lambda_div
        self.s = s
        self.delta_thr = delta_thr
        self.use_ema = use_ema
        self.use_diversity = use_diversity

        # Configurable loss composition
        if loss_names is None:
            loss_names = ["CE", "FL", "CBL", "LDAM"]
        self.loss_names = loss_names
        num_heads = len(loss_names)

        loss_builders = {
            "CE":      lambda: nn.CrossEntropyLoss(),
            "wCE":     lambda: WeightedCrossEntropy(class_counts),
            "FL":      lambda: FocalLoss(gamma=gamma, class_counts=class_counts),
            "CBL":     lambda: ClassBalancedLoss(class_counts, beta=beta),
            "LDAM":    lambda: LDAMLoss(class_counts, C=ldam_C),
        }
        self.loss_fns = nn.ModuleList([
            loss_builders[name]() for name in loss_names
        ])

        # Learnable coefficients (initialised so w_j ≈ 1/num_heads each)
        init_c = np.sqrt(1.0 / num_heads - alpha) if (1.0 / num_heads) > alpha else 0.1
        self.coeffs = nn.Parameter(torch.full((num_heads,), init_c))

        # EMA running estimates (not parameters — buffers)
        self.register_buffer("ema", torch.ones(num_heads))

    def forward(
        self,
        logits_list: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits_list: list of N tensors (B, K) from each head.
            targets:     (B,) class labels.
        Returns:
            total_loss, info_dict
        """
        N = len(self.loss_fns)
        assert len(logits_list) == N, f"WELDE requires exactly {N} head outputs."

        # --- Compute raw losses ---
        raw = []
        for j, (fn, logits) in enumerate(zip(self.loss_fns, logits_list)):
            raw.append(fn(logits, targets))

        # --- EMA normalisation ---
        if self.use_ema:
            with torch.no_grad():
                for j in range(N):
                    self.ema[j] = self.s * raw[j].detach() + (1.0 - self.s) * self.ema[j]
            ema_mean = self.ema.mean()
            # τ_j = ℓ̄ / ℓ̄_j  (stop-gradient: ema values are detached)
            tau = ema_mean / (self.ema + 1e-8)
            normalised = [tau[j] * raw[j] for j in range(N)]
        else:
            normalised = raw

        # --- Adaptive weighting ---
        w = self.coeffs ** 2 + self.alpha                   # (N,)
        welde_loss = sum(w[j] * normalised[j] for j in range(N))
        penalty = self.eta * (w.sum() - 1.0) ** 2
        total = welde_loss + penalty

        # --- Diversity regularisation ---
        div_val = torch.tensor(0.0, device=targets.device)
        if self.use_diversity and self.lambda_div > 0:
            probs = [F.softmax(logits, dim=1) for logits in logits_list]
            div_val = self._diversity_loss(probs)
            total = total + self.lambda_div * div_val

        info = {
            "total": total.item(),
            "welde": welde_loss.item(),
            "penalty": penalty.item(),
            "diversity": div_val.item(),
            "raw_losses": [l.item() for l in raw],
            "norm_losses": [l.item() for l in normalised],
            "weights": w.detach().cpu().numpy().tolist(),
            "ema": self.ema.detach().cpu().numpy().tolist(),
        }
        return total, info

    def _diversity_loss(self, probs_list: list[torch.Tensor]) -> torch.Tensor:
        """Hinge on squared mean-pairwise angular distance."""
        # L2-normalise each probability vector
        normed = [p / (p.norm(dim=1, keepdim=True) + 1e-8) for p in probs_list]
        M = len(normed)
        D_sq = torch.tensor(0.0, device=probs_list[0].device)
        count = 0
        for j in range(M):
            for m in range(j + 1, M):
                cos_sim = (normed[j] * normed[m]).sum(dim=1)       # (B,)
                angular_dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0)
                D_sq = D_sq + angular_dist_sq.mean()
                count += 1
        D_sq = D_sq / count
        return torch.clamp(self.delta_thr - D_sq, min=0)


def get_loss_fn(name: str, class_counts: np.ndarray, **kwargs) -> nn.Module:
    """Factory function for single-loss baselines."""
    lookup = {
        "CE":      lambda: nn.CrossEntropyLoss(),
        "wCE":     lambda: WeightedCrossEntropy(class_counts),
        "FL":      lambda: FocalLoss(gamma=kwargs.get("gamma", 2.0), class_counts=class_counts),
        "CBL":     lambda: ClassBalancedLoss(class_counts, beta=kwargs.get("beta", 0.999)),
        "LDAM":    lambda: LDAMLoss(class_counts, C=kwargs.get("ldam_C", 0.5)),
        "DB_Loss": lambda: DistributionBalancedLoss(class_counts),
        "EqL_v2":  lambda: EqualizationLossV2(class_counts),
    }
    if name not in lookup:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(lookup.keys())}")
    return lookup[name]()
