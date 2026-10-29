"""
ParticleVision — Loss Functions
================================
Combo loss: Weighted Focal Loss + Per-class Dice Loss.
Designed for severe class imbalance (Fines ≈ 0.14 % of pixels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .config import CFG


class FocalLoss(nn.Module):
    """
    α-balanced focal loss for multi-class segmentation.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Parameters
    ----------
    alpha : list of float
        Per-class weight (length = num_classes).
    gamma : float
        Focusing parameter (default 2.0).
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = None,
    ):
        super().__init__()
        if alpha is None:
            alpha = CFG.focal_alpha
        if gamma is None:
            gamma = CFG.focal_gamma
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.float32)
        )
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W) raw model output
        targets : (B, H, W) int64 class indices

        Returns
        -------
        Scalar focal loss (mean over batch).
        """
        C = logits.shape[1]
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Gather the probability of the correct class
        targets_one_hot = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

        p_t = (probs * targets_one_hot).sum(dim=1)  # (B, H, W)
        p_t = p_t.clamp(min=1e-8)

        # Per-pixel focal modulation
        focal_weight = (1 - p_t) ** self.gamma  # (B, H, W)

        # Class-specific alpha
        alpha_t = self.alpha[targets]  # (B, H, W)

        # Cross-entropy term
        ce = -torch.log(p_t)

        loss = alpha_t * focal_weight * ce  # (B, H, W)
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Per-class Dice loss, averaged across classes with optional extra weight
    on the Fines class.

    Dice = (2·TP + smooth) / (2·TP + FP + FN + smooth)
    Loss = 1 - Dice
    """

    def __init__(
        self,
        smooth: float = None,
        fines_weight: float = None,
        num_classes: int = None,
    ):
        super().__init__()
        self.smooth = smooth if smooth is not None else CFG.dice_smooth
        self.fines_weight = (
            fines_weight if fines_weight is not None else CFG.dice_fines_weight
        )
        self.num_classes = num_classes or CFG.num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)
        targets : (B, H, W) int64

        Returns
        -------
        Scalar Dice loss.
        """
        C = self.num_classes
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = (
            F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        )  # (B, C, H, W)

        # Per-class Dice
        dims = (0, 2, 3)  # reduce over batch, H, W
        intersection = (probs * targets_one_hot).sum(dim=dims)  # (C,)
        cardinality = (probs + targets_one_hot).sum(dim=dims)   # (C,)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Weighted average: extra weight on fines (index 3)
        weights = torch.ones(C, device=logits.device)
        weights[3] = self.fines_weight  # Fines class gets extra weight
        weighted_dice = (dice * weights).sum() / weights.sum()

        return 1.0 - weighted_dice


class ComboLoss(nn.Module):
    """
    Combined loss: w_focal · FocalLoss + w_dice · DiceLoss

    Default: 50/50 weighting.
    """

    def __init__(
        self,
        focal_weight: float = None,
        dice_weight: float = None,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = None,
        dice_smooth: float = None,
        dice_fines_weight: float = None,
    ):
        super().__init__()
        self.w_focal = focal_weight if focal_weight is not None else CFG.loss_focal_weight
        self.w_dice = dice_weight if dice_weight is not None else CFG.loss_dice_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=dice_smooth, fines_weight=dice_fines_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fl = self.focal(logits, targets)
        dl = self.dice(logits, targets)
        return self.w_focal * fl + self.w_dice * dl


class TverskyLoss(nn.Module):
    """
    Per-class Tversky loss — generalisation of Dice that lets you trade
    precision for recall by setting β > α.

    Tversky(α, β) = TP / (TP + α·FP + β·FN)
    Loss = 1 - Tversky

    When α = β = 0.5, this equals Dice loss.
    When β > α (e.g. 0.7/0.3), the loss penalises false negatives
    (missed fines) more than false positives (hallucinated fines).

    Parameters
    ----------
    alpha : float  — weight for FP (default 0.3 → lenient on FP)
    beta  : float  — weight for FN (default 0.7 → strict on FN)
    fines_weight : float — extra multiplier for the fines class in the
                           weighted average across classes.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        fines_weight: float = None,
        num_classes: int = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.fines_weight = fines_weight if fines_weight is not None else CFG.dice_fines_weight
        self.num_classes = num_classes or CFG.num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)
        targets : (B, H, W) int64

        Returns
        -------
        Scalar Tversky loss.
        """
        C = self.num_classes
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = (
            F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        )  # (B, C, H, W)

        dims = (0, 2, 3)  # reduce over batch, H, W
        TP = (probs * targets_one_hot).sum(dim=dims)          # (C,)
        FP = (probs * (1 - targets_one_hot)).sum(dim=dims)    # (C,)
        FN = ((1 - probs) * targets_one_hot).sum(dim=dims)    # (C,)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )  # (C,)

        # Weighted average with extra weight on fines
        weights = torch.ones(C, device=logits.device)
        weights[3] = self.fines_weight
        weighted_tversky = (tversky * weights).sum() / weights.sum()

        return 1.0 - weighted_tversky


class ComboLossV2(nn.Module):
    """
    Improved combo loss: Focal + Tversky (recall-biased).

    Replaces DiceLoss with TverskyLoss to push recall on the fines class.
    """

    def __init__(
        self,
        focal_weight: float = None,
        tversky_weight: float = None,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = None,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_fines_weight: float = None,
    ):
        super().__init__()
        self.w_focal = focal_weight if focal_weight is not None else CFG.loss_focal_weight
        self.w_tversky = tversky_weight if tversky_weight is not None else CFG.loss_dice_weight

        # Bump fines focal alpha for V2
        if focal_alpha is None:
            focal_alpha = list(CFG.focal_alpha)
            focal_alpha[3] = 100.0  # Stronger fines penalty

        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta,
            fines_weight=tversky_fines_weight,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fl = self.focal(logits, targets)
        tl = self.tversky(logits, targets)
        return self.w_focal * fl + self.w_tversky * tl


class DualHeadLoss(nn.Module):
    """
    Combined loss for the dual-head architecture:
    Total = SegLoss(Focal+Dice) + heatmap_weight * HeatmapLoss(BCE)

    Uses pos_weight on BCE to counteract the extreme sparsity of
    heatmap targets (typically <3% of pixels are positive).
    """
    def __init__(self, heatmap_weight: float = 0.5, heatmap_pos_weight: float = 50.0, **combo_kwargs):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.seg_loss = ComboLoss(**combo_kwargs)
        # pos_weight rebalances the rare positive pixels so the head
        # actually learns to detect centroids instead of "predict all zeros"
        self.heatmap_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([heatmap_pos_weight])
        )

    def forward(self, pred: dict, seg_targets: torch.Tensor, heatmap_targets: torch.Tensor) -> torch.Tensor:
        """
        pred: dict with 'seg_logits' and 'heatmap_logits'
        seg_targets: (B, H, W) int64
        heatmap_targets: (B, 1, H, W) float32
        """
        seg_logits = pred["seg_logits"]
        heatmap_logits = pred["heatmap_logits"]

        # Segmentation loss
        loss_seg = self.seg_loss(seg_logits, seg_targets)

        # Heatmap loss
        loss_hm = self.heatmap_loss(heatmap_logits, heatmap_targets)

        return loss_seg + self.heatmap_weight * loss_hm


class SoftIoULoss(nn.Module):
    """
    Per-class Soft IoU (Jaccard) loss.
    IoU = Intersection / Union
    Union = Sum(Probs) + Sum(Targets) - Intersection
    Loss = 1 - IoU
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        fines_weight: float = None,
        num_classes: int = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.fines_weight = (
            fines_weight if fines_weight is not None else CFG.dice_fines_weight
        )
        self.num_classes = num_classes or CFG.num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = self.num_classes
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = (
            F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        )  # (B, C, H, W)

        dims = (0, 2, 3)  # reduce over batch, H, W
        intersection = (probs * targets_one_hot).sum(dim=dims)  # (C,)
        total = (probs + targets_one_hot).sum(dim=dims)         # (C,)
        union = total - intersection                            # (C,)

        iou = (intersection + self.smooth) / (union + self.smooth)

        # Weighted average: extra weight on fines (index 3)
        weights = torch.ones(C, device=logits.device)
        weights[3] = self.fines_weight
        weighted_iou = (iou * weights).sum() / weights.sum()

        return 1.0 - weighted_iou


class ComboLossV3(nn.Module):
    """
    Combo loss: w_focal · FocalLoss + w_iou · SoftIoULoss
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        iou_weight: float = 0.5,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = None,
        iou_smooth: float = 1e-6,
        iou_fines_weight: float = None,
    ):
        super().__init__()
        self.w_focal = focal_weight
        self.w_iou = iou_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou = SoftIoULoss(smooth=iou_smooth, fines_weight=iou_fines_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fl = self.focal(logits, targets)
        il = self.iou(logits, targets)
        return self.w_focal * fl + self.w_iou * il

