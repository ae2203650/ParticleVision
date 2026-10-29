"""
ParticleVision — Evaluation Metrics
=====================================
Per-class and aggregate segmentation metrics:
  - Pixel accuracy (overall)
  - Per-class IoU, Dice (F1), Precision, Recall
  - Mean IoU (mIoU)
  - Confusion matrix
"""

import numpy as np
import torch
from typing import Dict, Optional
from .config import CFG


class SegmentationMetrics:
    """
    Accumulates predictions over multiple batches, then computes
    all metrics at once from the aggregated confusion matrix.
    """

    def __init__(self, num_classes: int = None, class_names=None):
        self.num_classes = num_classes or CFG.num_classes
        self.class_names = class_names or CFG.class_names
        self.reset()

    def reset(self):
        """Clear accumulated counts."""
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        Accumulate a batch of predictions.

        Parameters
        ----------
        preds   : int array, any shape — flattened to 1-D.
        targets : int array, same shape as preds.
        """
        preds = preds.ravel()
        targets = targets.ravel()

        # Build partial confusion matrix efficiently
        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        indices = targets * self.num_classes + preds
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion += cm.reshape(self.num_classes, self.num_classes)

    def update_torch(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Convenience: accept (B, C, H, W) logits and (B, H, W) targets,
        argmax, move to numpy, and call update().
        """
        preds = logits.argmax(dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        self.update(preds, targets_np)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def pixel_accuracy(self) -> float:
        """Overall pixel accuracy = correct / total."""
        correct = np.diag(self.confusion).sum()
        total = self.confusion.sum()
        return float(correct / max(total, 1))

    def per_class_iou(self) -> np.ndarray:
        """Per-class Intersection over Union."""
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, 0.0)
        return iou.astype(np.float64)

    def mean_iou(self) -> float:
        """Mean IoU across all classes."""
        return float(self.per_class_iou().mean())

    def per_class_dice(self) -> np.ndarray:
        """Per-class Dice coefficient (= F1 score)."""
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        dice = np.where(denom > 0, 2 * tp / denom, 0.0)
        return dice.astype(np.float64)

    def per_class_precision(self) -> np.ndarray:
        """Per-class precision = TP / (TP + FP)."""
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        denom = tp + fp
        return np.where(denom > 0, tp / denom, 0.0).astype(np.float64)

    def per_class_recall(self) -> np.ndarray:
        """Per-class recall = TP / (TP + FN)."""
        tp = np.diag(self.confusion)
        fn = self.confusion.sum(axis=1) - tp
        denom = tp + fn
        return np.where(denom > 0, tp / denom, 0.0).astype(np.float64)

    def summary(self) -> Dict:
        """
        Return a dict with all metrics, ready for printing / logging.
        """
        iou = self.per_class_iou()
        dice = self.per_class_dice()
        prec = self.per_class_precision()
        rec = self.per_class_recall()

        result = {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": self.mean_iou(),
        }
        for i, name in enumerate(self.class_names):
            result[f"{name}_iou"] = float(iou[i])
            result[f"{name}_dice"] = float(dice[i])
            result[f"{name}_precision"] = float(prec[i])
            result[f"{name}_recall"] = float(rec[i])

        return result

    def print_summary(self, title: str = "Evaluation Metrics"):
        """Pretty-print all metrics."""
        s = self.summary()
        print(f"\n{'=' * 65}")
        print(f"  {title}")
        print(f"{'=' * 65}")
        print(f"  Overall pixel accuracy : {s['pixel_accuracy']:.4f}")
        print(f"  Mean IoU (mIoU)        : {s['mean_iou']:.4f}")
        print(f"{'─' * 65}")
        header = f"  {'Class':<10} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}"
        print(header)
        print(f"  {'─' * 42}")
        for name in self.class_names:
            iou  = s[f"{name}_iou"]
            dice = s[f"{name}_dice"]
            prec = s[f"{name}_precision"]
            rec  = s[f"{name}_recall"]
            marker = " ◀" if name == "Fines" else ""
            print(f"  {name:<10} {iou:>8.4f} {dice:>8.4f} {prec:>8.4f} {rec:>8.4f}{marker}")
        print(f"{'=' * 65}\n")

    def confusion_matrix(self) -> np.ndarray:
        """Return the raw confusion matrix."""
        return self.confusion.copy()

class BinaryMetrics:
    """
    Accumulates binary predictions (0 or 1) and computes metrics.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        
    def update(self, preds: np.ndarray, targets: np.ndarray):
        preds = preds.ravel() == 1
        targets = targets.ravel() == 1
        
        self.tp += np.logical_and(preds, targets).sum()
        self.fp += np.logical_and(preds, np.logical_not(targets)).sum()
        self.tn += np.logical_and(np.logical_not(preds), np.logical_not(targets)).sum()
        self.fn += np.logical_and(np.logical_not(preds), targets).sum()
        
    def update_torch(self, preds: torch.Tensor, targets: torch.Tensor):
        self.update(preds.cpu().numpy(), targets.cpu().numpy())
        
    def summary(self):
        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'dice': dice
        }
