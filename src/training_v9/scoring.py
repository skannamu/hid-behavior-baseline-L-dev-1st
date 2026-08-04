"""Per-window scoring and calibration helpers for ReCon-HID v9."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.models_v9 import ReConHIDV9


@dataclass(frozen=True)
class V9ScoreBatch:
    sequence_reconstruction: np.ndarray
    context_reconstruction: np.ndarray
    reconstruction_combined: np.ndarray
    classifier_probability: np.ndarray
    prototype_distance: np.ndarray | None

    def as_dict(self) -> dict[str, np.ndarray | None]:
        return {
            "sequence_reconstruction": self.sequence_reconstruction,
            "context_reconstruction": self.context_reconstruction,
            "reconstruction_combined": self.reconstruction_combined,
            "classifier_probability": self.classifier_probability,
            "prototype_distance": self.prototype_distance,
        }


@torch.no_grad()
def compute_v9_scores(
    model: ReConHIDV9,
    sequence: Tensor,
    context: Tensor,
    *,
    context_weight: float = 0.5,
) -> V9ScoreBatch:
    if context_weight < 0:
        raise ValueError("context_weight must be non-negative")

    model.eval()
    output = model(sequence, context)

    sequence_score = (
        output.sequence_reconstruction.sub(sequence).pow(2).mean(dim=(1, 2))
    )
    context_score = (
        output.context_reconstruction.sub(context).pow(2).mean(dim=1)
    )
    combined = sequence_score + context_weight * context_score
    classifier_probability = torch.sigmoid(output.classifier_logit)

    prototype = output.prototype_distance
    return V9ScoreBatch(
        sequence_reconstruction=sequence_score.detach().cpu().numpy(),
        context_reconstruction=context_score.detach().cpu().numpy(),
        reconstruction_combined=combined.detach().cpu().numpy(),
        classifier_probability=classifier_probability.detach().cpu().numpy(),
        prototype_distance=(
            prototype.detach().cpu().numpy() if prototype is not None else None
        ),
    )


def quantile_threshold(
    normal_scores: np.ndarray,
    *,
    target_fpr: float,
) -> float:
    """Return a calibration-only threshold using a conservative order statistic.

    Detection uses score > threshold. NumPy's 'higher' quantile method chooses
    an observed score at or above the requested quantile and avoids interpolation
    that may imply unsupported precision.
    """
    values = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot calibrate a threshold from zero scores")
    if not np.isfinite(values).all():
        raise ValueError("Calibration scores contain NaN or infinity")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")

    return float(
        np.quantile(values, 1.0 - target_fpr, method="higher")
    )


def empirical_fpr(normal_scores: np.ndarray, threshold: float) -> float:
    values = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute FPR from zero scores")
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    return float(np.mean(values > threshold))
