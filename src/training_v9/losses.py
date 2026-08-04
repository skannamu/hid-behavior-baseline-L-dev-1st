"""Training losses for ReCon-HID v9."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models_v9.recon_hid_v9 import ReConHIDV9Output


@dataclass(frozen=True)
class ReConHIDV9LossConfig:
    sequence_reconstruction_weight: float = 1.0
    context_reconstruction_weight: float = 0.5
    classifier_weight: float = 1.0
    latent_l2_weight: float = 1.0e-4

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class ReConHIDV9LossOutput:
    total: Tensor
    sequence_reconstruction: Tensor
    context_reconstruction: Tensor
    classifier: Tensor
    latent_l2: Tensor


class ReConHIDV9Loss(nn.Module):
    def __init__(
        self,
        config: ReConHIDV9LossConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ReConHIDV9LossConfig()
        self.config.validate()

    def forward(
        self,
        output: ReConHIDV9Output,
        *,
        sequence_target: Tensor,
        context_target: Tensor,
        binary_labels: Tensor | None = None,
    ) -> ReConHIDV9LossOutput:
        sequence_loss = F.mse_loss(
            output.sequence_reconstruction,
            sequence_target,
        )
        context_loss = F.mse_loss(
            output.context_reconstruction,
            context_target,
        )

        if binary_labels is None:
            classifier_loss = output.classifier_logit.new_zeros(())
        else:
            labels = binary_labels.float().view_as(output.classifier_logit)
            classifier_loss = F.binary_cross_entropy_with_logits(
                output.classifier_logit,
                labels,
            )

        latent_l2 = output.fused_latent.pow(2).mean()

        total = (
            self.config.sequence_reconstruction_weight * sequence_loss
            + self.config.context_reconstruction_weight * context_loss
            + self.config.classifier_weight * classifier_loss
            + self.config.latent_l2_weight * latent_l2
        )

        return ReConHIDV9LossOutput(
            total=total,
            sequence_reconstruction=sequence_loss,
            context_reconstruction=context_loss,
            classifier=classifier_loss,
            latent_l2=latent_l2,
        )
