"""Dual-input ReCon-HID v9 model for Feature Schema v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.features.schema import (
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
)


@dataclass(frozen=True)
class ReConHIDV9Config:
    sequence_input_dim: int = len(SEQUENCE_FEATURES)
    context_input_dim: int = len(WINDOW_CONTEXT_FEATURES)
    window_size: int = WINDOW_SIZE

    sequence_hidden_dim: int = 64
    sequence_latent_dim: int = 24
    context_hidden_dim: int = 32
    context_latent_dim: int = 12
    fused_latent_dim: int = 32

    sequence_num_layers: int = 1
    dropout: float = 0.10
    bidirectional_encoder: bool = False

    classifier_hidden_dim: int = 32

    def validate(self) -> None:
        positive = {
            "sequence_input_dim": self.sequence_input_dim,
            "context_input_dim": self.context_input_dim,
            "window_size": self.window_size,
            "sequence_hidden_dim": self.sequence_hidden_dim,
            "sequence_latent_dim": self.sequence_latent_dim,
            "context_hidden_dim": self.context_hidden_dim,
            "context_latent_dim": self.context_latent_dim,
            "fused_latent_dim": self.fused_latent_dim,
            "sequence_num_layers": self.sequence_num_layers,
            "classifier_hidden_dim": self.classifier_hidden_dim,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.sequence_input_dim != len(SEQUENCE_FEATURES):
            raise ValueError(
                "sequence_input_dim must match Feature Schema v2: "
                f"expected={len(SEQUENCE_FEATURES)}, "
                f"actual={self.sequence_input_dim}"
            )
        if self.context_input_dim != len(WINDOW_CONTEXT_FEATURES):
            raise ValueError(
                "context_input_dim must match Feature Schema v2: "
                f"expected={len(WINDOW_CONTEXT_FEATURES)}, "
                f"actual={self.context_input_dim}"
            )
        if self.window_size != WINDOW_SIZE:
            raise ValueError(
                f"window_size mismatch: expected={WINDOW_SIZE}, "
                f"actual={self.window_size}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReConHIDV9Output:
    sequence_reconstruction: Tensor
    context_reconstruction: Tensor
    classifier_logit: Tensor
    sequence_latent: Tensor
    context_latent: Tensor
    fused_latent: Tensor
    prototype_distance: Tensor | None

    @property
    def classifier_probability(self) -> Tensor:
        return torch.sigmoid(self.classifier_logit)


class ReConHIDV9(nn.Module):
    """Sequence autoencoder + context autoencoder + fused decision heads.

    Input:
        sequence: [B, 50, 12]
        context:  [B, 12]

    Output:
        reconstructed sequence/context, classifier logit, latent vectors,
        and optional Euclidean distance to a stored normal prototype.
    """

    def __init__(self, config: ReConHIDV9Config | None = None) -> None:
        super().__init__()
        self.config = config or ReConHIDV9Config()
        self.config.validate()

        directions = 2 if self.config.bidirectional_encoder else 1
        encoder_dropout = (
            self.config.dropout
            if self.config.sequence_num_layers > 1
            else 0.0
        )

        self.sequence_encoder = nn.LSTM(
            input_size=self.config.sequence_input_dim,
            hidden_size=self.config.sequence_hidden_dim,
            num_layers=self.config.sequence_num_layers,
            batch_first=True,
            dropout=encoder_dropout,
            bidirectional=self.config.bidirectional_encoder,
        )
        self.sequence_latent_head = nn.Sequential(
            nn.Linear(
                self.config.sequence_hidden_dim * directions,
                self.config.sequence_latent_dim,
            ),
            nn.LayerNorm(self.config.sequence_latent_dim),
            nn.GELU(),
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(
                self.config.context_input_dim,
                self.config.context_hidden_dim,
            ),
            nn.LayerNorm(self.config.context_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.context_hidden_dim,
                self.config.context_latent_dim,
            ),
            nn.LayerNorm(self.config.context_latent_dim),
            nn.GELU(),
        )

        concatenated_dim = (
            self.config.sequence_latent_dim
            + self.config.context_latent_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(concatenated_dim, self.config.fused_latent_dim),
            nn.LayerNorm(self.config.fused_latent_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # The fused latent initializes a causal decoder. It does not receive
        # the original sequence, preventing a trivial identity shortcut.
        decoder_state_dim = (
            self.config.sequence_num_layers
            * self.config.sequence_hidden_dim
        )
        self.decoder_hidden_init = nn.Linear(
            self.config.fused_latent_dim,
            decoder_state_dim,
        )
        self.decoder_cell_init = nn.Linear(
            self.config.fused_latent_dim,
            decoder_state_dim,
        )
        self.sequence_decoder = nn.LSTM(
            input_size=self.config.fused_latent_dim,
            hidden_size=self.config.sequence_hidden_dim,
            num_layers=self.config.sequence_num_layers,
            batch_first=True,
            dropout=encoder_dropout,
            bidirectional=False,
        )
        self.sequence_output_head = nn.Linear(
            self.config.sequence_hidden_dim,
            self.config.sequence_input_dim,
        )

        self.context_decoder = nn.Sequential(
            nn.Linear(
                self.config.fused_latent_dim,
                self.config.context_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.config.context_hidden_dim,
                self.config.context_input_dim,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.fused_latent_dim,
                self.config.classifier_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden_dim, 1),
        )

        self.register_buffer(
            "normal_prototype",
            torch.zeros(self.config.fused_latent_dim),
            persistent=True,
        )
        self.register_buffer(
            "normal_prototype_ready",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    def encode(self, sequence: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_inputs(sequence, context)

        _, (hidden, _) = self.sequence_encoder(sequence)
        directions = 2 if self.config.bidirectional_encoder else 1

        if directions == 1:
            final_hidden = hidden[-1]
        else:
            # Last layer's forward and backward states.
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)

        sequence_latent = self.sequence_latent_head(final_hidden)
        context_latent = self.context_encoder(context)
        fused = self.fusion(
            torch.cat((sequence_latent, context_latent), dim=-1)
        )
        return sequence_latent, context_latent, fused

    def decode_sequence(self, fused: Tensor) -> Tensor:
        batch_size = fused.shape[0]
        layers = self.config.sequence_num_layers
        hidden_dim = self.config.sequence_hidden_dim

        h0 = self.decoder_hidden_init(fused).view(
            batch_size, layers, hidden_dim
        ).transpose(0, 1).contiguous()
        c0 = self.decoder_cell_init(fused).view(
            batch_size, layers, hidden_dim
        ).transpose(0, 1).contiguous()

        repeated_fused = fused.unsqueeze(1).expand(
            -1, self.config.window_size, -1
        )
        decoded, _ = self.sequence_decoder(repeated_fused, (h0, c0))
        return self.sequence_output_head(decoded)

    def prototype_distance(self, fused: Tensor) -> Tensor | None:
        if not bool(self.normal_prototype_ready.item()):
            return None
        return torch.linalg.vector_norm(
            fused - self.normal_prototype.unsqueeze(0),
            ord=2,
            dim=-1,
        )

    @torch.no_grad()
    def set_normal_prototype(self, fused_latents: Tensor) -> Tensor:
        if fused_latents.ndim != 2:
            raise ValueError(
                "fused_latents must have shape [N, fused_latent_dim]"
            )
        if fused_latents.shape[0] == 0:
            raise ValueError("Cannot build a prototype from zero samples")
        if fused_latents.shape[1] != self.config.fused_latent_dim:
            raise ValueError(
                "Prototype latent dimension mismatch: "
                f"expected={self.config.fused_latent_dim}, "
                f"actual={fused_latents.shape[1]}"
            )
        prototype = fused_latents.mean(dim=0)
        self.normal_prototype.copy_(prototype)
        self.normal_prototype_ready.fill_(True)
        return prototype.detach().clone()

    def clear_normal_prototype(self) -> None:
        self.normal_prototype.zero_()
        self.normal_prototype_ready.fill_(False)

    def forward(self, sequence: Tensor, context: Tensor) -> ReConHIDV9Output:
        sequence_latent, context_latent, fused = self.encode(
            sequence, context
        )
        sequence_reconstruction = self.decode_sequence(fused)
        context_reconstruction = self.context_decoder(fused)
        classifier_logit = self.classifier(fused).squeeze(-1)

        return ReConHIDV9Output(
            sequence_reconstruction=sequence_reconstruction,
            context_reconstruction=context_reconstruction,
            classifier_logit=classifier_logit,
            sequence_latent=sequence_latent,
            context_latent=context_latent,
            fused_latent=fused,
            prototype_distance=self.prototype_distance(fused),
        )

    def _validate_inputs(self, sequence: Tensor, context: Tensor) -> None:
        if sequence.ndim != 3:
            raise ValueError(
                f"sequence must be rank 3 [B,T,F], got {tuple(sequence.shape)}"
            )
        if context.ndim != 2:
            raise ValueError(
                f"context must be rank 2 [B,C], got {tuple(context.shape)}"
            )
        if sequence.shape[0] != context.shape[0]:
            raise ValueError("sequence/context batch size mismatch")
        if sequence.shape[1] != self.config.window_size:
            raise ValueError(
                f"sequence length mismatch: expected={self.config.window_size}, "
                f"actual={sequence.shape[1]}"
            )
        if sequence.shape[2] != self.config.sequence_input_dim:
            raise ValueError(
                "sequence feature dimension mismatch: "
                f"expected={self.config.sequence_input_dim}, "
                f"actual={sequence.shape[2]}"
            )
        if context.shape[1] != self.config.context_input_dim:
            raise ValueError(
                "context feature dimension mismatch: "
                f"expected={self.config.context_input_dim}, "
                f"actual={context.shape[1]}"
            )
        if not torch.isfinite(sequence).all():
            raise ValueError("sequence contains NaN or infinity")
        if not torch.isfinite(context).all():
            raise ValueError("context contains NaN or infinity")
