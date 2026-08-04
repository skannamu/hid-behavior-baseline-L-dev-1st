"""Train-only feature normalization with a serializable schema contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
)


@dataclass
class FeatureNormalizerV2:
    """Median/IQR normalizer fitted only on the training split.

    Binary sequence features are left unchanged. Other sequence features and
    all context features are robustly scaled. Zero-IQR dimensions use scale 1.
    """

    sequence_center: np.ndarray | None = None
    sequence_scale: np.ndarray | None = None
    context_center: np.ndarray | None = None
    context_scale: np.ndarray | None = None
    fitted: bool = False

    BINARY_SEQUENCE_NAMES = {
        "release_inversion_flag",
        "correction_key_flag",
        "repeat_flag",
        "shift_at_press",
        "ctrl_at_press",
        "alt_at_press",
        "meta_at_press",
    }

    def fit(
        self,
        sequence: np.ndarray,
        context: np.ndarray,
    ) -> "FeatureNormalizerV2":
        self._validate_arrays(sequence, context)

        flattened = sequence.reshape(-1, sequence.shape[-1]).astype(
            np.float64,
            copy=False,
        )
        seq_center = np.median(flattened, axis=0)
        seq_q1 = np.quantile(flattened, 0.25, axis=0)
        seq_q3 = np.quantile(flattened, 0.75, axis=0)
        seq_scale = seq_q3 - seq_q1

        for index, name in enumerate(SEQUENCE_FEATURES):
            if name in self.BINARY_SEQUENCE_NAMES:
                seq_center[index] = 0.0
                seq_scale[index] = 1.0

        seq_scale[seq_scale < 1.0e-9] = 1.0

        context64 = context.astype(np.float64, copy=False)
        ctx_center = np.median(context64, axis=0)
        ctx_q1 = np.quantile(context64, 0.25, axis=0)
        ctx_q3 = np.quantile(context64, 0.75, axis=0)
        ctx_scale = ctx_q3 - ctx_q1
        ctx_scale[ctx_scale < 1.0e-9] = 1.0

        self.sequence_center = seq_center.astype(np.float32)
        self.sequence_scale = seq_scale.astype(np.float32)
        self.context_center = ctx_center.astype(np.float32)
        self.context_scale = ctx_scale.astype(np.float32)
        self.fitted = True
        return self

    def transform(
        self,
        sequence: np.ndarray,
        context: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        self._validate_arrays(sequence, context)

        normalized_sequence = (
            sequence.astype(np.float32, copy=False) - self.sequence_center
        ) / self.sequence_scale
        normalized_context = (
            context.astype(np.float32, copy=False) - self.context_center
        ) / self.context_scale

        if not np.isfinite(normalized_sequence).all():
            raise ValueError("Normalized sequence contains NaN or infinity")
        if not np.isfinite(normalized_context).all():
            raise ValueError("Normalized context contains NaN or infinity")

        return normalized_sequence.astype(np.float32), normalized_context.astype(
            np.float32
        )

    def fit_transform(
        self,
        sequence: np.ndarray,
        context: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(sequence, context).transform(sequence, context)

    def state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "schema_hash": EXPECTED_SCHEMA_SHA256,
            "sequence_features": list(SEQUENCE_FEATURES),
            "window_context_features": list(WINDOW_CONTEXT_FEATURES),
            "sequence_center": self.sequence_center.tolist(),
            "sequence_scale": self.sequence_scale.tolist(),
            "context_center": self.context_center.tolist(),
            "context_scale": self.context_scale.tolist(),
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FeatureNormalizerV2":
        if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Normalizer schema version mismatch")
        if payload.get("schema_hash") != EXPECTED_SCHEMA_SHA256:
            raise ValueError("Normalizer schema hash mismatch")
        if list(payload.get("sequence_features", [])) != list(
            SEQUENCE_FEATURES
        ):
            raise ValueError("Normalizer sequence feature order mismatch")
        if list(payload.get("window_context_features", [])) != list(
            WINDOW_CONTEXT_FEATURES
        ):
            raise ValueError("Normalizer context feature order mismatch")

        normalizer = cls(
            sequence_center=np.asarray(
                payload["sequence_center"], dtype=np.float32
            ),
            sequence_scale=np.asarray(
                payload["sequence_scale"], dtype=np.float32
            ),
            context_center=np.asarray(
                payload["context_center"], dtype=np.float32
            ),
            context_scale=np.asarray(
                payload["context_scale"], dtype=np.float32
            ),
            fitted=True,
        )
        return normalizer

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("FeatureNormalizerV2 is not fitted")

    @staticmethod
    def _validate_arrays(
        sequence: np.ndarray,
        context: np.ndarray,
    ) -> None:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [N, 50, 12]")
        if context.ndim != 2:
            raise ValueError("context must have shape [N, 12]")
        if sequence.shape[0] != context.shape[0]:
            raise ValueError("sequence/context sample count mismatch")
        if sequence.shape[2] != len(SEQUENCE_FEATURES):
            raise ValueError("sequence feature count mismatch")
        if context.shape[1] != len(WINDOW_CONTEXT_FEATURES):
            raise ValueError("context feature count mismatch")
        if not np.isfinite(sequence).all():
            raise ValueError("sequence contains NaN or infinity")
        if not np.isfinite(context).all():
            raise ValueError("context contains NaN or infinity")
