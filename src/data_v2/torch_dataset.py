"""PyTorch adapter for the strict Feature Schema v2 dataset."""

from __future__ import annotations

from typing import Callable

from .window_dataset import FeatureV2WindowDataset


class TorchFeatureV2Dataset:
    """Lazy torch adapter without making torch a loader-level dependency."""

    def __init__(
        self,
        dataset: FeatureV2WindowDataset,
        *,
        label_encoder: Callable[[str], int] | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "TorchFeatureV2Dataset requires PyTorch"
            ) from exc

        self._torch = torch
        self.dataset = dataset
        self.label_encoder = label_encoder or self._default_label_encoder

    @staticmethod
    def _default_label_encoder(label: str) -> int:
        normalized = label.strip().lower()
        if normalized in {"normal", "benign", "0"}:
            return 0
        if normalized in {"attack", "malicious", "1"}:
            return 1
        raise ValueError(f"Unknown binary label: {label!r}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return {
            "sequence": self._torch.from_numpy(sample.sequence.copy()).float(),
            "context": self._torch.from_numpy(sample.context.copy()).float(),
            "label": self._torch.tensor(
                self.label_encoder(sample.label),
                dtype=self._torch.long,
            ),
            "participant_id": sample.participant_id,
            "session_id": sample.session_id,
            "window_id": sample.window_id,
        }
