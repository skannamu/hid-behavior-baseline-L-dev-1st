"""Load and score schema-bound Stable-v9 defender artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_v2.window_dataset import FeatureV2WindowDataset
from src.models_v9 import ReConHIDV9, load_v9_checkpoint
from src.training_v9 import FeatureNormalizerV2, compute_v9_scores

from .common import read_json


_EPSILON = 1.0e-12


@dataclass(frozen=True)
class DefenderScoreSet:
    sequence_reconstruction: np.ndarray
    context_reconstruction: np.ndarray
    reconstruction_combined: np.ndarray
    classifier_probability: np.ndarray
    prototype_distance: np.ndarray
    decision_score: np.ndarray
    predicted_attack: np.ndarray

    def __len__(self) -> int:
        return int(self.decision_score.size)


@dataclass
class DefenderBundle:
    run_dir: Path
    checkpoint_path: Path
    model: ReConHIDV9
    checkpoint_payload: dict[str, Any]
    normalizer: FeatureNormalizerV2
    calibration: dict[str, Any]
    split_manifest: dict[str, Any]
    config: dict[str, Any]
    classifier_trained: bool

    @property
    def defender_id(self) -> str:
        extra = self.checkpoint_payload.get("extra") or {}
        return str(extra.get("defender_id") or self.run_dir.name)

    @property
    def policy_components(self) -> tuple[str, ...]:
        policy = self.calibration.get("decision_policy")
        if isinstance(policy, dict):
            components = policy.get("components")
            if isinstance(components, list) and components:
                return tuple(str(value) for value in components)
        components = ["reconstruction", "prototype"]
        if self.classifier_trained and self._classifier_threshold() is not None:
            components.append("classifier")
        return tuple(components)

    def _reconstruction_threshold(self) -> float:
        section = self.calibration.get("reconstruction")
        if not isinstance(section, dict) or section.get("threshold") is None:
            raise ValueError("Calibration is missing reconstruction.threshold")
        return float(section["threshold"])

    def _prototype_threshold(self) -> float:
        section = self.calibration.get("prototype")
        if not isinstance(section, dict) or section.get("threshold") is None:
            raise ValueError("Calibration is missing prototype.threshold")
        return float(section["threshold"])

    def _classifier_threshold(self) -> float | None:
        section = self.calibration.get("classifier")
        if isinstance(section, dict) and section.get("threshold") is not None:
            return float(section["threshold"])
        legacy = self.calibration.get("classifier_threshold")
        return float(legacy) if legacy is not None else None

    def _decision_threshold(self) -> float:
        policy = self.calibration.get("decision_policy")
        if isinstance(policy, dict) and policy.get("threshold") is not None:
            return float(policy["threshold"])
        # D0 artifacts predate the fused calibration policy. A component ratio
        # above one means that one calibration-only component threshold was
        # crossed. This fallback is used only until D1 writes a fused policy.
        return 1.0

    def decision_score_from_components(
        self,
        *,
        reconstruction: np.ndarray,
        prototype: np.ndarray,
        classifier: np.ndarray,
    ) -> np.ndarray:
        ratios: list[np.ndarray] = []
        components = self.policy_components
        if "reconstruction" in components:
            ratios.append(
                reconstruction / max(abs(self._reconstruction_threshold()), _EPSILON)
            )
        if "prototype" in components:
            ratios.append(
                prototype / max(abs(self._prototype_threshold()), _EPSILON)
            )
        if "classifier" in components:
            classifier_threshold = self._classifier_threshold()
            if classifier_threshold is None:
                raise ValueError(
                    "Classifier is part of the decision policy but no threshold exists"
                )
            ratios.append(classifier / max(abs(classifier_threshold), _EPSILON))
        if not ratios:
            raise ValueError("Decision policy has no components")
        return np.maximum.reduce(ratios)

    def score_dataset(
        self,
        dataset: FeatureV2WindowDataset,
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
        context_weight: float | None = None,
    ) -> DefenderScoreSet:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        sequence, context = self.normalizer.transform(
            dataset.sequence_array,
            dataset.context_array,
        )
        seq_tensor = torch.from_numpy(sequence).float()
        ctx_tensor = torch.from_numpy(context).float()
        loader = DataLoader(
            TensorDataset(seq_tensor, ctx_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        weight = context_weight
        if weight is None:
            normal_cfg = self.config.get("normal_pretrain")
            if isinstance(normal_cfg, dict):
                weight = float(normal_cfg.get("score_context_weight", 0.5))
            else:
                hardened_cfg = self.config.get("hardened_training")
                weight = float(
                    hardened_cfg.get("score_context_weight", 0.5)
                    if isinstance(hardened_cfg, dict)
                    else 0.5
                )

        self.model.to(resolved_device)
        self.model.eval()

        sequence_scores: list[np.ndarray] = []
        context_scores: list[np.ndarray] = []
        reconstruction_scores: list[np.ndarray] = []
        classifier_scores: list[np.ndarray] = []
        prototype_scores: list[np.ndarray] = []

        for seq_batch, ctx_batch in loader:
            scores = compute_v9_scores(
                self.model,
                seq_batch.to(resolved_device),
                ctx_batch.to(resolved_device),
                context_weight=float(weight),
            )
            if scores.prototype_distance is None:
                raise RuntimeError("Defender checkpoint has no normal prototype")
            sequence_scores.append(scores.sequence_reconstruction)
            context_scores.append(scores.context_reconstruction)
            reconstruction_scores.append(scores.reconstruction_combined)
            classifier_scores.append(scores.classifier_probability)
            prototype_scores.append(scores.prototype_distance)

        if not reconstruction_scores:
            raise ValueError("Cannot score an empty dataset")

        sequence_array = np.concatenate(sequence_scores).astype(np.float64)
        context_array = np.concatenate(context_scores).astype(np.float64)
        reconstruction_array = np.concatenate(reconstruction_scores).astype(np.float64)
        classifier_array = np.concatenate(classifier_scores).astype(np.float64)
        prototype_array = np.concatenate(prototype_scores).astype(np.float64)
        decision_score = self.decision_score_from_components(
            reconstruction=reconstruction_array,
            prototype=prototype_array,
            classifier=classifier_array,
        )
        predicted = (decision_score > self._decision_threshold()).astype(np.int64)

        return DefenderScoreSet(
            sequence_reconstruction=sequence_array,
            context_reconstruction=context_array,
            reconstruction_combined=reconstruction_array,
            classifier_probability=classifier_array,
            prototype_distance=prototype_array,
            decision_score=decision_score,
            predicted_attack=predicted,
        )


def load_defender_bundle(
    run_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> DefenderBundle:
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)

    checkpoint_path = root / "best_model.pt"
    normalizer_path = root / "normalizer.json"
    calibration_path = root / "calibration.json"
    split_path = root / "split_manifest.json"
    config_path = root / "config.json"
    for path in (
        checkpoint_path,
        normalizer_path,
        calibration_path,
        split_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    model, payload = load_v9_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    extra = payload.get("extra") or {}
    classifier_trained = bool(extra.get("classifier_trained", False))
    normalizer = FeatureNormalizerV2.from_state_dict(read_json(normalizer_path))

    return DefenderBundle(
        run_dir=root,
        checkpoint_path=checkpoint_path,
        model=model,
        checkpoint_payload=payload,
        normalizer=normalizer,
        calibration=read_json(calibration_path),
        split_manifest=read_json(split_path),
        config=read_json(config_path),
        classifier_trained=classifier_trained,
    )
