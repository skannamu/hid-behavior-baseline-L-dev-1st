"""Leakage-safe hard-negative fine-tuning for ReCon-HID v9."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data_v2 import (
    BalancedSamplingConfig,
    FeatureV2WindowDataset,
    balanced_subsample,
    group_key_for_mode,
    load_manifest_entries,
    resolve_manifest_window_paths,
)
from src.features.schema import EXPECTED_SCHEMA_SHA256, FEATURE_SCHEMA_VERSION
from src.models_v9 import save_v9_checkpoint
from src.training_v9 import (
    FeatureNormalizerV2,
    compute_v9_scores,
    empirical_fpr,
    quantile_threshold,
)

from .bundle import load_defender_bundle
from .common import atomic_json, set_reproducibility, sha256_file, stable_rank


ATTACK_LABELS = {"attack", "malicious", "1"}
NORMAL_LABELS = {"normal", "benign", "0"}


@dataclass(frozen=True)
class HardenedTrainingConfig:
    seed: int = 20260807
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    patience: int = 5
    min_delta: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    num_workers: int = 0

    sequence_reconstruction_weight: float = 0.5
    context_reconstruction_weight: float = 0.25
    classifier_weight: float = 1.0
    latent_l2_weight: float = 1.0e-4

    score_context_weight: float = 0.5
    target_fpr: float = 0.01
    attack_dev_ratio: float = 0.20
    max_attack_train_windows: int = 4000
    device: str = "cuda"
    deterministic: bool = True

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "patience", "max_attack_train_windows"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for name in (
            "weight_decay",
            "min_delta",
            "gradient_clip_norm",
            "sequence_reconstruction_weight",
            "context_reconstruction_weight",
            "classifier_weight",
            "latent_l2_weight",
            "score_context_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        if not 0.0 <= self.attack_dev_ratio < 1.0:
            raise ValueError("attack_dev_ratio must be in [0, 1)")


@dataclass(frozen=True)
class HardenedTrainingResult:
    run_dir: str
    defender_id: str
    parent_run_dir: str
    best_epoch: int
    normal_train_windows: int
    normal_calibration_windows: int
    attack_train_windows: int
    attack_dev_windows: int
    reconstruction_threshold: float
    prototype_threshold: float
    classifier_threshold: float
    decision_threshold: float
    calibration_decision_fpr: float
    attack_dev_detection_rate: float | None


def _assert_labels(
    dataset: FeatureV2WindowDataset,
    allowed: set[str],
    description: str,
) -> None:
    labels = {str(value).strip().lower() for value in dataset.labels.tolist()}
    unsupported = labels - allowed
    if unsupported:
        raise ValueError(f"{description} has unsupported labels: {sorted(unsupported)}")


def _subset_by_group_names(
    dataset: FeatureV2WindowDataset,
    *,
    group_mode: str,
    group_names: Sequence[str],
) -> FeatureV2WindowDataset:
    selected = set(str(value) for value in group_names)
    indices = [
        index
        for index, sample in enumerate(dataset)
        if group_key_for_mode(sample, group_mode) in selected
    ]
    if not indices:
        raise ValueError(
            f"No normal windows matched groups={sorted(selected)} "
            f"under group_mode={group_mode}"
        )
    return dataset.subset(indices)


def _load_frozen_normal_splits(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    parent_split_manifest: dict[str, Any],
) -> tuple[
    FeatureV2WindowDataset,
    FeatureV2WindowDataset,
    FeatureV2WindowDataset,
]:
    entries = load_manifest_entries(
        manifest_path,
        dataset_root=dataset_root,
        verify_files=True,
        verify_hashes=True,
    )
    paths = resolve_manifest_window_paths(entries, dataset_root=dataset_root)
    dataset, _ = FeatureV2WindowDataset.from_paths(paths)
    _assert_labels(dataset, NORMAL_LABELS, "Normal manifest")

    group_mode = str(parent_split_manifest.get("group_mode", "participant"))
    train = _subset_by_group_names(
        dataset,
        group_mode=group_mode,
        group_names=parent_split_manifest["train_groups"],
    )
    calibration = _subset_by_group_names(
        dataset,
        group_mode=group_mode,
        group_names=parent_split_manifest["calibration_groups"],
    )
    test = _subset_by_group_names(
        dataset,
        group_mode=group_mode,
        group_names=parent_split_manifest["test_groups"],
    )
    return train, calibration, test


def _normal_balance_config(parent_config: dict[str, Any], seed: int) -> BalancedSamplingConfig:
    normal = parent_config.get("normal_sampling")
    if not isinstance(normal, dict):
        normal = parent_config.get("normal_pretrain")
    if not isinstance(normal, dict):
        hardened = parent_config.get("hardened_training")
        normal = hardened if isinstance(hardened, dict) else {}
    return BalancedSamplingConfig(
        enabled=bool(normal.get("balance_training", True)),
        window_stride=int(normal.get("training_window_stride", 5)),
        balance_mode=str(normal.get("training_balance_mode", "equal")),
        target_windows_per_group=normal.get("training_target_windows_per_group"),
        max_windows_per_group=normal.get("training_max_windows_per_group", 3000),
        seed=seed,
    )


def _load_attack_dataset(paths: str | Path | Sequence[str | Path]) -> tuple[FeatureV2WindowDataset, list[Path]]:
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_file():
            resolved.append(path)
        elif path.is_dir():
            resolved.extend(sorted(path.rglob("window.csv")))
        else:
            raise FileNotFoundError(path)
    if not resolved:
        raise FileNotFoundError("No attack window.csv files were resolved")
    dataset, _ = FeatureV2WindowDataset.from_paths(resolved)
    _assert_labels(dataset, ATTACK_LABELS, "Attack hard negatives")
    return dataset, resolved


def _split_attack_by_session(
    dataset: FeatureV2WindowDataset,
    *,
    dev_ratio: float,
    seed: int,
) -> tuple[FeatureV2WindowDataset, FeatureV2WindowDataset | None, tuple[str, ...], tuple[str, ...]]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset):
        groups.setdefault(sample.session_id, []).append(index)
    names = sorted(groups, key=lambda name: (stable_rank(name, seed=seed), name))
    if len(names) < 2 or dev_ratio <= 0:
        return dataset, None, tuple(names), tuple()

    n_dev = max(1, int(round(len(names) * dev_ratio)))
    n_dev = min(n_dev, len(names) - 1)
    dev_names = tuple(names[-n_dev:])
    train_names = tuple(names[:-n_dev])
    train_indices = [index for name in train_names for index in groups[name]]
    dev_indices = [index for name in dev_names for index in groups[name]]
    return (
        dataset.subset(train_indices),
        dataset.subset(dev_indices),
        train_names,
        dev_names,
    )


def _cap_attack_dataset(
    dataset: FeatureV2WindowDataset,
    *,
    maximum: int,
    seed: int,
) -> FeatureV2WindowDataset:
    if len(dataset) <= maximum:
        return dataset
    indices = sorted(
        range(len(dataset)),
        key=lambda index: (
            stable_rank(
                dataset[index].session_id,
                dataset[index].window_id,
                index,
                seed=seed,
            ),
            index,
        ),
    )[:maximum]
    return dataset.subset(sorted(indices))


def _normal_only_training_loss(
    output: Any,
    sequence: Tensor,
    context: Tensor,
    labels: Tensor,
    *,
    config: HardenedTrainingConfig,
    pos_weight: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    sequence_per = (
        output.sequence_reconstruction.sub(sequence).pow(2).mean(dim=(1, 2))
    )
    context_per = (
        output.context_reconstruction.sub(context).pow(2).mean(dim=1)
    )
    normal_mask = labels < 0.5
    if bool(normal_mask.any()):
        sequence_loss = sequence_per[normal_mask].mean()
        context_loss = context_per[normal_mask].mean()
    else:
        sequence_loss = sequence_per.mean() * 0.0
        context_loss = context_per.mean() * 0.0

    classifier_loss = F.binary_cross_entropy_with_logits(
        output.classifier_logit,
        labels,
        pos_weight=pos_weight,
    )
    latent_l2 = output.fused_latent.pow(2).mean()
    total = (
        config.sequence_reconstruction_weight * sequence_loss
        + config.context_reconstruction_weight * context_loss
        + config.classifier_weight * classifier_loss
        + config.latent_l2_weight * latent_l2
    )
    return total, {
        "sequence_reconstruction": sequence_loss,
        "context_reconstruction": context_loss,
        "classifier": classifier_loss,
        "latent_l2": latent_l2,
    }


def _make_mixed_loader(
    normal_sequence: np.ndarray,
    normal_context: np.ndarray,
    attack_sequence: np.ndarray,
    attack_context: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    sequence = np.concatenate([normal_sequence, attack_sequence], axis=0)
    context = np.concatenate([normal_context, attack_context], axis=0)
    labels = np.concatenate([
        np.zeros(len(normal_sequence), dtype=np.float32),
        np.ones(len(attack_sequence), dtype=np.float32),
    ])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(sequence).float(),
            torch.from_numpy(context).float(),
            torch.from_numpy(labels).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def _validation_objective(
    model: Any,
    *,
    normal_sequence: Tensor,
    normal_context: Tensor,
    attack_sequence: Tensor | None,
    attack_context: Tensor | None,
    batch_size: int,
    device: torch.device,
    config: HardenedTrainingConfig,
) -> float:
    model.eval()
    total = 0.0
    count = 0

    normal_loader = DataLoader(
        TensorDataset(normal_sequence, normal_context),
        batch_size=batch_size,
        shuffle=False,
    )
    for sequence, context in normal_loader:
        sequence = sequence.to(device)
        context = context.to(device)
        output = model(sequence, context)
        recon = output.sequence_reconstruction.sub(sequence).pow(2).mean()
        ctx = output.context_reconstruction.sub(context).pow(2).mean()
        cls = F.binary_cross_entropy_with_logits(
            output.classifier_logit,
            torch.zeros_like(output.classifier_logit),
        )
        loss = (
            config.sequence_reconstruction_weight * recon
            + config.context_reconstruction_weight * ctx
            + config.classifier_weight * cls
        )
        batch_count = sequence.shape[0]
        total += float(loss.detach().cpu()) * batch_count
        count += batch_count

    if attack_sequence is not None and attack_context is not None:
        attack_loader = DataLoader(
            TensorDataset(attack_sequence, attack_context),
            batch_size=batch_size,
            shuffle=False,
        )
        for sequence, context in attack_loader:
            sequence = sequence.to(device)
            context = context.to(device)
            output = model(sequence, context)
            cls = F.binary_cross_entropy_with_logits(
                output.classifier_logit,
                torch.ones_like(output.classifier_logit),
            )
            batch_count = sequence.shape[0]
            total += float(config.classifier_weight * cls.detach().cpu()) * batch_count
            count += batch_count

    if count == 0:
        raise ValueError("Validation objective received zero windows")
    return total / count


@torch.no_grad()
def _encode_array(
    model: Any,
    sequence: np.ndarray,
    context: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(sequence).float(),
            torch.from_numpy(context).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    latents: list[Tensor] = []
    model.eval()
    for seq_batch, ctx_batch in loader:
        _, _, fused = model.encode(seq_batch.to(device), ctx_batch.to(device))
        latents.append(fused.detach())
    if not latents:
        raise ValueError("Cannot encode zero normal training windows")
    return torch.cat(latents, dim=0)


@torch.no_grad()
def _score_arrays(
    model: Any,
    sequence: np.ndarray,
    context: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    context_weight: float,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(sequence).float(),
            torch.from_numpy(context).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    reconstruction: list[np.ndarray] = []
    prototype: list[np.ndarray] = []
    classifier: list[np.ndarray] = []
    model.eval()
    for seq_batch, ctx_batch in loader:
        scores = compute_v9_scores(
            model,
            seq_batch.to(device),
            ctx_batch.to(device),
            context_weight=context_weight,
        )
        if scores.prototype_distance is None:
            raise RuntimeError("Normal prototype is not ready")
        reconstruction.append(scores.reconstruction_combined)
        prototype.append(scores.prototype_distance)
        classifier.append(scores.classifier_probability)
    if not reconstruction:
        raise ValueError("Cannot score zero windows")
    return {
        "reconstruction": np.concatenate(reconstruction).astype(np.float64),
        "prototype": np.concatenate(prototype).astype(np.float64),
        "classifier": np.concatenate(classifier).astype(np.float64),
    }


def _decision_scores(
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> np.ndarray:
    ratios = [
        scores[name] / max(abs(float(thresholds[name])), 1.0e-12)
        for name in ("reconstruction", "prototype", "classifier")
    ]
    return np.maximum.reduce(ratios)


def train_hardened_defender_v9(
    *,
    normal_dataset_root: str | Path,
    normal_manifest_path: str | Path,
    parent_run_dir: str | Path,
    attack_paths: str | Path | Sequence[str | Path],
    output_root: str | Path,
    defender_id: str,
    run_name: str | None = None,
    config: HardenedTrainingConfig | None = None,
) -> HardenedTrainingResult:
    cfg = config or HardenedTrainingConfig()
    cfg.validate()
    set_reproducibility(cfg.seed, cfg.deterministic)

    if not defender_id.strip():
        raise ValueError("defender_id must not be empty")
    parent = load_defender_bundle(parent_run_dir, map_location="cpu")
    normal_train_raw, normal_calibration, _normal_test = _load_frozen_normal_splits(
        dataset_root=normal_dataset_root,
        manifest_path=normal_manifest_path,
        parent_split_manifest=parent.split_manifest,
    )
    normal_sampling_config = _normal_balance_config(parent.config, cfg.seed)
    balance_result = balanced_subsample(
        normal_train_raw,
        config=normal_sampling_config,
    )
    normal_train = balance_result.dataset

    attacks, resolved_attack_paths = _load_attack_dataset(attack_paths)
    attack_train, attack_dev, attack_train_groups, attack_dev_groups = (
        _split_attack_by_session(
            attacks,
            dev_ratio=cfg.attack_dev_ratio,
            seed=cfg.seed,
        )
    )
    attack_train = _cap_attack_dataset(
        attack_train,
        maximum=cfg.max_attack_train_windows,
        seed=cfg.seed,
    )

    normalizer: FeatureNormalizerV2 = parent.normalizer
    normal_train_sequence, normal_train_context = normalizer.transform(
        normal_train.sequence_array,
        normal_train.context_array,
    )
    normal_cal_sequence, normal_cal_context = normalizer.transform(
        normal_calibration.sequence_array,
        normal_calibration.context_array,
    )
    attack_train_sequence, attack_train_context = normalizer.transform(
        attack_train.sequence_array,
        attack_train.context_array,
    )
    if attack_dev is not None:
        attack_dev_sequence, attack_dev_context = normalizer.transform(
            attack_dev.sequence_array,
            attack_dev.context_array,
        )
    else:
        attack_dev_sequence = attack_dev_context = None

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {cfg.device!r}, but CUDA is unavailable")
    device = torch.device(cfg.device)
    model = parent.model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    normal_count = len(normal_train)
    attack_count = len(attack_train)
    pos_weight = torch.tensor(
        normal_count / max(attack_count, 1),
        dtype=torch.float32,
        device=device,
    )
    train_loader = _make_mixed_loader(
        normal_train_sequence,
        normal_train_context,
        attack_train_sequence,
        attack_train_context,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_name = run_name or f"{defender_id}_{timestamp}"
    run_dir = Path(output_root) / resolved_name
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_model.pt"

    atomic_json(run_dir / "config.json", {
        "hardened_training": asdict(cfg),
        "model": model.config.to_dict(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "defender_id": defender_id,
        "parent_run_dir": str(Path(parent_run_dir)),
        "normal_sampling": asdict(normal_sampling_config),
    })
    atomic_json(run_dir / "normalizer.json", normalizer.state_dict())
    atomic_json(run_dir / "split_manifest.json", {
        "group_mode": parent.split_manifest.get("group_mode", "participant"),
        "train_groups": list(parent.split_manifest["train_groups"]),
        "calibration_groups": list(parent.split_manifest["calibration_groups"]),
        "test_groups": list(parent.split_manifest["test_groups"]),
        "normal_source_manifest": str(Path(normal_manifest_path)),
        "normal_raw_train_windows": len(normal_train_raw),
        "normal_train_windows": len(normal_train),
        "normal_training_balance": balance_result.report,
        "normal_calibration_windows": len(normal_calibration),
        "normal_test_status": "frozen_untouched",
        "attack_train_groups": list(attack_train_groups),
        "attack_dev_groups": list(attack_dev_groups),
        "attack_train_windows": len(attack_train),
        "attack_dev_windows": len(attack_dev) if attack_dev is not None else 0,
        "attack_paths": [str(path) for path in resolved_attack_paths],
        "attack_sha256": {
            str(path): sha256_file(path) for path in resolved_attack_paths
        },
        "random_window_split": False,
    })

    normal_cal_seq_tensor = torch.from_numpy(normal_cal_sequence).float()
    normal_cal_ctx_tensor = torch.from_numpy(normal_cal_context).float()
    attack_dev_seq_tensor = (
        torch.from_numpy(attack_dev_sequence).float()
        if attack_dev_sequence is not None
        else None
    )
    attack_dev_ctx_tensor = (
        torch.from_numpy(attack_dev_context).float()
        if attack_dev_context is not None
        else None
    )

    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for sequence, context, labels in train_loader:
            sequence = sequence.to(device, non_blocking=device.type == "cuda")
            context = context.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence, context)
            loss, _parts = _normal_only_training_loss(
                output,
                sequence,
                context,
                labels,
                config=cfg,
                pos_weight=pos_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.gradient_clip_norm,
            )
            optimizer.step()
            batch_count = sequence.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count

        train_loss = total_loss / max(total_count, 1)
        validation_loss = _validation_objective(
            model,
            normal_sequence=normal_cal_seq_tensor,
            normal_context=normal_cal_ctx_tensor,
            attack_sequence=attack_dev_seq_tensor,
            attack_context=attack_dev_ctx_tensor,
            batch_size=cfg.batch_size,
            device=device,
            config=cfg,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })
        atomic_json(run_dir / "history.json", history)

        if validation_loss < best_validation - cfg.min_delta:
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            save_v9_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    "stage": "hard_negative_finetuning",
                    "defender_id": defender_id,
                    "parent_run_dir": str(Path(parent_run_dir)),
                    "normalizer_state": normalizer.state_dict(),
                    "classifier_trained": True,
                    "best_validation_loss": best_validation,
                },
            )
        else:
            stale_epochs += 1

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} best_epoch={best_epoch:03d}"
        )
        if stale_epochs >= cfg.patience:
            print(f"early_stopping epoch={epoch} patience={cfg.patience}")
            break

    if best_epoch == 0:
        raise RuntimeError("Hard-negative training produced no checkpoint")

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    normal_latents = _encode_array(
        model,
        normal_train_sequence,
        normal_train_context,
        batch_size=cfg.batch_size,
        device=device,
    )
    model.set_normal_prototype(normal_latents)
    save_v9_checkpoint(
        best_path,
        model=model,
        optimizer=None,
        epoch=best_epoch,
        extra={
            "stage": "hard_negative_finetuning",
            "defender_id": defender_id,
            "parent_run_dir": str(Path(parent_run_dir)),
            "normalizer_state": normalizer.state_dict(),
            "classifier_trained": True,
            "best_validation_loss": best_validation,
            "prototype_source": "balanced_normal_train_only",
        },
    )

    normal_scores = _score_arrays(
        model,
        normal_cal_sequence,
        normal_cal_context,
        batch_size=cfg.batch_size,
        device=device,
        context_weight=cfg.score_context_weight,
    )
    thresholds = {
        name: quantile_threshold(values, target_fpr=cfg.target_fpr)
        for name, values in normal_scores.items()
    }
    normal_decision = _decision_scores(normal_scores, thresholds)
    decision_threshold = quantile_threshold(
        normal_decision,
        target_fpr=cfg.target_fpr,
    )
    decision_fpr = empirical_fpr(normal_decision, decision_threshold)

    attack_dev_detection_rate: float | None = None
    if attack_dev_sequence is not None and attack_dev_context is not None:
        attack_scores = _score_arrays(
            model,
            attack_dev_sequence,
            attack_dev_context,
            batch_size=cfg.batch_size,
            device=device,
            context_weight=cfg.score_context_weight,
        )
        attack_decision = _decision_scores(attack_scores, thresholds)
        attack_dev_detection_rate = float(
            np.mean(attack_decision > decision_threshold)
        )

    atomic_json(run_dir / "calibration.json", {
        "source_split": "normal_calibration_only",
        "target_fpr": cfg.target_fpr,
        "detection_rule": "decision_score > threshold",
        "reconstruction": {
            "threshold": thresholds["reconstruction"],
            "empirical_fpr": empirical_fpr(
                normal_scores["reconstruction"],
                thresholds["reconstruction"],
            ),
        },
        "prototype": {
            "threshold": thresholds["prototype"],
            "empirical_fpr": empirical_fpr(
                normal_scores["prototype"],
                thresholds["prototype"],
            ),
        },
        "classifier": {
            "threshold": thresholds["classifier"],
            "empirical_fpr": empirical_fpr(
                normal_scores["classifier"],
                thresholds["classifier"],
            ),
            "trained": True,
        },
        "decision_policy": {
            "type": "max_component_threshold_ratio",
            "components": ["reconstruction", "prototype", "classifier"],
            "threshold": decision_threshold,
            "empirical_fpr": decision_fpr,
            "selection_source": "normal_calibration_only",
        },
        "attack_dev_metrics": {
            "source": "current_round_nonfinal_attack_dev",
            "detection_rate": attack_dev_detection_rate,
            "sample_count": len(attack_dev) if attack_dev is not None else 0,
        },
    })
    atomic_json(run_dir / "untouched_test.json", {
        "status": "reserved",
        "reason": (
            "The frozen normal test participants are not evaluated during "
            "iterative D1/D2/... training. Evaluate them only after D_final is frozen."
        ),
        "test_groups": list(parent.split_manifest["test_groups"]),
    })

    result = HardenedTrainingResult(
        run_dir=str(run_dir),
        defender_id=defender_id,
        parent_run_dir=str(Path(parent_run_dir)),
        best_epoch=best_epoch,
        normal_train_windows=len(normal_train),
        normal_calibration_windows=len(normal_calibration),
        attack_train_windows=len(attack_train),
        attack_dev_windows=len(attack_dev) if attack_dev is not None else 0,
        reconstruction_threshold=thresholds["reconstruction"],
        prototype_threshold=thresholds["prototype"],
        classifier_threshold=thresholds["classifier"],
        decision_threshold=decision_threshold,
        calibration_decision_fpr=decision_fpr,
        attack_dev_detection_rate=attack_dev_detection_rate,
    )
    atomic_json(run_dir / "result.json", asdict(result))
    return result
