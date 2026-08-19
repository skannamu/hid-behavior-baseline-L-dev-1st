"""Leakage-safe normal-only pretraining for ReCon-HID v9."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.data_v2 import (
    BalancedSamplingConfig,
    FeatureV2WindowDataset,
    GroupSplitConfig,
    balanced_subsample,
    load_manifest_entries,
    resolve_manifest_window_paths,
    split_by_group,
)
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
)
from src.models_v9 import (
    ReConHIDV9,
    ReConHIDV9Config,
    save_v9_checkpoint,
)
from .losses import ReConHIDV9Loss, ReConHIDV9LossConfig
from .normalizer import FeatureNormalizerV2
from .scoring import compute_v9_scores, empirical_fpr, quantile_threshold


NORMAL_LABELS = {"normal", "benign", "0"}


@dataclass(frozen=True)
class NormalPretrainConfig:
    seed: int = 20260804
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 8
    min_delta: float = 1.0e-5
    num_workers: int = 0

    train_ratio: float = 0.70
    calibration_ratio: float = 0.15
    test_ratio: float = 0.15
    split_group_mode: str = "participant"

    explicit_train_groups: tuple[str, ...] | None = None
    explicit_calibration_groups: tuple[str, ...] | None = None
    explicit_test_groups: tuple[str, ...] | None = None

    # Main-paper experiments keep the held-out Test participant
    # completely unscored until D_final has been selected.
    # Legacy/smoke execution may deliberately enable this.
    evaluate_test_during_pretraining: bool = True

    balance_training: bool = True
    training_window_stride: int = 5
    training_balance_mode: str = "equal"
    training_target_windows_per_group: int | None = None
    training_max_windows_per_group: int | None = 3000

    sequence_reconstruction_weight: float = 1.0
    context_reconstruction_weight: float = 0.5
    latent_l2_weight: float = 1.0e-4

    score_context_weight: float = 0.5
    target_fpr: float = 0.01

    device: str = "cuda"
    deterministic: bool = True

    def validate(self) -> None:
        integer_positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
        for name, value in integer_positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        nonnegative = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "min_delta": self.min_delta,
            "sequence_reconstruction_weight": (
                self.sequence_reconstruction_weight
            ),
            "context_reconstruction_weight": (
                self.context_reconstruction_weight
            ),
            "latent_l2_weight": self.latent_l2_weight,
            "score_context_weight": self.score_context_weight,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")

        GroupSplitConfig(
            train_ratio=self.train_ratio,
            calibration_ratio=self.calibration_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
            group_mode=self.split_group_mode,
            explicit_train_groups=self.explicit_train_groups,
            explicit_calibration_groups=(
                self.explicit_calibration_groups
            ),
            explicit_test_groups=self.explicit_test_groups,
        ).validate()
        BalancedSamplingConfig(
            enabled=self.balance_training,
            window_stride=self.training_window_stride,
            balance_mode=self.training_balance_mode,
            target_windows_per_group=self.training_target_windows_per_group,
            max_windows_per_group=self.training_max_windows_per_group,
            seed=self.seed,
        ).validate()


@dataclass(frozen=True)
class NormalPretrainResult:
    run_dir: str
    best_epoch: int
    train_windows: int
    calibration_windows: int
    test_windows: int
    reconstruction_threshold: float
    prototype_threshold: float
    calibration_reconstruction_fpr: float
    calibration_prototype_fpr: float
    test_reconstruction_fpr: float | None
    test_prototype_fpr: float | None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def _assert_normal_only(dataset: FeatureV2WindowDataset) -> None:
    labels = {str(label).strip().lower() for label in dataset.labels.tolist()}
    unsupported = labels - NORMAL_LABELS
    if unsupported:
        raise ValueError(
            "Normal-only pretraining received non-normal labels: "
            f"{sorted(unsupported)}"
        )


def _to_tensor_dataset(
    sequence: np.ndarray,
    context: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(sequence).float(),
        torch.from_numpy(context).float(),
    )


def _evaluate_reconstruction_loss(
    model: ReConHIDV9,
    sequence: Tensor,
    context: Tensor,
    loss_fn: ReConHIDV9Loss,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    dataset = TensorDataset(sequence, context)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total = 0.0
    count = 0
    with torch.no_grad():
        for seq_batch, ctx_batch in loader:
            seq_batch = seq_batch.to(device)
            ctx_batch = ctx_batch.to(device)
            output = model(seq_batch, ctx_batch)
            losses = loss_fn(
                output,
                sequence_target=seq_batch,
                context_target=ctx_batch,
                binary_labels=None,
            )
            batch_count = seq_batch.shape[0]
            total += float(losses.total.detach().cpu()) * batch_count
            count += batch_count

    if count == 0:
        raise ValueError("Evaluation split contains zero windows")
    return total / count


@torch.no_grad()
def _encode_all(
    model: ReConHIDV9,
    sequence: Tensor,
    context: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    loader = DataLoader(
        TensorDataset(sequence, context),
        batch_size=batch_size,
        shuffle=False,
    )
    latents: list[Tensor] = []
    for seq_batch, ctx_batch in loader:
        _, _, fused = model.encode(
            seq_batch.to(device),
            ctx_batch.to(device),
        )
        latents.append(fused.detach())
    if not latents:
        raise ValueError("Cannot encode zero windows")
    return torch.cat(latents, dim=0)


@torch.no_grad()
def _score_all(
    model: ReConHIDV9,
    sequence: Tensor,
    context: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    context_weight: float,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TensorDataset(sequence, context),
        batch_size=batch_size,
        shuffle=False,
    )

    reconstruction: list[np.ndarray] = []
    prototype: list[np.ndarray] = []

    for seq_batch, ctx_batch in loader:
        scores = compute_v9_scores(
            model,
            seq_batch.to(device),
            ctx_batch.to(device),
            context_weight=context_weight,
        )
        reconstruction.append(scores.reconstruction_combined)
        if scores.prototype_distance is None:
            raise RuntimeError("Normal prototype is not ready")
        prototype.append(scores.prototype_distance)

    if not reconstruction:
        raise ValueError("Cannot score zero windows")

    return {
        "reconstruction_combined": np.concatenate(reconstruction),
        "prototype_distance": np.concatenate(prototype),
    }


def run_normal_pretraining(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    config: NormalPretrainConfig | None = None,
    model_config: ReConHIDV9Config | None = None,
    run_name: str | None = None,
    manifest_path: str | Path | None = None,
    verify_manifest_hashes: bool = False,
) -> NormalPretrainResult:
    cfg = config or NormalPretrainConfig()
    cfg.validate()
    model_cfg = model_config or ReConHIDV9Config()
    model_cfg.validate()

    _set_reproducibility(cfg.seed, cfg.deterministic)

    if manifest_path is not None:
        manifest_entries = load_manifest_entries(
            manifest_path,
            dataset_root=dataset_root,
            verify_files=True,
            verify_hashes=verify_manifest_hashes,
        )
        window_paths = resolve_manifest_window_paths(
            manifest_entries,
            dataset_root=dataset_root,
        )
        dataset, file_reports = FeatureV2WindowDataset.from_paths(window_paths)
    else:
        manifest_entries = None
        dataset, file_reports = FeatureV2WindowDataset.discover(dataset_root)
    _assert_normal_only(dataset)

    split = split_by_group(
        dataset,
        config=GroupSplitConfig(
            train_ratio=cfg.train_ratio,
            calibration_ratio=cfg.calibration_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.seed,
            group_mode=cfg.split_group_mode,
            explicit_train_groups=(
                cfg.explicit_train_groups
            ),
            explicit_calibration_groups=(
                cfg.explicit_calibration_groups
            ),
            explicit_test_groups=(
                cfg.explicit_test_groups
            ),
        ),
    )

    balance_result = balanced_subsample(
        split.train,
        config=BalancedSamplingConfig(
            enabled=cfg.balance_training,
            window_stride=cfg.training_window_stride,
            balance_mode=cfg.training_balance_mode,
            target_windows_per_group=cfg.training_target_windows_per_group,
            max_windows_per_group=cfg.training_max_windows_per_group,
            seed=cfg.seed,
        ),
    )
    training_dataset = balance_result.dataset

    normalizer = FeatureNormalizerV2()
    train_sequence, train_context = normalizer.fit_transform(
        training_dataset.sequence_array,
        training_dataset.context_array,
    )
    calibration_sequence, calibration_context = normalizer.transform(
        split.calibration.sequence_array,
        split.calibration.context_array,
    )
    test_sequence = None
    test_context = None

    if cfg.evaluate_test_during_pretraining:
        test_sequence, test_context = normalizer.transform(
            split.test.sequence_array,
            split.test.context_array,
        )

    train_tensor_sequence = torch.from_numpy(train_sequence).float()
    train_tensor_context = torch.from_numpy(train_context).float()
    calibration_tensor_sequence = torch.from_numpy(
        calibration_sequence
    ).float()
    calibration_tensor_context = torch.from_numpy(
        calibration_context
    ).float()
    test_tensor_sequence = None
    test_tensor_context = None

    if cfg.evaluate_test_during_pretraining:
        assert test_sequence is not None
        assert test_context is not None

        test_tensor_sequence = torch.from_numpy(
            test_sequence
        ).float()

        test_tensor_context = torch.from_numpy(
            test_context
        ).float()

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {cfg.device!r}, but CUDA is unavailable"
        )
    device = torch.device(cfg.device)

    model = ReConHIDV9(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = ReConHIDV9Loss(
        ReConHIDV9LossConfig(
            sequence_reconstruction_weight=(
                cfg.sequence_reconstruction_weight
            ),
            context_reconstruction_weight=(
                cfg.context_reconstruction_weight
            ),
            classifier_weight=0.0,
            latent_l2_weight=cfg.latent_l2_weight,
        )
    )

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(cfg.seed)
    train_loader = DataLoader(
        _to_tensor_dataset(train_sequence, train_context),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=train_loader_generator,
        pin_memory=device.type == "cuda",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_name = run_name or f"normal_pretrain_{timestamp}"
    run_dir = Path(output_root) / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    _atomic_json(run_dir / "config.json", {
        "normal_pretrain": asdict(cfg),
        "model": model_cfg.to_dict(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "sequence_features": list(SEQUENCE_FEATURES),
        "window_context_features": list(WINDOW_CONTEXT_FEATURES),
    })
    _atomic_json(run_dir / "split_manifest.json", {
        "group_mode": split.group_mode,
        "assignment_mode": split.assignment_mode,
        "train_groups": list(split.train_groups),
        "calibration_groups": list(split.calibration_groups),
        "test_groups": list(split.test_groups),
        "raw_train_windows": len(split.train),
        "train_windows": len(training_dataset),
        "training_balance": balance_result.report,
        "calibration_windows": len(split.calibration),
        "test_windows": len(split.test),
        "source_files": [asdict(report) for report in file_reports],
        "source_manifest": str(manifest_path) if manifest_path is not None else None,
        "manifest_entry_count": (
            len(manifest_entries) if manifest_entries is not None else None
        ),
        "split_seed": cfg.seed,
        "random_window_split": False,
        "test_evaluation_timing": (
            "during_pretraining"
            if cfg.evaluate_test_during_pretraining
            else "deferred_until_final"
        ),
    })
    _atomic_json(run_dir / "normalizer.json", normalizer.state_dict())

    history: list[dict[str, float | int]] = []
    best_calibration_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0

        for sequence_batch, context_batch in train_loader:
            sequence_batch = sequence_batch.to(
                device, non_blocking=device.type == "cuda"
            )
            context_batch = context_batch.to(
                device, non_blocking=device.type == "cuda"
            )

            optimizer.zero_grad(set_to_none=True)
            output = model(sequence_batch, context_batch)
            losses = loss_fn(
                output,
                sequence_target=sequence_batch,
                context_target=context_batch,
                binary_labels=None,
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_count = sequence_batch.shape[0]
            train_total += float(losses.total.detach().cpu()) * batch_count
            train_count += batch_count

        train_loss = train_total / train_count
        calibration_loss = _evaluate_reconstruction_loss(
            model,
            calibration_tensor_sequence,
            calibration_tensor_context,
            loss_fn,
            batch_size=cfg.batch_size,
            device=device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "calibration_loss": calibration_loss,
        })
        _atomic_json(run_dir / "history.json", history)

        improved = (
            calibration_loss
            < best_calibration_loss - cfg.min_delta
        )
        if improved:
            best_calibration_loss = calibration_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_v9_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    "stage": "normal_pretraining",
                    "normalizer_state": normalizer.state_dict(),
                    "best_calibration_loss": best_calibration_loss,
                    "classifier_trained": False,
                },
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"calibration_loss={calibration_loss:.6f} "
            f"best_epoch={best_epoch:03d}"
        )

        if epochs_without_improvement >= cfg.patience:
            print(
                f"early_stopping epoch={epoch} "
                f"patience={cfg.patience}"
            )
            break

    if best_epoch == 0:
        raise RuntimeError("Training ended without producing a checkpoint")

    payload = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)

    train_latents = _encode_all(
        model,
        train_tensor_sequence,
        train_tensor_context,
        batch_size=cfg.batch_size,
        device=device,
    )
    model.set_normal_prototype(train_latents)

    # Save again after fitting the normal prototype from train only.
    save_v9_checkpoint(
        best_path,
        model=model,
        optimizer=None,
        epoch=best_epoch,
        extra={
            "stage": "normal_pretraining",
            "normalizer_state": normalizer.state_dict(),
            "best_calibration_loss": best_calibration_loss,
            "classifier_trained": False,
            "prototype_source": "train_only",
        },
    )

    calibration_scores = _score_all(
        model,
        calibration_tensor_sequence,
        calibration_tensor_context,
        batch_size=cfg.batch_size,
        device=device,
        context_weight=cfg.score_context_weight,
    )
    reconstruction_threshold = quantile_threshold(
        calibration_scores["reconstruction_combined"],
        target_fpr=cfg.target_fpr,
    )
    prototype_threshold = quantile_threshold(
        calibration_scores["prototype_distance"],
        target_fpr=cfg.target_fpr,
    )

    calibration_reconstruction_fpr = empirical_fpr(
        calibration_scores["reconstruction_combined"],
        reconstruction_threshold,
    )
    calibration_prototype_fpr = empirical_fpr(
        calibration_scores["prototype_distance"],
        prototype_threshold,
    )

    _atomic_json(run_dir / "calibration.json", {
        "source_split": "calibration",
        "target_fpr": cfg.target_fpr,
        "detection_rule": "score > threshold",
        "reconstruction": {
            "threshold": reconstruction_threshold,
            "empirical_fpr": calibration_reconstruction_fpr,
            "sample_count": int(
                calibration_scores["reconstruction_combined"].size
            ),
        },
        "prototype": {
            "threshold": prototype_threshold,
            "empirical_fpr": calibration_prototype_fpr,
            "sample_count": int(
                calibration_scores["prototype_distance"].size
            ),
        },
        "classifier_threshold": None,
        "classifier_trained": False,
    })

    
    test_reconstruction_fpr: float | None = None
    test_prototype_fpr: float | None = None

    if cfg.evaluate_test_during_pretraining:
        assert test_tensor_sequence is not None
        assert test_tensor_context is not None

        # Legacy/smoke behavior. Thresholds remain calibration-only.
        test_scores = _score_all(
            model,
            test_tensor_sequence,
            test_tensor_context,
            batch_size=cfg.batch_size,
            device=device,
            context_weight=cfg.score_context_weight,
        )

        test_reconstruction_fpr = empirical_fpr(
            test_scores["reconstruction_combined"],
            reconstruction_threshold,
        )

        test_prototype_fpr = empirical_fpr(
            test_scores["prototype_distance"],
            prototype_threshold,
        )

        _atomic_json(
            run_dir / "test_metrics.json",
            {
                "source_split": "untouched_test",
                "status": "evaluated",
                "threshold_source": "calibration_only",
                "test_windows": len(split.test),
                "reconstruction_fpr": (
                    test_reconstruction_fpr
                ),
                "prototype_fpr": (
                    test_prototype_fpr
                ),
                "classifier_metrics": None,
                "attack_detection_metrics": None,
            },
        )

    else:
        # Main-paper protocol:
        # Test is known only as a reserved participant identity here.
        # No Test feature normalization, scoring, or FPR computation
        # occurs before D_final.
        _atomic_json(
            run_dir / "test_metrics.json",
            {
                "source_split": "untouched_test",
                "status": "deferred_until_final",
                "threshold_source": "calibration_only",
                "test_windows": len(split.test),
                "reconstruction_fpr": None,
                "prototype_fpr": None,
                "classifier_metrics": None,
                "attack_detection_metrics": None,
                "reason": (
                    "Main-experiment protocol defers all "
                    "normal Test scoring until D_final is frozen."
                ),
            },
        )



    result = NormalPretrainResult(
        run_dir=str(run_dir),
        best_epoch=best_epoch,
        train_windows=len(training_dataset),
        calibration_windows=len(split.calibration),
        test_windows=len(split.test),
        reconstruction_threshold=reconstruction_threshold,
        prototype_threshold=prototype_threshold,
        calibration_reconstruction_fpr=(
            calibration_reconstruction_fpr
        ),
        calibration_prototype_fpr=calibration_prototype_fpr,
        test_reconstruction_fpr=test_reconstruction_fpr,
        test_prototype_fpr=test_prototype_fpr,
    )
    _atomic_json(run_dir / "result.json", asdict(result))
    return result
