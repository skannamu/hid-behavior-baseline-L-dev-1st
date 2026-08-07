"""Deterministic participant-by-scenario balancing for normal training data."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .window_dataset import FeatureV2WindowDataset, WindowSample


SUPPORTED_BALANCE_MODES = {"equal", "cap"}


@dataclass(frozen=True)
class BalancedSamplingConfig:
    enabled: bool = True
    window_stride: int = 5
    balance_mode: str = "equal"
    target_windows_per_group: int | None = None
    max_windows_per_group: int | None = 3000
    seed: int = 20260804

    def validate(self) -> None:
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if self.balance_mode not in SUPPORTED_BALANCE_MODES:
            raise ValueError(
                f"Unsupported balance_mode={self.balance_mode!r}; "
                f"expected one of {sorted(SUPPORTED_BALANCE_MODES)}"
            )
        for name, value in (
            ("target_windows_per_group", self.target_windows_per_group),
            ("max_windows_per_group", self.max_windows_per_group),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")


@dataclass(frozen=True)
class BalancedSamplingResult:
    dataset: FeatureV2WindowDataset
    selected_indices: tuple[int, ...]
    report: dict[str, Any]


def _stable_rank(sample: WindowSample, index: int, seed: int) -> str:
    payload = (
        f"{seed}::{sample.participant_id}::{sample.scenario}::"
        f"{sample.session_id}::{sample.window_id}::{index}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _session_phase(session_id: str, stride: int, seed: int) -> int:
    payload = f"{seed}::{session_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % stride


def participant_scenario_key(sample: WindowSample) -> str:
    return f"{sample.participant_id}::{sample.scenario}"


def balanced_subsample(
    dataset: FeatureV2WindowDataset,
    *,
    config: BalancedSamplingConfig | None = None,
) -> BalancedSamplingResult:
    cfg = config or BalancedSamplingConfig()
    cfg.validate()

    if not cfg.enabled:
        indices = tuple(range(len(dataset)))
        return BalancedSamplingResult(
            dataset=dataset,
            selected_indices=indices,
            report={
                "enabled": False,
                "input_windows": len(dataset),
                "selected_windows": len(dataset),
                "config": asdict(cfg),
                "groups": {},
            },
        )

    original_counts: dict[str, int] = {}
    session_indices: dict[tuple[str, str, str], list[int]] = {}
    for index, sample in enumerate(dataset):
        key = participant_scenario_key(sample)
        original_counts[key] = original_counts.get(key, 0) + 1
        session_key = (sample.participant_id, sample.scenario, sample.session_id)
        session_indices.setdefault(session_key, []).append(index)

    grouped: dict[str, list[int]] = {}
    for (participant, scenario, session_id), indices in sorted(session_indices.items()):
        ordered = sorted(indices, key=lambda idx: dataset[idx].window_id)
        offset_bound = min(cfg.window_stride, len(ordered))
        offset = _session_phase(session_id, offset_bound, cfg.seed)
        selected_session = ordered[offset::cfg.window_stride]
        if not selected_session:
            selected_session = [ordered[0]]
        grouped.setdefault(f"{participant}::{scenario}", []).extend(selected_session)

    after_stride_counts = {
        key: len(grouped.get(key, [])) for key in sorted(original_counts)
    }
    for key, count in after_stride_counts.items():
        if count == 0:
            raise ValueError(
                f"No training windows remain after stride sampling for group {key}"
            )

    if cfg.target_windows_per_group is not None:
        target = cfg.target_windows_per_group
    elif cfg.balance_mode == "equal":
        target = min(after_stride_counts.values())
    else:
        target = max(after_stride_counts.values())

    if cfg.max_windows_per_group is not None:
        target = min(target, cfg.max_windows_per_group)
    if target <= 0:
        raise ValueError("Resolved target_windows_per_group must be positive")

    selected: list[int] = []
    group_rows: dict[str, dict[str, Any]] = {}
    for key in sorted(original_counts):
        candidates = grouped[key]
        ordered = sorted(
            candidates,
            key=lambda index: (
                _stable_rank(dataset[index], index, cfg.seed),
                dataset[index].session_id,
                dataset[index].window_id,
            ),
        )
        selected_for_group = ordered[: min(target, len(ordered))]
        selected.extend(selected_for_group)
        group_rows[key] = {
            "original_windows": original_counts[key],
            "after_stride_windows": after_stride_counts[key],
            "selected_windows": len(selected_for_group),
        }

    selected_indices = tuple(sorted(selected))
    selected_dataset = dataset.subset(selected_indices)
    return BalancedSamplingResult(
        dataset=selected_dataset,
        selected_indices=selected_indices,
        report={
            "enabled": True,
            "input_windows": len(dataset),
            "selected_windows": len(selected_dataset),
            "resolved_target_windows_per_group": target,
            "config": asdict(cfg),
            "groups": group_rows,
        },
    )
