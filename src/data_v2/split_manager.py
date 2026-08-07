"""Leakage-safe group splitting for Feature Schema v2 datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

from .window_dataset import FeatureV2WindowDataset, WindowSample


SUPPORTED_GROUP_MODES = {"participant_session", "participant"}


@dataclass(frozen=True)
class GroupSplitConfig:
    train_ratio: float = 0.70
    calibration_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 20260804
    group_mode: str = "participant_session"

    def validate(self) -> None:
        values = (self.train_ratio, self.calibration_ratio, self.test_ratio)
        if any(value <= 0 for value in values):
            raise ValueError("All split ratios must be positive")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError(
                "train_ratio + calibration_ratio + test_ratio must equal 1"
            )
        if self.group_mode not in SUPPORTED_GROUP_MODES:
            raise ValueError(
                f"Unsupported group_mode={self.group_mode!r}; "
                f"expected one of {sorted(SUPPORTED_GROUP_MODES)}"
            )


@dataclass(frozen=True)
class GroupSplit:
    train: FeatureV2WindowDataset
    calibration: FeatureV2WindowDataset
    test: FeatureV2WindowDataset

    train_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    group_mode: str = "custom"

    def assert_disjoint(self) -> None:
        train = set(self.train_groups)
        calibration = set(self.calibration_groups)
        test = set(self.test_groups)

        if train & calibration or train & test or calibration & test:
            raise ValueError("Group leakage detected across splits")


def _stable_group_rank(group: str, seed: int) -> str:
    payload = f"{seed}::{group}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def group_key_for_mode(sample: WindowSample, mode: str) -> str:
    if mode == "participant_session":
        return sample.normal_group_key
    if mode == "participant":
        return sample.participant_id
    raise ValueError(
        f"Unsupported group mode {mode!r}; expected one of "
        f"{sorted(SUPPORTED_GROUP_MODES)}"
    )


def split_by_group(
    dataset: FeatureV2WindowDataset,
    *,
    config: GroupSplitConfig = GroupSplitConfig(),
    group_key: Callable[[WindowSample], str] | None = None,
) -> GroupSplit:
    config.validate()
    key_fn = group_key or (
        lambda sample: group_key_for_mode(sample, config.group_mode)
    )
    resolved_mode = "custom" if group_key is not None else config.group_mode

    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset):
        groups.setdefault(key_fn(sample), []).append(index)

    group_names = sorted(
        groups,
        key=lambda name: (_stable_group_rank(name, config.seed), name),
    )

    # Never fall back to random windows. Stride-1 windows overlap by 49/50.
    if len(group_names) < 3:
        raise ValueError(
            "At least 3 independent groups are required for "
            "train/calibration/test. "
            f"Found {len(group_names)} group(s): {group_names}. "
            "Collect more independent groups; random window splitting is "
            "prohibited because stride-1 windows overlap by 49/50."
        )

    n_groups = len(group_names)
    n_train = max(1, int(n_groups * config.train_ratio))
    n_calibration = max(1, int(n_groups * config.calibration_ratio))

    # Preserve at least one final-test group.
    if n_train + n_calibration >= n_groups:
        overflow = n_train + n_calibration - (n_groups - 1)
        if n_train > n_calibration:
            n_train -= overflow
        else:
            n_calibration -= overflow

    if n_train < 1 or n_calibration < 1:
        raise ValueError(
            "Not enough groups to create non-empty train/calibration/test splits"
        )

    train_groups = tuple(group_names[:n_train])
    calibration_groups = tuple(
        group_names[n_train : n_train + n_calibration]
    )
    test_groups = tuple(group_names[n_train + n_calibration :])

    if not test_groups:
        raise ValueError("Final-test group set is empty")

    def indices_for(selected: Sequence[str]) -> list[int]:
        return [
            index
            for group in selected
            for index in groups[group]
        ]

    result = GroupSplit(
        train=dataset.subset(indices_for(train_groups)),
        calibration=dataset.subset(indices_for(calibration_groups)),
        test=dataset.subset(indices_for(test_groups)),
        train_groups=train_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
        group_mode=resolved_mode,
    )
    result.assert_disjoint()
    return result
