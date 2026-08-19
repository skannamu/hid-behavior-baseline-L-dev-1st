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

    # Optional exact role assignment.
    #
    # When supplied, all three must be supplied and together
    # must exactly cover every observed group in the dataset.
    explicit_train_groups: tuple[str, ...] | None = None
    explicit_calibration_groups: tuple[str, ...] | None = None
    explicit_test_groups: tuple[str, ...] | None = None

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

        explicit = (
            self.explicit_train_groups,
            self.explicit_calibration_groups,
            self.explicit_test_groups,
        )

        if any(value is not None for value in explicit):
            if any(value is None for value in explicit):
                raise ValueError(
                    "Explicit group assignment requires train, "
                    "calibration, and test groups together"
                )

            assert self.explicit_train_groups is not None
            assert self.explicit_calibration_groups is not None
            assert self.explicit_test_groups is not None

            named = {
                "train": self.explicit_train_groups,
                "calibration": self.explicit_calibration_groups,
                "test": self.explicit_test_groups,
            }

            resolved: dict[str, set[str]] = {}

            for name, values in named.items():
                if not values:
                    raise ValueError(
                        f"Explicit {name} groups must not be empty"
                    )

                if len(set(values)) != len(values):
                    raise ValueError(
                        f"Explicit {name} groups contain duplicates"
                    )

                resolved[name] = set(values)

            if (
                resolved["train"] & resolved["calibration"]
                or resolved["train"] & resolved["test"]
                or resolved["calibration"] & resolved["test"]
            ):
                raise ValueError(
                    "Explicit group assignment contains leakage "
                    "across train/calibration/test"
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
    assignment_mode: str = "deterministic_ratio"

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

    if config.explicit_train_groups is not None:
        assert config.explicit_calibration_groups is not None
        assert config.explicit_test_groups is not None

        train_groups = tuple(
            config.explicit_train_groups
        )
        calibration_groups = tuple(
            config.explicit_calibration_groups
        )
        test_groups = tuple(
            config.explicit_test_groups
        )

        observed = set(groups)

        assigned = (
            set(train_groups)
            | set(calibration_groups)
            | set(test_groups)
        )

        unknown = sorted(assigned - observed)
        missing = sorted(observed - assigned)

        if unknown:
            raise ValueError(
                "Explicit split references unknown groups: "
                f"{unknown}"
            )

        if missing:
            raise ValueError(
                "Explicit split does not assign every observed group: "
                f"{missing}"
            )

        assignment_mode = "explicit"

    else:
        n_groups = len(group_names)
        n_train = max(
            1,
            int(n_groups * config.train_ratio),
        )
        n_calibration = max(
            1,
            int(
                n_groups
                * config.calibration_ratio
            ),
        )

        # Preserve at least one final-test group.
        if (
            n_train + n_calibration
            >= n_groups
        ):
            overflow = (
                n_train
                + n_calibration
                - (n_groups - 1)
            )

            if n_train > n_calibration:
                n_train -= overflow
            else:
                n_calibration -= overflow

        if (
            n_train < 1
            or n_calibration < 1
        ):
            raise ValueError(
                "Not enough groups to create non-empty "
                "train/calibration/test splits"
            )

        train_groups = tuple(
            group_names[:n_train]
        )

        calibration_groups = tuple(
            group_names[
                n_train:
                n_train + n_calibration
            ]
        )

        test_groups = tuple(
            group_names[
                n_train + n_calibration:
            ]
        )

        if not test_groups:
            raise ValueError(
                "Final-test group set is empty"
            )

        assignment_mode = (
            "deterministic_ratio"
        )

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
        assignment_mode=assignment_mode,
    )
    result.assert_disjoint()
    return result
