from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.data_v2.split_manager import (
    GroupSplitConfig,
    split_by_group,
)


@dataclass(frozen=True)
class FakeSample:
    participant_id: str

    @property
    def normal_group_key(self) -> str:
        return self.participant_id


class FakeDataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def subset(self, indices):
        return FakeDataset(
            self.samples[index]
            for index in indices
        )


def dataset():
    return FakeDataset(
        FakeSample(f"p{i:03d}")
        for i in range(1, 7)
    )


def test_explicit_participant_roles_are_exact():
    result = split_by_group(
        dataset(),
        config=GroupSplitConfig(
            group_mode="participant",
            explicit_train_groups=(
                "p001",
                "p002",
                "p003",
                "p004",
            ),
            explicit_calibration_groups=(
                "p005",
            ),
            explicit_test_groups=(
                "p006",
            ),
        ),
    )

    assert result.assignment_mode == "explicit"

    assert result.train_groups == (
        "p001",
        "p002",
        "p003",
        "p004",
    )

    assert result.calibration_groups == (
        "p005",
    )

    assert result.test_groups == (
        "p006",
    )

    result.assert_disjoint()


def test_explicit_roles_must_cover_all_groups():
    with pytest.raises(
        ValueError,
        match="does not assign every observed group",
    ):
        split_by_group(
            dataset(),
            config=GroupSplitConfig(
                group_mode="participant",
                explicit_train_groups=(
                    "p001",
                    "p002",
                    "p003",
                ),
                explicit_calibration_groups=(
                    "p004",
                ),
                explicit_test_groups=(
                    "p005",
                ),
            ),
        )


def test_explicit_roles_reject_unknown_group():
    with pytest.raises(
        ValueError,
        match="unknown groups",
    ):
        split_by_group(
            dataset(),
            config=GroupSplitConfig(
                group_mode="participant",
                explicit_train_groups=(
                    "p001",
                    "p002",
                    "p003",
                    "p004",
                ),
                explicit_calibration_groups=(
                    "p005",
                ),
                explicit_test_groups=(
                    "p999",
                ),
            ),
        )


def test_explicit_roles_reject_leakage():
    with pytest.raises(
        ValueError,
        match="leakage",
    ):
        GroupSplitConfig(
            group_mode="participant",
            explicit_train_groups=(
                "p001",
                "p002",
                "p003",
                "p004",
            ),
            explicit_calibration_groups=(
                "p004",
            ),
            explicit_test_groups=(
                "p006",
            ),
        ).validate()


def test_partial_explicit_assignment_rejected():
    with pytest.raises(
        ValueError,
        match="requires train, calibration, and test",
    ):
        GroupSplitConfig(
            group_mode="participant",
            explicit_train_groups=(
                "p001",
                "p002",
            ),
        ).validate()
