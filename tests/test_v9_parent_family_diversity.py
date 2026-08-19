from __future__ import annotations

import pytest

from src.coevolution_v9.selection_strategies import (
    family_diverse_take,
    random_parent_entries,
)


FAMILIES = (
    "constant_fast",
    "jittered_mimic",
    "burst_pause",
    "overlap_dense",
    "shortcut_heavy",
    "correction_heavy",
)


def entries():
    result = []

    for index, family in enumerate(FAMILIES):
        result.append({
            "candidate_id": f"{family}_a",
            "family": family,
        })
        result.append({
            "candidate_id": f"{family}_b",
            "family": family,
        })

    return result


def test_family_diverse_take_covers_every_family():
    selected = family_diverse_take(
        entries(),
        limit=8,
        require_full_coverage=True,
    )

    assert len(selected) == 8

    assert {
        entry["family"]
        for entry in selected
    } == set(FAMILIES)


def test_family_coverage_rejects_insufficient_budget():
    with pytest.raises(ValueError):
        family_diverse_take(
            entries(),
            limit=5,
            require_full_coverage=True,
        )


def test_random_parent_selection_preserves_family_coverage():
    first = random_parent_entries(
        entries(),
        limit=8,
        seed=1234,
        preserve_family_coverage=True,
    )

    second = random_parent_entries(
        entries(),
        limit=8,
        seed=1234,
        preserve_family_coverage=True,
    )

    assert len(first) == 8

    assert {
        entry["family"]
        for entry in first
    } == set(FAMILIES)

    assert [
        entry["candidate_id"]
        for entry in first
    ] == [
        entry["candidate_id"]
        for entry in second
    ]
