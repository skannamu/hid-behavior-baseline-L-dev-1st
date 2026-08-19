from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.coevolution_v9.paper_protocol import (
    assert_paper_protocol,
    build_paper_jobs,
    load_paper_protocol,
    method_budget_signatures,
    ordered_participant_assignments,
)


PROTOCOL = Path(
    "configs/evaluation_protocol_v9_paper.json"
)

PARTICIPANTS = tuple(
    f"p{i:03d}"
    for i in range(1, 7)
)


def test_frozen_protocol_audit_passes():
    protocol = load_paper_protocol(
        PROTOCOL
    )

    result = assert_paper_protocol(
        protocol,
        participants=PARTICIPANTS,
    )

    assert result["status"] == "PASS"


def test_six_participants_produce_30_ordered_splits():
    assignments = (
        ordered_participant_assignments(
            PARTICIPANTS
        )
    )

    assert len(assignments) == 30


def test_every_calibration_test_pair_is_unique():
    assignments = (
        ordered_participant_assignments(
            PARTICIPANTS
        )
    )

    pairs = {
        (
            row["calibration_group"],
            row["test_group"],
        )
        for row in assignments
    }

    assert len(pairs) == 30


def test_role_frequencies_are_balanced():
    assignments = (
        ordered_participant_assignments(
            PARTICIPANTS
        )
    )

    test = Counter(
        row["test_group"]
        for row in assignments
    )

    calibration = Counter(
        row["calibration_group"]
        for row in assignments
    )

    train = Counter()

    for row in assignments:
        train.update(
            row["train_groups"]
        )

    assert set(test.values()) == {5}
    assert set(
        calibration.values()
    ) == {5}
    assert set(train.values()) == {20}


def test_method_computational_budgets_are_identical():
    protocol = load_paper_protocol(
        PROTOCOL
    )

    signatures = (
        method_budget_signatures(
            protocol
        )
    )

    values = list(
        signatures.values()
    )

    assert values[0] == values[1]
    assert values[1] == values[2]


def test_full_three_method_matrix_has_90_jobs():
    protocol = load_paper_protocol(
        PROTOCOL
    )

    jobs = build_paper_jobs(
        protocol,
        PARTICIPANTS,
    )

    assert len(jobs) == 90


def test_same_split_uses_same_seeds_across_methods():
    protocol = load_paper_protocol(
        PROTOCOL
    )

    jobs = build_paper_jobs(
        protocol,
        PARTICIPANTS,
    )

    first_split = [
        row
        for row in jobs
        if row["split_index"] == 0
    ]

    assert len(first_split) == 3

    assert len({
        row["seed"]
        for row in first_split
    }) == 1

    assert len({
        row["final_seed"]
        for row in first_split
    }) == 1
