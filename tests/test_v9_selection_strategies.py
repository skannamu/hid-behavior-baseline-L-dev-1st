from __future__ import annotations

from collections import Counter

from src.coevolution_v9.selection_strategies import (
    random_diverse_select,
    random_parent_entries,
)
from src.coevolution_v9.weakness_miner import WeaknessMiningConfig


def _prediction_rows():
    rows = []
    global_index = 0

    for candidate in range(24):
        for window in range(25):
            rows.append({
                "global_index": global_index,
                "candidate_id": f"c{candidate:02d}",
                "window_id": window,
                "decision_score": float(candidate * 1000 + window),
                "is_bypass": int(window % 2 == 0),
            })
            global_index += 1

    return rows


def _entries():
    return [
        {
            "candidate_id": f"c{candidate:02d}",
            "policy": {"family": "dummy"},
        }
        for candidate in range(24)
    ]


def _identity(rows):
    return [
        (
            str(row["candidate_id"]),
            int(row["window_id"]),
            int(row["global_index"]),
        )
        for row in rows
    ]


def test_random_diverse_selection_exact_budget_and_session_balance():
    selected = random_diverse_select(
        _prediction_rows(),
        limit=192,
        seed=20260807,
    )

    assert len(selected) == 192

    counts = Counter(str(row["candidate_id"]) for row in selected)
    assert len(counts) == 24
    assert set(counts.values()) == {8}


def test_random_selection_is_deterministic():
    rows = _prediction_rows()

    first = random_diverse_select(
        rows,
        limit=192,
        seed=20260807,
    )
    second = random_diverse_select(
        rows,
        limit=192,
        seed=20260807,
    )

    assert _identity(first) == _identity(second)


def test_random_selection_does_not_depend_on_defender_scores():
    original = _prediction_rows()

    modified = [
        {
            **row,
            "decision_score": -float(row["decision_score"]) * 1_000_000.0,
            "is_bypass": 1 - int(row["is_bypass"]),
        }
        for row in original
    ]

    first = random_diverse_select(
        original,
        limit=192,
        seed=12345,
    )
    second = random_diverse_select(
        modified,
        limit=192,
        seed=12345,
    )

    assert _identity(first) == _identity(second)


def test_random_parent_selection_is_deterministic_and_exact():
    parents = random_parent_entries(
        _entries(),
        limit=8,
        seed=20260808,
    )

    assert len(parents) == 8
    assert len({entry["candidate_id"] for entry in parents}) == 8

    again = random_parent_entries(
        _entries(),
        limit=8,
        seed=20260808,
    )

    assert [
        entry["candidate_id"] for entry in parents
    ] == [
        entry["candidate_id"] for entry in again
    ]


def test_weakness_config_accepts_experimental_modes():
    WeaknessMiningConfig(
        max_hard_windows=192,
        max_parent_sessions=8,
        selection_mode="random",
        parent_selection_mode="random",
        selection_seed=1,
        require_full_budget=True,
    ).validate()


def test_weakness_config_accepts_static_no_parent():
    WeaknessMiningConfig(
        max_hard_windows=192,
        max_parent_sessions=8,
        selection_mode="random",
        parent_selection_mode="none",
        selection_seed=1,
        require_full_budget=True,
    ).validate()
