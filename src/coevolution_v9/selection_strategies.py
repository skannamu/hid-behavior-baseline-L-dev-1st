"""Score-independent control selection strategies for Stable-v9 experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


def family_diverse_take(
    entries: list[dict[str, Any]],
    *,
    limit: int,
    require_full_coverage: bool = False,
) -> list[dict[str, Any]]:
    """Take preferred entries while preserving family coverage when possible.

    Input order represents preference. One entry from each observed family is
    taken first, in first-occurrence order, and remaining capacity is filled
    from the original preference order.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    if not entries:
        return []

    families: list[str] = []

    for entry in entries:
        family = str(entry.get("family", "")).strip()
        if not family:
            raise ValueError(
                "Parent candidate is missing family metadata"
            )
        if family not in families:
            families.append(family)

    if require_full_coverage and limit < len(families):
        raise ValueError(
            "Parent budget cannot preserve all observed families: "
            f"limit={limit}, families={len(families)}"
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # First pass: one preferred candidate from each family.
    for family in families:
        if len(selected) >= limit:
            break

        for entry in entries:
            candidate_id = str(entry["candidate_id"])

            if candidate_id in selected_ids:
                continue

            if str(entry["family"]) != family:
                continue

            selected.append(entry)
            selected_ids.add(candidate_id)
            break

    # Second pass: fill remaining capacity in original preference order.
    for entry in entries:
        if len(selected) >= min(limit, len(entries)):
            break

        candidate_id = str(entry["candidate_id"])

        if candidate_id in selected_ids:
            continue

        selected.append(entry)
        selected_ids.add(candidate_id)

    return selected


def random_diverse_select(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded random selection with candidate-session diversity.

    Defender scores and predictions are intentionally ignored.
    Candidate sessions and windows within each session are shuffled
    deterministically, then selected round-robin across sessions.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    if not rows:
        return []

    by_session: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        by_session.setdefault(
            str(row["candidate_id"]),
            [],
        ).append(row)

    rng = np.random.default_rng(seed)

    session_names = sorted(by_session)
    session_perm = rng.permutation(len(session_names))

    session_order = [
        session_names[int(index)]
        for index in session_perm
    ]

    shuffled: dict[str, list[dict[str, Any]]] = {}

    for session in session_order:
        values = sorted(
            by_session[session],
            key=lambda row: (
                int(row["window_id"]),
                int(row["global_index"]),
            ),
        )

        permutation = rng.permutation(len(values))

        shuffled[session] = [
            values[int(index)]
            for index in permutation
        ]

    positions = {
        session: 0
        for session in session_order
    }

    selected: list[dict[str, Any]] = []
    target = min(limit, len(rows))

    while len(selected) < target:
        progressed = False

        for session in session_order:
            position = positions[session]
            values = shuffled[session]

            if position >= len(values):
                continue

            selected.append(values[position])
            positions[session] = position + 1
            progressed = True

            if len(selected) >= target:
                break

        if not progressed:
            break

    return selected


def random_parent_entries(
    entries: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    preserve_family_coverage: bool = False,
) -> list[dict[str, Any]]:
    """Select parent sessions without using defender scores."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    if not entries:
        return []

    ordered = sorted(
        entries,
        key=lambda entry: str(entry["candidate_id"]),
    )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))

    shuffled = [
        ordered[int(index)]
        for index in permutation
    ]

    if preserve_family_coverage:
        return family_diverse_take(
            shuffled,
            limit=limit,
            require_full_coverage=True,
        )

    return shuffled[: min(limit, len(shuffled))]
