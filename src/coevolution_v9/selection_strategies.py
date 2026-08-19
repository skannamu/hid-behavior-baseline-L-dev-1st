"""Score-independent control selection strategies for Stable-v9 experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


def random_diverse_select(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded random window selection with candidate-session diversity.

    Defender scores are intentionally ignored. Candidate sessions and the
    windows within each session are randomly ordered, after which selection
    proceeds round-robin across sessions.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not rows:
        return []

    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_session.setdefault(str(row["candidate_id"]), []).append(row)

    rng = np.random.default_rng(seed)

    session_names = sorted(by_session)
    session_perm = rng.permutation(len(session_names))
    session_order = [session_names[int(index)] for index in session_perm]

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
        shuffled[session] = [values[int(index)] for index in permutation]

    positions = {session: 0 for session in session_order}
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
) -> list[dict[str, Any]]:
    """Select parent attack sessions without using defender scores."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not entries:
        return []

    ordered = sorted(entries, key=lambda entry: str(entry["candidate_id"]))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))

    return [
        ordered[int(index)]
        for index in permutation[: min(limit, len(ordered))]
    ]
