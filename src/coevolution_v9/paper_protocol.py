"""Frozen paper-protocol validation and experiment-matrix utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.data_v2 import CORE_SCENARIOS
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
)

from .experiment_loop import METHOD_PROFILES
from .raw_attack_generator import (
    RawAttackGenerationConfig,
)


EXPECTED_METHODS = (
    "recon_hid",
    "random_iterative",
    "static_mixed",
)


EXPECTED_METHOD_PROFILES = {
    "recon_hid": (
        "guided",
        "guided",
        True,
    ),
    "random_iterative": (
        "random",
        "random",
        True,
    ),
    "static_mixed": (
        "random",
        "none",
        False,
    ),
}


def load_paper_protocol(
    path: str | Path,
) -> dict[str, Any]:
    source = Path(path)

    payload = json.loads(
        source.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Paper protocol must be a JSON object"
        )

    return payload


def ordered_participant_assignments(
    participants: Sequence[str],
) -> list[dict[str, Any]]:
    resolved = tuple(
        sorted(
            {
                str(value)
                for value in participants
            }
        )
    )

    if len(resolved) < 3:
        raise ValueError(
            "At least three participants are required"
        )

    assignments: list[
        dict[str, Any]
    ] = []

    index = 0

    for test in resolved:
        for calibration in resolved:
            if calibration == test:
                continue

            train = tuple(
                participant
                for participant in resolved
                if participant
                not in {
                    calibration,
                    test,
                }
            )

            assignments.append({
                "split_index": index,
                "assignment_id": (
                    f"test_{test}__cal_{calibration}"
                ),
                "train_groups": train,
                "calibration_group": calibration,
                "test_group": test,
            })

            index += 1

    return assignments


def method_budget_signatures(
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    c = protocol["coevolution"]

    shared = {
        "candidate_sessions_per_probe": (
            int(
                c[
                    "candidate_sessions_per_probe"
                ]
            )
        ),
        "keystrokes_per_candidate": int(
            c["keystrokes_per_candidate"]
        ),
        "mutation_strength": float(
            c["mutation_strength"]
        ),
        "max_hard_windows_per_adaptation": int(
            c[
                "max_hard_windows_per_adaptation"
            ]
        ),
        "max_parent_sessions": int(
            c["max_parent_sessions"]
        ),
        "hardening_epochs": int(
            c["hardening_epochs"]
        ),
        "hardening_batch_size": int(
            c["hardening_batch_size"]
        ),
        "hardening_learning_rate": float(
            c["hardening_learning_rate"]
        ),
    }

    return {
        method: dict(shared)
        for method in protocol["methods"]
    }


def build_paper_jobs(
    protocol: dict[str, Any],
    participants: Sequence[str],
    *,
    methods: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    assignments = (
        ordered_participant_assignments(
            participants
        )
    )

    selected_methods = tuple(
        methods
        if methods is not None
        else protocol["methods"]
    )

    base_seed = int(
        protocol["coevolution"]["base_seed"]
    )

    final_base_seed = int(
        protocol[
            "final_evaluation"
        ]["base_seed"]
    )

    seed_stride = 1_000_000

    jobs: list[dict[str, Any]] = []

    for split in assignments:
        split_index = int(
            split["split_index"]
        )

        split_seed = (
            base_seed
            + split_index * seed_stride
        )

        final_seed = (
            final_base_seed
            + split_index * seed_stride
        )

        for method in selected_methods:
            if method not in EXPECTED_METHODS:
                raise ValueError(
                    f"Unsupported method {method!r}"
                )

            jobs.append({
                **split,
                "method": method,
                "job_id": (
                    f'{split["assignment_id"]}'
                    f"__{method}"
                ),
                # Same seeds for every method
                # within the same participant split.
                "seed": split_seed,
                "final_seed": final_seed,
            })

    return jobs


def audit_paper_protocol(
    protocol: dict[str, Any],
    *,
    participants: Sequence[str] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []

    def require(
        condition: bool,
        message: str,
    ) -> None:
        if not condition:
            issues.append(message)

    require(
        protocol.get("status")
        == "frozen_before_main_results",
        "protocol is not marked frozen_before_main_results",
    )

    schema = protocol.get(
        "feature_schema",
        {},
    )

    require(
        schema.get("version")
        == FEATURE_SCHEMA_VERSION,
        "Feature Schema version mismatch",
    )

    require(
        schema.get("sha256")
        == EXPECTED_SCHEMA_SHA256,
        "Feature Schema hash mismatch",
    )

    normal = protocol.get(
        "normal_data",
        {},
    )

    expected_scenarios = set(
        CORE_SCENARIOS
    )

    require(
        set(
            normal.get(
                "required_scenarios",
                [],
            )
        )
        == expected_scenarios,
        "normal scenario set does not match implementation",
    )

    participant_count = int(
        normal.get(
            "required_participants",
            -1,
        )
    )

    require(
        participant_count == 6,
        "paper protocol requires exactly six participants",
    )

    require(
        int(
            normal.get(
                "train_count",
                -1,
            )
        )
        == 4,
        "paper train participant count must be 4",
    )

    require(
        int(
            normal.get(
                "calibration_count",
                -1,
            )
        )
        == 1,
        "paper calibration participant count must be 1",
    )

    require(
        int(
            normal.get(
                "test_count",
                -1,
            )
        )
        == 1,
        "paper Test participant count must be 1",
    )

    require(
        int(
            normal.get(
                "ordered_split_count",
                -1,
            )
        )
        == 30,
        "six-participant ordered split count must be 30",
    )

    require(
        normal.get(
            "test_evaluation_timing"
        )
        == "after_D_final_only",
        "normal Test is not deferred until D_final",
    )

    methods = tuple(
        protocol.get(
            "methods",
            [],
        )
    )

    require(
        methods == EXPECTED_METHODS,
        "paper method list/order mismatch",
    )

    for method, expected in (
        EXPECTED_METHOD_PROFILES.items()
    ):
        profile = METHOD_PROFILES[
            method
        ]

        actual = (
            profile.selection_mode,
            profile.parent_selection_mode,
            profile.inherit_parent_policy,
        )

        require(
            actual == expected,
            (
                "method profile mismatch for "
                f"{method}: {actual} != {expected}"
            ),
        )

    coevolution = protocol.get(
        "coevolution",
        {},
    )

    configured_families = tuple(
        coevolution.get(
            "required_families",
            [],
        )
    )

    implemented_families = tuple(
        RawAttackGenerationConfig().families
    )

    require(
        configured_families
        == implemented_families,
        (
            "paper adaptive family list does not "
            "match generator implementation"
        ),
    )

    require(
        int(
            coevolution.get(
                "max_parent_sessions",
                0,
            )
        )
        >= len(configured_families),
        (
            "parent budget cannot preserve "
            "full attack-family coverage"
        ),
    )

    convergence = protocol.get(
        "convergence",
        {},
    )

    global_threshold = float(
        convergence.get(
            "global_bypass_threshold",
            -1.0,
        )
    )

    family_threshold = float(
        convergence.get(
            "family_bypass_threshold",
            -1.0,
        )
    )

    require(
        0.0
        <= global_threshold
        < 1.0,
        "invalid global bypass threshold",
    )

    require(
        0.0
        <= family_threshold
        < 1.0,
        "invalid family bypass threshold",
    )

    require(
        global_threshold
        <= family_threshold,
        (
            "global threshold should not be "
            "looser than family threshold"
        ),
    )

    min_stages = int(
        convergence.get(
            "min_defender_stages",
            0,
        )
    )

    max_stages = int(
        convergence.get(
            "max_defender_stages",
            0,
        )
    )

    require(
        1 <= min_stages <= max_stages,
        "invalid defender-stage bounds",
    )

    fresh_probes = int(
        convergence.get(
            "fresh_probes_per_defender",
            0,
        )
    )

    require(
        fresh_probes >= 2,
        (
            "same-defender convergence requires "
            "at least two fresh probes"
        ),
    )

    require(
        convergence.get(
            "patience_scope"
        )
        == "same_defender_fresh_probes",
        "convergence patience scope mismatch",
    )

    final = protocol.get(
        "final_evaluation",
        {},
    )

    final_families = set(
        final.get(
            "heldout_families",
            [],
        )
    )

    require(
        bool(final_families),
        "A_final held-out family set is empty",
    )

    require(
        final_families.isdisjoint(
            configured_families
        ),
        (
            "A_final families overlap adaptive "
            "training/probe families"
        ),
    )

    require(
        int(final.get("base_seed", -1))
        != int(
            coevolution.get(
                "base_seed",
                -1,
            )
        ),
        "A_final seed must differ from adaptive seed",
    )

    require(
        final.get("parent_policy")
        is None,
        "A_final must have no inherited parent policy",
    )

    matched = protocol.get(
        "matched_budget",
        {},
    )

    checkpoints = [
        int(value)
        for value in matched.get(
            "fixed_round_checkpoints",
            [],
        )
    ]

    require(
        bool(checkpoints),
        "matched-budget checkpoints are empty",
    )

    require(
        checkpoints
        == sorted(set(checkpoints)),
        (
            "matched-budget checkpoints must be "
            "sorted and unique"
        ),
    )

    require(
        all(
            1 <= value <= max_stages
            for value in checkpoints
        ),
        "matched-budget checkpoint exceeds stage cap",
    )

    signatures = (
        method_budget_signatures(
            protocol
        )
    )

    unique_signatures = {
        json.dumps(
            value,
            sort_keys=True,
        )
        for value in signatures.values()
    }

    require(
        len(unique_signatures) == 1,
        "method computational budgets are not matched",
    )

    assignment_summary = None

    if participants is not None:
        resolved = tuple(
            sorted(
                {
                    str(value)
                    for value in participants
                }
            )
        )

        require(
            len(resolved)
            == participant_count,
            (
                "observed participant count does "
                "not match frozen protocol"
            ),
        )

        if len(resolved) == participant_count:
            assignments = (
                ordered_participant_assignments(
                    resolved
                )
            )

            require(
                len(assignments) == 30,
                (
                    "observed participants do not "
                    "produce 30 ordered assignments"
                ),
            )

            test_counts = Counter(
                row["test_group"]
                for row in assignments
            )

            calibration_counts = Counter(
                row["calibration_group"]
                for row in assignments
            )

            train_counts = Counter()

            for row in assignments:
                train_counts.update(
                    row["train_groups"]
                )

            assignment_summary = {
                "assignment_count": len(
                    assignments
                ),
                "test_role_counts": dict(
                    test_counts
                ),
                "calibration_role_counts": dict(
                    calibration_counts
                ),
                "train_role_counts": dict(
                    train_counts
                ),
            }

            require(
                set(test_counts.values())
                == {5},
                (
                    "each participant must appear "
                    "as Test exactly five times"
                ),
            )

            require(
                set(
                    calibration_counts.values()
                )
                == {5},
                (
                    "each participant must appear "
                    "as Calibration exactly five times"
                ),
            )

            require(
                set(train_counts.values())
                == {20},
                (
                    "each participant must appear "
                    "in Train exactly twenty times"
                ),
            )

    return {
        "status": (
            "PASS"
            if not issues
            else "FAIL"
        ),
        "issues": issues,
        "method_budget_signatures": signatures,
        "assignment_summary": assignment_summary,
    }


def assert_paper_protocol(
    protocol: dict[str, Any],
    *,
    participants: Sequence[str] | None = None,
) -> dict[str, Any]:
    result = audit_paper_protocol(
        protocol,
        participants=participants,
    )

    if result["status"] != "PASS":
        raise ValueError(
            "Paper protocol audit failed: "
            + "; ".join(
                result["issues"]
            )
        )

    return result
