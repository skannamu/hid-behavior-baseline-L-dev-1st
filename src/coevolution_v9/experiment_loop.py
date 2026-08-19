"""Top-level fixed-budget and convergence-driven Stable-v9 experiment loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .common import atomic_json
from .convergence import (
    ConvergenceConfig,
    ConvergenceDecision,
    ConvergenceTracker,
    summarize_convergence_history,
)
from .round_runner import (
    CoevolutionRoundConfig,
    harden_coevolution_probe,
    probe_coevolution_round,
    run_coevolution_round,
)


@dataclass(frozen=True)
class MethodProfile:
    selection_mode: str
    parent_selection_mode: str
    inherit_parent_policy: bool


METHOD_PROFILES: dict[str, MethodProfile] = {
    "recon_hid": MethodProfile(
        selection_mode="guided",
        parent_selection_mode="guided",
        inherit_parent_policy=True,
    ),
    "random_iterative": MethodProfile(
        selection_mode="random",
        parent_selection_mode="random",
        inherit_parent_policy=True,
    ),
    "static_mixed": MethodProfile(
        selection_mode="random",
        parent_selection_mode="none",
        inherit_parent_policy=False,
    ),
}


@dataclass(frozen=True)
class ExperimentLoopConfig:
    stopping_mode: str
    seed: int
    fixed_rounds: int | None = None
    convergence: ConvergenceConfig | None = None

    def validate(self) -> None:
        if self.stopping_mode not in {
            "fixed",
            "convergence",
        }:
            raise ValueError(
                "stopping_mode must be 'fixed' or 'convergence'"
            )

        if self.stopping_mode == "fixed":
            if self.fixed_rounds is None:
                raise ValueError(
                    "fixed_rounds is required in fixed mode"
                )
            if self.fixed_rounds <= 0:
                raise ValueError(
                    "fixed_rounds must be positive"
                )

        if self.stopping_mode == "convergence":
            if self.convergence is None:
                raise ValueError(
                    "convergence config is required"
                )
            self.convergence.validate()


def _round_config_for_index(
    base: CoevolutionRoundConfig,
    *,
    round_index: int,
    seed: int,
) -> CoevolutionRoundConfig:
    round_seed = seed + round_index * 10_000

    return CoevolutionRoundConfig(
        method=base.method,
        attack_generation=replace(
            base.attack_generation,
            seed=round_seed,
        ),
        weakness_mining=replace(
            base.weakness_mining,
            selection_seed=round_seed,
        ),
        hardened_training=replace(
            base.hardened_training,
            seed=round_seed,
        ),
    )


def _profile_for_config(
    round_config: CoevolutionRoundConfig,
) -> MethodProfile:
    try:
        profile = METHOD_PROFILES[
            round_config.method
        ]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported method: {round_config.method}"
        ) from exc

    if (
        round_config.weakness_mining.selection_mode
        != profile.selection_mode
    ):
        raise ValueError(
            "Round config selection_mode does not match "
            f"method={round_config.method}"
        )

    if (
        round_config.weakness_mining.parent_selection_mode
        != profile.parent_selection_mode
    ):
        raise ValueError(
            "Round config parent_selection_mode does not match "
            f"method={round_config.method}"
        )

    return profile


def run_experiment_loop(
    *,
    normal_dataset_root: str | Path,
    normal_manifest_path: str | Path,
    d0_run_dir: str | Path,
    output_root: str | Path,
    base_round_config: CoevolutionRoundConfig,
    loop_config: ExperimentLoopConfig,
) -> dict[str, Any]:
    """Run either a matched fixed budget or convergence-driven loop."""

    loop_config.validate()

    profile = _profile_for_config(
        base_round_config
    )

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    timeline_path = (
        root / "coevolution_timeline.json"
    )

    if timeline_path.exists():
        raise FileExistsError(timeline_path)

    current_defender = Path(
        d0_run_dir
    ).resolve()

    current_parent_policy: Path | None = None

    timeline: list[dict[str, Any]] = []

    # ==========================================================
    # FIXED-BUDGET MODE
    # ==========================================================

    if loop_config.stopping_mode == "fixed":
        assert loop_config.fixed_rounds is not None

        for round_index in range(
            loop_config.fixed_rounds
        ):
            round_config = _round_config_for_index(
                base_round_config,
                round_index=round_index,
                seed=loop_config.seed,
            )

            inherited_policy = (
                current_parent_policy
                if profile.inherit_parent_policy
                else None
            )

            result = run_coevolution_round(
                round_index=round_index,
                parent_defender_run_dir=(
                    current_defender
                ),
                normal_dataset_root=(
                    normal_dataset_root
                ),
                normal_manifest_path=(
                    normal_manifest_path
                ),
                output_root=root,
                parent_policy_path=(
                    inherited_policy
                ),
                config=round_config,
            )

            weakness = result[
                "weakness_mining"
            ]

            timeline.append({
                "round_index": round_index,
                "seed": (
                    loop_config.seed
                    + round_index * 10_000
                ),
                "probe_manifest_path": (
                    result.get(
                        "probe_manifest_path"
                    )
                ),
                "round_manifest_path": (
                    result[
                        "round_manifest_path"
                    ]
                ),
                "evaluated_defender_run_dir": (
                    str(current_defender)
                ),
                "next_defender_run_dir": (
                    result[
                        "next_defender_run_dir"
                    ]
                ),
                "attack_window_count": (
                    weakness[
                        "attack_window_count"
                    ]
                ),
                "bypass_rate": (
                    weakness["bypass_rate"]
                ),
                "worst_family_bypass_rate": (
                    weakness.get(
                        "worst_family_bypass_rate"
                    )
                ),
                "hard_negative_count": (
                    weakness[
                        "hard_negative_count"
                    ]
                ),
                "selected_parent_session_count": (
                    weakness[
                        "selected_parent_session_count"
                    ]
                ),
                "stop_decision": None,
            })

            current_defender = Path(
                result[
                    "next_defender_run_dir"
                ]
            ).resolve()

            if profile.inherit_parent_policy:
                current_parent_policy = Path(
                    result[
                        "next_parent_policy_path"
                    ]
                ).resolve()
            else:
                current_parent_policy = None

        payload = {
            "status": "PASS",
            "method": base_round_config.method,
            "stopping_mode": "fixed",
            "fixed_rounds": (
                loop_config.fixed_rounds
            ),
            "converged": None,
            "stop_reason": (
                "fixed_budget_completed"
            ),
            "initial_defender_run_dir": str(
                Path(d0_run_dir).resolve()
            ),
            "terminal_defender_run_dir": str(
                current_defender
            ),
            # A fixed-budget endpoint is not automatically
            # promoted to D_final.
            "final_defender_run_dir": None,
            "timeline": timeline,
        }

        path = atomic_json(
            timeline_path,
            payload,
        )

        return {
            **payload,
            "timeline_path": str(path),
        }

    # ==========================================================
    # CONVERGENCE MODE
    # ==========================================================

    assert loop_config.convergence is not None

    tracker = ConvergenceTracker(
        loop_config.convergence
    )

    decisions: list[
        ConvergenceDecision
    ] = []

    final_defender: str | None = None
    stop_reason: str | None = None

    max_probes = (
        loop_config.convergence.max_rounds
    )

    for round_index in range(max_probes):
        round_config = _round_config_for_index(
            base_round_config,
            round_index=round_index,
            seed=loop_config.seed,
        )

        inherited_policy = (
            current_parent_policy
            if profile.inherit_parent_policy
            else None
        )

        probe = probe_coevolution_round(
            round_index=round_index,
            parent_defender_run_dir=(
                current_defender
            ),
            normal_dataset_root=(
                normal_dataset_root
            ),
            normal_manifest_path=(
                normal_manifest_path
            ),
            output_root=root,
            parent_policy_path=(
                inherited_policy
            ),
            config=round_config,
        )

        weakness = probe[
            "weakness_mining"
        ]

        decision = tracker.evaluate(
            probe_index=round_index,
            weakness_summary=weakness,
        )

        decisions.append(decision)

        entry: dict[str, Any] = {
            "round_index": round_index,
            "seed": (
                loop_config.seed
                + round_index * 10_000
            ),
            "evaluated_defender_run_dir": str(
                current_defender
            ),
            "probe_manifest_path": (
                probe[
                    "probe_manifest_path"
                ]
            ),
            "round_manifest_path": None,
            "next_defender_run_dir": None,
            "attack_window_count": (
                weakness[
                    "attack_window_count"
                ]
            ),
            "bypass_rate": (
                weakness["bypass_rate"]
            ),
            "family_bypass": (
                weakness["family_bypass"]
            ),
            "worst_family_bypass_rate": (
                weakness[
                    "worst_family_bypass_rate"
                ]
            ),
            "hard_negative_count": (
                weakness[
                    "hard_negative_count"
                ]
            ),
            "selected_parent_session_count": (
                weakness[
                    "selected_parent_session_count"
                ]
            ),
            "stop_decision": (
                decision.to_dict()
            ),
        }

        # Crucial semantics:
        # The defender that was actually probed is the one
        # eligible to become D_final.
        if decision.should_stop:
            stop_reason = decision.stop_reason

            if (
                decision.stop_reason
                == "converged"
            ):
                final_defender = str(
                    current_defender
                )

            timeline.append(entry)
            break

        hardened = harden_coevolution_probe(
            probe_result=probe,
            config=round_config,
        )

        entry[
            "round_manifest_path"
        ] = hardened[
            "round_manifest_path"
        ]

        entry[
            "next_defender_run_dir"
        ] = hardened[
            "next_defender_run_dir"
        ]

        timeline.append(entry)

        current_defender = Path(
            hardened[
                "next_defender_run_dir"
            ]
        ).resolve()

        if profile.inherit_parent_policy:
            current_parent_policy = Path(
                hardened[
                    "next_parent_policy_path"
                ]
            ).resolve()
        else:
            current_parent_policy = None

    convergence_summary = (
        summarize_convergence_history(
            decisions
        )
    )

    payload = {
        "status": "PASS",
        "method": base_round_config.method,
        "stopping_mode": "convergence",
        "convergence_config": {
            "min_rounds": (
                loop_config.convergence.min_rounds
            ),
            "max_rounds": (
                loop_config.convergence.max_rounds
            ),
            "global_bypass_threshold": (
                loop_config.convergence
                .global_bypass_threshold
            ),
            "family_bypass_threshold": (
                loop_config.convergence
                .family_bypass_threshold
            ),
            "patience": (
                loop_config.convergence.patience
            ),
            "required_families": list(
                loop_config.convergence
                .required_families
            ),
        },
        "converged": (
            stop_reason == "converged"
        ),
        "stop_reason": stop_reason,
        "initial_defender_run_dir": str(
            Path(d0_run_dir).resolve()
        ),
        "terminal_defender_run_dir": str(
            current_defender
        ),
        "final_defender_run_dir": (
            final_defender
        ),
        "convergence": (
            convergence_summary
        ),
        "timeline": timeline,
    }

    path = atomic_json(
        timeline_path,
        payload,
    )

    return {
        **payload,
        "timeline_path": str(path),
    }
