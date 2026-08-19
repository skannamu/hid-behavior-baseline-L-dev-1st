from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coevolution_v9 import (
    CoevolutionRoundConfig,
    HardenedTrainingConfig,
    RawAttackGenerationConfig,
    WeaknessMiningConfig,
    run_coevolution_round,
)


METHOD_PROFILES = {
    "recon_hid": {
        "selection_mode": "guided",
        "parent_selection_mode": "guided",
        "inherit_parent_policy": True,
    },
    "random_iterative": {
        "selection_mode": "random",
        "parent_selection_mode": "random",
        "inherit_parent_policy": True,
    },
    "static_mixed": {
        "selection_mode": "random",
        "parent_selection_mode": "none",
        "inherit_parent_policy": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched-budget Stable-v9 attack/defense hardening "
            "rounds. Final held-out evaluation is separate."
        )
    )

    parser.add_argument("--normal-root", required=True)
    parser.add_argument("--normal-manifest", required=True)
    parser.add_argument("--d0-run-dir", required=True)
    parser.add_argument("--output-root", required=True)

    parser.add_argument(
        "--method",
        choices=tuple(METHOD_PROFILES),
        default="recon_hid",
    )

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=24)

    parser.add_argument(
        "--keystrokes-per-candidate",
        type=int,
        default=72,
    )

    parser.add_argument(
        "--mutation-strength",
        type=float,
        default=0.18,
    )

    parser.add_argument(
        "--max-hard-windows",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--max-parent-sessions",
        type=int,
        default=8,
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3.0e-4,
    )

    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    profile = METHOD_PROFILES[args.method]

    output_root = Path(args.output_root).resolve()
    parent_run = Path(args.d0_run_dir).resolve()

    parent_policy: Path | None = None
    timeline: list[dict[str, object]] = []

    base_config = CoevolutionRoundConfig(
        method=args.method,
        attack_generation=RawAttackGenerationConfig(
            candidates=args.candidates,
            keystrokes_per_candidate=(
                args.keystrokes_per_candidate
            ),
            seed=args.seed,
            mutation_strength=args.mutation_strength,
        ),
        weakness_mining=WeaknessMiningConfig(
            max_hard_windows=args.max_hard_windows,
            max_parent_sessions=args.max_parent_sessions,
            selection_mode=str(
                profile["selection_mode"]
            ),
            parent_selection_mode=str(
                profile["parent_selection_mode"]
            ),
            selection_seed=args.seed,
            require_full_budget=True,
            batch_size=args.batch_size,
            device=args.device,
        ),
        hardened_training=HardenedTrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            target_fpr=args.target_fpr,
            device=args.device,
        ),
    )

    for round_index in range(args.rounds):
        round_seed = args.seed + round_index * 10_000

        round_config = CoevolutionRoundConfig(
            method=args.method,
            attack_generation=replace(
                base_config.attack_generation,
                seed=round_seed,
            ),
            weakness_mining=replace(
                base_config.weakness_mining,
                selection_seed=round_seed,
            ),
            hardened_training=replace(
                base_config.hardened_training,
                seed=round_seed,
            ),
        )

        inherited_policy = (
            parent_policy
            if bool(profile["inherit_parent_policy"])
            else None
        )

        result = run_coevolution_round(
            round_index=round_index,
            parent_defender_run_dir=parent_run,
            normal_dataset_root=args.normal_root,
            normal_manifest_path=args.normal_manifest,
            output_root=output_root,
            parent_policy_path=inherited_policy,
            config=round_config,
        )

        weakness = result["weakness_mining"]

        timeline.append({
            "round_index": round_index,
            "method": args.method,
            "seed": round_seed,
            "round_manifest_path": (
                result["round_manifest_path"]
            ),
            "defender_run_dir": (
                result["next_defender_run_dir"]
            ),
            "generated_parent_policy_path": (
                result["next_parent_policy_path"]
            ),
            "parent_policy_inherited": bool(
                profile["inherit_parent_policy"]
            ),
            "selection_mode": (
                weakness["selection_mode"]
            ),
            "parent_selection_mode": (
                weakness["parent_selection_mode"]
            ),
            "attack_window_count": (
                weakness["attack_window_count"]
            ),
            "hard_negative_count": (
                weakness["hard_negative_count"]
            ),
            "selected_parent_session_count": (
                weakness["selected_parent_session_count"]
            ),
            "bypass_rate": weakness["bypass_rate"],
        })

        parent_run = Path(
            result["next_defender_run_dir"]
        )

        if bool(profile["inherit_parent_policy"]):
            parent_policy = Path(
                result["next_parent_policy_path"]
            )
        else:
            parent_policy = None

    output = output_root / "coevolution_timeline.json"
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "status": "PASS",
        "method": args.method,
        "round_count": args.rounds,
        "protocol_budget": {
            "candidates_per_round": args.candidates,
            "keystrokes_per_candidate": (
                args.keystrokes_per_candidate
            ),
            "mutation_strength": (
                args.mutation_strength
            ),
            "hard_windows_per_round": (
                args.max_hard_windows
            ),
            "parent_sessions_per_round": (
                args.max_parent_sessions
            ),
            "hardening_epochs_per_round": (
                args.epochs
            ),
        },
        "selection_policy": {
            "window_selection": (
                profile["selection_mode"]
            ),
            "parent_selection": (
                profile["parent_selection_mode"]
            ),
            "parent_policy_inheritance": bool(
                profile["inherit_parent_policy"]
            ),
        },
        "rounds": timeline,
        "latest_defender_run_dir": str(parent_run),
        "latest_parent_policy_path": (
            str(parent_policy)
            if parent_policy is not None
            else None
        ),
        "final_holdout_status": (
            "not_run_by_design"
        ),
    }

    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        output.read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()
