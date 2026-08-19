from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )

from src.coevolution_v9 import (
    CoevolutionRoundConfig,
    ConvergenceConfig,
    ExperimentLoopConfig,
    HardenedTrainingConfig,
    METHOD_PROFILES,
    RawAttackGenerationConfig,
    WeaknessMiningConfig,
    run_experiment_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stable-v9 hardening with either a fixed "
            "matched budget or convergence-driven stopping."
        )
    )

    parser.add_argument(
        "--normal-root",
        required=True,
    )
    parser.add_argument(
        "--normal-manifest",
        required=True,
    )
    parser.add_argument(
        "--d0-run-dir",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--method",
        choices=tuple(METHOD_PROFILES),
        default="recon_hid",
    )

    parser.add_argument(
        "--stopping-mode",
        choices=("fixed", "convergence"),
        default="fixed",
    )

    # Fixed-budget mode only.
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
    )

    # Convergence mode.
    parser.add_argument(
        "--min-rounds",
        type=int,
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
    )
    parser.add_argument(
        "--global-bypass-threshold",
        type=float,
    )
    parser.add_argument(
        "--family-bypass-threshold",
        type=float,
    )
    parser.add_argument(
        "--convergence-patience",
        type=int,
    )

    # Shared per-round budget.
    parser.add_argument(
        "--candidates",
        type=int,
        default=24,
    )
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

    # Shared training budget.
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
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

    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260807,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    profile = METHOD_PROFILES[
        args.method
    ]

    base_round_config = CoevolutionRoundConfig(
        method=args.method,
        attack_generation=(
            RawAttackGenerationConfig(
                candidates=args.candidates,
                keystrokes_per_candidate=(
                    args.keystrokes_per_candidate
                ),
                seed=args.seed,
                mutation_strength=(
                    args.mutation_strength
                ),
            )
        ),
        weakness_mining=WeaknessMiningConfig(
            max_hard_windows=(
                args.max_hard_windows
            ),
            max_parent_sessions=(
                args.max_parent_sessions
            ),
            selection_mode=(
                profile.selection_mode
            ),
            parent_selection_mode=(
                profile.parent_selection_mode
            ),
            selection_seed=args.seed,
            require_full_budget=True,
            preserve_parent_family_coverage=(
                profile.parent_selection_mode
                != "none"
            ),
            batch_size=args.batch_size,
            device=args.device,
        ),
        hardened_training=HardenedTrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=(
                args.learning_rate
            ),
            target_fpr=args.target_fpr,
            device=args.device,
        ),
    )

    if args.stopping_mode == "fixed":
        loop_config = ExperimentLoopConfig(
            stopping_mode="fixed",
            seed=args.seed,
            fixed_rounds=args.rounds,
        )

    else:
        required = {
            "--min-rounds": args.min_rounds,
            "--max-rounds": args.max_rounds,
            "--global-bypass-threshold": (
                args.global_bypass_threshold
            ),
            "--family-bypass-threshold": (
                args.family_bypass_threshold
            ),
            "--convergence-patience": (
                args.convergence_patience
            ),
        }

        missing = [
            name
            for name, value in required.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                "Convergence mode requires: "
                + ", ".join(missing)
            )

        convergence = ConvergenceConfig(
            min_rounds=args.min_rounds,
            max_rounds=args.max_rounds,
            global_bypass_threshold=(
                args.global_bypass_threshold
            ),
            family_bypass_threshold=(
                args.family_bypass_threshold
            ),
            patience=(
                args.convergence_patience
            ),
            required_families=tuple(
                base_round_config
                .attack_generation
                .families
            ),
        )

        loop_config = ExperimentLoopConfig(
            stopping_mode="convergence",
            seed=args.seed,
            convergence=convergence,
        )

    result = run_experiment_loop(
        normal_dataset_root=(
            args.normal_root
        ),
        normal_manifest_path=(
            args.normal_manifest
        ),
        d0_run_dir=args.d0_run_dir,
        output_root=args.output_root,
        base_round_config=(
            base_round_config
        ),
        loop_config=loop_config,
    )

    import json

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
