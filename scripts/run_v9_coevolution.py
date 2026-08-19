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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stable-v9 raw-event A_k -> weakness mining -> D_{k+1} rounds. "
            "The final held-out attack evaluation is intentionally separate."
        )
    )
    parser.add_argument("--normal-root", required=True)
    parser.add_argument("--normal-manifest", required=True)
    parser.add_argument("--d0-run-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--keystrokes-per-candidate", type=int, default=72)
    parser.add_argument("--max-hard-windows", type=int, default=512)
    parser.add_argument("--max-parent-sessions", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    output_root = Path(args.output_root).resolve()
    parent_run = Path(args.d0_run_dir).resolve()
    parent_policy: Path | None = None
    timeline = []
    base_config = CoevolutionRoundConfig(
        attack_generation=RawAttackGenerationConfig(
            candidates=args.candidates,
            keystrokes_per_candidate=args.keystrokes_per_candidate,
            seed=args.seed,
        ),
        weakness_mining=WeaknessMiningConfig(
            max_hard_windows=args.max_hard_windows,
            max_parent_sessions=args.max_parent_sessions,
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
        round_config = CoevolutionRoundConfig(
            attack_generation=replace(
                base_config.attack_generation,
                seed=args.seed + round_index * 10_000,
            ),
            weakness_mining=base_config.weakness_mining,
            hardened_training=replace(
                base_config.hardened_training,
                seed=args.seed + round_index * 10_000,
            ),
        )
        result = run_coevolution_round(
            round_index=round_index,
            parent_defender_run_dir=parent_run,
            normal_dataset_root=args.normal_root,
            normal_manifest_path=args.normal_manifest,
            output_root=output_root,
            parent_policy_path=parent_policy,
            config=round_config,
        )
        timeline.append({
            "round_index": round_index,
            "round_manifest_path": result["round_manifest_path"],
            "defender_run_dir": result["next_defender_run_dir"],
            "parent_policy_path": result["next_parent_policy_path"],
            "bypass_rate": result["weakness_mining"]["bypass_rate"],
        })
        parent_run = Path(result["next_defender_run_dir"])
        parent_policy = Path(result["next_parent_policy_path"])

    output = output_root / "coevolution_timeline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "PASS",
        "rounds": timeline,
        "latest_defender_run_dir": str(parent_run),
        "latest_parent_policy_path": str(parent_policy),
        "final_holdout_status": "not_run_by_design",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
