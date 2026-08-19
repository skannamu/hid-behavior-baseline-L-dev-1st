"""Run one complete Stable-v9 ReCon-HID experiment.

This script only orchestrates existing validated components:

normal validation
 -> frozen manifest
 -> normal-only D0
 -> iterative co-evolution
 -> converged D_final
 -> one-shot held-out A_final / V_final

It intentionally contains no duplicate feature extraction, attack generation,
weakness mining, defender training, or final-evaluation implementation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
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
    FinalEvaluationConfig,
    HardenedTrainingConfig,
    METHOD_PROFILES,
    RawAttackGenerationConfig,
    WeaknessMiningConfig,
    run_experiment_loop,
    run_final_evaluation,
)
from src.data_v2 import (
    SessionValidationConfig,
    build_dataset_manifest,
    validate_dataset_root,
    write_dataset_manifest,
)
from src.training_v9 import (
    NormalPretrainConfig,
    run_normal_pretraining,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one full Stable-v9 normal -> D0 -> co-evolution "
            "-> D_final -> A_final experiment."
        )
    )

    p.add_argument("--normal-root", required=True)
    p.add_argument("--output-root", required=True)

    # Paper-grade explicit participant roles.
    p.add_argument(
        "--train-groups",
        nargs="+",
        required=True,
    )
    p.add_argument(
        "--calibration-group",
        required=True,
    )
    p.add_argument(
        "--test-group",
        required=True,
    )

    p.add_argument(
        "--method",
        choices=tuple(METHOD_PROFILES),
        default="recon_hid",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260807)

    # ----------------------------------------------------------
    # D0
    # ----------------------------------------------------------
    p.add_argument("--d0-epochs", type=int, default=50)
    p.add_argument("--d0-batch-size", type=int, default=128)
    p.add_argument(
        "--d0-learning-rate",
        type=float,
        default=1.0e-3,
    )
    p.add_argument(
        "--d0-weight-decay",
        type=float,
        default=1.0e-4,
    )
    p.add_argument("--d0-patience", type=int, default=8)
    p.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
    )

    # Must be deliberately declared for the main experiment.
    p.add_argument(
        "--training-window-stride",
        type=int,
        required=True,
    )
    p.add_argument(
        "--training-max-windows-per-group",
        type=int,
        required=True,
    )

    # ----------------------------------------------------------
    # Co-evolution budget
    # ----------------------------------------------------------
    p.add_argument("--candidates", type=int, default=24)
    p.add_argument(
        "--keystrokes-per-candidate",
        type=int,
        default=72,
    )
    p.add_argument(
        "--mutation-strength",
        type=float,
        default=0.18,
    )
    p.add_argument(
        "--max-hard-windows",
        type=int,
        default=192,
    )
    p.add_argument(
        "--max-parent-sessions",
        type=int,
        default=8,
    )

    p.add_argument(
        "--hardening-epochs",
        type=int,
        default=20,
    )
    p.add_argument(
        "--hardening-batch-size",
        type=int,
        default=128,
    )
    p.add_argument(
        "--hardening-learning-rate",
        type=float,
        default=3.0e-4,
    )

    # ----------------------------------------------------------
    # Convergence protocol.
    #
    # NO defaults by design. These must be frozen before results.
    # ----------------------------------------------------------
    p.add_argument(
        "--min-rounds",
        type=int,
        required=True,
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        required=True,
    )
    p.add_argument(
        "--global-bypass-threshold",
        type=float,
        required=True,
    )
    p.add_argument(
        "--family-bypass-threshold",
        type=float,
        required=True,
    )
    p.add_argument(
        "--convergence-patience",
        type=int,
        required=True,
    )

    # ----------------------------------------------------------
    # Held-out final attack
    # ----------------------------------------------------------
    p.add_argument(
        "--final-seed",
        type=int,
        default=920260807,
    )
    p.add_argument(
        "--final-candidates",
        type=int,
        default=24,
    )
    p.add_argument(
        "--final-keystrokes-per-candidate",
        type=int,
        default=84,
    )
    p.add_argument(
        "--final-mutation-strength",
        type=float,
        default=0.10,
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.train_groups) != 4:
        raise ValueError(
            "Paper protocol requires exactly "
            "four Train participants"
        )

    train_groups = tuple(
        args.train_groups
    )

    if len(set(train_groups)) != 4:
        raise ValueError(
            "Train participant IDs must be unique"
        )

    calibration_group = str(
        args.calibration_group
    )

    test_group = str(
        args.test_group
    )

    all_roles = (
        set(train_groups)
        | {calibration_group}
        | {test_group}
    )

    if len(all_roles) != 6:
        raise ValueError(
            "Train/Calibration/Test participant "
            "roles must be mutually disjoint"
        )

    normal_root = Path(
        args.normal_root
    ).resolve()

    if not normal_root.is_dir():
        raise FileNotFoundError(normal_root)

    output = Path(
        args.output_root
    ).resolve()

    # Main runs are immutable.
    if output.exists():
        raise FileExistsError(
            "Experiment output already exists; "
            f"refusing to overwrite: {output}"
        )

    output.mkdir(parents=True)

    started_at = datetime.now().isoformat()

    # ==========================================================
    # 1. Strict normal-data validation
    # ==========================================================
    report = validate_dataset_root(
        normal_root,
        config=SessionValidationConfig(
            require_complete_core_scenarios=True
        ),
    )

    validation_path = (
        output / "normal_validation.json"
    )
    validation_path.write_text(
        json.dumps(
            report.summary_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if report.has_errors:
        raise RuntimeError(
            "Normal-data validation failed. "
            f"See {validation_path}"
        )

    # ==========================================================
    # 2. Immutable dataset manifest
    # ==========================================================
    dataset_id = (
        "stable_v9_main_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    manifest = build_dataset_manifest(
        report,
        dataset_id=dataset_id,
    )

    manifest_paths = write_dataset_manifest(
        manifest,
        output / "normal_manifest",
    )

    # ==========================================================
    # 3. Normal-only D0
    # ==========================================================
    d0 = run_normal_pretraining(
        dataset_root=normal_root,
        manifest_path=manifest_paths["jsonl"],
        verify_manifest_hashes=True,
        output_root=output / "defenders",
        run_name="D0",
        config=NormalPretrainConfig(
            seed=args.seed,
            epochs=args.d0_epochs,
            batch_size=args.d0_batch_size,
            learning_rate=args.d0_learning_rate,
            weight_decay=args.d0_weight_decay,
            patience=args.d0_patience,
            target_fpr=args.target_fpr,
            split_group_mode="participant",
            explicit_train_groups=(
                train_groups
            ),
            explicit_calibration_groups=(
                calibration_group,
            ),
            explicit_test_groups=(
                test_group,
            ),
            evaluate_test_during_pretraining=False,
            balance_training=True,
            training_window_stride=(
                args.training_window_stride
            ),
            training_max_windows_per_group=(
                args.training_max_windows_per_group
            ),
            device=args.device,
            deterministic=True,
        ),
    )

    # ==========================================================
    # 4. Existing co-evolution implementation
    # ==========================================================
    profile = METHOD_PROFILES[
        args.method
    ]

    attack_cfg = RawAttackGenerationConfig(
        candidates=args.candidates,
        keystrokes_per_candidate=(
            args.keystrokes_per_candidate
        ),
        seed=args.seed,
        mutation_strength=(
            args.mutation_strength
        ),
    )

    round_cfg = CoevolutionRoundConfig(
        method=args.method,
        attack_generation=attack_cfg,
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
            batch_size=(
                args.hardening_batch_size
            ),
            device=args.device,
        ),
        hardened_training=HardenedTrainingConfig(
            seed=args.seed,
            epochs=args.hardening_epochs,
            batch_size=(
                args.hardening_batch_size
            ),
            learning_rate=(
                args.hardening_learning_rate
            ),
            target_fpr=args.target_fpr,
            device=args.device,
            deterministic=True,
        ),
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
        patience=args.convergence_patience,
        required_families=tuple(
            attack_cfg.families
        ),
    )

    loop_result = run_experiment_loop(
        normal_dataset_root=normal_root,
        normal_manifest_path=(
            manifest_paths["jsonl"]
        ),
        d0_run_dir=d0.run_dir,
        output_root=output / "coevolution",
        base_round_config=round_cfg,
        loop_config=ExperimentLoopConfig(
            stopping_mode="convergence",
            seed=args.seed,
            convergence=convergence,
        ),
    )

    # Do NOT call A_final on a failed convergence run.
    if not loop_result["converged"]:
        summary = {
            "status": "NOT_CONVERGED",
            "started_at": started_at,
            "dataset_id": dataset_id,
            "method": args.method,
            "d0_run_dir": d0.run_dir,
            "coevolution": loop_result,
            "A_final_status": "NOT_RUN",
        }

        (output / "experiment_summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Co-evolution reached its stopping cap "
            "without convergence; A_final was not run."
        )

    # ==========================================================
    # 5. Existing frozen held-out final evaluation
    # ==========================================================
    final_result = run_final_evaluation(
        final_defender_run_dir=(
            loop_result[
                "final_defender_run_dir"
            ]
        ),
        normal_dataset_root=normal_root,
        normal_manifest_path=(
            manifest_paths["jsonl"]
        ),
        coevolution_timeline_path=(
            loop_result["timeline_path"]
        ),
        output_dir=(
            output / "final_evaluation"
        ),
        config=FinalEvaluationConfig(
            seed=args.final_seed,
            candidates=args.final_candidates,
            keystrokes_per_candidate=(
                args.final_keystrokes_per_candidate
            ),
            mutation_strength=(
                args.final_mutation_strength
            ),
            batch_size=(
                args.hardening_batch_size
            ),
            device=args.device,
        ),
    )

    if final_result["status"] != "PASS":
        raise RuntimeError(
            "Final evaluation did not PASS"
        )

    if (
        final_result["lineage_audit_status"]
        != "PASS"
    ):
        raise RuntimeError(
            "Final lineage audit did not PASS"
        )

    summary = {
        "status": "PASS",
        "started_at": started_at,
        "completed_at": (
            datetime.now().isoformat()
        ),
        "dataset_id": dataset_id,
        "normal_root": str(normal_root),
        "normal_manifest": manifest_paths,
        "normal_validation": (
            report.summary_dict()
        ),
        "method": args.method,
        "d0_run_dir": d0.run_dir,
        "coevolution": loop_result,
        "d_final_run_dir": (
            loop_result[
                "final_defender_run_dir"
            ]
        ),
        "final_evaluation": final_result,
    }

    summary_path = (
        output / "experiment_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(output),
                "dataset_id": dataset_id,
                "d0_run_dir": d0.run_dir,
                "d_final_run_dir": (
                    loop_result[
                        "final_defender_run_dir"
                    ]
                ),
                "convergence_probe_count": (
                    loop_result[
                        "convergence"
                    ]["probe_count"]
                ),
                "lineage_audit_status": (
                    final_result[
                        "lineage_audit_status"
                    ]
                ),
                "summary_path": str(
                    summary_path
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
