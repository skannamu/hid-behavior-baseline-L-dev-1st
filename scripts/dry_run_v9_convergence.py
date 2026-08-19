from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )

from scripts.dry_run_v9_data_readiness import (
    PROTOCOL,
    _write_session,
)

from src.coevolution_v9 import (
    CoevolutionRoundConfig,
    ConvergenceConfig,
    ExperimentLoopConfig,
    FinalEvaluationConfig,
    HardenedTrainingConfig,
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

from src.models_v9 import ReConHIDV9Config
from src.training_v9 import (
    NormalPretrainConfig,
    run_normal_pretraining,
)


def main() -> None:
    output_root = Path(
        "experiments/v9_convergence_dry_run"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    workspace = Path(
        tempfile.mkdtemp(
            prefix="v9_convergence_",
            dir=output_root,
        )
    )

    normal_root = workspace / "normal"

    try:
        # ------------------------------------------------------
        # Synthetic stable-v2 normal dataset
        # ------------------------------------------------------
        for participant_index in range(6):
            participant = (
                f"p{participant_index + 1:03d}"
            )

            for scenario in PROTOCOL:
                _write_session(
                    normal_root,
                    participant=participant,
                    scenario=scenario,
                    participant_index=participant_index,
                    windows=6,
                )

        validation = validate_dataset_root(
            normal_root,
            config=SessionValidationConfig(
                require_complete_core_scenarios=True
            ),
        )

        if validation.has_errors:
            raise RuntimeError(
                validation.summary_dict()
            )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        manifest = build_dataset_manifest(
            validation,
            dataset_id=(
                f"convergence_dry_{timestamp}"
            ),
        )

        manifest_paths = write_dataset_manifest(
            manifest,
            workspace / "manifest",
        )

        # ------------------------------------------------------
        # Tiny CPU D0
        # ------------------------------------------------------
        d0 = run_normal_pretraining(
            dataset_root=normal_root,
            manifest_path=manifest_paths["jsonl"],
            verify_manifest_hashes=True,
            output_root=workspace / "defenders",
            run_name="D0",
            config=NormalPretrainConfig(
                seed=77,
                epochs=1,
                batch_size=8,
                learning_rate=1.0e-3,
                weight_decay=0.0,
                patience=1,
                target_fpr=0.20,
                split_group_mode="participant",
                balance_training=True,
                training_window_stride=2,
                training_max_windows_per_group=10,
                device="cpu",
                deterministic=True,
            ),
            model_config=ReConHIDV9Config(
                sequence_hidden_dim=8,
                sequence_latent_dim=4,
                context_hidden_dim=8,
                context_latent_dim=4,
                fused_latent_dim=8,
                classifier_hidden_dim=8,
                dropout=0.0,
            ),
        )

        # ------------------------------------------------------
        # Real coevolution components, tiny budget.
        #
        # Threshold=1.0 is ONLY a control-flow smoke setting.
        #
        # patience=2 forces:
        # D0 probe -> harden D1 -> D1 probe -> converge.
        # ------------------------------------------------------
        families = RawAttackGenerationConfig().families

        base_round = CoevolutionRoundConfig(
            method="recon_hid",
            attack_generation=(
                RawAttackGenerationConfig(
                    candidates=6,
                    keystrokes_per_candidate=60,
                    seed=88,
                    mutation_strength=0.05,
                )
            ),
            weakness_mining=(
                WeaknessMiningConfig(
                    max_hard_windows=24,
                    max_parent_sessions=6,
                    selection_mode="guided",
                    parent_selection_mode="guided",
                    selection_seed=88,
                    require_full_budget=True,
                    preserve_parent_family_coverage=True,
                    batch_size=8,
                    device="cpu",
                )
            ),
            hardened_training=(
                HardenedTrainingConfig(
                    seed=99,
                    epochs=1,
                    batch_size=8,
                    learning_rate=5.0e-4,
                    weight_decay=0.0,
                    patience=1,
                    target_fpr=0.20,
                    attack_dev_ratio=0.34,
                    max_attack_train_windows=24,
                    device="cpu",
                    deterministic=True,
                )
            ),
        )

        loop_config = ExperimentLoopConfig(
            stopping_mode="convergence",
            seed=88,
            convergence=ConvergenceConfig(
                min_rounds=2,
                max_rounds=3,
                global_bypass_threshold=1.0,
                family_bypass_threshold=1.0,
                patience=2,
                required_families=tuple(
                    families
                ),
            ),
        )

        result = run_experiment_loop(
            normal_dataset_root=normal_root,
            normal_manifest_path=(
                manifest_paths["jsonl"]
            ),
            d0_run_dir=d0.run_dir,
            output_root=(
                workspace / "coevolution"
            ),
            base_round_config=base_round,
            loop_config=loop_config,
        )

        # ------------------------------------------------------
        # Structural assertions
        # ------------------------------------------------------
        if not result["converged"]:
            raise RuntimeError(
                "Expected convergence smoke to converge"
            )

        if result["stop_reason"] != "converged":
            raise RuntimeError(
                "Unexpected stop reason: "
                f'{result["stop_reason"]}'
            )

        if len(result["timeline"]) != 2:
            raise RuntimeError(
                "Expected exactly two probes"
            )

        first = result["timeline"][0]
        second = result["timeline"][1]

        if first["next_defender_run_dir"] is None:
            raise RuntimeError(
                "First probe should harden D1"
            )

        if second["next_defender_run_dir"] is not None:
            raise RuntimeError(
                "Converged probe must not create D2"
            )

        final_dir = Path(
            result["final_defender_run_dir"]
        )

        if final_dir.name != "D1":
            raise RuntimeError(
                "Expected probed D1 to become final "
                f"candidate, got {final_dir}"
            )

        # ------------------------------------------------------
        # One-shot A_final / V_final.
        # Only after the convergence loop has selected D_final.
        # ------------------------------------------------------
        final_result = run_final_evaluation(
            final_defender_run_dir=(
                result["final_defender_run_dir"]
            ),
            normal_dataset_root=normal_root,
            normal_manifest_path=(
                manifest_paths["jsonl"]
            ),
            coevolution_timeline_path=(
                result["timeline_path"]
            ),
            output_dir=(
                workspace / "final_evaluation"
            ),
            config=FinalEvaluationConfig(
                seed=990260807,
                candidates=6,
                keystrokes_per_candidate=60,
                mutation_strength=0.0,
                batch_size=8,
                device="cpu",
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
                "A_final lineage audit did not PASS"
            )

        payload = {
            "status": "PASS",
            "normal_validation": (
                validation.summary_dict()
            ),
            "d0": asdict(d0),
            "coevolution": result,
            "final_evaluation": final_result,
            "assertions": {
                "probe_count": 2,
                "D0_hardened_to_D1": True,
                "D1_was_probed": True,
                "D2_was_not_created": True,
                "final_defender_is_D1": True,
                "A_final_used_only_after_convergence": True,
                "A_final_lineage_audit_passed": True,
            },
            "note": (
                "Synthetic execution smoke only. "
                "A_final is executed only after convergence. "
                "Thresholds of 1.0 are deliberately permissive "
                "to verify convergence control flow and are not "
                "research protocol values."
            ),
            "workspace": str(workspace),
        }

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        shutil.rmtree(
            workspace,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
