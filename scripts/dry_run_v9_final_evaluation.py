from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dry_run_v9_data_readiness import PROTOCOL, _write_session
from src.coevolution_v9 import (
    CoevolutionRoundConfig,
    FinalEvaluationConfig,
    HardenedTrainingConfig,
    RawAttackGenerationConfig,
    WeaknessMiningConfig,
    run_coevolution_round,
    run_final_evaluation,
)
from src.data_v2 import (
    SessionValidationConfig,
    build_dataset_manifest,
    validate_dataset_root,
    write_dataset_manifest,
)
from src.models_v9 import ReConHIDV9Config
from src.training_v9 import NormalPretrainConfig, run_normal_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny CPU smoke for D0 -> A0 -> D1 -> frozen V_final."
    )
    parser.add_argument(
        "--output-root",
        default="experiments/v9_final_evaluation_dry_run",
    )
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = Path(tempfile.mkdtemp(prefix="v9_final_eval_", dir=output_root))
    normal_root = workspace / "normal"

    try:
        for participant_index in range(6):
            participant = f"p{participant_index + 1:03d}"
            for scenario in PROTOCOL:
                _write_session(
                    normal_root,
                    participant=participant,
                    scenario=scenario,
                    participant_index=participant_index,
                    windows=6,
                )

        report = validate_dataset_root(
            normal_root,
            config=SessionValidationConfig(require_complete_core_scenarios=True),
        )
        if report.has_errors:
            raise RuntimeError(report.summary_dict())
        manifest = build_dataset_manifest(
            report,
            dataset_id=f"final_eval_dry_{timestamp}",
        )
        manifest_paths = write_dataset_manifest(manifest, workspace / "manifest")

        d0 = run_normal_pretraining(
            dataset_root=normal_root,
            manifest_path=manifest_paths["jsonl"],
            verify_manifest_hashes=True,
            output_root=workspace / "defenders",
            run_name="D0",
            config=NormalPretrainConfig(
                seed=501,
                epochs=1,
                batch_size=8,
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
        round_result = run_coevolution_round(
            round_index=0,
            parent_defender_run_dir=d0.run_dir,
            normal_dataset_root=normal_root,
            normal_manifest_path=manifest_paths["jsonl"],
            output_root=workspace / "rounds",
            config=CoevolutionRoundConfig(
                attack_generation=RawAttackGenerationConfig(
                    candidates=6,
                    keystrokes_per_candidate=60,
                    seed=601,
                    mutation_strength=0.05,
                ),
                weakness_mining=WeaknessMiningConfig(
                    max_hard_windows=24,
                    max_parent_sessions=3,
                    batch_size=8,
                    device="cpu",
                ),
                hardened_training=HardenedTrainingConfig(
                    seed=701,
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
                ),
            ),
        )
        timeline = workspace / "rounds" / "coevolution_timeline.json"
        timeline.write_text(json.dumps({
            "status": "PASS",
            "rounds": [{
                "round_index": 0,
                "round_manifest_path": round_result["round_manifest_path"],
                "defender_run_dir": round_result["next_defender_run_dir"],
                "parent_policy_path": round_result["next_parent_policy_path"],
                "bypass_rate": round_result["weakness_mining"]["bypass_rate"],
            }],
            "latest_defender_run_dir": round_result["next_defender_run_dir"],
            "latest_parent_policy_path": round_result["next_parent_policy_path"],
            "final_holdout_status": "not_run_by_design",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        final_result = run_final_evaluation(
            final_defender_run_dir=round_result["next_defender_run_dir"],
            normal_dataset_root=normal_root,
            normal_manifest_path=manifest_paths["jsonl"],
            coevolution_timeline_path=timeline,
            output_dir=workspace / "final_evaluation",
            config=FinalEvaluationConfig(
                seed=990260807,
                candidates=6,
                keystrokes_per_candidate=60,
                mutation_strength=0.0,
                batch_size=8,
                device="cpu",
            ),
        )
        print(json.dumps({
            "status": "PASS",
            "normal_validation": report.summary_dict(),
            "d_final_run_dir": round_result["next_defender_run_dir"],
            "training_attack_families": [
                "constant_fast",
                "jittered_mimic",
                "burst_pause",
                "overlap_dense",
                "shortcut_heavy",
                "correction_heavy",
            ],
            "A_final_families": list(FinalEvaluationConfig().families),
            "final_evaluation": final_result,
            "workspace": str(workspace),
            "note": (
                "Synthetic data and one-epoch models are execution smoke only; "
                "the displayed FPR/TPR values are not research results."
            ),
        }, ensure_ascii=False, indent=2))
    finally:
        if not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
