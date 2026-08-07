from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dry_run_v9_data_readiness import PROTOCOL, _write_session
from src.coevolution_v9 import (
    CoevolutionRoundConfig,
    HardenedTrainingConfig,
    RawAttackGenerationConfig,
    WeaknessMiningConfig,
    run_coevolution_round,
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
        description=(
            "Tiny CPU smoke for Stable-v9 D0 -> raw A0 -> weakness mining -> D1."
        )
    )
    parser.add_argument(
        "--output-root",
        default="experiments/v9_coevolution_dry_run",
    )
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = Path(tempfile.mkdtemp(prefix="v9_coevolution_", dir=output_root))
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
            config=SessionValidationConfig(
                require_complete_core_scenarios=True
            ),
        )
        if report.has_errors:
            raise RuntimeError(report.summary_dict())
        manifest = build_dataset_manifest(
            report,
            dataset_id=f"coevolution_dry_{timestamp}",
        )
        manifest_paths = write_dataset_manifest(
            manifest,
            workspace / "manifest",
        )

        d0_result = run_normal_pretraining(
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

        round_result = run_coevolution_round(
            round_index=0,
            parent_defender_run_dir=d0_result.run_dir,
            normal_dataset_root=normal_root,
            normal_manifest_path=manifest_paths["jsonl"],
            output_root=workspace / "rounds",
            config=CoevolutionRoundConfig(
                attack_generation=RawAttackGenerationConfig(
                    candidates=6,
                    keystrokes_per_candidate=60,
                    seed=88,
                    mutation_strength=0.05,
                ),
                weakness_mining=WeaknessMiningConfig(
                    max_hard_windows=24,
                    max_parent_sessions=3,
                    batch_size=8,
                    device="cpu",
                ),
                hardened_training=HardenedTrainingConfig(
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
                ),
            ),
        )

        output = {
            "status": "PASS",
            "normal_validation": report.summary_dict(),
            "normal_manifest": manifest.summary_dict(),
            "d0": asdict(d0_result),
            "round_0": {
                "round_manifest_path": round_result["round_manifest_path"],
                "candidate_count": round_result["attack_generation"]["candidate_count"],
                "attack_window_count": round_result["weakness_mining"]["attack_window_count"],
                "bypass_rate": round_result["weakness_mining"]["bypass_rate"],
                "hard_negative_count": round_result["weakness_mining"]["hard_negative_count"],
                "d1_run_dir": round_result["next_defender_run_dir"],
            },
            "workspace": str(workspace),
            "note": (
                "Synthetic normal windows and raw-event attack candidates are "
                "execution smoke data only; the displayed metrics are not research results."
            ),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        if not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
