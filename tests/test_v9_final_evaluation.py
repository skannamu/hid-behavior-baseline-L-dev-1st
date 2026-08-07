from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.dry_run_v9_data_readiness import PROTOCOL, _write_session
from src.coevolution_v9 import (
    CoevolutionRoundConfig,
    FinalEvaluationConfig,
    HardenedTrainingConfig,
    RawAttackGenerationConfig,
    WeaknessMiningConfig,
    run_coevolution_round,
    audit_final_lineage,
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


class V9FinalEvaluationTest(unittest.TestCase):
    def _build_one_round(self, root: Path):
        normal_root = root / "normal"
        for participant_index in range(6):
            participant = f"p{participant_index + 1:03d}"
            for scenario in PROTOCOL:
                _write_session(
                    normal_root,
                    participant=participant,
                    scenario=scenario,
                    participant_index=participant_index,
                    windows=5,
                )
        report = validate_dataset_root(
            normal_root,
            config=SessionValidationConfig(require_complete_core_scenarios=True),
        )
        manifest = build_dataset_manifest(report, dataset_id="final_eval_test")
        manifest_paths = write_dataset_manifest(manifest, root / "manifest")
        d0 = run_normal_pretraining(
            dataset_root=normal_root,
            manifest_path=manifest_paths["jsonl"],
            verify_manifest_hashes=True,
            output_root=root / "defenders",
            run_name="D0",
            config=NormalPretrainConfig(
                seed=31,
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
            output_root=root / "rounds",
            config=CoevolutionRoundConfig(
                attack_generation=RawAttackGenerationConfig(
                    candidates=3,
                    keystrokes_per_candidate=56,
                    seed=37,
                    mutation_strength=0.02,
                ),
                weakness_mining=WeaknessMiningConfig(
                    max_hard_windows=12,
                    max_parent_sessions=2,
                    batch_size=8,
                    device="cpu",
                ),
                hardened_training=HardenedTrainingConfig(
                    seed=41,
                    epochs=1,
                    batch_size=8,
                    learning_rate=5.0e-4,
                    weight_decay=0.0,
                    patience=1,
                    target_fpr=0.20,
                    attack_dev_ratio=0.34,
                    max_attack_train_windows=12,
                    device="cpu",
                    deterministic=True,
                ),
            ),
        )
        timeline_path = root / "rounds" / "coevolution_timeline.json"
        timeline_path.write_text(json.dumps({
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
        }, indent=2), encoding="utf-8")
        return normal_root, Path(manifest_paths["jsonl"]), Path(
            round_result["next_defender_run_dir"]
        ), timeline_path

    def test_final_eval_freezes_and_uses_disjoint_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normal_root, manifest_path, d_final, timeline = self._build_one_round(root)
            output = root / "final_eval"
            result = run_final_evaluation(
                final_defender_run_dir=d_final,
                normal_dataset_root=normal_root,
                normal_manifest_path=manifest_path,
                coevolution_timeline_path=timeline,
                output_dir=output,
                config=FinalEvaluationConfig(
                    seed=99001,
                    candidates=3,
                    keystrokes_per_candidate=56,
                    mutation_strength=0.0,
                    batch_size=8,
                    device="cpu",
                ),
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["lineage_audit_status"], "PASS")
            self.assertTrue((output / "frozen_defender" / "best_model.pt").is_file())
            self.assertTrue((output / "V_final" / "final_metrics.json").is_file())
            audit = json.loads(
                (output / "V_final" / "lineage_audit.json").read_text()
            )
            self.assertEqual(audit["overlaps"]["families"], [])
            self.assertEqual(audit["overlaps"]["seeds"], [])
            self.assertEqual(
                audit["overlaps"]["non_null_parent_candidate_ids"], []
            )
            metrics = json.loads(
                (output / "V_final" / "final_metrics.json").read_text()
            )
            self.assertTrue(metrics["methodology"]["defender_frozen_before_scoring"])
            self.assertFalse(
                metrics["methodology"]["threshold_recalibrated_on_final_data"]
            )
            self.assertGreater(metrics["A_final"]["window_count"], 0)
            self.assertGreater(metrics["normal_test"]["test_window_count"], 0)

            with self.assertRaises(FileExistsError):
                run_final_evaluation(
                    final_defender_run_dir=d_final,
                    normal_dataset_root=normal_root,
                    normal_manifest_path=manifest_path,
                    coevolution_timeline_path=timeline,
                    output_dir=output,
                    config=FinalEvaluationConfig(
                        seed=99002,
                        candidates=3,
                        keystrokes_per_candidate=56,
                        batch_size=8,
                        device="cpu",
                    ),
                )

    def test_lineage_audit_rejects_training_family_overlap(self):
        training = {
            "candidate_ids": ["A0_g0_c0000"],
            "seeds": [101],
            "families": ["constant_fast"],
            "window_hashes": ["abc"],
            "latest_defender_run_dir": "/tmp/D1",
            "round_count": 1,
        }
        final_entries = [{
            "candidate_id": "A_final_g10000_c0000",
            "seed": 202,
            "family": "constant_fast",
            "window_sha256": "def",
            "parent_candidate_id": None,
        }]
        with self.assertRaises(ValueError):
            audit_final_lineage(
                final_entries=final_entries,
                training_lineage=training,
                final_defender_source_dir="/tmp/D1",
                config=FinalEvaluationConfig(),
            )


if __name__ == "__main__":
    unittest.main()
