from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.dry_run_v9_data_readiness import PROTOCOL, _write_session
from src.coevolution_v9 import (
    CoevolutionRoundConfig,
    HardenedTrainingConfig,
    RawAttackGenerationConfig,
    WeaknessMiningConfig,
    generate_raw_attack_pool,
    run_coevolution_round,
)
from src.data_v2 import (
    FeatureV2WindowDataset,
    SessionValidationConfig,
    build_dataset_manifest,
    validate_dataset_root,
    write_dataset_manifest,
)
from src.features.schema import WINDOW_SIZE, expected_window_columns
from src.models_v9 import ReConHIDV9Config
from src.training_v9 import NormalPretrainConfig, run_normal_pretraining


class V9CoevolutionTest(unittest.TestCase):
    def test_raw_generator_uses_stable_extractor_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = generate_raw_attack_pool(
                output_root=tmp,
                round_name="A0",
                config=RawAttackGenerationConfig(
                    candidates=3,
                    keystrokes_per_candidate=56,
                    seed=123,
                    mutation_strength=0.0,
                ),
            )
            self.assertEqual(info["candidate_count"], 3)
            self.assertGreater(info["total_windows"], 0)
            entries = [
                json.loads(line)
                for line in Path(info["manifest_path"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(entries), 3)
            for entry in entries:
                metadata = json.loads(
                    (Path(entry["session_dir"]) / "metadata.json")
                    .read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["generation_level"], "raw_event")
                self.assertTrue(Path(entry["raw_path"]).is_file())
                self.assertTrue(Path(entry["state_path"]).is_file())
                self.assertTrue(Path(entry["window_path"]).is_file())
                dataset, reports = FeatureV2WindowDataset.from_paths(
                    [entry["window_path"]]
                )
                self.assertGreater(len(dataset), 0)
                self.assertEqual(reports[0].labels, ("attack",))

    def test_one_round_tiny_cpu_preserves_frozen_normal_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
                config=SessionValidationConfig(
                    require_complete_core_scenarios=True
                ),
            )
            self.assertFalse(report.has_errors)
            manifest = build_dataset_manifest(report, dataset_id="coevo_test")
            manifest_paths = write_dataset_manifest(manifest, root / "manifest")

            d0 = run_normal_pretraining(
                dataset_root=normal_root,
                manifest_path=manifest_paths["jsonl"],
                verify_manifest_hashes=True,
                output_root=root / "defenders",
                run_name="D0",
                config=NormalPretrainConfig(
                    seed=7,
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

            result = run_coevolution_round(
                round_index=0,
                parent_defender_run_dir=d0.run_dir,
                normal_dataset_root=normal_root,
                normal_manifest_path=manifest_paths["jsonl"],
                output_root=root / "rounds",
                config=CoevolutionRoundConfig(
                    attack_generation=RawAttackGenerationConfig(
                        candidates=3,
                        keystrokes_per_candidate=56,
                        seed=17,
                        mutation_strength=0.02,
                    ),
                    weakness_mining=WeaknessMiningConfig(
                        max_hard_windows=12,
                        max_parent_sessions=2,
                        batch_size=8,
                        device="cpu",
                    ),
                    hardened_training=HardenedTrainingConfig(
                        seed=19,
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
            self.assertTrue(Path(result["round_manifest_path"]).is_file())
            self.assertGreater(result["attack_generation"]["total_windows"], 0)
            self.assertEqual(result["methodology"]["attack_generation_level"], "raw_event")
            self.assertFalse(result["methodology"]["normal_test_used_during_round"])

            d1_dir = Path(result["next_defender_run_dir"])
            untouched = json.loads(
                (d1_dir / "untouched_test.json").read_text(encoding="utf-8")
            )
            self.assertEqual(untouched["status"], "reserved")
            calibration = json.loads(
                (d1_dir / "calibration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                calibration["decision_policy"]["selection_source"],
                "normal_calibration_only",
            )

            hard_path = Path(result["weakness_mining"]["hard_window_path"])
            with hard_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, expected_window_columns(WINDOW_SIZE))
                rows = list(reader)
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(row["label"] == "attack" for row in rows))


if __name__ == "__main__":
    unittest.main()
