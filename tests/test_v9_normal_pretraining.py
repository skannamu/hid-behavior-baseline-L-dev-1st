from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    expected_window_columns,
)
from src.models_v9 import ReConHIDV9Config
from src.training_v9 import (
    NormalPretrainConfig,
    empirical_fpr,
    quantile_threshold,
    run_normal_pretraining,
)


def make_row(
    *,
    participant: str,
    session: str,
    window_id: int,
    offset: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "participant_id": participant,
        "session_id": session,
        "window_id": window_id,
        "start_keystroke_index": window_id,
        "end_keystroke_index": window_id + WINDOW_SIZE - 1,
        "label": "normal",
        "scenario": "typing",
        "input_source": "human",
    }
    for t in range(WINDOW_SIZE):
        for feature_index, feature in enumerate(SEQUENCE_FEATURES):
            if feature_index >= 5:
                value = float((window_id + t + feature_index) % 2)
            else:
                value = offset + 0.001 * (window_id + t + feature_index)
            row[f"t{t}_{feature}"] = value

    for feature_index, feature in enumerate(WINDOW_CONTEXT_FEATURES):
        row[f"ctx_{feature}"] = (
            offset + 0.01 * (window_id + feature_index)
        )
    return row


def write_group(root: Path, index: int, windows: int = 4) -> None:
    participant = f"p{index:03d}"
    session = f"s{index:03d}"
    path = root / participant / session / "window" / "window.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=expected_window_columns(WINDOW_SIZE),
        )
        writer.writeheader()
        writer.writerows(
            make_row(
                participant=participant,
                session=session,
                window_id=window_id,
                offset=index * 0.1,
            )
            for window_id in range(windows)
        )


class NormalPretrainingTest(unittest.TestCase):
    def test_threshold_and_empirical_fpr(self):
        scores = np.arange(100, dtype=np.float64)
        threshold = quantile_threshold(scores, target_fpr=0.05)
        self.assertEqual(threshold, 95.0)
        self.assertEqual(empirical_fpr(scores, threshold), 0.04)

    def test_end_to_end_tiny_cpu_run(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_root = base / "data"
            output_root = base / "out"

            for index in range(6):
                write_group(data_root, index, windows=3)

            result = run_normal_pretraining(
                dataset_root=data_root,
                output_root=output_root,
                run_name="tiny",
                config=NormalPretrainConfig(
                    seed=123,
                    epochs=2,
                    batch_size=4,
                    learning_rate=1.0e-3,
                    weight_decay=0.0,
                    patience=2,
                    target_fpr=0.20,
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

            run_dir = Path(result.run_dir)
            self.assertTrue((run_dir / "best_model.pt").exists())
            self.assertTrue((run_dir / "normalizer.json").exists())
            self.assertTrue((run_dir / "calibration.json").exists())
            self.assertTrue((run_dir / "test_metrics.json").exists())

            split = json.loads(
                (run_dir / "split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            train = set(split["train_groups"])
            calibration = set(split["calibration_groups"])
            test = set(split["test_groups"])
            self.assertFalse(train & calibration)
            self.assertFalse(train & test)
            self.assertFalse(calibration & test)

            metrics = json.loads(
                (run_dir / "test_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metrics["threshold_source"],
                "calibration_only",
            )
            self.assertIsNone(metrics["attack_detection_metrics"])


if __name__ == "__main__":
    unittest.main()
