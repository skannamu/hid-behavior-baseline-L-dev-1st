from __future__ import annotations

import json

from scripts.dry_run_v9_data_readiness import (
    PROTOCOL,
    _write_session,
)
from src.models_v9 import ReConHIDV9Config
from src.training_v9 import (
    NormalPretrainConfig,
    run_normal_pretraining,
)


def test_paper_mode_defers_test_scoring_until_final(
    tmp_path,
):
    normal_root = tmp_path / "normal"

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

    result = run_normal_pretraining(
        dataset_root=normal_root,
        output_root=tmp_path / "defenders",
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
            explicit_train_groups=(
                "p001", "p002", "p003", "p004",
            ),
            explicit_calibration_groups=(
                "p005",
            ),
            explicit_test_groups=(
                "p006",
            ),
            evaluate_test_during_pretraining=False,
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

    run_dir = tmp_path / "defenders" / "D0"

    split = json.loads(
        (run_dir / "split_manifest.json")
        .read_text(encoding="utf-8")
    )

    metrics = json.loads(
        (run_dir / "test_metrics.json")
        .read_text(encoding="utf-8")
    )

    assert split["assignment_mode"] == "explicit"

    assert split["train_groups"] == [
        "p001", "p002", "p003", "p004",
    ]

    assert split["calibration_groups"] == [
        "p005",
    ]

    assert split["test_groups"] == [
        "p006",
    ]

    assert (
        split["test_evaluation_timing"]
        == "deferred_until_final"
    )

    assert (
        metrics["status"]
        == "deferred_until_final"
    )

    assert metrics["reconstruction_fpr"] is None
    assert metrics["prototype_fpr"] is None

    assert result.test_reconstruction_fpr is None
    assert result.test_prototype_fpr is None
