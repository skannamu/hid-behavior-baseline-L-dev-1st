from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict

from src.models_v9 import ReConHIDV9Config
from src.training_v9 import NormalPretrainConfig, run_normal_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe normal-only pretraining for ReCon-HID v9. "
            "Final evaluation defaults to participant-disjoint splitting."
        )
    )
    parser.add_argument("dataset_root")
    parser.add_argument("--manifest")
    parser.add_argument("--verify-manifest-hashes", action="store_true")
    parser.add_argument(
        "--output-root",
        default="experiments/v9_normal_pretrain",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--split-group-mode",
        choices=("participant", "participant_session"),
        default="participant",
    )
    parser.add_argument("--disable-training-balance", action="store_true")
    parser.add_argument("--training-window-stride", type=int, default=5)
    parser.add_argument(
        "--training-balance-mode",
        choices=("equal", "cap"),
        default="equal",
    )
    parser.add_argument("--training-target-windows-per-group", type=int)
    parser.add_argument("--training-max-windows-per-group", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--non-deterministic", action="store_true")
    args = parser.parse_args()

    result = run_normal_pretraining(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        run_name=args.run_name,
        manifest_path=args.manifest,
        verify_manifest_hashes=args.verify_manifest_hashes,
        config=NormalPretrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            target_fpr=args.target_fpr,
            split_group_mode=args.split_group_mode,
            balance_training=not args.disable_training_balance,
            training_window_stride=args.training_window_stride,
            training_balance_mode=args.training_balance_mode,
            training_target_windows_per_group=(
                args.training_target_windows_per_group
            ),
            training_max_windows_per_group=(
                args.training_max_windows_per_group
            ),
            device=args.device,
            deterministic=not args.non_deterministic,
        ),
        model_config=ReConHIDV9Config(),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
