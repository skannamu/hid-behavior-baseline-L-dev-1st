from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.models_v9 import ReConHIDV9Config
from src.training_v9 import NormalPretrainConfig, run_normal_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe normal-only pretraining for ReCon-HID v9. "
            "Requires at least 3 independent participant/session groups."
        )
    )
    parser.add_argument("dataset_root")
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
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
    )
    args = parser.parse_args()

    result = run_normal_pretraining(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        run_name=args.run_name,
        config=NormalPretrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            target_fpr=args.target_fpr,
            device=args.device,
            deterministic=not args.non_deterministic,
        ),
        model_config=ReConHIDV9Config(),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
