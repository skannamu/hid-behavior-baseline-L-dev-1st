from __future__ import annotations

import argparse
import json

from src.data_v2 import (
    FeatureV2WindowDataset,
    GroupSplitConfig,
    split_by_group,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview leakage-safe participant/session group splits."
    )
    parser.add_argument("path")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    dataset, _ = FeatureV2WindowDataset.discover(args.path)
    split = split_by_group(
        dataset,
        config=GroupSplitConfig(
            train_ratio=args.train_ratio,
            calibration_ratio=args.calibration_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        ),
    )

    payload = {
        "train": {
            "window_count": len(split.train),
            "groups": list(split.train_groups),
        },
        "calibration": {
            "window_count": len(split.calibration),
            "groups": list(split.calibration_groups),
        },
        "test": {
            "window_count": len(split.test),
            "groups": list(split.test_groups),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
