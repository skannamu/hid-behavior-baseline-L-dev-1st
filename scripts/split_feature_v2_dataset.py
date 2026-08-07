from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import (
    BalancedSamplingConfig,
    FeatureV2WindowDataset,
    GroupSplitConfig,
    balanced_subsample,
    load_manifest_entries,
    resolve_manifest_window_paths,
    split_by_group,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview leakage-safe participant or participant/session splits."
    )
    parser.add_argument("path")
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--group-mode",
        choices=("participant", "participant_session"),
        default="participant",
    )
    parser.add_argument("--training-window-stride", type=int, default=5)
    parser.add_argument("--training-max-windows-per-group", type=int, default=3000)
    parser.add_argument("--disable-training-balance", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        entries = load_manifest_entries(
            args.manifest,
            dataset_root=args.path,
            verify_files=True,
        )
        paths = resolve_manifest_window_paths(entries, dataset_root=args.path)
        dataset, _ = FeatureV2WindowDataset.from_paths(paths)
    else:
        dataset, _ = FeatureV2WindowDataset.discover(args.path)

    split = split_by_group(
        dataset,
        config=GroupSplitConfig(
            train_ratio=args.train_ratio,
            calibration_ratio=args.calibration_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            group_mode=args.group_mode,
        ),
    )
    balance = balanced_subsample(
        split.train,
        config=BalancedSamplingConfig(
            enabled=not args.disable_training_balance,
            window_stride=args.training_window_stride,
            max_windows_per_group=args.training_max_windows_per_group,
            seed=args.seed,
        ),
    )

    payload = {
        "group_mode": split.group_mode,
        "train": {
            "raw_window_count": len(split.train),
            "sampled_window_count": len(balance.dataset),
            "groups": list(split.train_groups),
            "balance": balance.report,
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
