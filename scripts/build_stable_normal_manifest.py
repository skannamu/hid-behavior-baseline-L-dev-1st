from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import (
    SessionValidationConfig,
    build_dataset_manifest,
    validate_dataset_root,
    write_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate stable-v2 normal sessions and freeze the valid session "
            "set as a reproducible dataset manifest."
        )
    )
    parser.add_argument("dataset_root")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="data/normal/manifests/latest",
    )
    parser.add_argument("--exclude-warn", action="store_true")
    parser.add_argument("--require-complete-core", action="store_true")
    parser.add_argument("--expected-collector-version")
    parser.add_argument("--expected-protocol-version")
    parser.add_argument("--duration-tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--fixed-overrun-warning-seconds", type=float, default=60.0)
    args = parser.parse_args()

    report = validate_dataset_root(
        args.dataset_root,
        config=SessionValidationConfig(
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            fixed_overrun_warning_seconds=args.fixed_overrun_warning_seconds,
            expected_collector_version=args.expected_collector_version,
            expected_protocol_version=args.expected_protocol_version,
            require_complete_core_scenarios=args.require_complete_core,
        ),
    )
    manifest = build_dataset_manifest(
        report,
        dataset_id=args.dataset_id,
        include_warn=not args.exclude_warn,
    )
    paths = write_dataset_manifest(manifest, Path(args.output_dir))

    print(json.dumps({
        "summary": manifest.summary_dict(),
        "outputs": paths,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
