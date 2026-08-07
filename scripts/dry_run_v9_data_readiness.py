from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
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
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    expected_window_columns,
)
from src.models_v9 import ReConHIDV9Config
from src.training_v9 import NormalPretrainConfig, run_normal_pretraining


PROTOCOL = {
    "korean_typing": (8, "fixed", 480.0),
    "english_typing": (8, "fixed", 480.0),
    "coding_controlled": (10, "fixed", 600.0),
    "free_writing": (10, "minimum", 600.0),
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _window_row(
    participant: str,
    session: str,
    scenario: str,
    window_id: int,
    participant_offset: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "participant_id": participant,
        "session_id": session,
        "window_id": window_id,
        "start_keystroke_index": window_id + 1,
        "end_keystroke_index": window_id + WINDOW_SIZE,
        "label": "normal",
        "scenario": scenario,
        "input_source": "human",
    }
    binary = {
        "release_inversion_flag",
        "correction_key_flag",
        "repeat_flag",
        "shift_at_press",
        "ctrl_at_press",
        "alt_at_press",
        "meta_at_press",
    }
    for t in range(WINDOW_SIZE):
        for feature_index, feature in enumerate(SEQUENCE_FEATURES):
            value = (
                float((window_id + t + feature_index) % 2)
                if feature in binary
                else participant_offset + 0.001 * (window_id + t + feature_index)
            )
            row[f"t{t}_{feature}"] = value
    for feature_index, feature in enumerate(WINDOW_CONTEXT_FEATURES):
        row[f"ctx_{feature}"] = (
            participant_offset + 0.01 * (window_id + feature_index)
        )
    return row


def _write_session(
    root: Path,
    *,
    participant: str,
    scenario: str,
    participant_index: int,
    windows: int,
) -> None:
    minutes, rule, duration = PROTOCOL[scenario]
    session = f"dry_{participant}_{scenario}"
    session_dir = root / participant / session
    for child in ("raw", "state", "window"):
        (session_dir / child).mkdir(parents=True, exist_ok=True)

    common = {
        "collector_version": "dry-run-collector",
        "collection_protocol_version": "dry-run-protocol",
        "extractor_version": "feature_core_v2_3",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "participant_id": participant,
        "session_id": session,
        "scenario": scenario,
        "planned_duration_min": minutes,
        "planned_duration_rule": rule,
    }
    _write_json(session_dir / "metadata.json", {
        **common,
        "status": "finalized",
        "label": "normal",
        "input_source": "human",
    })
    _write_json(session_dir / "session_summary.json", {
        **common,
        "close_reason": "dry_run",
        "duration_seconds": duration,
        "raw_event_count": 2,
        "paired_keystroke_count": 1,
        "window_count": windows,
        "quality_event_count": 0,
        "raw_validation_ok": True,
        "state_validation_ok": True,
        "window_validation_ok": True,
    })
    (session_dir / "quality_events.jsonl").write_text("", encoding="utf-8")

    raw_fields = [
        "session_id", "raw_event_index", "timestamp_ns", "wall_time_ns",
        "device_id_session_local", "make_code", "extended_flag", "vkey",
        "message", "raw_flags", "action",
    ]
    with (session_dir / "raw" / "raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        for index, action in enumerate(("down", "up")):
            writer.writerow({
                "session_id": session,
                "raw_event_index": index,
                "timestamp_ns": 100 + index * 100,
                "wall_time_ns": 1000 + index * 100,
                "device_id_session_local": "kbd001",
                "make_code": 30,
                "extended_flag": 0,
                "vkey": 65,
                "message": 0,
                "raw_flags": index,
                "action": action,
            })

    state_fields = [
        "session_id", "keystroke_index", "source_raw_down_index",
        "source_raw_up_index", "device_id_session_local", "make_code",
        "extended_flag", "vkey", "key_category", "press_timestamp_ns",
        "release_timestamp_ns", "hold_time_s", "press_interval_s",
        "signed_flight_time_s", "release_interval_s",
        "overlap_union_duration_s", "overlap_fraction",
        "concurrent_keys_at_press", "release_inversion_flag",
        "correction_key_flag", "repeat_count", "repeat_flag",
        "shift_at_press", "ctrl_at_press", "alt_at_press", "meta_at_press",
        "modifier_count_at_press", "command_shortcut_flag", "pairing_status",
        "quality_flags", "participant_id", "scenario", "input_source", "label",
    ]
    with (session_dir / "state" / "state.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=state_fields)
        writer.writeheader()
        row = {field: 0 for field in state_fields}
        row.update({
            "session_id": session,
            "keystroke_index": 0,
            "source_raw_down_index": 0,
            "source_raw_up_index": 1,
            "device_id_session_local": "kbd001",
            "make_code": 30,
            "vkey": 65,
            "key_category": "letter",
            "press_timestamp_ns": 100,
            "release_timestamp_ns": 200,
            "hold_time_s": 0.1,
            "press_interval_s": "",
            "signed_flight_time_s": "",
            "release_interval_s": "",
            "release_inversion_flag": "",
            "pairing_status": "paired",
            "quality_flags": "[]",
            "participant_id": participant,
            "scenario": scenario,
            "input_source": "human",
            "label": "normal",
        })
        writer.writerow(row)

    with (session_dir / "window" / "window.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=expected_window_columns(WINDOW_SIZE),
        )
        writer.writeheader()
        writer.writerows(
            _window_row(
                participant,
                session,
                scenario,
                window_id,
                participant_index * 0.05,
            )
            for window_id in range(windows)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation -> manifest -> participant split/balancing -> "
            "tiny CPU D0 training on synthetic stable-v2 sessions."
        )
    )
    parser.add_argument(
        "--output-root",
        default="experiments/v9_data_readiness_dry_run",
    )
    parser.add_argument("--windows-per-session", type=int, default=6)
    parser.add_argument("--keep-synthetic-data", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = Path(tempfile.mkdtemp(prefix="v9_readiness_", dir=output_root))
    data_root = workspace / "data"

    try:
        for participant_index in range(6):
            participant = f"p{participant_index + 1:03d}"
            for scenario in PROTOCOL:
                _write_session(
                    data_root,
                    participant=participant,
                    scenario=scenario,
                    participant_index=participant_index,
                    windows=args.windows_per_session,
                )

        report = validate_dataset_root(
            data_root,
            config=SessionValidationConfig(
                require_complete_core_scenarios=True
            ),
        )
        if report.has_errors:
            raise RuntimeError(report.summary_dict())

        manifest = build_dataset_manifest(
            report,
            dataset_id=f"dry_run_{timestamp}",
        )
        manifest_dir = workspace / "manifest"
        manifest_paths = write_dataset_manifest(manifest, manifest_dir)

        result = run_normal_pretraining(
            dataset_root=data_root,
            manifest_path=manifest_paths["jsonl"],
            verify_manifest_hashes=True,
            output_root=output_root,
            run_name=f"d0_dry_run_{timestamp}",
            config=NormalPretrainConfig(
                seed=20260804,
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

        print(json.dumps({
            "status": "PASS",
            "validation": report.summary_dict(),
            "manifest": manifest.summary_dict(),
            "d0": asdict(result),
            "synthetic_workspace": str(workspace),
        }, ensure_ascii=False, indent=2))
    finally:
        if not args.keep_synthetic_data:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
