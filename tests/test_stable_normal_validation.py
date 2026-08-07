from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data_v2.session_validator import (
    SessionValidationConfig,
    validate_dataset_root,
    validate_session_dir,
)
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    expected_window_columns,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _window_row(participant: str, session: str, scenario: str, window_id: int) -> dict:
    row = {
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
    for t in range(WINDOW_SIZE):
        for feature in SEQUENCE_FEATURES:
            row[f"t{t}_{feature}"] = 0.0
    for feature in WINDOW_CONTEXT_FEATURES:
        row[f"ctx_{feature}"] = 0.0
    return row


def write_valid_session(
    root: Path,
    *,
    participant: str = "p001",
    session: str = "s001",
    scenario: str = "korean_typing",
    duration_seconds: float = 480.0,
) -> Path:
    session_dir = root / participant / session
    (session_dir / "raw").mkdir(parents=True)
    (session_dir / "state").mkdir()
    (session_dir / "window").mkdir()

    metadata = {
        "collector_version": "collector-test",
        "collection_protocol_version": "protocol-test",
        "extractor_version": "feature_core_v2_3",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "status": "finalized",
        "participant_id": participant,
        "session_id": session,
        "scenario": scenario,
        "planned_duration_min": 8,
        "planned_duration_rule": "fixed",
        "label": "normal",
        "input_source": "human",
    }
    summary = {
        **{k: metadata[k] for k in (
            "collector_version",
            "collection_protocol_version",
            "extractor_version",
            "feature_schema_version",
            "schema_hash",
            "participant_id",
            "session_id",
            "scenario",
            "planned_duration_min",
            "planned_duration_rule",
        )},
        "close_reason": "ctrl_c",
        "duration_seconds": duration_seconds,
        "raw_event_count": 2,
        "paired_keystroke_count": 1,
        "window_count": 1,
        "quality_event_count": 0,
        "raw_validation_ok": True,
        "state_validation_ok": True,
        "window_validation_ok": True,
    }
    _write_json(session_dir / "metadata.json", metadata)
    _write_json(session_dir / "session_summary.json", summary)
    (session_dir / "quality_events.jsonl").write_text("", encoding="utf-8")

    with (session_dir / "raw" / "raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "session_id",
            "raw_event_index",
            "timestamp_ns",
            "wall_time_ns",
            "device_id_session_local",
            "make_code",
            "extended_flag",
            "vkey",
            "message",
            "raw_flags",
            "action",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "session_id": session,
            "raw_event_index": 0,
            "timestamp_ns": 100,
            "wall_time_ns": 1000,
            "device_id_session_local": "kbd001",
            "make_code": 30,
            "extended_flag": 0,
            "vkey": 65,
            "message": 0,
            "raw_flags": 0,
            "action": "down",
        })
        writer.writerow({
            "session_id": session,
            "raw_event_index": 1,
            "timestamp_ns": 200,
            "wall_time_ns": 1100,
            "device_id_session_local": "kbd001",
            "make_code": 30,
            "extended_flag": 0,
            "vkey": 65,
            "message": 0,
            "raw_flags": 1,
            "action": "up",
        })

    state_fields = [
        "session_id",
        "keystroke_index",
        "source_raw_down_index",
        "source_raw_up_index",
        "device_id_session_local",
        "make_code",
        "extended_flag",
        "vkey",
        "key_category",
        "press_timestamp_ns",
        "release_timestamp_ns",
        "hold_time_s",
        "press_interval_s",
        "signed_flight_time_s",
        "release_interval_s",
        "overlap_union_duration_s",
        "overlap_fraction",
        "concurrent_keys_at_press",
        "release_inversion_flag",
        "correction_key_flag",
        "repeat_count",
        "repeat_flag",
        "shift_at_press",
        "ctrl_at_press",
        "alt_at_press",
        "meta_at_press",
        "modifier_count_at_press",
        "command_shortcut_flag",
        "pairing_status",
        "quality_flags",
        "participant_id",
        "scenario",
        "input_source",
        "label",
    ]
    with (session_dir / "state" / "state.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=state_fields)
        writer.writeheader()
        writer.writerow({
            "session_id": session,
            "keystroke_index": 0,
            "source_raw_down_index": 0,
            "source_raw_up_index": 1,
            "device_id_session_local": "kbd001",
            "make_code": 30,
            "extended_flag": 0,
            "vkey": 65,
            "key_category": "letter",
            "press_timestamp_ns": 100,
            "release_timestamp_ns": 200,
            "hold_time_s": 0.1,
            "press_interval_s": "",
            "signed_flight_time_s": "",
            "release_interval_s": "",
            "overlap_union_duration_s": 0.0,
            "overlap_fraction": 0.0,
            "concurrent_keys_at_press": 0,
            "release_inversion_flag": "",
            "correction_key_flag": 0,
            "repeat_count": 0,
            "repeat_flag": 0,
            "shift_at_press": 0,
            "ctrl_at_press": 0,
            "alt_at_press": 0,
            "meta_at_press": 0,
            "modifier_count_at_press": 0,
            "command_shortcut_flag": 0,
            "pairing_status": "paired",
            "quality_flags": "[]",
            "participant_id": participant,
            "scenario": scenario,
            "input_source": "human",
            "label": "normal",
        })

    with (session_dir / "window" / "window.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=expected_window_columns(WINDOW_SIZE),
        )
        writer.writeheader()
        writer.writerow(_window_row(participant, session, scenario, 0))
    return session_dir


class StableNormalValidationTest(unittest.TestCase):
    def test_valid_session_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = write_valid_session(Path(tmp))
            result = validate_session_dir(session_dir)
            self.assertEqual(result.status, "PASS", result.to_dict())
            self.assertEqual(result.window_count, 1)

    def test_short_fixed_session_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = write_valid_session(
                Path(tmp), duration_seconds=400.0
            )
            result = validate_session_dir(session_dir)
            self.assertEqual(result.status, "FAIL")
            self.assertIn(
                "duration_too_short",
                {issue.code for issue in result.issues},
            )

    def test_duplicate_session_id_fails_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_valid_session(root, participant="p001", session="same")
            second = write_valid_session(root, participant="p002", session="other")
            metadata = json.loads((second / "metadata.json").read_text())
            summary = json.loads((second / "session_summary.json").read_text())
            metadata["session_id"] = "same"
            summary["session_id"] = "same"
            _write_json(second / "metadata.json", metadata)
            _write_json(second / "session_summary.json", summary)
            # Keep folder name different to ensure both sessions are discovered.
            report = validate_dataset_root(root)
            self.assertTrue(report.has_errors)
            self.assertTrue(any(
                issue.code == "duplicate_session_id"
                for issue in report.dataset_issues
            ))
            self.assertTrue(first.exists())

    def test_incomplete_core_is_warning_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_session(root)
            report = validate_dataset_root(root)
            self.assertFalse(report.has_errors)
            self.assertTrue(report.missing_core_scenarios["p001"])

    def test_incomplete_core_can_be_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_valid_session(root)
            report = validate_dataset_root(
                root,
                config=SessionValidationConfig(
                    require_complete_core_scenarios=True
                ),
            )
            self.assertTrue(report.has_errors)


if __name__ == "__main__":
    unittest.main()
