from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from src.features.keystroke_builder import KeystrokeBuilder
from src.features.models import RawKeyEvent
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    WINDOW_CONTEXT_FEATURES,
    assert_schema_file,
    expected_window_columns,
)
from src.features.validators import validate_states, validate_windows
from src.features.window_builder import WindowBuilder


ROOT = Path(__file__).resolve().parents[1]
VECTOR_PATH = ROOT / "specs" / "feature_schema_v2" / "feature_schema_v2_test_vectors.json"


def event(row: dict) -> RawKeyEvent:
    return RawKeyEvent(
        session_id="s-test",
        raw_event_index=row.setdefault("_index", event.counter),
        timestamp_ns=row["t"],
        device_id_session_local=str(row["device"]),
        make_code=row["make"],
        extended_flag=row["ext"],
        vkey=row["vkey"],
        message=0,
        action=row["action"],
    )


event.counter = 0


class FeatureSchemaV2Test(unittest.TestCase):
    def setUp(self) -> None:
        event.counter = 0
        self.builder = KeystrokeBuilder()

    def build(self, rows):
        converted = []
        for index, row in enumerate(rows):
            row = dict(row)
            row["_index"] = index
            converted.append(event(row))
        return self.builder.build(
            converted,
            participant_id="p-test",
            scenario="test",
            input_source="human",
            label="normal",
        )

    def assert_close(self, actual, expected, tolerance=1e-9):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
            f"actual={actual}, expected={expected}",
        )

    def test_simple_sequential(self):
        result = self.build([
            {"t": 0, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 100_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "up"},
            {"t": 150_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "down"},
            {"t": 250_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "up"},
        ])
        self.assertEqual(len(result.states), 2)
        self.assert_close(result.states[1].press_interval_s, 0.15)
        self.assert_close(result.states[1].signed_flight_time_s, 0.05)
        self.assertEqual(result.states[1].release_inversion_flag, 0)

    def test_overlap_and_release_inversion(self):
        result = self.build([
            {"t": 0, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 50_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "down"},
            {"t": 100_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "up"},
            {"t": 150_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "up"},
        ])
        a, b = result.states
        self.assert_close(a.overlap_fraction, 1.0 / 3.0)
        self.assert_close(b.overlap_fraction, 1.0)
        self.assert_close(b.signed_flight_time_s, -0.10)
        self.assertEqual(b.release_inversion_flag, 1)
        self.assertEqual(b.concurrent_keys_at_press, 1)

    def test_modifier_snapshot_survives_release_order(self):
        result = self.build([
            {"t": 0, "device": 0, "make": 29, "ext": 0, "vkey": 17, "action": "down"},
            {"t": 20_000_000, "device": 0, "make": 46, "ext": 0, "vkey": 67, "action": "down"},
            {"t": 40_000_000, "device": 0, "make": 29, "ext": 0, "vkey": 17, "action": "up"},
            {"t": 60_000_000, "device": 0, "make": 46, "ext": 0, "vkey": 67, "action": "up"},
        ])
        ctrl_state = next(s for s in result.states if s.vkey == 17)
        c_state = next(s for s in result.states if s.vkey == 67)
        self.assertEqual(ctrl_state.ctrl_at_press, 0)
        self.assertEqual(ctrl_state.modifier_count_at_press, 0)
        self.assertEqual(ctrl_state.command_shortcut_flag, 0)
        self.assertEqual(c_state.ctrl_at_press, 1)
        self.assertEqual(c_state.command_shortcut_flag, 1)
        self.assertEqual(c_state.concurrent_keys_at_press, 1)

    def test_modifier_self_excluded_and_prior_modifier_visible(self):
        result = self.build([
            {"t": 0, "device": 0, "make": 42, "ext": 0, "vkey": 16, "action": "down"},
            {"t": 10_000_000, "device": 0, "make": 29, "ext": 0, "vkey": 17, "action": "down"},
            {"t": 20_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 40_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "up"},
            {"t": 50_000_000, "device": 0, "make": 29, "ext": 0, "vkey": 17, "action": "up"},
            {"t": 60_000_000, "device": 0, "make": 42, "ext": 0, "vkey": 16, "action": "up"},
        ])
        shift_state = next(s for s in result.states if s.vkey == 16)
        ctrl_state = next(s for s in result.states if s.vkey == 17)
        a_state = next(s for s in result.states if s.vkey == 65)

        self.assertEqual(shift_state.shift_at_press, 0)
        self.assertEqual(shift_state.ctrl_at_press, 0)
        self.assertEqual(ctrl_state.shift_at_press, 1)
        self.assertEqual(ctrl_state.ctrl_at_press, 0)
        self.assertEqual(a_state.shift_at_press, 1)
        self.assertEqual(a_state.ctrl_at_press, 1)
        self.assertEqual(a_state.command_shortcut_flag, 1)

    def test_schema_yaml_hash_and_version(self):
        schema_path = ROOT / "specs" / "feature_schema_v2" / "feature_schema_v2.yaml"
        self.assertEqual(FEATURE_SCHEMA_VERSION, "2.0.0")
        self.assertEqual(assert_schema_file(schema_path), EXPECTED_SCHEMA_SHA256)

    def test_os_repeat_is_one_keystroke(self):
        result = self.build([
            {"t": 0, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 400_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 450_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 500_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "up"},
        ])
        self.assertEqual(len(result.states), 1)
        self.assertEqual(result.states[0].repeat_count, 2)
        self.assertEqual(result.states[0].repeat_flag, 1)

    def test_overlap_union_is_bounded(self):
        result = self.build([
            {"t": 0, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 20_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "down"},
            {"t": 40_000_000, "device": 0, "make": 46, "ext": 0, "vkey": 67, "action": "down"},
            {"t": 60_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "up"},
            {"t": 80_000_000, "device": 0, "make": 46, "ext": 0, "vkey": 67, "action": "up"},
            {"t": 100_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "up"},
        ])
        a_state = next(s for s in result.states if s.vkey == 65)
        self.assert_close(a_state.overlap_union_duration_s, 0.06)
        self.assert_close(a_state.overlap_fraction, 0.6)
        self.assertTrue(all(0.0 <= s.overlap_fraction <= 1.0 for s in result.states))

    def test_command_shortcut_rate_uses_non_modifier_denominator(self):
        rows = []
        raw_index = 0
        current_ns = 0

        def add(vkey, make, hold_ns=30_000_000):
            nonlocal raw_index, current_ns
            rows.append({
                "session_id": "s-shortcut-rate",
                "raw_event_index": raw_index,
                "timestamp_ns": current_ns,
                "device_id_session_local": "0",
                "make_code": make,
                "extended_flag": 0,
                "vkey": vkey,
                "message": 0,
                "action": "down",
            })
            raw_index += 1
            rows.append({
                "session_id": "s-shortcut-rate",
                "raw_event_index": raw_index,
                "timestamp_ns": current_ns + hold_ns,
                "device_id_session_local": "0",
                "make_code": make,
                "extended_flag": 0,
                "vkey": vkey,
                "message": 0,
                "action": "up",
            })
            raw_index += 1
            current_ns += 100_000_000

        # First throwaway predecessor.
        add(65, 30)

        # One Ctrl+C shortcut: Ctrl down, C down/up, Ctrl up.
        rows.append({
            "session_id": "s-shortcut-rate",
            "raw_event_index": raw_index,
            "timestamp_ns": current_ns,
            "device_id_session_local": "0",
            "make_code": 29,
            "extended_flag": 0,
            "vkey": 17,
            "message": 0,
            "action": "down",
        })
        raw_index += 1
        rows.append({
            "session_id": "s-shortcut-rate",
            "raw_event_index": raw_index,
            "timestamp_ns": current_ns + 10_000_000,
            "device_id_session_local": "0",
            "make_code": 46,
            "extended_flag": 0,
            "vkey": 67,
            "message": 0,
            "action": "down",
        })
        raw_index += 1
        rows.append({
            "session_id": "s-shortcut-rate",
            "raw_event_index": raw_index,
            "timestamp_ns": current_ns + 30_000_000,
            "device_id_session_local": "0",
            "make_code": 46,
            "extended_flag": 0,
            "vkey": 67,
            "message": 0,
            "action": "up",
        })
        raw_index += 1
        rows.append({
            "session_id": "s-shortcut-rate",
            "raw_event_index": raw_index,
            "timestamp_ns": current_ns + 40_000_000,
            "device_id_session_local": "0",
            "make_code": 29,
            "extended_flag": 0,
            "vkey": 17,
            "message": 0,
            "action": "up",
        })
        raw_index += 1
        current_ns += 100_000_000

        # Add 48 ordinary non-modifier keys. Eligible window will have:
        # 1 modifier row + 49 non-modifier rows, exactly 1 shortcut.
        for i in range(48):
            add(65 + (i % 3), 30 + (i % 3))

        result = self.builder.build(
            [RawKeyEvent.from_mapping(r) for r in rows],
            participant_id="p-shortcut",
            scenario="typing",
            input_source="human",
            label="normal",
        )
        windows = WindowBuilder().build(result.states)
        self.assertEqual(len(windows), 1)

        shortcut_index = list(
            __import__("src.features.schema", fromlist=["WINDOW_CONTEXT_FEATURES"])
            .WINDOW_CONTEXT_FEATURES
        ).index("command_shortcut_rate")
        self.assert_close(windows[0].context[shortcut_index], 1.0 / 49.0)


    def test_window_build_and_validation(self):
        rows = []
        raw_index = 0
        for i in range(52):
            press = i * 100_000_000
            rows.append({
                "session_id": "s-window",
                "raw_event_index": raw_index,
                "timestamp_ns": press,
                "device_id_session_local": "0",
                "make_code": 30 + (i % 3),
                "extended_flag": 0,
                "vkey": 65 + (i % 3),
                "message": 0,
                "action": "down",
            })
            raw_index += 1
            rows.append({
                "session_id": "s-window",
                "raw_event_index": raw_index,
                "timestamp_ns": press + 50_000_000,
                "device_id_session_local": "0",
                "make_code": 30 + (i % 3),
                "extended_flag": 0,
                "vkey": 65 + (i % 3),
                "message": 0,
                "action": "up",
            })
            raw_index += 1

        result = self.builder.build(
            [RawKeyEvent.from_mapping(r) for r in rows],
            participant_id="p-window",
            scenario="typing",
            input_source="human",
            label="normal",
        )
        self.assertTrue(validate_states(result.states).ok)

        windows = WindowBuilder().build(result.states)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].schema_hash, EXPECTED_SCHEMA_SHA256)
        self.assertTrue(validate_windows(windows).ok)
        self.assert_close(windows[0].context[0], 10.0)

    def test_trailing_incomplete_boundary_removes_shutdown_chord(self):
        result = self.build([
            {"t": 0, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "down"},
            {"t": 50_000_000, "device": 0, "make": 30, "ext": 0, "vkey": 65, "action": "up"},
            {"t": 100_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "down"},
            {"t": 150_000_000, "device": 0, "make": 48, "ext": 0, "vkey": 66, "action": "up"},
            {"t": 200_000_000, "device": 0, "make": 29, "ext": 0, "vkey": 17, "action": "down"},
            {"t": 210_000_000, "device": 0, "make": 46, "ext": 0, "vkey": 67, "action": "down"},
            {"t": 230_000_000, "device": 0, "make": 46, "ext": 0, "vkey": 67, "action": "up"},
        ])
        self.assertEqual([state.vkey for state in result.states], [65, 66])
        self.assertEqual(result.incomplete_key_count, 1)
        self.assertEqual(result.trailing_completed_keystrokes_trimmed, 1)
        self.assertEqual(result.boundary_cutoff_raw_event_index, 4)
        self.assertEqual(result.boundary_cutoff_timestamp_ns, 200_000_000)
        codes = [event.code for event in result.quality_events]
        self.assertIn("incomplete_key_at_end", codes)
        self.assertIn("trailing_boundary_trim", codes)

    def test_context_replaces_exact_mean_burst_duplicate(self):
        self.assertIn("hold_time_robust_cv", WINDOW_CONTEXT_FEATURES)
        self.assertNotIn("mean_burst_length_keys", WINDOW_CONTEXT_FEATURES)
        self.assertEqual(len(WINDOW_CONTEXT_FEATURES), 12)
        self.assertEqual(len(expected_window_columns()), 622)

        rows = []
        raw_index = 0
        for i in range(51):
            press = i * 100_000_000
            hold = 30_000_000 if i % 2 == 0 else 70_000_000
            make = 30 + (i % 3)
            vkey = 65 + (i % 3)
            rows.append({
                "session_id": "s-hold-cv",
                "raw_event_index": raw_index,
                "timestamp_ns": press,
                "device_id_session_local": "0",
                "make_code": make,
                "extended_flag": 0,
                "vkey": vkey,
                "message": 0,
                "action": "down",
            })
            raw_index += 1
            rows.append({
                "session_id": "s-hold-cv",
                "raw_event_index": raw_index,
                "timestamp_ns": press + hold,
                "device_id_session_local": "0",
                "make_code": make,
                "extended_flag": 0,
                "vkey": vkey,
                "message": 0,
                "action": "up",
            })
            raw_index += 1

        result = self.builder.build(
            [RawKeyEvent.from_mapping(row) for row in rows],
            participant_id="p-hold-cv",
            scenario="typing",
            input_source="human",
            label="normal",
        )
        windows = WindowBuilder().build(result.states)
        self.assertEqual(len(windows), 1)
        context = dict(zip(WINDOW_CONTEXT_FEATURES, windows[0].context, strict=True))
        self.assert_close(context["pause_rate"], 0.0)
        self.assertGreater(context["hold_time_robust_cv"], 0.0)



if __name__ == "__main__":
    unittest.main()
