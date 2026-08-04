from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.keystroke_builder import KeystrokeBuilder
from src.features.models import RawKeyEvent
from src.features.validators import validate_states, validate_windows
from src.features.window_builder import WindowBuilder


def main() -> None:
    events = []
    raw_index = 0

    for i in range(60):
        press_ns = i * 120_000_000
        make = 30 + (i % 4)
        vkey = 65 + (i % 4)

        events.append(
            RawKeyEvent(
                session_id="smoke-session",
                raw_event_index=raw_index,
                timestamp_ns=press_ns,
                device_id_session_local="kbd0",
                make_code=make,
                extended_flag=0,
                vkey=vkey,
                message=0,
                action="down",
            )
        )
        raw_index += 1

        events.append(
            RawKeyEvent(
                session_id="smoke-session",
                raw_event_index=raw_index,
                timestamp_ns=press_ns + 70_000_000,
                device_id_session_local="kbd0",
                make_code=make,
                extended_flag=0,
                vkey=vkey,
                message=0,
                action="up",
            )
        )
        raw_index += 1

    result = KeystrokeBuilder().build(
        events,
        participant_id="p-smoke",
        scenario="smoke",
        input_source="human",
        label="normal",
    )
    state_report = validate_states(result.states)
    state_report.raise_for_errors()

    windows = WindowBuilder().build(result.states)
    window_report = validate_windows(windows)
    window_report.raise_for_errors()

    print(f"states={len(result.states)}")
    print(f"quality_events={len(result.quality_events)}")
    print(f"windows={len(windows)}")
    print(f"sequence_shape={len(windows[0].sequence)}x{len(windows[0].sequence[0])}")
    print(f"context_dim={len(windows[0].context)}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
