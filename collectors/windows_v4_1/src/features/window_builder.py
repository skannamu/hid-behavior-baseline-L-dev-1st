"""Build deterministic 50-keystroke windows and context features."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from .models import KeystrokeState, WindowRecord
from .schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    PAUSE_THRESHOLD_SECONDS,
    ROBUST_CV_EPSILON,
    SEQUENCE_FEATURES,
    STRIDE,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    expected_window_columns,
)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mad(values: Sequence[float], center: float) -> float:
    return _median([abs(v - center) for v in values]) if values else 0.0


class WindowBuilder:
    def __init__(
        self,
        *,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        pause_threshold_seconds: float = PAUSE_THRESHOLD_SECONDS,
        schema_hash: str = EXPECTED_SCHEMA_SHA256,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if stride < 1:
            raise ValueError("stride must be positive")
        if pause_threshold_seconds <= 0:
            raise ValueError("pause_threshold_seconds must be positive")

        self.window_size = window_size
        self.stride = stride
        self.pause_threshold_seconds = pause_threshold_seconds
        self.schema_hash = schema_hash

    def build(self, states: Iterable[KeystrokeState]) -> tuple[WindowRecord, ...]:
        grouped: dict[str, list[KeystrokeState]] = {}
        for state in states:
            grouped.setdefault(state.session_id, []).append(state)

        records: list[WindowRecord] = []
        next_window_id = 0

        for session_id in sorted(grouped):
            session_states = sorted(
                grouped[session_id],
                key=lambda s: (s.press_timestamp_ns, s.source_raw_down_index),
            )

            # Transition-bearing model rows require a valid press-ordered predecessor.
            eligible = [
                state
                for state in session_states
                if state.press_interval_s is not None
                and state.signed_flight_time_s is not None
                and state.release_inversion_flag is not None
            ]

            for start in range(
                0,
                max(0, len(eligible) - self.window_size + 1),
                self.stride,
            ):
                chunk = eligible[start : start + self.window_size]
                if len(chunk) != self.window_size:
                    continue

                sequence = tuple(
                    tuple(float(getattr(state, name)) for name in SEQUENCE_FEATURES)
                    for state in chunk
                )
                context_map = self._context(chunk)
                context = tuple(
                    float(context_map[name]) for name in WINDOW_CONTEXT_FEATURES
                )

                first = chunk[0]
                last = chunk[-1]
                records.append(
                    WindowRecord(
                        feature_schema_version=FEATURE_SCHEMA_VERSION,
                        schema_hash=self.schema_hash,
                        participant_id=first.participant_id,
                        session_id=session_id,
                        window_id=next_window_id,
                        start_keystroke_index=first.keystroke_index,
                        end_keystroke_index=last.keystroke_index,
                        label=first.label,
                        scenario=first.scenario,
                        input_source=first.input_source,
                        sequence=sequence,
                        context=context,
                    )
                )
                next_window_id += 1

        return tuple(records)

    def _context(self, chunk: Sequence[KeystrokeState]) -> dict[str, float]:
        n = len(chunk)
        internal_intervals = [
            (chunk[i].press_timestamp_ns - chunk[i - 1].press_timestamp_ns) / 1e9
            for i in range(1, n)
        ]

        span_s = (
            chunk[-1].press_timestamp_ns - chunk[0].press_timestamp_ns
        ) / 1e9
        press_rate = (n - 1) / span_s if span_s > 0 else 0.0

        active = [
            value
            for value in internal_intervals
            if value < self.pause_threshold_seconds
        ]
        active_median = _median(active)
        active_mad = _mad(active, active_median)
        robust_cv = (
            1.4826 * active_mad / max(active_median, ROBUST_CV_EPSILON)
            if active
            else 0.0
        )

        pauses = [
            value
            for value in internal_intervals
            if value >= self.pause_threshold_seconds
        ]
        transition_count = max(n - 1, 1)
        pause_rate = len(pauses) / transition_count
        total_interval_time = sum(internal_intervals)
        pause_time_fraction = (
            sum(pauses) / total_interval_time if total_interval_time > 0 else 0.0
        )
        max_pause = max(pauses, default=0.0)

        burst_lengths = [1]
        for interval in internal_intervals:
            if interval >= self.pause_threshold_seconds:
                burst_lengths.append(1)
            else:
                burst_lengths[-1] += 1

        mean_burst = sum(burst_lengths) / len(burst_lengths)
        max_burst = max(burst_lengths)

        correction_rate = sum(s.correction_key_flag for s in chunk) / n

        non_modifier_states = [
            state for state in chunk if state.key_category != "MODIFIER"
        ]
        shortcut_rate = (
            sum(state.command_shortcut_flag for state in non_modifier_states)
            / len(non_modifier_states)
            if non_modifier_states
            else 0.0
        )

        overlap_rate = sum(s.overlap_fraction > 0 for s in chunk) / n

        # Only transitions fully internal to this window are counted.
        inversion_count = sum(
            chunk[i].release_timestamp_ns < chunk[i - 1].release_timestamp_ns
            for i in range(1, n)
        )
        inversion_rate = inversion_count / transition_count

        values = {
            "press_rate_hz": press_rate,
            "active_press_interval_median_s": active_median,
            "active_press_interval_robust_cv": robust_cv,
            "pause_rate": pause_rate,
            "pause_time_fraction": pause_time_fraction,
            "max_pause_interval_s": max_pause,
            "mean_burst_length_keys": mean_burst,
            "max_burst_length_keys": float(max_burst),
            "correction_rate": correction_rate,
            "command_shortcut_rate": shortcut_rate,
            "overlap_key_rate": overlap_rate,
            "release_inversion_rate": inversion_rate,
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"Non-finite context feature {name}={value}")

        return values


def write_windows_csv(
    records: Iterable[WindowRecord],
    path: str | Path,
    *,
    window_size: int = WINDOW_SIZE,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = expected_window_columns(window_size)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                record.to_flat_dict(
                    sequence_feature_names=SEQUENCE_FEATURES,
                    context_feature_names=WINDOW_CONTEXT_FEATURES,
                )
            )

    return output
