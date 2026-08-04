"""Validation utilities for raw events, states, and model windows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Any

from .models import KeystrokeState, RawKeyEvent, WindowRecord
from .schema import (
    BINARY_SEQUENCE_FEATURES,
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    UNIT_INTERVAL_CONTEXT_FEATURES,
    UNIT_INTERVAL_SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    index: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "; ".join(f"{i.code}: {i.message}" for i in self.errors)
            raise ValueError(details)


def validate_raw_events(
    events: Iterable[RawKeyEvent | Mapping[str, Any]],
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    seen: set[int] = set()

    for idx, raw in enumerate(events):
        try:
            event = raw if isinstance(raw, RawKeyEvent) else RawKeyEvent.from_mapping(raw)
        except Exception as exc:
            issues.append(
                ValidationIssue("error", "invalid_raw_event", str(exc), idx)
            )
            continue

        if event.raw_event_index in seen:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_raw_event_index",
                    str(event.raw_event_index),
                    idx,
                )
            )
        seen.add(event.raw_event_index)

        if event.timestamp_ns < 0:
            issues.append(
                ValidationIssue(
                    "error", "negative_timestamp", str(event.timestamp_ns), idx
                )
            )
        if event.extended_flag not in {0, 1}:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_extended_flag",
                    str(event.extended_flag),
                    idx,
                )
            )

    return ValidationReport(tuple(issues))


def validate_states(states: Iterable[KeystrokeState]) -> ValidationReport:
    rows = list(states)
    issues: list[ValidationIssue] = []

    for idx, state in enumerate(rows):
        if state.hold_time_s < 0:
            issues.append(
                ValidationIssue("error", "negative_hold_time", "", idx)
            )
        if not 0.0 <= state.overlap_fraction <= 1.0:
            issues.append(
                ValidationIssue(
                    "error",
                    "overlap_fraction_out_of_range",
                    str(state.overlap_fraction),
                    idx,
                )
            )
        if state.concurrent_keys_at_press < 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "negative_concurrent_key_count",
                    str(state.concurrent_keys_at_press),
                    idx,
                )
            )
        if idx > 0:
            previous = rows[idx - 1]
            if state.press_timestamp_ns < previous.press_timestamp_ns:
                issues.append(
                    ValidationIssue(
                        "error",
                        "state_order_not_press_order",
                        "",
                        idx,
                    )
                )
            if state.press_interval_s is None or state.press_interval_s < 0:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_press_interval",
                        str(state.press_interval_s),
                        idx,
                    )
                )
            if state.signed_flight_time_s is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_signed_flight_time",
                        "",
                        idx,
                    )
                )
            if state.release_inversion_flag not in {0, 1}:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_release_inversion_flag",
                        str(state.release_inversion_flag),
                        idx,
                    )
                )

    return ValidationReport(tuple(issues))


def validate_windows(records: Iterable[WindowRecord]) -> ValidationReport:
    issues: list[ValidationIssue] = []

    for idx, record in enumerate(records):
        if record.feature_schema_version != FEATURE_SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    "error",
                    "schema_version_mismatch",
                    record.feature_schema_version,
                    idx,
                )
            )
        if record.schema_hash != EXPECTED_SCHEMA_SHA256:
            issues.append(
                ValidationIssue(
                    "error",
                    "schema_hash_mismatch",
                    record.schema_hash,
                    idx,
                )
            )
        if len(record.sequence) != WINDOW_SIZE:
            issues.append(
                ValidationIssue(
                    "error",
                    "wrong_sequence_length",
                    str(len(record.sequence)),
                    idx,
                )
            )
            continue
        if len(record.context) != len(WINDOW_CONTEXT_FEATURES):
            issues.append(
                ValidationIssue(
                    "error",
                    "wrong_context_length",
                    str(len(record.context)),
                    idx,
                )
            )

        for t, vector in enumerate(record.sequence):
            if len(vector) != len(SEQUENCE_FEATURES):
                issues.append(
                    ValidationIssue(
                        "error",
                        "wrong_sequence_feature_count",
                        f"t={t}, count={len(vector)}",
                        idx,
                    )
                )
                continue

            for name, value in zip(SEQUENCE_FEATURES, vector, strict=True):
                if not math.isfinite(value):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "non_finite_sequence_value",
                            f"{name}={value}",
                            idx,
                        )
                    )
                if name in BINARY_SEQUENCE_FEATURES and value not in {0.0, 1.0}:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "non_binary_sequence_value",
                            f"{name}={value}",
                            idx,
                        )
                    )
                if name in UNIT_INTERVAL_SEQUENCE_FEATURES and not 0.0 <= value <= 1.0:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "sequence_value_out_of_unit_interval",
                            f"{name}={value}",
                            idx,
                        )
                    )

        for name, value in zip(WINDOW_CONTEXT_FEATURES, record.context, strict=True):
            if not math.isfinite(value):
                issues.append(
                    ValidationIssue(
                        "error",
                        "non_finite_context_value",
                        f"{name}={value}",
                        idx,
                    )
                )
            if name in UNIT_INTERVAL_CONTEXT_FEATURES and not 0.0 <= value <= 1.0:
                issues.append(
                    ValidationIssue(
                        "error",
                        "context_value_out_of_unit_interval",
                        f"{name}={value}",
                        idx,
                    )
                )

    return ValidationReport(tuple(issues))
