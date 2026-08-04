"""Typed data models used by the Feature Schema v2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class RawKeyEvent:
    session_id: str
    raw_event_index: int
    timestamp_ns: int
    device_id_session_local: str
    make_code: int
    extended_flag: int
    vkey: int
    message: int
    action: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "RawKeyEvent":
        action = str(row["action"]).strip().lower()
        if action not in {"down", "up"}:
            raise ValueError(f"Unsupported action: {action!r}")

        return cls(
            session_id=str(row["session_id"]),
            raw_event_index=int(row["raw_event_index"]),
            timestamp_ns=int(row["timestamp_ns"]),
            device_id_session_local=str(row["device_id_session_local"]),
            make_code=int(row["make_code"]),
            extended_flag=int(row.get("extended_flag", 0)),
            vkey=int(row["vkey"]),
            message=int(row.get("message", 0)),
            action=action,
        )

    @property
    def physical_key_id(self) -> tuple[str, int, int]:
        return (
            self.device_id_session_local,
            self.make_code,
            self.extended_flag,
        )


@dataclass(frozen=True)
class QualityEvent:
    code: str
    severity: str
    message: str
    raw_event_index: Optional[int] = None
    physical_key_id: Optional[tuple[str, int, int]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KeystrokeState:
    session_id: str
    keystroke_index: int
    source_raw_down_index: int
    source_raw_up_index: int
    device_id_session_local: str
    make_code: int
    extended_flag: int
    vkey: int
    key_category: str

    press_timestamp_ns: int
    release_timestamp_ns: int

    hold_time_s: float
    press_interval_s: Optional[float]
    signed_flight_time_s: Optional[float]
    release_interval_s: Optional[float]

    overlap_union_duration_s: float
    overlap_fraction: float
    concurrent_keys_at_press: int
    release_inversion_flag: Optional[int]

    correction_key_flag: int
    repeat_count: int
    repeat_flag: int

    shift_at_press: int
    ctrl_at_press: int
    alt_at_press: int
    meta_at_press: int
    modifier_count_at_press: int
    command_shortcut_flag: int

    pairing_status: str = "paired"
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    participant_id: str = ""
    scenario: str = ""
    input_source: str = ""
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quality_flags"] = list(self.quality_flags)
        return data


@dataclass(frozen=True)
class BuildResult:
    states: tuple[KeystrokeState, ...]
    quality_events: tuple[QualityEvent, ...]


@dataclass(frozen=True)
class WindowRecord:
    feature_schema_version: str
    schema_hash: str
    participant_id: str
    session_id: str
    window_id: int
    start_keystroke_index: int
    end_keystroke_index: int
    label: str
    scenario: str
    input_source: str

    sequence: tuple[tuple[float, ...], ...]
    context: tuple[float, ...]

    def to_flat_dict(
        self,
        sequence_feature_names: tuple[str, ...],
        context_feature_names: tuple[str, ...],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "feature_schema_version": self.feature_schema_version,
            "schema_hash": self.schema_hash,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "window_id": self.window_id,
            "start_keystroke_index": self.start_keystroke_index,
            "end_keystroke_index": self.end_keystroke_index,
            "label": self.label,
            "scenario": self.scenario,
            "input_source": self.input_source,
        }

        for t, values in enumerate(self.sequence):
            for name, value in zip(sequence_feature_names, values, strict=True):
                row[f"t{t}_{name}"] = value

        for name, value in zip(context_feature_names, self.context, strict=True):
            row[f"ctx_{name}"] = value

        return row
