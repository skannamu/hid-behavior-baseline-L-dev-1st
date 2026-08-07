"""Session-level validation for stable Feature Schema v2 normal data.

This module validates the complete collector output for each session rather than
only checking ``window/window.csv``.  It is intentionally read-only: source
files are never rewritten or moved.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.features.schema import EXPECTED_SCHEMA_SHA256, FEATURE_SCHEMA_VERSION, WINDOW_SIZE

from .window_dataset import load_window_csv


CORE_SCENARIOS: tuple[str, ...] = (
    "korean_typing",
    "english_typing",
    "coding_controlled",
    "free_writing",
)

SCENARIO_PROTOCOL: dict[str, tuple[float, str]] = {
    "korean_typing": (8.0, "fixed"),
    "english_typing": (8.0, "fixed"),
    "coding_controlled": (10.0, "fixed"),
    "free_writing": (10.0, "minimum"),
    "terminal_safe_commands": (6.0, "fixed"),
    "mixed_real_use": (6.0, "fixed"),
}

REQUIRED_SESSION_FILES: tuple[str, ...] = (
    "metadata.json",
    "session_summary.json",
    "quality_events.jsonl",
    "raw/raw.csv",
    "state/state.csv",
    "window/window.csv",
)

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "session_id",
    "raw_event_index",
    "timestamp_ns",
    "device_id_session_local",
    "make_code",
    "extended_flag",
    "vkey",
    "action",
)

STATE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "session_id",
    "keystroke_index",
    "source_raw_down_index",
    "source_raw_up_index",
    "participant_id",
    "scenario",
    "input_source",
    "label",
    "pairing_status",
)


@dataclass(frozen=True)
class SessionValidationConfig:
    duration_tolerance_seconds: float = 3.0
    fixed_overrun_warning_seconds: float = 60.0
    expected_collector_version: str | None = None
    expected_protocol_version: str | None = None
    require_complete_core_scenarios: bool = False

    def validate(self) -> None:
        if self.duration_tolerance_seconds < 0:
            raise ValueError("duration_tolerance_seconds must be non-negative")
        if self.fixed_overrun_warning_seconds < 0:
            raise ValueError("fixed_overrun_warning_seconds must be non-negative")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionValidationResult:
    session_dir: str
    participant_id: str = ""
    session_id: str = ""
    scenario: str = ""
    collector_version: str = ""
    collection_protocol_version: str = ""
    extractor_version: str = ""
    feature_schema_version: str = ""
    schema_hash: str = ""
    duration_seconds: float | None = None
    planned_duration_min: float | None = None
    planned_duration_rule: str = ""
    raw_event_count: int | None = None
    state_count: int | None = None
    window_count: int | None = None
    quality_event_count: int | None = None
    quality_warning_count: int = 0
    quality_error_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        severities = {issue.severity for issue in self.issues}
        if "error" in severities:
            return "FAIL"
        if "warning" in severities:
            return "WARN"
        return "PASS"

    @property
    def is_valid_for_training(self) -> bool:
        return self.status != "FAIL"

    def add_error(self, code: str, message: str, **details: Any) -> None:
        self.issues.append(ValidationIssue("error", code, message, details))

    def add_warning(self, code: str, message: str, **details: Any) -> None:
        self.issues.append(ValidationIssue("warning", code, message, details))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        payload["is_valid_for_training"] = self.is_valid_for_training
        payload["error_count"] = sum(
            issue.severity == "error" for issue in self.issues
        )
        payload["warning_count"] = sum(
            issue.severity == "warning" for issue in self.issues
        )
        return payload


@dataclass(frozen=True)
class DatasetValidationReport:
    root: str
    sessions: tuple[SessionValidationResult, ...]
    participant_scenarios: dict[str, tuple[str, ...]]
    missing_core_scenarios: dict[str, tuple[str, ...]]
    dataset_issues: tuple[ValidationIssue, ...]

    @property
    def pass_count(self) -> int:
        return sum(session.status == "PASS" for session in self.sessions)

    @property
    def warn_count(self) -> int:
        return sum(session.status == "WARN" for session in self.sessions)

    @property
    def fail_count(self) -> int:
        return sum(session.status == "FAIL" for session in self.sessions)

    @property
    def has_errors(self) -> bool:
        return self.fail_count > 0 or any(
            issue.severity == "error" for issue in self.dataset_issues
        )

    def summary_dict(self) -> dict[str, Any]:
        valid_sessions = [s for s in self.sessions if s.is_valid_for_training]
        participant_ids = sorted(
            {s.participant_id for s in self.sessions if s.participant_id}
        )
        scenarios = sorted({s.scenario for s in self.sessions if s.scenario})
        return {
            "dataset_root": self.root,
            "session_count": len(self.sessions),
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "valid_for_training_count": len(valid_sessions),
            "participant_count": len(participant_ids),
            "participants": participant_ids,
            "scenarios": scenarios,
            "total_windows_valid_sessions": sum(
                s.window_count or 0 for s in valid_sessions
            ),
            "participant_scenarios": {
                participant: list(values)
                for participant, values in self.participant_scenarios.items()
            },
            "missing_core_scenarios": {
                participant: list(values)
                for participant, values in self.missing_core_scenarios.items()
            },
            "dataset_issues": [asdict(issue) for issue in self.dataset_issues],
            "has_errors": self.has_errors,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "schema_hash": EXPECTED_SCHEMA_SHA256,
        }


def _read_json(path: Path, result: SessionValidationResult, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        result.add_error(code, f"Cannot parse JSON: {path}", error=str(exc))
        return {}
    if not isinstance(payload, dict):
        result.add_error(code, f"JSON root must be an object: {path}")
        return {}
    return payload


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check_equal(
    result: SessionValidationResult,
    *,
    code: str,
    field_name: str,
    values: Iterable[tuple[str, Any]],
) -> None:
    filtered = [(source, value) for source, value in values if value not in (None, "")]
    if not filtered:
        result.add_error(code, f"Missing {field_name} in all metadata sources")
        return
    unique = {str(value) for _, value in filtered}
    if len(unique) > 1:
        result.add_error(
            code,
            f"Inconsistent {field_name}",
            values={source: value for source, value in filtered},
        )


def _scan_raw_csv(path: Path, result: SessionValidationResult) -> int | None:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            missing = [name for name in RAW_REQUIRED_COLUMNS if name not in columns]
            if missing:
                result.add_error(
                    "raw_header_missing",
                    "raw.csv is missing required columns",
                    missing=missing,
                )
                return None

            count = 0
            previous_timestamp: int | None = None
            observed_session_ids: set[str] = set()
            for expected_index, row in enumerate(reader):
                count += 1
                observed_session_ids.add(row.get("session_id", "").strip())
                try:
                    actual_index = int(row["raw_event_index"])
                except Exception:
                    result.add_error(
                        "raw_index_invalid",
                        "raw_event_index is not an integer",
                        row=expected_index + 2,
                        value=row.get("raw_event_index"),
                    )
                    continue
                if actual_index != expected_index:
                    result.add_error(
                        "raw_index_not_contiguous",
                        "raw_event_index is not contiguous from zero",
                        row=expected_index + 2,
                        expected=expected_index,
                        actual=actual_index,
                    )
                    break
                try:
                    timestamp = int(row["timestamp_ns"])
                except Exception:
                    result.add_error(
                        "raw_timestamp_invalid",
                        "timestamp_ns is not an integer",
                        row=expected_index + 2,
                    )
                    continue
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    result.add_error(
                        "raw_timestamp_decreased",
                        "Raw monotonic timestamp decreased",
                        row=expected_index + 2,
                        previous=previous_timestamp,
                        actual=timestamp,
                    )
                    break
                previous_timestamp = timestamp
                if row.get("action") not in {"down", "up"}:
                    result.add_error(
                        "raw_action_invalid",
                        "Raw action must be down or up",
                        row=expected_index + 2,
                        value=row.get("action"),
                    )
                    break

            observed_session_ids.discard("")
            if len(observed_session_ids) != 1:
                result.add_error(
                    "raw_session_id_mixed",
                    "raw.csv must contain exactly one non-empty session_id",
                    values=sorted(observed_session_ids),
                )
            elif result.session_id and result.session_id not in observed_session_ids:
                result.add_error(
                    "raw_session_id_mismatch",
                    "raw.csv session_id differs from session metadata",
                    values=sorted(observed_session_ids),
                    expected=result.session_id,
                )
            return count
    except Exception as exc:
        result.add_error("raw_csv_unreadable", f"Cannot read {path}", error=str(exc))
        return None


def _scan_state_csv(path: Path, result: SessionValidationResult) -> int | None:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            missing = [name for name in STATE_REQUIRED_COLUMNS if name not in columns]
            if missing:
                result.add_error(
                    "state_header_missing",
                    "state.csv is missing required columns",
                    missing=missing,
                )
                return None

            count = 0
            participants: set[str] = set()
            sessions: set[str] = set()
            scenarios: set[str] = set()
            labels: set[str] = set()
            sources: set[str] = set()
            for expected_index, row in enumerate(reader):
                count += 1
                participants.add(row.get("participant_id", "").strip())
                sessions.add(row.get("session_id", "").strip())
                scenarios.add(row.get("scenario", "").strip())
                labels.add(row.get("label", "").strip().lower())
                sources.add(row.get("input_source", "").strip().lower())
                try:
                    actual_index = int(row["keystroke_index"])
                except Exception:
                    result.add_error(
                        "state_index_invalid",
                        "keystroke_index is not an integer",
                        row=expected_index + 2,
                    )
                    continue
                if actual_index != expected_index:
                    result.add_error(
                        "state_index_not_contiguous",
                        "keystroke_index is not contiguous from zero",
                        row=expected_index + 2,
                        expected=expected_index,
                        actual=actual_index,
                    )
                    break
                if row.get("pairing_status") != "paired":
                    result.add_error(
                        "state_not_paired",
                        "Stable state rows must have pairing_status=paired",
                        row=expected_index + 2,
                        value=row.get("pairing_status"),
                    )
                    break

            for values in (participants, sessions, scenarios, labels, sources):
                values.discard("")
            expected_sets = (
                ("state_participant_mismatch", participants, result.participant_id),
                ("state_session_mismatch", sessions, result.session_id),
                ("state_scenario_mismatch", scenarios, result.scenario),
                ("state_label_invalid", labels, "normal"),
                ("state_input_source_invalid", sources, "human"),
            )
            for code, values, expected in expected_sets:
                if values != {expected}:
                    result.add_error(
                        code,
                        "state.csv metadata is inconsistent",
                        values=sorted(values),
                        expected=expected,
                    )
            return count
    except Exception as exc:
        result.add_error("state_csv_unreadable", f"Cannot read {path}", error=str(exc))
        return None


def _scan_quality_events(path: Path, result: SessionValidationResult) -> tuple[int, int, int]:
    count = 0
    warning_count = 0
    error_count = 0
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                count += 1
                try:
                    payload = json.loads(line)
                except Exception as exc:
                    result.add_error(
                        "quality_jsonl_invalid",
                        "quality_events.jsonl contains invalid JSON",
                        line=line_no,
                        error=str(exc),
                    )
                    continue
                severity = str(payload.get("severity", "")).lower()
                warning_count += severity == "warning"
                error_count += severity == "error"
    except Exception as exc:
        result.add_error(
            "quality_file_unreadable",
            f"Cannot read {path}",
            error=str(exc),
        )
    return count, warning_count, error_count


def discover_session_dirs(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    if base.is_file():
        raise ValueError(f"Dataset root must be a directory: {base}")

    markers = (
        "session_summary.json",
        "metadata.json",
        "window/window.csv",
        "raw/raw.csv",
    )
    if any((base / marker).exists() for marker in markers):
        return [base]

    discovered: set[Path] = set()
    for name in ("session_summary.json", "metadata.json"):
        discovered.update(path.parent for path in base.rglob(name) if path.is_file())
    discovered.update(
        path.parent.parent
        for path in base.rglob("window.csv")
        if path.is_file() and path.parent.name == "window"
    )
    discovered.update(
        path.parent.parent
        for path in base.rglob("raw.csv")
        if path.is_file() and path.parent.name == "raw"
    )
    return sorted(discovered)


def validate_session_dir(
    session_dir: str | Path,
    *,
    config: SessionValidationConfig | None = None,
) -> SessionValidationResult:
    cfg = config or SessionValidationConfig()
    cfg.validate()
    session = Path(session_dir)
    result = SessionValidationResult(session_dir=str(session))

    for relative in REQUIRED_SESSION_FILES:
        if not (session / relative).is_file():
            result.add_error(
                "required_file_missing",
                f"Required session file is missing: {relative}",
                relative_path=relative,
            )

    metadata_path = session / "metadata.json"
    summary_path = session / "session_summary.json"
    metadata = _read_json(metadata_path, result, "metadata_json_invalid") if metadata_path.is_file() else {}
    summary = _read_json(summary_path, result, "summary_json_invalid") if summary_path.is_file() else {}

    folder_participant = session.parent.name
    folder_session = session.name
    result.participant_id = str(
        summary.get("participant_id") or metadata.get("participant_id") or folder_participant
    )
    result.session_id = str(
        summary.get("session_id") or metadata.get("session_id") or folder_session
    )
    result.scenario = str(summary.get("scenario") or metadata.get("scenario") or "")
    result.collector_version = str(
        summary.get("collector_version") or metadata.get("collector_version") or ""
    )
    result.collection_protocol_version = str(
        summary.get("collection_protocol_version")
        or metadata.get("collection_protocol_version")
        or ""
    )
    result.extractor_version = str(
        summary.get("extractor_version") or metadata.get("extractor_version") or ""
    )
    result.feature_schema_version = str(
        summary.get("feature_schema_version")
        or metadata.get("feature_schema_version")
        or ""
    )
    result.schema_hash = str(
        summary.get("schema_hash") or metadata.get("schema_hash") or ""
    )
    result.duration_seconds = _coerce_float(summary.get("duration_seconds"))
    result.planned_duration_min = _coerce_float(
        summary.get("planned_duration_min", metadata.get("planned_duration_min"))
    )
    result.planned_duration_rule = str(
        summary.get(
            "planned_duration_rule",
            metadata.get("planned_duration_rule", ""),
        )
    )

    _check_equal(
        result,
        code="participant_id_inconsistent",
        field_name="participant_id",
        values=(
            ("folder", folder_participant),
            ("metadata", metadata.get("participant_id")),
            ("summary", summary.get("participant_id")),
        ),
    )
    _check_equal(
        result,
        code="session_id_inconsistent",
        field_name="session_id",
        values=(
            ("folder", folder_session),
            ("metadata", metadata.get("session_id")),
            ("summary", summary.get("session_id")),
        ),
    )
    _check_equal(
        result,
        code="scenario_inconsistent",
        field_name="scenario",
        values=(
            ("metadata", metadata.get("scenario")),
            ("summary", summary.get("scenario")),
        ),
    )

    if metadata and metadata.get("status") != "finalized":
        result.add_error(
            "session_not_finalized",
            "metadata.status must be finalized",
            actual=metadata.get("status"),
        )

    if result.feature_schema_version != FEATURE_SCHEMA_VERSION:
        result.add_error(
            "schema_version_mismatch",
            "Feature schema version mismatch",
            expected=FEATURE_SCHEMA_VERSION,
            actual=result.feature_schema_version,
        )
    if result.schema_hash != EXPECTED_SCHEMA_SHA256:
        result.add_error(
            "schema_hash_mismatch",
            "Feature schema hash mismatch",
            expected=EXPECTED_SCHEMA_SHA256,
            actual=result.schema_hash,
        )

    if cfg.expected_collector_version and result.collector_version != cfg.expected_collector_version:
        result.add_error(
            "collector_version_mismatch",
            "Collector version differs from the requested version",
            expected=cfg.expected_collector_version,
            actual=result.collector_version,
        )
    if cfg.expected_protocol_version and result.collection_protocol_version != cfg.expected_protocol_version:
        result.add_error(
            "protocol_version_mismatch",
            "Collection protocol version differs from the requested version",
            expected=cfg.expected_protocol_version,
            actual=result.collection_protocol_version,
        )

    if str(summary.get("close_reason", "")).strip() == "":
        result.add_warning(
            "close_reason_missing",
            "session_summary.json has no close_reason",
        )

    if result.scenario not in SCENARIO_PROTOCOL:
        result.add_error(
            "scenario_unknown",
            "Unknown scenario",
            actual=result.scenario,
            allowed=sorted(SCENARIO_PROTOCOL),
        )
    else:
        expected_minutes, expected_rule = SCENARIO_PROTOCOL[result.scenario]
        if result.planned_duration_min is None:
            result.add_error("planned_duration_missing", "planned_duration_min is missing")
        elif abs(result.planned_duration_min - expected_minutes) > 1e-9:
            result.add_error(
                "planned_duration_mismatch",
                "Planned duration differs from protocol",
                expected=expected_minutes,
                actual=result.planned_duration_min,
            )
        if result.planned_duration_rule != expected_rule:
            result.add_error(
                "duration_rule_mismatch",
                "Duration rule differs from protocol",
                expected=expected_rule,
                actual=result.planned_duration_rule,
            )

        required_seconds = expected_minutes * 60.0
        if result.duration_seconds is None:
            result.add_error("duration_missing", "duration_seconds is missing or invalid")
        elif result.duration_seconds < required_seconds - cfg.duration_tolerance_seconds:
            result.add_error(
                "duration_too_short",
                "Session is shorter than the protocol minimum",
                required_seconds=required_seconds,
                tolerance_seconds=cfg.duration_tolerance_seconds,
                actual_seconds=result.duration_seconds,
            )
        elif (
            expected_rule == "fixed"
            and result.duration_seconds
            > required_seconds + cfg.fixed_overrun_warning_seconds
        ):
            result.add_warning(
                "fixed_duration_overrun",
                "Fixed-duration session substantially exceeded planned time",
                planned_seconds=required_seconds,
                actual_seconds=result.duration_seconds,
            )

    for flag in ("raw_validation_ok", "state_validation_ok", "window_validation_ok"):
        if summary and summary.get(flag) is not True:
            result.add_error(
                f"{flag}_false",
                f"Collector summary reports {flag}=false or missing",
                actual=summary.get(flag),
            )

    raw_path = session / "raw" / "raw.csv"
    state_path = session / "state" / "state.csv"
    window_path = session / "window" / "window.csv"
    quality_path = session / "quality_events.jsonl"

    if raw_path.is_file():
        result.raw_event_count = _scan_raw_csv(raw_path, result)
    if state_path.is_file():
        result.state_count = _scan_state_csv(state_path, result)

    if window_path.is_file():
        try:
            samples, report = load_window_csv(window_path)
            result.window_count = report.row_count
            if report.participant_ids != (result.participant_id,):
                result.add_error(
                    "window_participant_mismatch",
                    "window.csv participant_id differs from session metadata",
                    actual=list(report.participant_ids),
                    expected=result.participant_id,
                )
            if report.session_ids != (result.session_id,):
                result.add_error(
                    "window_session_mismatch",
                    "window.csv session_id differs from session metadata",
                    actual=list(report.session_ids),
                    expected=result.session_id,
                )
            if report.scenarios != (result.scenario,):
                result.add_error(
                    "window_scenario_mismatch",
                    "window.csv scenario differs from session metadata",
                    actual=list(report.scenarios),
                    expected=result.scenario,
                )
            normalized_labels = tuple(sorted(label.lower() for label in report.labels))
            if normalized_labels != ("normal",):
                result.add_error(
                    "window_label_invalid",
                    "Normal dataset window labels must all be normal",
                    actual=list(report.labels),
                )
            input_sources = {sample.input_source.strip().lower() for sample in samples}
            if input_sources != {"human"}:
                result.add_error(
                    "window_input_source_invalid",
                    "Normal dataset input_source must all be human",
                    actual=sorted(input_sources),
                )
            for expected_id, sample in enumerate(samples):
                if sample.window_id != expected_id:
                    result.add_error(
                        "window_id_not_contiguous",
                        "window_id is not contiguous from zero",
                        row=sample.row_number,
                        expected=expected_id,
                        actual=sample.window_id,
                    )
                    break
                if sample.start_keystroke_index != expected_id + 1:
                    # Stable windows exclude the first transitionless state, so
                    # the first valid 50-state window normally starts at index 1.
                    result.add_error(
                        "window_start_index_unexpected",
                        "start_keystroke_index does not match stable stride-1 policy",
                        row=sample.row_number,
                        expected=expected_id + 1,
                        actual=sample.start_keystroke_index,
                    )
                    break
        except Exception as exc:
            result.add_error(
                "window_strict_validation_failed",
                "Strict Feature Schema v2 window validation failed",
                error=str(exc),
            )

    if quality_path.is_file():
        (
            result.quality_event_count,
            result.quality_warning_count,
            result.quality_error_count,
        ) = _scan_quality_events(quality_path, result)
        if result.quality_error_count:
            result.add_warning(
                "quality_error_events_present",
                "Collector recorded error-severity quality events",
                count=result.quality_error_count,
            )

    count_checks = (
        ("raw_event_count", result.raw_event_count),
        ("paired_keystroke_count", result.state_count),
        ("window_count", result.window_count),
        ("quality_event_count", result.quality_event_count),
    )
    for summary_name, actual_count in count_checks:
        expected_count = _coerce_int(summary.get(summary_name))
        if expected_count is None:
            result.add_error(
                f"summary_{summary_name}_missing",
                f"session_summary.json has no valid {summary_name}",
            )
        elif actual_count is not None and expected_count != actual_count:
            result.add_error(
                f"summary_{summary_name}_mismatch",
                f"{summary_name} differs from the actual file row count",
                expected=expected_count,
                actual=actual_count,
            )

    if (result.window_count or 0) <= 0:
        result.add_error("window_count_zero", "Session contains no model windows")

    return result


def validate_dataset_root(
    root: str | Path,
    *,
    config: SessionValidationConfig | None = None,
) -> DatasetValidationReport:
    cfg = config or SessionValidationConfig()
    cfg.validate()
    session_dirs = discover_session_dirs(root)
    sessions = [validate_session_dir(path, config=cfg) for path in session_dirs]
    dataset_issues: list[ValidationIssue] = []

    if not sessions:
        dataset_issues.append(
            ValidationIssue(
                "error",
                "no_sessions_found",
                "No collector session directories were found",
                {"root": str(Path(root))},
            )
        )

    by_session_id: dict[str, list[SessionValidationResult]] = {}
    for session in sessions:
        if session.session_id:
            by_session_id.setdefault(session.session_id, []).append(session)
    for session_id, duplicates in by_session_id.items():
        if len(duplicates) > 1:
            issue = ValidationIssue(
                "error",
                "duplicate_session_id",
                "The same session_id appears in multiple directories",
                {"session_id": session_id, "paths": [s.session_dir for s in duplicates]},
            )
            dataset_issues.append(issue)
            for session in duplicates:
                session.issues.append(issue)

    version_fields = (
        ("collector_version", "mixed_collector_versions"),
        ("collection_protocol_version", "mixed_protocol_versions"),
        ("extractor_version", "mixed_extractor_versions"),
        ("feature_schema_version", "mixed_schema_versions"),
        ("schema_hash", "mixed_schema_hashes"),
    )
    for attribute, code in version_fields:
        values = sorted(
            {getattr(session, attribute) for session in sessions if getattr(session, attribute)}
        )
        if len(values) > 1:
            dataset_issues.append(
                ValidationIssue(
                    "error",
                    code,
                    f"Dataset contains multiple {attribute} values",
                    {"values": values},
                )
            )

    participant_scenarios: dict[str, tuple[str, ...]] = {}
    for participant in sorted({s.participant_id for s in sessions if s.participant_id}):
        participant_scenarios[participant] = tuple(
            sorted({s.scenario for s in sessions if s.participant_id == participant and s.scenario})
        )
    missing_core: dict[str, tuple[str, ...]] = {}
    for participant, scenarios in participant_scenarios.items():
        missing = tuple(scenario for scenario in CORE_SCENARIOS if scenario not in scenarios)
        if missing:
            missing_core[participant] = missing
            severity = "error" if cfg.require_complete_core_scenarios else "warning"
            dataset_issues.append(
                ValidationIssue(
                    severity,
                    "participant_core_scenarios_incomplete",
                    "Participant has not submitted every core scenario",
                    {"participant_id": participant, "missing": list(missing)},
                )
            )

    return DatasetValidationReport(
        root=str(Path(root)),
        sessions=tuple(sessions),
        participant_scenarios=participant_scenarios,
        missing_core_scenarios=missing_core,
        dataset_issues=tuple(dataset_issues),
    )
