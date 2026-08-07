"""Versioned dataset manifests for validated stable-v2 normal sessions."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.features.schema import EXPECTED_SCHEMA_SHA256, FEATURE_SCHEMA_VERSION

from .session_validator import CORE_SCENARIOS, DatasetValidationReport


MANIFEST_FORMAT_VERSION = "stable_normal_manifest_v1"


@dataclass(frozen=True)
class ManifestEntry:
    participant_id: str
    session_id: str
    scenario: str
    session_dir: str
    window_path: str
    window_sha256: str
    window_count: int
    duration_seconds: float
    validation_status: str
    include_for_training: bool
    feature_schema_version: str
    schema_hash: str
    collector_version: str
    collection_protocol_version: str
    extractor_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    dataset_root: str
    entries: tuple[ManifestEntry, ...]
    excluded_sessions: tuple[dict[str, Any], ...]
    dataset_digest: str

    @property
    def training_entries(self) -> tuple[ManifestEntry, ...]:
        return tuple(entry for entry in self.entries if entry.include_for_training)

    def summary_dict(self) -> dict[str, Any]:
        included = self.training_entries
        participants = sorted({entry.participant_id for entry in included})
        scenarios = sorted({entry.scenario for entry in included})
        return {
            "manifest_format_version": MANIFEST_FORMAT_VERSION,
            "dataset_id": self.dataset_id,
            "dataset_root": self.dataset_root,
            "dataset_digest": self.dataset_digest,
            "entry_count": len(self.entries),
            "included_session_count": len(included),
            "excluded_session_count": len(self.excluded_sessions),
            "participant_count": len(participants),
            "participants": participants,
            "scenarios": scenarios,
            "total_windows": sum(entry.window_count for entry in included),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "schema_hash": EXPECTED_SCHEMA_SHA256,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(entries: Iterable[ManifestEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: (item.participant_id, item.session_id)):
        payload = json.dumps(
            entry.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_dataset_manifest(
    report: DatasetValidationReport,
    *,
    dataset_id: str,
    include_warn: bool = True,
    allow_dataset_errors: bool = False,
) -> DatasetManifest:
    """Build a frozen manifest from a complete validation report.

    Failed sessions are always excluded.  Warning sessions are included by
    default because warnings are informational unless the caller chooses a
    stricter policy.
    """

    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")
    if report.has_errors and not allow_dataset_errors:
        dataset_error_codes = [
            issue.code for issue in report.dataset_issues if issue.severity == "error"
        ]
        if dataset_error_codes:
            raise ValueError(
                "Dataset-level validation errors must be resolved before manifest "
                f"creation: {dataset_error_codes}"
            )

    root = Path(report.root)
    entries: list[ManifestEntry] = []
    excluded: list[dict[str, Any]] = []

    for session in sorted(
        report.sessions,
        key=lambda item: (item.participant_id, item.session_id, item.session_dir),
    ):
        include = session.status == "PASS" or (
            include_warn and session.status == "WARN"
        )
        session_path = Path(session.session_dir)
        window_path = session_path / "window" / "window.csv"

        if not include:
            excluded.append(
                {
                    "participant_id": session.participant_id,
                    "session_id": session.session_id,
                    "scenario": session.scenario,
                    "validation_status": session.status,
                    "issue_codes": [issue.code for issue in session.issues],
                    "session_dir": _relative_or_absolute(session_path, root),
                }
            )
            continue

        if not window_path.is_file():
            raise FileNotFoundError(window_path)
        if session.window_count is None or session.duration_seconds is None:
            raise ValueError(
                f"Validated session lacks counts/duration: {session.session_dir}"
            )

        entries.append(
            ManifestEntry(
                participant_id=session.participant_id,
                session_id=session.session_id,
                scenario=session.scenario,
                session_dir=_relative_or_absolute(session_path, root),
                window_path=_relative_or_absolute(window_path, root),
                window_sha256=_sha256_file(window_path),
                window_count=int(session.window_count),
                duration_seconds=float(session.duration_seconds),
                validation_status=session.status,
                include_for_training=True,
                feature_schema_version=session.feature_schema_version,
                schema_hash=session.schema_hash,
                collector_version=session.collector_version,
                collection_protocol_version=session.collection_protocol_version,
                extractor_version=session.extractor_version,
            )
        )

    if not entries:
        raise ValueError("No validated sessions are available for training")

    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_root=str(root.resolve()),
        entries=tuple(entries),
        excluded_sessions=tuple(excluded),
        dataset_digest=_canonical_digest(entries),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_manifest(manifest: DatasetManifest, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = [entry.to_dict() for entry in manifest.entries]
    jsonl_path = output / "dataset_manifest.jsonl"
    csv_path = output / "dataset_manifest.csv"
    summary_path = output / "dataset_summary.json"
    participant_path = output / "participant_scenario_summary.csv"
    exclusion_path = output / "exclusion_log.csv"

    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows, list(ManifestEntry.__dataclass_fields__))
    _write_json(summary_path, manifest.summary_dict())

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in manifest.training_entries:
        key = (entry.participant_id, entry.scenario)
        row = grouped.setdefault(
            key,
            {
                "participant_id": entry.participant_id,
                "scenario": entry.scenario,
                "session_count": 0,
                "window_count": 0,
                "duration_seconds": 0.0,
            },
        )
        row["session_count"] += 1
        row["window_count"] += entry.window_count
        row["duration_seconds"] += entry.duration_seconds

    participant_rows = [grouped[key] for key in sorted(grouped)]
    _write_csv(
        participant_path,
        participant_rows,
        [
            "participant_id",
            "scenario",
            "session_count",
            "window_count",
            "duration_seconds",
        ],
    )

    exclusion_rows = list(manifest.excluded_sessions)
    _write_csv(
        exclusion_path,
        exclusion_rows,
        [
            "participant_id",
            "session_id",
            "scenario",
            "validation_status",
            "issue_codes",
            "session_dir",
        ],
    )

    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "participant_scenario_summary": str(participant_path),
        "exclusion_log": str(exclusion_path),
    }


def load_manifest_entries(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    verify_files: bool = True,
    verify_hashes: bool = False,
) -> tuple[ManifestEntry, ...]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    root = Path(dataset_root).resolve() if dataset_root is not None else None
    entries: list[ManifestEntry] = []
    seen_sessions: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            entry = ManifestEntry(**payload)
            if entry.feature_schema_version != FEATURE_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_no}: feature schema version mismatch"
                )
            if entry.schema_hash != EXPECTED_SCHEMA_SHA256:
                raise ValueError(f"{path}:{line_no}: schema hash mismatch")
            if entry.session_id in seen_sessions:
                raise ValueError(
                    f"{path}:{line_no}: duplicate session_id={entry.session_id}"
                )
            seen_sessions.add(entry.session_id)

            window = Path(entry.window_path)
            if not window.is_absolute():
                if root is None:
                    raise ValueError(
                        "dataset_root is required for relative manifest paths"
                    )
                window = root / window
            if verify_files and not window.is_file():
                raise FileNotFoundError(window)
            if verify_hashes and _sha256_file(window) != entry.window_sha256:
                raise ValueError(
                    f"Manifest file hash mismatch: {window}"
                )
            entries.append(entry)

    if not entries:
        raise ValueError(f"Manifest contains no entries: {path}")
    return tuple(entries)


def resolve_manifest_window_paths(
    entries: Iterable[ManifestEntry],
    *,
    dataset_root: str | Path,
) -> list[Path]:
    root = Path(dataset_root).resolve()
    paths: list[Path] = []
    for entry in entries:
        if not entry.include_for_training:
            continue
        path = Path(entry.window_path)
        paths.append(path if path.is_absolute() else root / path)
    return paths


def participant_core_completion(entries: Iterable[ManifestEntry]) -> dict[str, list[str]]:
    present: dict[str, set[str]] = {}
    for entry in entries:
        present.setdefault(entry.participant_id, set()).add(entry.scenario)
    return {
        participant: sorted(set(CORE_SCENARIOS) - scenarios)
        for participant, scenarios in sorted(present.items())
    }
