from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.data_v2.session_validator import (
    DatasetValidationReport,
    SessionValidationConfig,
    validate_dataset_root,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_session_csv(path: Path, report: DatasetValidationReport) -> None:
    rows = []
    for session in report.sessions:
        payload = session.to_dict()
        rows.append(
            {
                "status": payload["status"],
                "is_valid_for_training": payload["is_valid_for_training"],
                "participant_id": session.participant_id,
                "session_id": session.session_id,
                "scenario": session.scenario,
                "duration_seconds": session.duration_seconds,
                "planned_duration_min": session.planned_duration_min,
                "planned_duration_rule": session.planned_duration_rule,
                "raw_event_count": session.raw_event_count,
                "state_count": session.state_count,
                "window_count": session.window_count,
                "quality_event_count": session.quality_event_count,
                "quality_warning_count": session.quality_warning_count,
                "quality_error_count": session.quality_error_count,
                "error_count": payload["error_count"],
                "warning_count": payload["warning_count"],
                "issue_codes": ";".join(issue.code for issue in session.issues),
                "session_dir": session.session_dir,
            }
        )

    fieldnames = [
        "status",
        "is_valid_for_training",
        "participant_id",
        "session_id",
        "scenario",
        "duration_seconds",
        "planned_duration_min",
        "planned_duration_rule",
        "raw_event_count",
        "state_count",
        "window_count",
        "quality_event_count",
        "quality_warning_count",
        "quality_error_count",
        "error_count",
        "warning_count",
        "issue_codes",
        "session_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_exclusion_csv(path: Path, report: DatasetValidationReport) -> None:
    rows = []
    for session in report.sessions:
        if session.status != "FAIL":
            continue
        for issue in session.issues:
            if issue.severity != "error":
                continue
            rows.append(
                {
                    "participant_id": session.participant_id,
                    "session_id": session.session_id,
                    "scenario": session.scenario,
                    "issue_code": issue.code,
                    "reason": issue.message,
                    "session_dir": session.session_dir,
                }
            )
    fieldnames = [
        "participant_id",
        "session_id",
        "scenario",
        "issue_code",
        "reason",
        "session_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate complete stable-v2 normal collector sessions and write "
            "machine-readable quality reports."
        )
    )
    parser.add_argument("dataset_root")
    parser.add_argument(
        "--output-dir",
        default="reports/data_validation/latest",
    )
    parser.add_argument("--duration-tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--fixed-overrun-warning-seconds", type=float, default=60.0)
    parser.add_argument("--expected-collector-version")
    parser.add_argument("--expected-protocol-version")
    parser.add_argument("--require-complete-core", action="store_true")
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Always exit zero after writing reports, even when validation fails.",
    )
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

    output = Path(args.output_dir)
    session_rows = [session.to_dict() for session in report.sessions]
    _write_session_csv(output / "session_validation.csv", report)
    _write_jsonl(output / "session_validation.jsonl", session_rows)
    _write_json(output / "dataset_validation_summary.json", report.summary_dict())
    _write_exclusion_csv(output / "exclusion_candidates.csv", report)

    for session in report.sessions:
        duration = (
            f"{session.duration_seconds:.1f}s"
            if session.duration_seconds is not None
            else "duration=?"
        )
        print(
            f"[{session.status}] {session.participant_id or '?'} / "
            f"{session.scenario or '?'} / {duration} / "
            f"windows={session.window_count if session.window_count is not None else '?'}"
        )
        for issue in session.issues:
            print(f"  - {issue.severity.upper()} {issue.code}: {issue.message}")

    for issue in report.dataset_issues:
        print(f"[DATASET {issue.severity.upper()}] {issue.code}: {issue.message}")

    summary = report.summary_dict()
    print("\n===== DATASET VALIDATION SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReports written to: {output}")

    if report.has_errors and not args.no_fail_exit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
