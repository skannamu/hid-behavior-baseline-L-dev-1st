"""Strict loader for Feature Schema v2 window CSV files."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    expected_window_columns,
)


@dataclass(frozen=True)
class WindowSample:
    source_file: str
    row_number: int

    participant_id: str
    session_id: str
    window_id: int
    start_keystroke_index: int
    end_keystroke_index: int
    label: str
    scenario: str
    input_source: str

    sequence: np.ndarray
    context: np.ndarray

    @property
    def normal_group_key(self) -> str:
        return f"{self.participant_id}::{self.session_id}"


@dataclass(frozen=True)
class WindowFileReport:
    path: str
    row_count: int
    participant_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    labels: tuple[str, ...]
    scenarios: tuple[str, ...]
    schema_version: str
    schema_hash: str


def discover_window_csvs(root: str | Path) -> list[Path]:
    base = Path(root)
    if base.is_file():
        return [base] if base.name == "window.csv" else []

    return sorted(
        p for p in base.rglob("window.csv")
        if p.is_file()
    )


def _parse_int(row: dict[str, str], name: str, *, path: Path, line_no: int) -> int:
    try:
        return int(row[name])
    except Exception as exc:
        raise ValueError(
            f"{path}:{line_no}: invalid integer field {name}={row.get(name)!r}"
        ) from exc


def _parse_float(
    row: dict[str, str],
    name: str,
    *,
    path: Path,
    line_no: int,
) -> float:
    try:
        value = float(row[name])
    except Exception as exc:
        raise ValueError(
            f"{path}:{line_no}: invalid float field {name}={row.get(name)!r}"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"{path}:{line_no}: non-finite value {name}={value}"
        )
    return value


def load_window_csv(path: str | Path) -> tuple[list[WindowSample], WindowFileReport]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    expected_columns = expected_window_columns(WINDOW_SIZE)
    samples: list[WindowSample] = []

    participant_ids: set[str] = set()
    session_ids: set[str] = set()
    labels: set[str] = set()
    scenarios: set[str] = set()

    detected_version: str | None = None
    detected_hash: str | None = None

    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        actual_columns = reader.fieldnames or []

        if actual_columns != expected_columns:
            missing = [c for c in expected_columns if c not in actual_columns]
            extra = [c for c in actual_columns if c not in expected_columns]
            first_mismatch = next(
                (
                    (idx, exp, act)
                    for idx, (exp, act) in enumerate(
                        zip(expected_columns, actual_columns)
                    )
                    if exp != act
                ),
                None,
            )
            raise ValueError(
                f"{source}: window header mismatch; "
                f"expected_count={len(expected_columns)}, "
                f"actual_count={len(actual_columns)}, "
                f"missing={missing[:10]}, extra={extra[:10]}, "
                f"first_mismatch={first_mismatch}"
            )

        for line_no, row in enumerate(reader, start=2):
            version = row["feature_schema_version"]
            schema_hash = row["schema_hash"]

            if version != FEATURE_SCHEMA_VERSION:
                raise ValueError(
                    f"{source}:{line_no}: schema version mismatch; "
                    f"expected={FEATURE_SCHEMA_VERSION}, actual={version}"
                )
            if schema_hash != EXPECTED_SCHEMA_SHA256:
                raise ValueError(
                    f"{source}:{line_no}: schema hash mismatch; "
                    f"expected={EXPECTED_SCHEMA_SHA256}, actual={schema_hash}"
                )

            if detected_version is None:
                detected_version = version
                detected_hash = schema_hash
            elif version != detected_version or schema_hash != detected_hash:
                raise ValueError(
                    f"{source}:{line_no}: mixed schema rows in one file"
                )

            participant_id = row["participant_id"].strip()
            session_id = row["session_id"].strip()
            label = row["label"].strip()
            scenario = row["scenario"].strip()
            input_source = row["input_source"].strip()

            if not participant_id:
                raise ValueError(f"{source}:{line_no}: empty participant_id")
            if not session_id:
                raise ValueError(f"{source}:{line_no}: empty session_id")
            if not label:
                raise ValueError(f"{source}:{line_no}: empty label")

            sequence = np.empty(
                (WINDOW_SIZE, len(SEQUENCE_FEATURES)),
                dtype=np.float32,
            )
            for t in range(WINDOW_SIZE):
                for feature_index, feature_name in enumerate(SEQUENCE_FEATURES):
                    column = f"t{t}_{feature_name}"
                    sequence[t, feature_index] = _parse_float(
                        row, column, path=source, line_no=line_no
                    )

            context = np.empty(
                (len(WINDOW_CONTEXT_FEATURES),),
                dtype=np.float32,
            )
            for feature_index, feature_name in enumerate(WINDOW_CONTEXT_FEATURES):
                column = f"ctx_{feature_name}"
                context[feature_index] = _parse_float(
                    row, column, path=source, line_no=line_no
                )

            start_index = _parse_int(
                row, "start_keystroke_index", path=source, line_no=line_no
            )
            end_index = _parse_int(
                row, "end_keystroke_index", path=source, line_no=line_no
            )
            if end_index - start_index + 1 != WINDOW_SIZE:
                raise ValueError(
                    f"{source}:{line_no}: invalid keystroke span "
                    f"{start_index}..{end_index}"
                )

            sample = WindowSample(
                source_file=str(source),
                row_number=line_no,
                participant_id=participant_id,
                session_id=session_id,
                window_id=_parse_int(
                    row, "window_id", path=source, line_no=line_no
                ),
                start_keystroke_index=start_index,
                end_keystroke_index=end_index,
                label=label,
                scenario=scenario,
                input_source=input_source,
                sequence=sequence,
                context=context,
            )
            samples.append(sample)

            participant_ids.add(participant_id)
            session_ids.add(session_id)
            labels.add(label)
            scenarios.add(scenario)

    if not samples:
        raise ValueError(f"{source}: no window rows")

    report = WindowFileReport(
        path=str(source),
        row_count=len(samples),
        participant_ids=tuple(sorted(participant_ids)),
        session_ids=tuple(sorted(session_ids)),
        labels=tuple(sorted(labels)),
        scenarios=tuple(sorted(scenarios)),
        schema_version=detected_version or "",
        schema_hash=detected_hash or "",
    )
    return samples, report


class FeatureV2WindowDataset:
    """In-memory strict dataset assembled from one or more window.csv files."""

    def __init__(self, samples: Sequence[WindowSample]) -> None:
        if not samples:
            raise ValueError("FeatureV2WindowDataset requires at least one sample")
        self._samples = tuple(samples)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
    ) -> tuple["FeatureV2WindowDataset", tuple[WindowFileReport, ...]]:
        all_samples: list[WindowSample] = []
        reports: list[WindowFileReport] = []

        for path in sorted(Path(p) for p in paths):
            samples, report = load_window_csv(path)
            all_samples.extend(samples)
            reports.append(report)

        if not all_samples:
            raise ValueError("No Feature Schema v2 windows were loaded")

        return cls(all_samples), tuple(reports)

    @classmethod
    def discover(
        cls,
        root: str | Path,
    ) -> tuple["FeatureV2WindowDataset", tuple[WindowFileReport, ...]]:
        paths = discover_window_csvs(root)
        if not paths:
            raise FileNotFoundError(
                f"No window.csv files found under {Path(root)}"
            )
        return cls.from_paths(paths)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> WindowSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[WindowSample]:
        return iter(self._samples)

    @property
    def samples(self) -> tuple[WindowSample, ...]:
        return self._samples

    @property
    def sequence_array(self) -> np.ndarray:
        return np.stack([sample.sequence for sample in self._samples], axis=0)

    @property
    def context_array(self) -> np.ndarray:
        return np.stack([sample.context for sample in self._samples], axis=0)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([sample.label for sample in self._samples], dtype=object)

    @property
    def normal_group_keys(self) -> np.ndarray:
        return np.asarray(
            [sample.normal_group_key for sample in self._samples],
            dtype=object,
        )

    def subset(self, indices: Sequence[int]) -> "FeatureV2WindowDataset":
        return FeatureV2WindowDataset([self._samples[i] for i in indices])

    def summary(self) -> dict[str, object]:
        participants = sorted({s.participant_id for s in self._samples})
        sessions = sorted({s.session_id for s in self._samples})
        labels = sorted({s.label for s in self._samples})
        scenarios = sorted({s.scenario for s in self._samples})

        return {
            "window_count": len(self._samples),
            "participant_count": len(participants),
            "session_count": len(sessions),
            "participants": participants,
            "sessions": sessions,
            "labels": labels,
            "scenarios": scenarios,
            "sequence_shape": [
                len(self._samples),
                WINDOW_SIZE,
                len(SEQUENCE_FEATURES),
            ],
            "context_shape": [
                len(self._samples),
                len(WINDOW_CONTEXT_FEATURES),
            ],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "schema_hash": EXPECTED_SCHEMA_SHA256,
        }
