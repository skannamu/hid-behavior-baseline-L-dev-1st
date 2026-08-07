from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data_v2 import (
    BalancedSamplingConfig,
    FeatureV2WindowDataset,
    GroupSplitConfig,
    balanced_subsample,
    build_dataset_manifest,
    load_manifest_entries,
    resolve_manifest_window_paths,
    split_by_group,
    validate_dataset_root,
    write_dataset_manifest,
)
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    expected_window_columns,
)
from tests.test_stable_normal_validation import write_valid_session


def _rewrite_protocol(session_dir: Path, scenario: str) -> None:
    rules = {
        "korean_typing": (8, "fixed", 480.0),
        "english_typing": (8, "fixed", 480.0),
        "coding_controlled": (10, "fixed", 600.0),
        "free_writing": (10, "minimum", 600.0),
    }
    minutes, rule, duration = rules[scenario]
    for name in ("metadata.json", "session_summary.json"):
        path = session_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["scenario"] = scenario
        payload["planned_duration_min"] = minutes
        payload["planned_duration_rule"] = rule
        if name == "session_summary.json":
            payload["duration_seconds"] = duration
        path.write_text(json.dumps(payload), encoding="utf-8")

    for relative in ("state/state.csv", "window/window.csv"):
        path = session_dir / relative
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        fields = list(rows[0])
        for row in rows:
            row["scenario"] = scenario
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _make_window_row(
    participant: str,
    session: str,
    scenario: str,
    window_id: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "participant_id": participant,
        "session_id": session,
        "window_id": window_id,
        "start_keystroke_index": window_id,
        "end_keystroke_index": window_id + WINDOW_SIZE - 1,
        "label": "normal",
        "scenario": scenario,
        "input_source": "human",
    }
    for t in range(WINDOW_SIZE):
        for feature in SEQUENCE_FEATURES:
            row[f"t{t}_{feature}"] = float(window_id) * 0.001
    for feature in WINDOW_CONTEXT_FEATURES:
        row[f"ctx_{feature}"] = float(window_id) * 0.01
    return row


def _write_window_file(
    root: Path,
    participant: str,
    session: str,
    scenario: str,
    count: int,
) -> Path:
    path = root / participant / session / "window" / "window.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=expected_window_columns(WINDOW_SIZE),
        )
        writer.writeheader()
        writer.writerows(
            _make_window_row(participant, session, scenario, index)
            for index in range(count)
        )
    return path


class V9DataReadinessTest(unittest.TestCase):
    def test_manifest_freezes_valid_sessions_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            first = write_valid_session(root, participant="p001", session="s001")
            second = write_valid_session(root, participant="p002", session="s002")
            failed = write_valid_session(
                root,
                participant="p003",
                session="s003",
                duration_seconds=300.0,
            )
            self.assertTrue(first.exists() and second.exists() and failed.exists())

            report = validate_dataset_root(root)
            manifest = build_dataset_manifest(report, dataset_id="unit_test")
            self.assertEqual(len(manifest.entries), 2)
            self.assertEqual(len(manifest.excluded_sessions), 1)

            out = Path(tmp) / "manifest"
            paths = write_dataset_manifest(manifest, out)
            entries = load_manifest_entries(
                paths["jsonl"],
                dataset_root=root,
                verify_files=True,
                verify_hashes=True,
            )
            self.assertEqual(len(entries), 2)
            self.assertTrue(manifest.dataset_digest)

    def test_participant_split_never_crosses_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for participant_index in range(6):
                participant = f"p{participant_index:03d}"
                for session_index in range(2):
                    paths.append(_write_window_file(
                        root,
                        participant,
                        f"s{participant_index:03d}_{session_index}",
                        "korean_typing",
                        3,
                    ))
            dataset, _ = FeatureV2WindowDataset.from_paths(paths)
            split = split_by_group(
                dataset,
                config=GroupSplitConfig(seed=7, group_mode="participant"),
            )
            train = {sample.participant_id for sample in split.train}
            calibration = {
                sample.participant_id for sample in split.calibration
            }
            test = {sample.participant_id for sample in split.test}
            self.assertFalse(train & calibration)
            self.assertFalse(train & test)
            self.assertFalse(calibration & test)
            self.assertEqual(split.group_mode, "participant")

    def test_balanced_sampler_equalizes_participant_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                _write_window_file(root, "p001", "s1", "korean_typing", 20),
                _write_window_file(root, "p001", "s2", "coding_controlled", 10),
                _write_window_file(root, "p002", "s3", "korean_typing", 15),
                _write_window_file(root, "p002", "s4", "coding_controlled", 8),
            ]
            dataset, _ = FeatureV2WindowDataset.from_paths(paths)
            result = balanced_subsample(
                dataset,
                config=BalancedSamplingConfig(
                    window_stride=1,
                    balance_mode="equal",
                    seed=11,
                ),
            )
            selected_counts = {
                key: row["selected_windows"]
                for key, row in result.report["groups"].items()
            }
            self.assertEqual(set(selected_counts.values()), {8})
            self.assertEqual(len(result.dataset), 32)

    def test_manifest_paths_load_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            for index in range(3):
                write_valid_session(
                    root,
                    participant=f"p{index:03d}",
                    session=f"s{index:03d}",
                )
            report = validate_dataset_root(root)
            manifest = build_dataset_manifest(report, dataset_id="paths")
            out = Path(tmp) / "manifest"
            paths = write_dataset_manifest(manifest, out)
            entries = load_manifest_entries(
                paths["jsonl"], dataset_root=root
            )
            window_paths = resolve_manifest_window_paths(
                entries, dataset_root=root
            )
            dataset, _ = FeatureV2WindowDataset.from_paths(window_paths)
            self.assertEqual(len(dataset), 3)


if __name__ == "__main__":
    unittest.main()
