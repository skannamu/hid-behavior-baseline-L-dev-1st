from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data_v2 import (
    FeatureV2WindowDataset,
    GroupSplitConfig,
    load_window_csv,
    split_by_group,
)
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    expected_window_columns,
)


def make_row(
    *,
    participant: str,
    session: str,
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
        "scenario": "typing",
        "input_source": "human",
    }
    for t in range(WINDOW_SIZE):
        for feature in SEQUENCE_FEATURES:
            row[f"t{t}_{feature}"] = 0.0
    for feature in WINDOW_CONTEXT_FEATURES:
        row[f"ctx_{feature}"] = 0.0
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=expected_window_columns(WINDOW_SIZE),
        )
        writer.writeheader()
        writer.writerows(rows)


class DataV2LoaderTest(unittest.TestCase):
    def test_loads_exact_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.csv"
            write_csv(
                path,
                [
                    make_row(
                        participant="p001",
                        session="s001",
                        window_id=0,
                    )
                ],
            )
            samples, report = load_window_csv(path)
            self.assertEqual(report.row_count, 1)
            self.assertEqual(samples[0].sequence.shape, (50, 12))
            self.assertEqual(samples[0].context.shape, (12,))

    def test_rejects_schema_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.csv"
            row = make_row(
                participant="p001",
                session="s001",
                window_id=0,
            )
            row["schema_hash"] = "bad"
            write_csv(path, [row])
            with self.assertRaisesRegex(ValueError, "schema hash mismatch"):
                load_window_csv(path)

    def test_rejects_header_reordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.csv"
            columns = expected_window_columns(WINDOW_SIZE)
            columns[10], columns[11] = columns[11], columns[10]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    make_row(
                        participant="p001",
                        session="s001",
                        window_id=0,
                    )
                )
            with self.assertRaisesRegex(ValueError, "header mismatch"):
                load_window_csv(path)

    def test_group_split_is_disjoint_and_deterministic(self):
        samples = []
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for group_index in range(10):
                session_dir = Path(tmp) / f"p{group_index:03d}" / f"s{group_index:03d}"
                session_dir.mkdir(parents=True)
                path = session_dir / "window.csv"
                write_csv(
                    path,
                    [
                        make_row(
                            participant=f"p{group_index:03d}",
                            session=f"s{group_index:03d}",
                            window_id=i,
                        )
                        for i in range(2)
                    ],
                )
                paths.append(path)

            dataset, _ = FeatureV2WindowDataset.from_paths(paths)
            config = GroupSplitConfig(seed=123)
            split_a = split_by_group(dataset, config=config)
            split_b = split_by_group(dataset, config=config)

            self.assertEqual(split_a.train_groups, split_b.train_groups)
            self.assertEqual(
                split_a.calibration_groups,
                split_b.calibration_groups,
            )
            self.assertEqual(split_a.test_groups, split_b.test_groups)
            split_a.assert_disjoint()

    def test_one_session_never_falls_back_to_window_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.csv"
            write_csv(
                path,
                [
                    make_row(
                        participant="p001",
                        session="s001",
                        window_id=i,
                    )
                    for i in range(100)
                ],
            )
            dataset, _ = FeatureV2WindowDataset.from_paths([path])
            with self.assertRaisesRegex(ValueError, "At least 3 independent groups"):
                split_by_group(dataset)


if __name__ == "__main__":
    unittest.main()
