from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_v2 import FeatureV2WindowDataset
from src.features.schema import (
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly validate and summarize Feature Schema v2 windows."
    )
    parser.add_argument(
        "path",
        help="A window.csv file or a directory containing session window.csv files",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path for a JSON summary",
    )
    args = parser.parse_args()

    dataset, reports = FeatureV2WindowDataset.discover(args.path)
    summary = dataset.summary()
    summary["files"] = [
        {
            "path": report.path,
            "row_count": report.row_count,
            "participant_ids": list(report.participant_ids),
            "session_ids": list(report.session_ids),
            "labels": list(report.labels),
            "scenarios": list(report.scenarios),
            "schema_version": report.schema_version,
            "schema_hash": report.schema_hash,
        }
        for report in reports
    ]
    summary["sequence_features"] = list(SEQUENCE_FEATURES)
    summary["window_context_features"] = list(WINDOW_CONTEXT_FEATURES)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
