from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(
            Path(__file__)
            .resolve()
            .parents[1]
        ),
    )

from src.coevolution_v9.paper_protocol import (
    assert_paper_protocol,
    load_paper_protocol,
)
from src.data_v2 import (
    SessionValidationConfig,
    validate_dataset_root,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        default=(
            "configs/"
            "evaluation_protocol_v9_paper.json"
        ),
    )

    parser.add_argument(
        "--normal-root",
    )

    parser.add_argument(
        "--output",
    )

    args = parser.parse_args()

    protocol_path = Path(
        args.protocol
    ).resolve()

    protocol = load_paper_protocol(
        protocol_path
    )

    participants = None
    normal_validation = None

    if args.normal_root:
        report = validate_dataset_root(
            args.normal_root,
            config=SessionValidationConfig(
                require_complete_core_scenarios=True
            ),
        )

        normal_validation = (
            report.summary_dict()
        )

        if report.has_errors:
            raise RuntimeError(
                "Normal dataset failed validation"
            )

        participants = (
            normal_validation[
                "participants"
            ]
        )

    audit = assert_paper_protocol(
        protocol,
        participants=participants,
    )

    payload = {
        "status": "PASS",
        "protocol_path": str(
            protocol_path
        ),
        "protocol_sha256": sha256(
            protocol_path
        ),
        "audit": audit,
        "normal_validation": (
            normal_validation
        ),
    }

    if args.output:
        target = Path(args.output)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
