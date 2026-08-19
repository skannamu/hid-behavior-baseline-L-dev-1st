from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path,
) -> None:
    print(
        "RUN:",
        " ".join(command),
    )

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--protocol",
        default=(
            "configs/"
            "evaluation_protocol_v9_paper.json"
        ),
    )

    p.add_argument(
        "--run-tests",
        action="store_true",
    )

    p.add_argument(
        "--run-smoke",
        action="store_true",
    )

    p.add_argument(
        "--require-clean-git",
        action="store_true",
    )

    args = p.parse_args()

    repo = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    protocol_path = (
        repo / args.protocol
    ).resolve()

    protocol = load_paper_protocol(
        protocol_path
    )

    protocol_audit = (
        assert_paper_protocol(
            protocol
        )
    )

    run(
        ["git", "diff", "--check"],
        cwd=repo,
    )

    if args.run_tests:
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--import-mode=importlib",
                "tests",
            ],
            cwd=repo,
        )

    if args.run_smoke:
        run(
            [
                sys.executable,
                "scripts/"
                "dry_run_v9_convergence.py",
            ],
            cwd=repo,
        )

    git_commit = (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=repo,
            text=True,
        )
        .strip()
    )

    git_status = (
        subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=repo,
            text=True,
        )
    )

    if (
        args.require_clean_git
        and git_status.strip()
    ):
        raise RuntimeError(
            "Git working tree is not clean"
        )

    payload = {
        "status": "PASS",
        "framework": "ReCon-HID Stable-v9",
        "feature_schema_version": "2.0.0",
        "protocol_id": (
            protocol["protocol_id"]
        ),
        "protocol_sha256": sha256(
            protocol_path
        ),
        "protocol_audit": (
            protocol_audit
        ),
        "git_commit": git_commit,
        "git_clean": (
            not bool(
                git_status.strip()
            )
        ),
        "tests_executed": (
            args.run_tests
        ),
        "e2e_smoke_executed": (
            args.run_smoke
        ),
    }

    output = (
        repo
        / "experiments"
        / "framework_readiness_audit.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
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
