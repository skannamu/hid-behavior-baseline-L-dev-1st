#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/home/js/hid_behavior_ai")

SOURCE_ROOT = (
    REPO
    / "experiments"
    / "seminar_n4_full_20260820_103101"
)

NORMAL_ROOT = Path(
    "/home/js/incoming_normal/paper_main_n4_clean"
)

PROTOCOL_PATH = (
    REPO
    / "configs"
    / "evaluation_protocol_v9_seminar_n4.json"
)

METHODS = (
    "recon_hid",
    "random_iterative",
    "static_mixed",
)

EXPECTED_SPLITS = 12

CHECKPOINTS = (1, 2, 3, 5)


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    obj: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def recursively_find_values(
    obj: Any,
    key: str,
) -> list[Any]:
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)

            found.extend(
                recursively_find_values(
                    v,
                    key,
                )
            )

    elif isinstance(obj, list):
        for v in obj:
            found.extend(
                recursively_find_values(
                    v,
                    key,
                )
            )

    return found


def recursively_collect_strings(
    obj: Any,
) -> list[str]:
    result = []

    if isinstance(obj, dict):
        for v in obj.values():
            result.extend(
                recursively_collect_strings(
                    v
                )
            )

    elif isinstance(obj, list):
        for v in obj:
            result.extend(
                recursively_collect_strings(
                    v
                )
            )

    elif isinstance(obj, str):
        result.append(obj)

    return result


def contains_string(
    obj: Any,
    needle: str,
) -> bool:
    if isinstance(obj, str):
        return (
            obj == needle
            or needle in obj
        )

    if isinstance(obj, dict):
        return any(
            contains_string(
                v,
                needle,
            )
            for v in obj.values()
        )

    if isinstance(obj, list):
        return any(
            contains_string(
                v,
                needle,
            )
            for v in obj
        )

    return False


def first_existing_path(
    values: list[Any],
) -> Path | None:
    for value in values:
        if not isinstance(
            value,
            str,
        ):
            continue

        p = Path(value)

        if p.exists():
            return p.resolve()

    return None


def extract_manifest(
    summary: dict[str, Any],
) -> Path:
    nm = summary.get(
        "normal_manifest"
    )

    candidates = []

    if isinstance(nm, dict):
        for key in (
            "jsonl",
            "manifest_jsonl",
            "path",
        ):
            if nm.get(key):
                candidates.append(
                    nm[key]
                )

    elif isinstance(nm, str):
        candidates.append(nm)

    candidates.extend(
        recursively_find_values(
            summary,
            "normal_manifest_path",
        )
    )

    path = first_existing_path(
        candidates
    )

    if path is None:
        raise RuntimeError(
            "Could not resolve normal manifest "
            "from source summary."
        )

    return path


def extract_seed(
    summary: dict[str, Any],
) -> int:
    co = summary.get(
        "coevolution",
        {},
    )

    # Preferred explicit locations.
    preferred = []

    if isinstance(co, dict):
        for key in (
            "seed",
            "base_seed",
        ):
            if key in co:
                preferred.append(
                    co[key]
                )

    for value in preferred:
        try:
            return int(value)
        except Exception:
            pass

    # Recover from first recorded round/probe metadata.
    timeline_path = None

    if isinstance(co, dict):
        tp = co.get(
            "timeline_path"
        )

        if tp:
            p = Path(tp)

            if p.exists():
                timeline_path = p

    if timeline_path:
        timeline = read_json(
            timeline_path
        )

        seeds = recursively_find_values(
            timeline,
            "base_seed",
        )

        for s in seeds:
            try:
                return int(s)
            except Exception:
                pass

    # N4 frozen protocol base seed.
    protocol = read_json(
        PROTOCOL_PATH
    )

    coevo = protocol[
        "coevolution"
    ]

    for key in (
        "base_seed",
        "seed",
    ):
        if key in coevo:
            return int(
                coevo[key]
            )

    raise RuntimeError(
        "Could not resolve co-evolution base seed."
    )


def extract_final_seed(
    summary: dict[str, Any],
    summary_path: Path,
) -> int:
    """
    Recover the ACTUAL A_final seed used by the frozen source run.

    Important:
    do not silently fall back to a config default because the runtime
    artifact is authoritative for reproducibility.
    """

    def direct_generation_seed(obj: Any) -> list[int]:
        found: list[int] = []

        if isinstance(obj, dict):
            generation = obj.get("generation_config")

            if (
                isinstance(generation, dict)
                and generation.get("seed") is not None
            ):
                try:
                    found.append(
                        int(generation["seed"])
                    )
                except Exception:
                    pass

            # Some summaries may explicitly record final_seed.
            if obj.get("final_seed") is not None:
                try:
                    found.append(
                        int(obj["final_seed"])
                    )
                except Exception:
                    pass

            for value in obj.values():
                found.extend(
                    direct_generation_seed(value)
                )

        elif isinstance(obj, list):
            for value in obj:
                found.extend(
                    direct_generation_seed(value)
                )

        return found


    # --------------------------------------------------------
    # 1. First inspect the experiment summary itself.
    # --------------------------------------------------------
    seeds = direct_generation_seed(
        summary.get(
            "final_evaluation",
            {}
        )
    )

    if seeds:
        unique = sorted(set(seeds))

        if len(unique) == 1:
            return unique[0]


    # --------------------------------------------------------
    # 2. Inspect JSON files explicitly referenced by the final
    #    evaluation result.
    # --------------------------------------------------------
    final_obj = summary.get(
        "final_evaluation",
        {}
    )

    referenced_jsons: list[Path] = []

    for value in recursively_collect_strings(
        final_obj
    ):
        try:
            candidate = Path(value)

            if (
                candidate.suffix.lower() == ".json"
                and candidate.is_file()
            ):
                referenced_jsons.append(
                    candidate.resolve()
                )
        except Exception:
            pass

    # --------------------------------------------------------
    # 3. Also inspect the frozen final_evaluation artifact tree
    #    belonging to this exact source job.
    # --------------------------------------------------------
    job_root = summary_path.parent.resolve()

    final_root = (
        job_root
        / "final_evaluation"
    )

    artifact_jsons: list[Path] = []

    if final_root.exists():
        artifact_jsons = sorted(
            final_root.rglob("*.json")
        )


    # Prefer likely final-summary files before arbitrary JSON.
    all_jsons = []

    seen = set()

    for candidate in (
        referenced_jsons
        + artifact_jsons
    ):
        key = str(candidate)

        if key not in seen:
            seen.add(key)
            all_jsons.append(candidate)

    all_jsons.sort(
        key=lambda path: (
            0
            if path.name in {
                "final_metrics.json",
                "final_evaluation.json",
                "evaluation_summary.json",
                "summary.json",
            }
            else 1,
            len(path.parts),
            str(path),
        )
    )


    artifact_hits: list[
        tuple[str, int]
    ] = []

    for json_path in all_jsons:
        try:
            obj = read_json(
                json_path
            )
        except Exception:
            continue

        values = direct_generation_seed(
            obj
        )

        for value in values:
            artifact_hits.append(
                (
                    str(json_path),
                    int(value),
                )
            )


    if artifact_hits:
        unique = sorted({
            seed
            for _, seed in artifact_hits
        })

        if len(unique) == 1:
            return unique[0]

        # If several values were found, prefer a direct
        # generation_config.seed from final_metrics.json.
        preferred = [
            seed
            for path, seed in artifact_hits
            if Path(path).name
            == "final_metrics.json"
        ]

        preferred_unique = sorted(
            set(preferred)
        )

        if len(preferred_unique) == 1:
            return preferred_unique[0]

        raise RuntimeError(
            "Ambiguous A_final seeds in frozen artifacts for "
            f"{summary_path}: "
            f"{artifact_hits}"
        )


    raise RuntimeError(
        "Could not resolve ACTUAL A_final seed from frozen "
        f"source experiment: {summary_path}"
    )


def discover_source_splits():
    split_dirs = sorted(
        p
        for p in SOURCE_ROOT.glob(
            "test_p*__cal_p*"
        )
        if p.is_dir()
    )

    if len(split_dirs) != EXPECTED_SPLITS:
        raise RuntimeError(
            f"Expected {EXPECTED_SPLITS} source splits, "
            f"found {len(split_dirs)}"
        )

    rows = []

    for split_dir in split_dirs:
        split = split_dir.name

        m = re.fullmatch(
            r"test_(p\d+)__cal_(p\d+)",
            split,
        )

        if not m:
            raise RuntimeError(
                f"Unexpected split name: {split}"
            )

        test_p, cal_p = m.groups()

        # Use one common D0 source for all three methods.
        source_summary_path = (
            split_dir
            / "recon_hid"
            / "experiment_summary.json"
        )

        if not source_summary_path.exists():
            raise FileNotFoundError(
                source_summary_path
            )

        summary = read_json(
            source_summary_path
        )

        if (
            summary.get("status")
            != "PASS"
        ):
            raise RuntimeError(
                f"Source job not PASS: "
                f"{source_summary_path}"
            )

        d0 = Path(
            summary["d0_run_dir"]
        ).resolve()

        if not d0.exists():
            raise FileNotFoundError(
                d0
            )

        manifest = extract_manifest(
            summary
        )

        base_seed = extract_seed(
            summary
        )

        final_seed = (
            extract_final_seed(
                summary,
                source_summary_path,
            )
        )

        # Verify method source jobs exist.
        method_summaries = {}

        for method in METHODS:
            p = (
                split_dir
                / method
                / "experiment_summary.json"
            )

            if not p.exists():
                raise FileNotFoundError(
                    p
                )

            x = read_json(p)

            if x.get("status") != "PASS":
                raise RuntimeError(
                    f"Source job not PASS: {p}"
                )

            method_summaries[
                method
            ] = x

        # Protocol says same final seed across methods.
        method_final_seeds = {
            method:
                extract_final_seed(
                    method_summaries[
                        method
                    ],
                    (
                        split_dir
                        / method
                        / "experiment_summary.json"
                    ),
                )
            for method in METHODS
        }

        if len(
            set(
                method_final_seeds.values()
            )
        ) != 1:
            raise RuntimeError(
                f"A_final seed mismatch across methods "
                f"for {split}: "
                f"{method_final_seeds}"
            )

        final_seed = next(
            iter(
                method_final_seeds.values()
            )
        )

        rows.append({
            "split": split,
            "test": test_p,
            "cal": cal_p,
            "d0_run_dir":
                str(d0),
            "normal_manifest":
                str(manifest),
            "base_seed":
                base_seed,
            "final_seed":
                final_seed,
        })

    return rows


def find_timeline_json(
    output_root: Path,
) -> Path:
    candidates = sorted(
        output_root.rglob(
            "*.json"
        )
    )

    matches = []

    for p in candidates:
        try:
            obj = read_json(p)
        except Exception:
            continue

        if (
            isinstance(obj, dict)
            and isinstance(
                obj.get("timeline"),
                list,
            )
        ):
            matches.append(p)

    if not matches:
        raise RuntimeError(
            f"No experiment-loop timeline found "
            f"under {output_root}"
        )

    # Prefer names explicitly containing timeline.
    matches.sort(
        key=lambda p: (
            "timeline" not in
            p.name.lower(),
            len(p.parts),
        )
    )

    return matches[0]


def find_defender_path(
    timeline: dict[str, Any],
    k: int,
) -> Path:
    pattern = re.compile(
        rf"(?:^|/|\\)D{k}(?:$|/|\\)"
    )

    candidates = []

    for s in recursively_collect_strings(
        timeline
    ):
        if not pattern.search(s):
            continue

        p = Path(s)

        if p.exists() and p.is_dir():
            candidates.append(
                p.resolve()
            )

    # De-duplicate.
    unique = []

    seen = set()

    for p in candidates:
        x = str(p)

        if x not in seen:
            seen.add(x)
            unique.append(p)

    if len(unique) == 1:
        return unique[0]

    # Prefer actual run directory named Dk.
    exact = [
        p
        for p in unique
        if p.name == f"D{k}"
    ]

    if len(exact) == 1:
        return exact[0]

    raise RuntimeError(
        f"Could not uniquely resolve D{k}. "
        f"Candidates={unique}"
    )


def make_prefix_timeline(
    *,
    original_path: Path,
    defender_path: Path,
    checkpoint: int,
    out_path: Path,
) -> Path:
    obj = read_json(
        original_path
    )

    items = obj.get(
        "timeline"
    )

    if not isinstance(
        items,
        list,
    ):
        raise RuntimeError(
            "Timeline has no list field 'timeline'."
        )

    target = str(
        defender_path
    )

    cut = None

    for i, item in enumerate(
        items
    ):
        if contains_string(
            item,
            target,
        ):
            cut = i + 1
            break

    if cut is None:
        # Fallback: locate any Dk path in item.
        token = f"D{checkpoint}"

        for i, item in enumerate(
            items
        ):
            strings = (
                recursively_collect_strings(
                    item
                )
            )

            if any(
                Path(s).name == token
                for s in strings
                if isinstance(s, str)
            ):
                cut = i + 1
                break

    if cut is None:
        raise RuntimeError(
            f"Could not determine timeline prefix "
            f"for D{checkpoint}"
        )

    prefix = copy.deepcopy(
        obj
    )

    prefix["timeline"] = (
        copy.deepcopy(
            items[:cut]
        )
    )

    # Critical:
    # fixed endpoints are NOT convergence-defined D_final.
    prefix.pop(
        "final_defender_run_dir",
        None,
    )

    prefix[
        "latest_defender_run_dir"
    ] = target

    prefix[
        "matched_budget_endpoint"
    ] = {
        "type":
            "fixed_round_endpoint",
        "checkpoint":
            checkpoint,
        "label":
            f"D{checkpoint}_fixed",
        "source_full_timeline":
            str(
                original_path.resolve()
            ),
    }

    write_json(
        out_path,
        prefix,
    )

    return out_path


def build_fixed_command(
    *,
    row,
    method,
    output_root,
    device,
):
    protocol = read_json(
        PROTOCOL_PATH
    )

    c = protocol[
        "coevolution"
    ]

    runner = (
        REPO
        / "scripts"
        / "run_v9_coevolution.py"
    )

    cmd = [
        sys.executable,
        str(runner),

        "--normal-root",
        str(NORMAL_ROOT),

        "--normal-manifest",
        row[
            "normal_manifest"
        ],

        "--d0-run-dir",
        row[
            "d0_run_dir"
        ],

        "--output-root",
        str(output_root),

        "--method",
        method,

        "--stopping-mode",
        "fixed",

        "--rounds",
        "5",

        "--candidates",
        str(
            c[
                "candidate_sessions_per_probe"
            ]
        ),

        "--keystrokes-per-candidate",
        str(
            c[
                "keystrokes_per_candidate"
            ]
        ),

        "--mutation-strength",
        str(
            c[
                "mutation_strength"
            ]
        ),

        "--max-hard-windows",
        str(
            c[
                "max_hard_windows_per_adaptation"
            ]
        ),

        "--max-parent-sessions",
        str(
            c[
                "max_parent_sessions"
            ]
        ),

        "--epochs",
        str(
            c[
                "hardening_epochs"
            ]
        ),

        "--batch-size",
        str(
            c[
                "hardening_batch_size"
            ]
        ),

        "--learning-rate",
        str(
            c[
                "hardening_learning_rate"
            ]
        ),

        "--target-fpr",
        "0.01",

        "--device",
        device,

        "--seed",
        str(
            row["base_seed"]
        ),
    ]

    return cmd


def run_fixed_trajectory(
    *,
    row,
    method,
    job_root,
    device,
):
    coevo_root = (
        job_root
        / "fixed_D5_trajectory"
    )

    coevo_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    marker = (
        coevo_root
        / "_MATCHED_FIXED_COMPLETE.json"
    )

    if marker.exists():
        info = read_json(marker)

        timeline = Path(
            info["timeline"]
        )

        if timeline.exists():
            print(
                f"[REUSE] {row['split']} "
                f"{method} fixed trajectory"
            )

            return timeline

    cmd = build_fixed_command(
        row=row,
        method=method,
        output_root=coevo_root,
        device=device,
    )

    command_path = (
        coevo_root
        / "command.json"
    )

    write_json(
        command_path,
        {
            "command": cmd,
            "split":
                row["split"],
            "method":
                method,
            "checkpoint_max":
                5,
        },
    )

    log_path = (
        coevo_root
        / "run.log"
    )

    print()
    print(
        "=" * 100
    )
    print(
        f"[RUN FIXED D5] "
        f"{row['split']} / {method}"
    )
    print(
        "=" * 100
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        tail = (
            log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            .splitlines()[-80:]
        )

        print(
            "\n".join(tail)
        )

        raise RuntimeError(
            f"Fixed trajectory failed: "
            f"{row['split']} {method}"
        )

    timeline = (
        find_timeline_json(
            coevo_root
        )
    )

    tl = read_json(
        timeline
    )

    defenders = {}

    for k in range(1, 6):
        defenders[
            str(k)
        ] = str(
            find_defender_path(
                tl,
                k,
            )
        )

    write_json(
        marker,
        {
            "status": "PASS",
            "timeline":
                str(
                    timeline.resolve()
                ),
            "timeline_sha256":
                sha256(timeline),
            "defenders":
                defenders,
        },
    )

    print(
        f"[PASS] fixed D1-D5 trajectory"
    )

    return timeline


def run_endpoint_evaluation(
    *,
    row,
    method,
    job_root,
    full_timeline,
    checkpoint,
    device,
):
    sys.path.insert(
        0,
        str(REPO),
    )

    from src.coevolution_v9 import (
        FinalEvaluationConfig,
        run_final_evaluation,
    )

    full_obj = read_json(
        full_timeline
    )

    defender = (
        find_defender_path(
            full_obj,
            checkpoint,
        )
    )

    endpoint_root = (
        job_root
        / "fixed_endpoints"
        / f"D{checkpoint}_fixed"
    )

    endpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        endpoint_root
        / "matched_budget_summary.json"
    )

    if result_path.exists():
        old = read_json(
            result_path
        )

        if old.get("status") == "PASS":
            print(
                f"[REUSE] "
                f"{row['split']} {method} "
                f"D{checkpoint}"
            )

            return old

    prefix_path = (
        endpoint_root
        / f"D{checkpoint}_fixed_timeline.json"
    )

    make_prefix_timeline(
        original_path=full_timeline,
        defender_path=defender,
        checkpoint=checkpoint,
        out_path=prefix_path,
    )

    protocol = read_json(
        PROTOCOL_PATH
    )

    final = protocol[
        "final_evaluation"
    ]

    print(
        f"[EVAL] {row['split']} "
        f"{method} D{checkpoint}_fixed"
    )

    result = run_final_evaluation(
        final_defender_run_dir=
            defender,

        normal_dataset_root=
            NORMAL_ROOT,

        normal_manifest_path=
            row[
                "normal_manifest"
            ],

        coevolution_timeline_path=
            prefix_path,

        output_dir=
            endpoint_root
            / "final_evaluation",

        config=FinalEvaluationConfig(
            seed=int(
                row["final_seed"]
            ),

            candidates=int(
                final[
                    "candidate_sessions"
                ]
            ),

            keystrokes_per_candidate=int(
                final[
                    "keystrokes_per_candidate"
                ]
            ),

            mutation_strength=float(
                final[
                    "mutation_strength"
                ]
            ),

            batch_size=int(
                protocol[
                    "coevolution"
                ][
                    "hardening_batch_size"
                ]
            ),

            device=device,
        ),
    )

    if result.get("status") != "PASS":
        raise RuntimeError(
            f"D{checkpoint} final evaluation "
            f"did not PASS"
        )

    if (
        result.get(
            "lineage_audit_status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            f"D{checkpoint} lineage audit "
            f"did not PASS"
        )

    summary = {
        "status":
            "PASS",

        "evaluation_type":
            "fixed_round_matched_budget",

        "endpoint":
            f"D{checkpoint}_fixed",

        "checkpoint":
            checkpoint,

        "split":
            row["split"],

        "test":
            row["test"],

        "cal":
            row["cal"],

        "method":
            method,

        "base_seed":
            row["base_seed"],

        "final_seed":
            row["final_seed"],

        "d0_run_dir":
            row["d0_run_dir"],

        "defender_run_dir":
            str(defender),

        "full_fixed_timeline":
            str(
                full_timeline
            ),

        "endpoint_timeline":
            str(
                prefix_path
            ),

        "final_evaluation":
            result,
    }

    write_json(
        result_path,
        summary,
    )

    print(
        f"[PASS] {row['split']} "
        f"{method} D{checkpoint}_fixed"
    )

    return summary


def run_one_job(
    *,
    row,
    method,
    output_root,
    device,
):
    job_root = (
        output_root
        / row["split"]
        / method
    )

    job_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_timeline = (
        run_fixed_trajectory(
            row=row,
            method=method,
            job_root=job_root,
            device=device,
        )
    )

    results = []

    for checkpoint in CHECKPOINTS:
        results.append(
            run_endpoint_evaluation(
                row=row,
                method=method,
                job_root=job_root,
                full_timeline=
                    full_timeline,
                checkpoint=
                    checkpoint,
                device=device,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--split",
        default=None,
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
        default=None,
    )

    args = parser.parse_args()

    output_root = Path(
        args.output_root
    ).resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    protocol = read_json(
        PROTOCOL_PATH
    )

    declared = tuple(
        int(x)
        for x in protocol[
            "matched_budget"
        ][
            "fixed_round_checkpoints"
        ]
    )

    if declared != CHECKPOINTS:
        raise RuntimeError(
            f"Protocol checkpoints {declared} "
            f"!= expected {CHECKPOINTS}"
        )

    rows = discover_source_splits()

    write_json(
        output_root
        / "matched_budget_protocol_snapshot.json",
        {
            "created_at":
                datetime.now().isoformat(),

            "source_protocol":
                str(PROTOCOL_PATH),

            "source_protocol_sha256":
                sha256(
                    PROTOCOL_PATH
                ),

            "source_convergence_experiment":
                str(SOURCE_ROOT),

            "normal_root":
                str(NORMAL_ROOT),

            "checkpoints":
                list(CHECKPOINTS),

            "methods":
                list(METHODS),

            "splits":
                rows,

            "primary_rule":
                (
                    "Same fixed adaptation round "
                    "budget across methods; "
                    "fixed endpoints are not D_final."
                ),
        },
    )

    if args.smoke:
        rows = rows[:1]
        methods = (
            args.method
            if args.method
            else "recon_hid",
        )

    else:
        if args.split:
            rows = [
                x
                for x in rows
                if x["split"]
                == args.split
            ]

            if not rows:
                raise RuntimeError(
                    f"Unknown split: "
                    f"{args.split}"
                )

        methods = (
            (args.method,)
            if args.method
            else METHODS
        )

    total_jobs = (
        len(rows)
        * len(methods)
    )

    print(
        "=" * 100
    )
    print(
        "N4 MATCHED-BUDGET"
    )
    print(
        "=" * 100
    )
    print(
        f"source root : {SOURCE_ROOT}"
    )
    print(
        f"output root : {output_root}"
    )
    print(
        f"splits      : {len(rows)}"
    )
    print(
        f"methods     : {methods}"
    )
    print(
        f"jobs        : {total_jobs}"
    )
    print(
        f"checkpoints : {CHECKPOINTS}"
    )
    print()

    completed = 0

    for row in rows:
        for method in methods:
            run_one_job(
                row=row,
                method=method,
                output_root=
                    output_root,
                device=args.device,
            )

            completed += 1

            print(
                f"[JOB PASS] "
                f"{completed}/{total_jobs}"
            )

    print()
    print(
        "[PASS] matched-budget runner complete"
    )


if __name__ == "__main__":
    main()
