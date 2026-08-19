from __future__ import annotations

import argparse
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
    build_paper_jobs,
    load_paper_protocol,
)
from src.data_v2 import (
    SessionValidationConfig,
    validate_dataset_root,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Plan or execute the frozen "
            "30-split ReCon-HID paper matrix."
        )
    )

    p.add_argument(
        "--normal-root",
        required=True,
    )

    p.add_argument(
        "--output-root",
        required=True,
    )

    p.add_argument(
        "--protocol",
        default=(
            "configs/"
            "evaluation_protocol_v9_paper.json"
        ),
    )

    p.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "recon_hid",
            "random_iterative",
            "static_mixed",
        ),
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--execute",
        action="store_true",
    )

    p.add_argument(
        "--resume",
        action="store_true",
    )

    p.add_argument(
        "--start-job",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-jobs",
        type=int,
    )

    args = p.parse_args()

    normal_root = Path(
        args.normal_root
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    protocol = load_paper_protocol(
        args.protocol
    )

    validation = validate_dataset_root(
        normal_root,
        config=SessionValidationConfig(
            require_complete_core_scenarios=True
        ),
    )

    if validation.has_errors:
        raise RuntimeError(
            "Normal dataset validation failed"
        )

    summary = validation.summary_dict()

    participants = tuple(
        summary["participants"]
    )

    audit = assert_paper_protocol(
        protocol,
        participants=participants,
    )

    methods = (
        tuple(args.methods)
        if args.methods
        else tuple(
            protocol["methods"]
        )
    )

    jobs = build_paper_jobs(
        protocol,
        participants,
        methods=methods,
    )

    d0 = protocol["d0"]
    normal = protocol["normal_data"]
    coevo = protocol["coevolution"]
    conv = protocol["convergence"]
    final = protocol[
        "final_evaluation"
    ]

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    runner = (
        repo_root
        / "scripts"
        / "run_v9_full_experiment.py"
    )

    planned: list[
        dict[str, object]
    ] = []

    for job_index, job in enumerate(jobs):
        job_output = (
            output_root
            / job["assignment_id"]
            / job["method"]
        )

        command = [
            sys.executable,
            str(runner),

            "--normal-root",
            str(normal_root),

            "--output-root",
            str(job_output),

            "--method",
            str(job["method"]),

            "--device",
            args.device,

            "--seed",
            str(job["seed"]),

            "--train-groups",
            *[
                str(value)
                for value in job[
                    "train_groups"
                ]
            ],

            "--calibration-group",
            str(
                job[
                    "calibration_group"
                ]
            ),

            "--test-group",
            str(
                job["test_group"]
            ),

            "--d0-epochs",
            str(d0["epochs"]),

            "--d0-batch-size",
            str(d0["batch_size"]),

            "--d0-learning-rate",
            str(d0["learning_rate"]),

            "--d0-weight-decay",
            str(d0["weight_decay"]),

            "--d0-patience",
            str(d0["patience"]),

            "--target-fpr",
            str(d0["target_fpr"]),

            "--training-window-stride",
            str(
                normal[
                    "training_window_stride"
                ]
            ),

            "--training-max-windows-per-group",
            str(
                normal[
                    "training_max_windows_per_group"
                ]
            ),

            "--candidates",
            str(
                coevo[
                    "candidate_sessions_per_probe"
                ]
            ),

            "--keystrokes-per-candidate",
            str(
                coevo[
                    "keystrokes_per_candidate"
                ]
            ),

            "--mutation-strength",
            str(
                coevo[
                    "mutation_strength"
                ]
            ),

            "--max-hard-windows",
            str(
                coevo[
                    "max_hard_windows_per_adaptation"
                ]
            ),

            "--max-parent-sessions",
            str(
                coevo[
                    "max_parent_sessions"
                ]
            ),

            "--hardening-epochs",
            str(
                coevo[
                    "hardening_epochs"
                ]
            ),

            "--hardening-batch-size",
            str(
                coevo[
                    "hardening_batch_size"
                ]
            ),

            "--hardening-learning-rate",
            str(
                coevo[
                    "hardening_learning_rate"
                ]
            ),

            "--min-rounds",
            str(
                conv[
                    "min_defender_stages"
                ]
            ),

            "--max-rounds",
            str(
                conv[
                    "max_defender_stages"
                ]
            ),

            "--global-bypass-threshold",
            str(
                conv[
                    "global_bypass_threshold"
                ]
            ),

            "--family-bypass-threshold",
            str(
                conv[
                    "family_bypass_threshold"
                ]
            ),

            "--convergence-patience",
            str(
                conv[
                    "fresh_probes_per_defender"
                ]
            ),

            "--final-seed",
            str(job["final_seed"]),

            "--final-candidates",
            str(
                final[
                    "candidate_sessions"
                ]
            ),

            "--final-keystrokes-per-candidate",
            str(
                final[
                    "keystrokes_per_candidate"
                ]
            ),

            "--final-mutation-strength",
            str(
                final[
                    "mutation_strength"
                ]
            ),
        ]

        planned.append({
            "job_index": job_index,
            **job,
            "output_root": str(
                job_output
            ),
            "command": command,
        })

    plan = {
        "status": "PLANNED",
        "protocol_id": (
            protocol["protocol_id"]
        ),
        "participants": list(
            participants
        ),
        "split_count": 30,
        "method_count": len(
            methods
        ),
        "job_count": len(
            planned
        ),
        "methods": list(methods),
        "protocol_audit": audit,
        "normal_validation": summary,
        "jobs": planned,
    }

    plan_path = (
        output_root
        / "paper_matrix_plan.json"
    )

    plan_path.write_text(
        json.dumps(
            plan,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "PLANNED",
                "split_count": 30,
                "method_count": len(
                    methods
                ),
                "job_count": len(
                    planned
                ),
                "plan_path": str(
                    plan_path
                ),
                "execute": args.execute,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not args.execute:
        return

    selected = planned[
        args.start_job:
    ]

    if args.max_jobs is not None:
        selected = selected[
            :args.max_jobs
        ]

    for row in selected:
        job_output = Path(
            row["output_root"]
        )

        summary_path = (
            job_output
            / "experiment_summary.json"
        )

        if job_output.exists():
            if (
                args.resume
                and summary_path.is_file()
            ):
                existing = json.loads(
                    summary_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    existing.get("status")
                    == "PASS"
                ):
                    print(
                        "SKIP PASS:",
                        row["job_id"],
                    )
                    continue

            raise FileExistsError(
                "Existing incomplete/non-PASS "
                f"job output: {job_output}"
            )

        print(
            "RUN:",
            row["job_index"],
            row["job_id"],
        )

        subprocess.run(
            row["command"],
            cwd=repo_root,
            check=True,
        )


if __name__ == "__main__":
    main()
