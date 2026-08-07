from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.coevolution_v9 import FinalEvaluationConfig, run_final_evaluation


def _tuple(value):
    if value is None:
        return None
    return tuple(str(item) for item in value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze D_final and run one-shot held-out V_final evaluation."
    )
    parser.add_argument("--final-defender-run-dir", required=True)
    parser.add_argument("--normal-dataset-root", required=True)
    parser.add_argument("--normal-manifest", required=True)
    parser.add_argument("--coevolution-timeline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default="configs/final_evaluation_v9.yaml",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    section = payload.get("final_evaluation", payload)
    cfg = FinalEvaluationConfig(
        seed=int(section.get("seed", 920260807)),
        candidates=int(section.get("candidates", 24)),
        keystrokes_per_candidate=int(
            section.get("keystrokes_per_candidate", 84)
        ),
        mutation_strength=float(section.get("mutation_strength", 0.10)),
        families=_tuple(section.get("families"))
        or FinalEvaluationConfig().families,
        batch_size=int(section.get("batch_size", 256)),
        device=str(args.device or section.get("device", "cpu")),
        verify_normal_manifest_hashes=bool(
            section.get("verify_normal_manifest_hashes", True)
        ),
        require_timeline_defender_match=bool(
            section.get("require_timeline_defender_match", True)
        ),
        require_disjoint_seeds=bool(
            section.get("require_disjoint_seeds", True)
        ),
        require_disjoint_families=bool(
            section.get("require_disjoint_families", True)
        ),
        require_null_parent_lineage=bool(
            section.get("require_null_parent_lineage", True)
        ),
    )
    result = run_final_evaluation(
        final_defender_run_dir=args.final_defender_run_dir,
        normal_dataset_root=args.normal_dataset_root,
        normal_manifest_path=args.normal_manifest,
        coevolution_timeline_path=args.coevolution_timeline,
        output_dir=args.output_dir,
        config=cfg,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
