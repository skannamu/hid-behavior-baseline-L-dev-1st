"""Orchestrate one Stable-v9 attack/defense co-evolution round."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .common import atomic_json, sha256_file
from .hardened_trainer import (
    HardenedTrainingConfig,
    train_hardened_defender_v9,
)
from .raw_attack_generator import (
    RawAttackGenerationConfig,
    generate_raw_attack_pool,
)
from .weakness_miner import WeaknessMiningConfig, mine_v9_weaknesses


ROUND_FORMAT_VERSION = "recon_hid_v9_coevolution_round_v1"


@dataclass(frozen=True)
class CoevolutionRoundConfig:
    method: str = "recon_hid"
    attack_generation: RawAttackGenerationConfig = RawAttackGenerationConfig()
    weakness_mining: WeaknessMiningConfig = WeaknessMiningConfig()
    hardened_training: HardenedTrainingConfig = HardenedTrainingConfig()


def run_coevolution_round(
    *,
    round_index: int,
    parent_defender_run_dir: str | Path,
    normal_dataset_root: str | Path,
    normal_manifest_path: str | Path,
    output_root: str | Path,
    parent_policy_path: str | Path | None = None,
    config: CoevolutionRoundConfig | None = None,
) -> dict[str, Any]:
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    cfg = config or CoevolutionRoundConfig()

    if cfg.method not in {
        "recon_hid",
        "random_iterative",
        "static_mixed",
    }:
        raise ValueError(
            f"Unsupported experiment method: {cfg.method}"
        )

    root = Path(output_root).resolve() / f"round_{round_index:02d}"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    attack_name = f"A{round_index}"
    defender_id = f"D{round_index + 1}"
    generation_cfg = replace(
        cfg.attack_generation,
        generation=round_index,
        parent_policy_path=(
            str(parent_policy_path) if parent_policy_path is not None else None
        ),
    )
    attack_info = generate_raw_attack_pool(
        output_root=root / "attacks",
        round_name=attack_name,
        config=generation_cfg,
    )
    weakness_info = mine_v9_weaknesses(
        defender_run_dir=parent_defender_run_dir,
        attack_manifest_path=attack_info["manifest_path"],
        output_dir=root / "weakness",
        config=cfg.weakness_mining,
    )
    training_result = train_hardened_defender_v9(
        normal_dataset_root=normal_dataset_root,
        normal_manifest_path=normal_manifest_path,
        parent_run_dir=parent_defender_run_dir,
        attack_paths=weakness_info["hard_window_path"],
        output_root=root / "defenders",
        defender_id=defender_id,
        run_name=defender_id,
        config=cfg.hardened_training,
    )

    manifest = {
        "round_format_version": ROUND_FORMAT_VERSION,
        "experiment_method": cfg.method,
        "round_index": round_index,
        "attack_id": attack_name,
        "defender_id": defender_id,
        "parent_defender_run_dir": str(Path(parent_defender_run_dir)),
        "normal_dataset_root": str(Path(normal_dataset_root)),
        "normal_manifest_path": str(Path(normal_manifest_path)),
        "normal_manifest_sha256": sha256_file(normal_manifest_path),
        "parent_policy_path": (
            str(Path(parent_policy_path)) if parent_policy_path is not None else None
        ),
        "config": {
            "attack_generation": asdict(generation_cfg),
            "weakness_mining": asdict(cfg.weakness_mining),
            "hardened_training": asdict(cfg.hardened_training),
        },
        "attack_generation": attack_info,
        "weakness_mining": weakness_info,
        "training": asdict(training_result),
        "next_parent_policy_path": weakness_info["parent_policy_path"],
        "next_defender_run_dir": training_result.run_dir,
        "methodology": {
            "attack_generation_level": "raw_event",
            "stable_extractor_required": True,
            "normal_threshold_source": "calibration_only",
            "normal_test_used_during_round": False,
            "final_holdout_attack_used_during_round": False,
            "window_selection_mode": (
                cfg.weakness_mining.selection_mode
            ),
            "parent_selection_mode": (
                cfg.weakness_mining.parent_selection_mode
            ),
            "parent_policy_used_for_generation": (
                generation_cfg.parent_policy_path is not None
            ),
        },
    }
    manifest_path = atomic_json(root / "round_manifest.json", manifest)
    return {
        **manifest,
        "round_dir": str(root),
        "round_manifest_path": str(manifest_path),
    }
