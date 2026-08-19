"""Orchestrate Stable-v9 attack/defense co-evolution rounds."""

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
from .weakness_miner import (
    WeaknessMiningConfig,
    mine_v9_weaknesses,
)


ROUND_FORMAT_VERSION = "recon_hid_v9_coevolution_round_v1"
PROBE_FORMAT_VERSION = "recon_hid_v9_coevolution_probe_v1"

SUPPORTED_METHODS = {
    "recon_hid",
    "random_iterative",
    "static_mixed",
}


@dataclass(frozen=True)
class CoevolutionRoundConfig:
    method: str = "recon_hid"
    attack_generation: RawAttackGenerationConfig = (
        RawAttackGenerationConfig()
    )
    weakness_mining: WeaknessMiningConfig = (
        WeaknessMiningConfig()
    )
    hardened_training: HardenedTrainingConfig = (
        HardenedTrainingConfig()
    )


def _validate_method(method: str) -> None:
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported experiment method: {method}"
        )


def probe_coevolution_round(
    *,
    round_index: int,
    parent_defender_run_dir: str | Path,
    normal_dataset_root: str | Path,
    normal_manifest_path: str | Path,
    output_root: str | Path,
    parent_policy_path: str | Path | None = None,
    config: CoevolutionRoundConfig | None = None,
    probe_label: str | None = None,
    generation_override: int | None = None,
) -> dict[str, Any]:
    """Generate A_k and evaluate/mine it against the current D_k.

    This phase never trains D_{k+1}. It therefore provides the point at
    which a convergence controller may terminate the loop and freeze D_k.
    """
    if round_index < 0:
        raise ValueError(
            "round_index must be non-negative"
        )

    cfg = config or CoevolutionRoundConfig()
    _validate_method(cfg.method)

    root_name = f"round_{round_index:02d}"

    if probe_label is not None:
        root_name += f"_{probe_label}"

    root = (
        Path(output_root).resolve()
        / root_name
    )

    if root.exists():
        raise FileExistsError(root)

    root.mkdir(parents=True)

    attack_name = f"A{round_index}"

    if probe_label is not None:
        attack_name += f"_{probe_label}"

    generation_index = (
        round_index
        if generation_override is None
        else generation_override
    )

    generation_cfg = replace(
        cfg.attack_generation,
        generation=generation_index,
        parent_policy_path=(
            str(parent_policy_path)
            if parent_policy_path is not None
            else None
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

    probe_manifest = {
        "probe_format_version": PROBE_FORMAT_VERSION,
        "experiment_method": cfg.method,
        "round_index": round_index,
        "probe_label": probe_label,
        "attack_generation_index": generation_index,
        "attack_id": attack_name,
        "parent_defender_run_dir": str(
            Path(parent_defender_run_dir)
        ),
        "normal_dataset_root": str(
            Path(normal_dataset_root)
        ),
        "normal_manifest_path": str(
            Path(normal_manifest_path)
        ),
        "normal_manifest_sha256": sha256_file(
            normal_manifest_path
        ),
        "parent_policy_path": (
            str(Path(parent_policy_path))
            if parent_policy_path is not None
            else None
        ),
        "config": {
            "attack_generation": asdict(
                generation_cfg
            ),
            "weakness_mining": asdict(
                cfg.weakness_mining
            ),
            "hardened_training": asdict(
                cfg.hardened_training
            ),
        },
        "attack_generation": attack_info,
        "weakness_mining": weakness_info,
        "next_parent_policy_path": (
            weakness_info["parent_policy_path"]
        ),
        "methodology": {
            "attack_generation_level": "raw_event",
            "stable_extractor_required": True,
            "normal_threshold_source": (
                "calibration_only"
            ),
            "normal_test_used_during_round": False,
            "final_holdout_attack_used_during_round": False,
            "window_selection_mode": (
                cfg.weakness_mining.selection_mode
            ),
            "parent_selection_mode": (
                cfg.weakness_mining.parent_selection_mode
            ),
            "parent_policy_used_for_generation": (
                generation_cfg.parent_policy_path
                is not None
            ),
            "defender_hardened": False,
        },
    }

    probe_path = atomic_json(
        root / "probe_manifest.json",
        probe_manifest,
    )

    return {
        **probe_manifest,
        "round_dir": str(root),
        "probe_manifest_path": str(probe_path),
    }


def harden_coevolution_probe(
    *,
    probe_result: dict[str, Any],
    config: CoevolutionRoundConfig | None = None,
) -> dict[str, Any]:
    """Train D_{k+1} from an already completed A_k probe."""
    cfg = config or CoevolutionRoundConfig()
    _validate_method(cfg.method)

    if (
        str(probe_result["experiment_method"])
        != cfg.method
    ):
        raise ValueError(
            "Probe/config experiment method mismatch"
        )

    round_index = int(
        probe_result["round_index"]
    )

    root = Path(
        str(probe_result["round_dir"])
    ).resolve()

    if not root.is_dir():
        raise FileNotFoundError(root)

    round_manifest_path = (
        root / "round_manifest.json"
    )

    if round_manifest_path.exists():
        raise FileExistsError(
            round_manifest_path
        )

    defender_id = f"D{round_index + 1}"

    weakness_info = probe_result[
        "weakness_mining"
    ]

    training_result = train_hardened_defender_v9(
        normal_dataset_root=(
            probe_result["normal_dataset_root"]
        ),
        normal_manifest_path=(
            probe_result["normal_manifest_path"]
        ),
        parent_run_dir=(
            probe_result["parent_defender_run_dir"]
        ),
        attack_paths=(
            weakness_info["hard_window_path"]
        ),
        output_root=root / "defenders",
        defender_id=defender_id,
        run_name=defender_id,
        config=cfg.hardened_training,
    )

    methodology = dict(
        probe_result["methodology"]
    )
    methodology[
        "defender_hardened"
    ] = True

    manifest = {
        "round_format_version": (
            ROUND_FORMAT_VERSION
        ),
        "experiment_method": cfg.method,
        "round_index": round_index,
        "attack_id": (
            probe_result["attack_id"]
        ),
        "defender_id": defender_id,
        "parent_defender_run_dir": (
            probe_result[
                "parent_defender_run_dir"
            ]
        ),
        "normal_dataset_root": (
            probe_result["normal_dataset_root"]
        ),
        "normal_manifest_path": (
            probe_result["normal_manifest_path"]
        ),
        "normal_manifest_sha256": (
            probe_result[
                "normal_manifest_sha256"
            ]
        ),
        "parent_policy_path": (
            probe_result["parent_policy_path"]
        ),
        "config": {
            "attack_generation": (
                probe_result["config"][
                    "attack_generation"
                ]
            ),
            "weakness_mining": (
                probe_result["config"][
                    "weakness_mining"
                ]
            ),
            "hardened_training": asdict(
                cfg.hardened_training
            ),
        },
        "attack_generation": (
            probe_result["attack_generation"]
        ),
        "weakness_mining": weakness_info,
        "training": asdict(
            training_result
        ),
        "probe_manifest_path": (
            probe_result[
                "probe_manifest_path"
            ]
        ),
        "next_parent_policy_path": (
            weakness_info[
                "parent_policy_path"
            ]
        ),
        "next_defender_run_dir": (
            training_result.run_dir
        ),
        "methodology": methodology,
    }

    manifest_path = atomic_json(
        round_manifest_path,
        manifest,
    )

    return {
        **manifest,
        "round_dir": str(root),
        "round_manifest_path": str(
            manifest_path
        ),
    }


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
    """Compatibility wrapper: probe A_k, then always harden to D_{k+1}."""
    cfg = config or CoevolutionRoundConfig()

    probe = probe_coevolution_round(
        round_index=round_index,
        parent_defender_run_dir=(
            parent_defender_run_dir
        ),
        normal_dataset_root=(
            normal_dataset_root
        ),
        normal_manifest_path=(
            normal_manifest_path
        ),
        output_root=output_root,
        parent_policy_path=(
            parent_policy_path
        ),
        config=cfg,
    )

    return harden_coevolution_probe(
        probe_result=probe,
        config=cfg,
    )
