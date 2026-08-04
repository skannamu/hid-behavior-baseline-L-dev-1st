from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import shutil

from src.framework.config import require_cfg, dump_json
from src.framework.artifacts import defender_dir


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def export_final_detector(
    cfg: Dict[str, Any],
    source_defender_id: str = "D1",
    final_id: str = "D_final",
) -> Dict[str, Any]:
    src_dir = defender_dir(cfg, source_defender_id)
    final_dir = defender_dir(cfg, final_id)

    model_src = src_dir / "model.pt"
    if not model_src.exists():
        raise FileNotFoundError(f"Source defender model not found: {model_src}")

    copied = {}

    copied["model"] = _copy_if_exists(model_src, final_dir / "model.pt")
    copied["thresholds"] = _copy_if_exists(src_dir / "thresholds.json", final_dir / "thresholds.json")
    copied["defender_manifest"] = _copy_if_exists(src_dir / "defender_manifest.json", final_dir / "defender_manifest.json")
    copied["policy_config"] = _copy_if_exists(src_dir / "policy_config.json", final_dir / "policy_config.json")

    scaler_src = src_dir / "scaler.pkl"
    if not scaler_src.exists():
        scaler_src = Path(require_cfg(cfg, "paths.checkpoints_root")) / "D0" / "scaler.pkl"

    copied["scaler"] = _copy_if_exists(scaler_src, final_dir / "scaler.pkl")

    manifest = {
        "final_id": final_id,
        "source_defender_id": source_defender_id,
        "source_dir": str(src_dir),
        "final_dir": str(final_dir),
        "model_path": str(final_dir / "model.pt"),
        "scaler_path": str(final_dir / "scaler.pkl"),
        "thresholds_path": str(final_dir / "thresholds.json"),
        "policy_config_path": str(final_dir / "policy_config.json"),
        "copied": copied,
    }

    manifest_path = final_dir / "final_detector_manifest.json"
    dump_json(manifest_path, manifest)

    print("[OK] D_final exported")
    print(f"  - source: {src_dir}")
    print(f"  - final : {final_dir}")

    return {
        "final_id": final_id,
        "source_defender_id": source_defender_id,
        "final_dir": str(final_dir),
        "model_path": str(final_dir / "model.pt"),
        "manifest_path": str(manifest_path),
        "policy_config_path": str(final_dir / "policy_config.json"),
    }
