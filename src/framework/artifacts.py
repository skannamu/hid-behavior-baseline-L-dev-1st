from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from src.framework.config import require_cfg


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_framework_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    checkpoints_root = ensure_dir(require_cfg(cfg, 'paths.checkpoints_root'))
    experiments_root = ensure_dir(require_cfg(cfg, 'paths.experiments_root'))
    attack_root = ensure_dir(require_cfg(cfg, 'paths.attack_generated_root'))
    paths = {
        'checkpoints_root': checkpoints_root,
        'experiments_root': experiments_root,
        'attack_generated_root': attack_root,
        'logs': ensure_dir(experiments_root / 'logs'),
        'rounds': ensure_dir(experiments_root / 'rounds'),
        'tables': ensure_dir(experiments_root / 'tables'),
        'figures': ensure_dir(experiments_root / 'figures'),
        'artifacts': ensure_dir(experiments_root / 'artifacts'),
    }
    for defender_id in ['D0', 'D1', 'D2', 'D_final']:
        ensure_dir(checkpoints_root / defender_id)
    return paths


def defender_dir(cfg: Dict[str, Any], defender_id: str) -> Path:
    return ensure_dir(Path(require_cfg(cfg, 'paths.checkpoints_root')) / defender_id)


def round_dir(cfg: Dict[str, Any], name: str) -> Path:
    return ensure_dir(Path(require_cfg(cfg, 'paths.experiments_root')) / 'rounds' / name)


def attack_round_dir(cfg: Dict[str, Any], name: str) -> Path:
    return ensure_dir(Path(require_cfg(cfg, 'paths.attack_generated_root')) / name)
