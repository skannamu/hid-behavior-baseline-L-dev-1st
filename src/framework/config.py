from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from exc


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    cfg['_config_path'] = str(path)
    return cfg


def get_cfg(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def require_cfg(cfg: Dict[str, Any], dotted_key: str) -> Any:
    value = get_cfg(cfg, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required config key: {dotted_key}")
    return value


def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
