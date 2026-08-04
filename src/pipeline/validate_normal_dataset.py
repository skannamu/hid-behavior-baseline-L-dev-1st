from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.framework.config import require_cfg
from src.core.dataset import SELECTED_FEATURES, WINDOW_SIZE


def validate_normal_dataset(cfg: Dict[str, Any]) -> dict:
    csv_path = Path(require_cfg(cfg, 'paths.normal_window_csv'))
    if not csv_path.exists():
        raise FileNotFoundError(f'Normal window CSV not found: {csv_path}')
    df = pd.read_csv(csv_path)
    required = [f't{t}_{f}' for t in range(WINDOW_SIZE) for f in SELECTED_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required feature columns: {missing[:20]}')
    nan_count = int(df[required].isna().sum().sum())
    summary = {
        'path': str(csv_path),
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'missing_feature_columns': len(missing),
        'nan_count_in_features': nan_count,
        'participant_count': int(df['participant_id'].nunique()) if 'participant_id' in df.columns else None,
        'session_count': int(df['session_id'].nunique()) if 'session_id' in df.columns else None,
    }
    print('[OK] Normal dataset validation')
    for k, v in summary.items():
        print(f'  - {k}: {v}')
    return summary
