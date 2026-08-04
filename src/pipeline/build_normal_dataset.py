from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from src.framework.config import require_cfg, get_cfg, dump_json
from src.core.dataset import SELECTED_FEATURES, WINDOW_SIZE


def _required_window_columns() -> List[str]:
    return [f't{t}_{feature}' for t in range(WINDOW_SIZE) for feature in SELECTED_FEATURES]


def find_window_csvs(normal_raw_root: str | Path) -> List[Path]:
    root = Path(normal_raw_root)
    if not root.exists():
        return []
    return sorted(root.glob('**/window/window.csv'))


def _infer_ids(root: Path, csv_path: Path) -> dict:
    rel = csv_path.relative_to(root)
    parts = rel.parts
    participant_id = parts[0] if len(parts) >= 4 else 'unknown_participant'
    session_id = parts[1] if len(parts) >= 4 else csv_path.parent.parent.name
    return {
        'participant_id': participant_id,
        'session_id': session_id,
        'source_path': str(csv_path),
        'source_file': csv_path.name,
    }


def build_normal_dataset(cfg: Dict[str, Any], allow_empty: bool = False) -> Path | None:
    normal_raw_root = Path(require_cfg(cfg, 'paths.normal_raw_root'))
    normal_window_csv = Path(require_cfg(cfg, 'paths.normal_window_csv'))
    dataset_id = get_cfg(cfg, 'normal_dataset_id', normal_window_csv.stem)

    manifest_dir = Path('data/normal/manifests')
    manifest_dir.mkdir(parents=True, exist_ok=True)

    window_files = find_window_csvs(normal_raw_root)
    txt_manifest = manifest_dir / f'{dataset_id}_window_files.txt'
    csv_manifest = manifest_dir / f'{dataset_id}_manifest.csv'
    summary_json = manifest_dir / f'{dataset_id}_summary.json'
    txt_manifest.write_text('\n'.join(str(p) for p in window_files), encoding='utf-8')

    if not window_files:
        if allow_empty:
            print(f'[WAIT] No window.csv files found under: {normal_raw_root}')
            return None
        raise FileNotFoundError(f'No window/window.csv files found under: {normal_raw_root}')

    rows = []
    manifest_rows = []
    required_cols = _required_window_columns()

    for csv_path in window_files:
        meta = _infer_ids(normal_raw_root, csv_path)
        df = pd.read_csv(csv_path)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f'Missing {len(missing)} required feature columns in {csv_path}. First: {missing[:10]}')
        for k, v in meta.items():
            df[k] = v
        if 'label' not in df.columns:
            df['label'] = 'normal'
        rows.append(df)
        manifest_rows.append({**meta, 'rows': int(len(df)), 'columns': int(len(df.columns))})

    combined = pd.concat(rows, ignore_index=True)
    normal_window_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(normal_window_csv, index=False, encoding='utf-8')

    pd.DataFrame(manifest_rows).to_csv(csv_manifest, index=False, encoding='utf-8')
    summary = {
        'dataset_id': dataset_id,
        'normal_raw_root': str(normal_raw_root),
        'normal_window_csv': str(normal_window_csv),
        'window_csv_count': len(window_files),
        'total_windows': int(len(combined)),
        'participants': sorted(combined['participant_id'].dropna().unique().tolist()),
        'participant_count': int(combined['participant_id'].nunique()),
        'session_count': int(combined['session_id'].nunique()),
        'window_size': WINDOW_SIZE,
        'feature_dim': len(SELECTED_FEATURES),
        'selected_features': SELECTED_FEATURES,
        'manifest_csv': str(csv_manifest),
        'manifest_txt': str(txt_manifest),
    }
    dump_json(summary_json, summary)
    print('[OK] Normal dataset built')
    print(f'  - window files : {len(window_files)}')
    print(f'  - total windows: {len(combined)}')
    print(f'  - output csv   : {normal_window_csv}')
    return normal_window_csv


def main():
    import argparse
    from src.framework.config import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--allow-empty', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_normal_dataset(cfg, allow_empty=args.allow_empty)


if __name__ == '__main__':
    main()
