from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd


def mine_weaknesses(predictions_csv: str | Path, out_dir: str | Path, pred_col: str = 'pred_primary') -> Dict[str, Any]:
    predictions_csv = Path(predictions_csv); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(predictions_csv)
    attack_df = df[df['true_label'] == 1].copy(); normal_df = df[df['true_label'] == 0].copy()
    false_negatives = attack_df[attack_df[pred_col] == 0].copy(); false_positives = normal_df[normal_df[pred_col] == 1].copy()
    fn_csv = out_dir / 'false_negatives_hard_candidates.csv'; fp_csv = out_dir / 'false_positives_normal.csv'; report_json = out_dir / 'weakness_report.json'
    false_negatives.to_csv(fn_csv, index=False, encoding='utf-8'); false_positives.to_csv(fp_csv, index=False, encoding='utf-8')
    def safe_div(a,b): return float(a/b) if b else 0.0
    report = {'predictions_csv': str(predictions_csv), 'pred_col': pred_col, 'attack_count': int(len(attack_df)), 'normal_count': int(len(normal_df)), 'false_negative_count': int(len(false_negatives)), 'false_positive_count': int(len(false_positives)), 'bypass_rate': safe_div(len(false_negatives), len(attack_df)), 'normal_fpr': safe_div(len(false_positives), len(normal_df)), 'false_negatives_csv': str(fn_csv), 'false_positives_csv': str(fp_csv)}
    with report_json.open('w', encoding='utf-8') as f: json.dump(report, f, indent=2, ensure_ascii=False)
    print('[OK] Weakness mining done')
    print(f"  - false negatives: {len(false_negatives)}")
    print(f"  - bypass rate    : {report['bypass_rate']:.4f}")
    return report
