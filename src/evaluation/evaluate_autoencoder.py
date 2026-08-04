from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.core.dataset import SELECTED_FEATURES, WINDOW_SIZE
from src.core.model import LSTMAutoencoder
from src.framework.artifacts import defender_dir, round_dir
from src.framework.config import require_cfg, get_cfg


def _resolve_csv_files(path: str | Path):
    path = Path(path)
    if path.is_file(): return [path]
    if path.is_dir():
        files = sorted(path.glob('*.csv'))
        if not files: files = sorted(path.glob('**/*.csv'))
        return files
    return []


def _df_to_array(df: pd.DataFrame) -> np.ndarray:
    seqs = []
    for _, row in df.iterrows():
        seq = []
        for t in range(WINDOW_SIZE):
            step = []
            for f in SELECTED_FEATURES:
                value = row.get(f't{t}_{f}', 0.0)
                if pd.isna(value): value = 0.0
                step.append(float(value))
            seq.append(step)
        seqs.append(seq)
    return np.asarray(seqs, dtype=np.float32)


def _load_labeled_windows(path: str | Path, label: str) -> pd.DataFrame:
    rows = []
    for csv_path in _resolve_csv_files(path):
        df = pd.read_csv(csv_path)
        if 'source_file' not in df.columns: df['source_file'] = csv_path.name
        df['source_path'] = str(csv_path); df['label'] = label
        rows.append(df)
    if not rows: raise FileNotFoundError(f'No CSV files found for evaluation: {path}')
    return pd.concat(rows, ignore_index=True)


def _load_d0_model(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = LSTMAutoencoder(input_dim=checkpoint['input_dim'], hidden_dim=checkpoint['hidden_dim'], latent_dim=checkpoint['latent_dim'], num_layers=checkpoint['num_layers']).to(device)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    return model, checkpoint


def evaluate_d0(cfg: Dict[str, Any], attack_path: Optional[str | Path] = None, out_name: str = 'round0_D0_eval') -> Dict[str, Any]:
    ddir = defender_dir(cfg, 'D0'); rdir = round_dir(cfg, out_name)
    model_path = ddir / 'model.pt'; scaler_path = ddir / 'scaler.pkl'; threshold_path = ddir / 'thresholds.json'
    if not model_path.exists(): raise FileNotFoundError(f'D0 model not found: {model_path}')
    with threshold_path.open('r', encoding='utf-8') as f: thresholds = json.load(f)
    primary_key = thresholds.get('primary_key', 'recon_95'); primary_threshold = float(thresholds[primary_key])
    normal_path = require_cfg(cfg, 'paths.normal_window_csv')
    normal_df = _load_labeled_windows(normal_path, 'normal')
    eval_dfs = [normal_df]
    if attack_path is not None and Path(attack_path).exists(): eval_dfs.append(_load_labeled_windows(attack_path, 'attack'))
    elif attack_path is not None: print(f'[WARN] attack_path does not exist. Skipping: {attack_path}')
    df = pd.concat(eval_dfs, ignore_index=True)
    arr = _df_to_array(df); n, s, f = arr.shape
    scaler = joblib.load(scaler_path)
    arr_scaled = scaler.transform(arr.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.tensor(arr_scaled, dtype=torch.float32)), batch_size=int(get_cfg(cfg, 'training.batch_size', 128)), shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = _load_d0_model(model_path, device); mse = nn.MSELoss(reduction='none'); errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device); reconstructed = model(batch)
            loss = mse(reconstructed, batch).mean(dim=(1, 2)); errors.extend(loss.cpu().numpy().tolist())
    pred_df = pd.DataFrame({'global_index': np.arange(len(df)), 'label': df['label'].tolist(), 'true_label': (df['label'] == 'attack').astype(int).tolist(), 'source_file': df.get('source_file', pd.Series(['unknown'] * len(df))).tolist(), 'source_path': df.get('source_path', pd.Series(['unknown'] * len(df))).tolist(), 'reconstruction_error': errors})
    for key, th in thresholds.items():
        if key.startswith('recon_'): pred_df[f'pred_{key}'] = (pred_df['reconstruction_error'] > float(th)).astype(int)
    pred_df['pred_primary'] = (pred_df['reconstruction_error'] > primary_threshold).astype(int)
    y_true = pred_df['true_label'].to_numpy(); y_pred = pred_df['pred_primary'].to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum()); fn = int(((y_true == 1) & (y_pred == 0)).sum())
    def safe_div(a,b): return float(a/b) if b else 0.0
    summary = {'defender_id': 'D0', 'model_path': str(model_path), 'normal_path': str(normal_path), 'attack_path': str(attack_path) if attack_path is not None else None, 'primary_threshold_key': primary_key, 'primary_threshold': primary_threshold, 'count': int(len(pred_df)), 'normal_count': int((pred_df['true_label'] == 0).sum()), 'attack_count': int((pred_df['true_label'] == 1).sum()), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'accuracy': safe_div(tp+tn, tp+tn+fp+fn), 'precision': safe_div(tp, tp+fp), 'recall_attack_detection_rate': safe_div(tp, tp+fn), 'normal_false_positive_rate': safe_div(fp, fp+tn)}
    predictions_csv = rdir / 'd0_predictions.csv'; summary_csv = rdir / 'd0_summary.csv'; summary_json = rdir / 'd0_summary.json'
    pred_df.to_csv(predictions_csv, index=False, encoding='utf-8'); pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding='utf-8')
    with summary_json.open('w', encoding='utf-8') as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    print('[OK] D0 evaluation done')
    print(f"  - attack detect recall: {summary['recall_attack_detection_rate']:.4f}")
    print(f"  - normal FPR          : {summary['normal_false_positive_rate']:.4f}")
    return {'predictions_csv': str(predictions_csv), 'summary_csv': str(summary_csv), 'summary_json': str(summary_json), 'summary': summary}
