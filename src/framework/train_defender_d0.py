from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.core.dataset import SELECTED_FEATURES, WINDOW_SIZE
from src.core.model import LSTMAutoencoder
from src.framework.config import require_cfg, get_cfg, dump_json
from src.framework.artifacts import defender_dir, round_dir


def dataframe_to_array(df: pd.DataFrame) -> np.ndarray:
    sequences = []
    for _, row in df.iterrows():
        seq = []
        for t in range(WINDOW_SIZE):
            step = []
            for feature in SELECTED_FEATURES:
                value = row.get(f't{t}_{feature}', 0.0)
                if pd.isna(value):
                    value = 0.0
                step.append(float(value))
            seq.append(step)
        sequences.append(seq)
    return np.asarray(sequences, dtype=np.float32)


def split_by_session(df: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int = 42):
    if 'session_id' not in df.columns:
        print('[WARN] session_id not found. Falling back to row-level split.')
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df)); rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        n_valid = int(len(idx) * valid_ratio)
        return (df.iloc[idx[:n_train]].reset_index(drop=True),
                df.iloc[idx[n_train:n_train+n_valid]].reset_index(drop=True),
                df.iloc[idx[n_train+n_valid:]].reset_index(drop=True))
    sessions = sorted(df['session_id'].dropna().unique().tolist())
    rng = np.random.default_rng(seed); rng.shuffle(sessions)
    n = len(sessions)
    n_train = max(1, int(n * train_ratio))
    n_valid = max(1, int(n * valid_ratio)) if n >= 3 else max(0, n - n_train)
    train_sessions = set(sessions[:n_train])
    valid_sessions = set(sessions[n_train:n_train+n_valid])
    test_sessions = set(sessions[n_train+n_valid:])
    if not test_sessions and len(valid_sessions) > 1:
        test_sessions.add(valid_sessions.pop())
    train_df = df[df['session_id'].isin(train_sessions)].reset_index(drop=True)
    valid_df = df[df['session_id'].isin(valid_sessions)].reset_index(drop=True)
    test_df = df[df['session_id'].isin(test_sessions)].reset_index(drop=True)
    if len(valid_df) == 0:
        valid_df = test_df.copy() if len(test_df) else train_df.sample(frac=0.2, random_state=seed)
    if len(test_df) == 0:
        test_df = valid_df.copy()
    return train_df, valid_df, test_df


def _make_loader(arr: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.tensor(arr, dtype=torch.float32)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=shuffle)


def compute_reconstruction_errors(model, loader, device) -> np.ndarray:
    model.eval(); mse = nn.MSELoss(reduction='none'); errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = mse(reconstructed, batch).mean(dim=(1, 2))
            errors.extend(loss.cpu().numpy().tolist())
    return np.asarray(errors, dtype=np.float64)


def train_d0(cfg: Dict[str, Any]) -> Dict[str, Any]:
    normal_csv = Path(require_cfg(cfg, 'paths.normal_window_csv'))
    if not normal_csv.exists():
        raise FileNotFoundError(f'Normal window CSV not found: {normal_csv}')
    out_dir = defender_dir(cfg, 'D0')
    rdir = round_dir(cfg, 'round0_D0')
    model_path = out_dir / 'model.pt'
    scaler_path = out_dir / 'scaler.pkl'
    threshold_path = out_dir / 'thresholds.json'
    history_path = rdir / 'train_history_D0.csv'
    split_summary_path = rdir / 'split_summary_D0.json'
    batch_size = int(get_cfg(cfg, 'training.batch_size', 128))
    epochs = int(get_cfg(cfg, 'training.epochs', 50))
    lr = float(get_cfg(cfg, 'training.learning_rate', 0.001))
    train_ratio = float(get_cfg(cfg, 'training.train_ratio', 0.8))
    valid_ratio = float(get_cfg(cfg, 'training.valid_ratio', 0.1))
    seed = int(get_cfg(cfg, 'training.seed', 42))
    hidden_dim = int(get_cfg(cfg, 'model.d0.hidden_dim', 64))
    latent_dim = int(get_cfg(cfg, 'model.d0.latent_dim', 16))
    num_layers = int(get_cfg(cfg, 'model.d0.num_layers', 1))
    df = pd.read_csv(normal_csv)
    train_df, valid_df, test_df = split_by_session(df, train_ratio, valid_ratio, seed=seed)
    train_arr = dataframe_to_array(train_df); valid_arr = dataframe_to_array(valid_df); test_arr = dataframe_to_array(test_df)
    n_train, seq_len, feat_dim = train_arr.shape
    scaler = StandardScaler(); scaler.fit(train_arr.reshape(-1, feat_dim))
    def transform(arr):
        n, s, f = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    train_arr = transform(train_arr); valid_arr = transform(valid_arr); test_arr = transform(test_arr)
    joblib.dump(scaler, scaler_path)
    train_loader = _make_loader(train_arr, batch_size, True)
    valid_loader = _make_loader(valid_arr, batch_size, False)
    test_loader = _make_loader(test_arr, batch_size, False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(input_dim=len(SELECTED_FEATURES), hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    criterion = nn.MSELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []; best_valid = float('inf'); best_state = None
    print('\n===== TRAIN D0 START =====')
    print(f'Device       : {device}')
    print(f'Normal CSV   : {normal_csv}')
    print(f'Train rows   : {len(train_df)}')
    print(f'Valid rows   : {len(valid_df)}')
    print(f'Test rows    : {len(test_df)}')
    print('==========================\n')
    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device); optimizer.zero_grad()
            reconstructed = model(batch); loss = criterion(reconstructed, batch)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= max(1, len(train_loader.dataset))
        valid_errors = compute_reconstruction_errors(model, valid_loader, device)
        valid_loss = float(valid_errors.mean())
        history.append({'epoch': epoch, 'train_loss': float(train_loss), 'valid_loss': valid_loss})
        print(f'Epoch [{epoch:03d}/{epochs}] Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f}')
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    valid_errors = compute_reconstruction_errors(model, valid_loader, device)
    test_errors = compute_reconstruction_errors(model, test_loader, device)
    q_list = get_cfg(cfg, 'thresholds.quantiles', [0.90, 0.95, 0.99])
    thresholds = {f'recon_{int(q * 100)}': float(np.quantile(valid_errors, q)) for q in q_list}
    thresholds['primary'] = float(get_cfg(cfg, 'thresholds.primary', 0.95))
    thresholds['primary_key'] = f"recon_{int(float(thresholds['primary']) * 100)}"
    torch.save({'model_state_dict': model.state_dict(), 'input_dim': len(SELECTED_FEATURES), 'hidden_dim': hidden_dim, 'latent_dim': latent_dim, 'num_layers': num_layers, 'selected_features': SELECTED_FEATURES, 'window_size': WINDOW_SIZE, 'scaler_path': str(scaler_path), 'normal_csv': str(normal_csv), 'thresholds': thresholds, 'defender_id': 'D0'}, model_path)
    pd.DataFrame(history).to_csv(history_path, index=False, encoding='utf-8')
    split_summary = {'train_rows': int(len(train_df)), 'valid_rows': int(len(valid_df)), 'test_rows': int(len(test_df)), 'best_valid_loss': best_valid, 'valid_error_mean': float(valid_errors.mean()), 'test_error_mean': float(test_errors.mean()), 'thresholds': thresholds}
    dump_json(split_summary_path, split_summary); dump_json(threshold_path, thresholds)
    print('\n===== TRAIN D0 DONE =====')
    print(f'Saved model      : {model_path}')
    print(f'Saved scaler     : {scaler_path}')
    print(f'Saved thresholds : {threshold_path}')
    return {'defender_id': 'D0', 'model_path': str(model_path), 'scaler_path': str(scaler_path), 'threshold_path': str(threshold_path), 'history_path': str(history_path), 'split_summary_path': str(split_summary_path)}
