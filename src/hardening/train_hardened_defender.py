from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Sequence
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.core.dataset import SELECTED_FEATURES, WINDOW_SIZE
from src.core.hybrid_v2_model import ReConHIDv2
from src.framework.config import require_cfg, get_cfg, dump_json
from src.framework.artifacts import defender_dir, round_dir


def _resolve_csvs(paths: str | Path | Sequence[str | Path]) -> List[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    out = []
    for item in paths:
        p = Path(item)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            files = sorted(p.glob("*.csv"))
            if not files:
                files = sorted(p.glob("**/*.csv"))
            out.extend(files)
        else:
            print(f"[WARN] attack path not found, skipping: {p}")
    return out


def _df_to_array(df: pd.DataFrame) -> np.ndarray:
    seqs = []
    for _, row in df.iterrows():
        seq = []
        for t in range(WINDOW_SIZE):
            step = []
            for f in SELECTED_FEATURES:
                v = row.get(f"t{t}_{f}", 0.0)
                if pd.isna(v):
                    v = 0.0
                step.append(float(v))
            seq.append(step)
        seqs.append(seq)
    return np.asarray(seqs, dtype=np.float32)


def _load_labeled_df(path_or_paths, label_value: int, group: str) -> pd.DataFrame:
    csvs = _resolve_csvs(path_or_paths)
    rows = []
    for p in csvs:
        df = pd.read_csv(p)
        df["true_label"] = label_value
        df["group"] = group if len(csvs) == 1 else f"{group}:{p.parent.name}"
        df["source_file"] = p.name
        df["source_path"] = str(p)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No CSV files found for group={group}: {path_or_paths}")
    return pd.concat(rows, ignore_index=True)


def supervised_contrastive_loss(features, labels, temperature=0.1):
    device = features.device
    batch_size = features.size(0)

    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = mask.sum(dim=1)
    valid = positive_count > 0

    if valid.sum() == 0:
        return features.sum() * 0.0

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-12)
    return -mean_log_prob_pos[valid].mean()


def normal_only_reconstruction_loss(reconstructed, target, labels):
    mse = F.mse_loss(reconstructed, target, reduction="none").mean(dim=(1, 2))
    normal_mask = labels == 0
    if normal_mask.sum() == 0:
        return mse.mean() * 0.0
    return mse[normal_mask].mean()


def _eval_epoch(model, loader, cls_loss_fn, proto_loss_fn, device, weights):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_long = batch_y.long()

            reconstructed, class_logits, latent, latent_norm, proto_logits, _ = model(batch_x)
            recon_loss = normal_only_reconstruction_loss(reconstructed, batch_x, batch_y)
            cls_loss = cls_loss_fn(class_logits, batch_y)
            con_loss = supervised_contrastive_loss(latent_norm, batch_y_long, temperature=weights["contrastive_temperature"])
            proto_loss = proto_loss_fn(proto_logits, batch_y_long)

            loss = (
                weights["reconstruction"] * recon_loss
                + weights["classification"] * cls_loss
                + weights["contrastive"] * con_loss
                + weights["prototype"] * proto_loss
            )
            total_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(class_logits) >= 0.5).float()
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}


def train_hardened_defender(
    cfg: Dict[str, Any],
    defender_id: str = "D1",
    normal_path: str | Path | None = None,
    attack_paths: str | Path | Sequence[str | Path] | None = None,
    attack_path: str | Path | None = None,
    out_name: str | None = None,
) -> Dict[str, Any]:
    normal_path = normal_path or require_cfg(cfg, "paths.normal_window_csv")
    if attack_paths is None:
        attack_paths = attack_path
    if attack_paths is None:
        raise ValueError("attack_paths or attack_path is required")

    out_dir = defender_dir(cfg, defender_id)
    rdir = round_dir(cfg, out_name or f"train_{defender_id}")

    model_path = out_dir / "model.pt"
    threshold_path = out_dir / "thresholds.json"
    manifest_path = out_dir / "defender_manifest.json"
    history_path = rdir / f"train_history_{defender_id}.csv"

    normal_df = _load_labeled_df(normal_path, 0, "normal")
    attack_df = _load_labeled_df(attack_paths, 1, "attack")
    df = pd.concat([normal_df, attack_df], ignore_index=True)

    arr = _df_to_array(df)

    import joblib
    scaler_path = Path(require_cfg(cfg, "paths.checkpoints_root")) / "D0" / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    n, s, f = arr.shape
    arr = scaler.transform(arr.reshape(-1, f)).reshape(n, s, f).astype(np.float32)

    x = torch.tensor(arr, dtype=torch.float32)
    y = torch.tensor(df["true_label"].to_numpy(dtype=np.float32), dtype=torch.float32)
    dataset = TensorDataset(x, y)

    batch_size = int(get_cfg(cfg, "training.batch_size", 128))
    epochs = int(get_cfg(cfg, "training.epochs", 50))
    lr = float(get_cfg(cfg, "training.learning_rate", 0.001))

    valid_size = max(1, int(len(dataset) * float(get_cfg(cfg, "training.valid_ratio", 0.1))))
    train_size = len(dataset) - valid_size
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = int(get_cfg(cfg, "model.d1.hidden_dim", 64))
    latent_dim = int(get_cfg(cfg, "model.d1.latent_dim", 32))
    num_layers = int(get_cfg(cfg, "model.d1.num_layers", 1))
    dropout = float(get_cfg(cfg, "model.d1.dropout", 0.2))
    proto_temperature = float(get_cfg(cfg, "model.d1.proto_temperature", 0.2))

    model = ReConHIDv2(
        input_dim=len(SELECTED_FEATURES),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        proto_temperature=proto_temperature,
    ).to(device)

    normal_count = int((y == 0).sum().item())
    attack_count = int((y == 1).sum().item())
    pos_weight = torch.tensor([normal_count / max(attack_count, 1)], dtype=torch.float32, device=device)

    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    proto_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights = {
        "reconstruction": float(get_cfg(cfg, "loss.reconstruction", 0.3)),
        "classification": float(get_cfg(cfg, "loss.classification", 1.0)),
        "contrastive": float(get_cfg(cfg, "loss.contrastive", 0.2)),
        "prototype": float(get_cfg(cfg, "loss.prototype", 0.5)),
        "contrastive_temperature": float(get_cfg(cfg, "loss.contrastive_temperature", 0.1)),
    }

    print(f"\n===== TRAIN {defender_id} HARDENED START =====")
    print(f"Device      : {device}")
    print(f"Normal path : {normal_path}")
    print(f"Attack paths: {attack_paths}")
    print(f"Samples     : {len(dataset)} normal={normal_count} attack={attack_count}")
    print(f"Train/Valid : {train_size}/{valid_size}")
    print(f"Model path  : {model_path}")
    print("==========================================\n")

    best_valid = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_correct = total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_long = batch_y.long()

            optimizer.zero_grad()

            reconstructed, class_logits, latent, latent_norm, proto_logits, _ = model(batch_x)
            recon_loss = normal_only_reconstruction_loss(reconstructed, batch_x, batch_y)
            cls_loss = cls_loss_fn(class_logits, batch_y)
            con_loss = supervised_contrastive_loss(latent_norm, batch_y_long, temperature=weights["contrastive_temperature"])
            proto_loss = proto_loss_fn(proto_logits, batch_y_long)

            loss = (
                weights["reconstruction"] * recon_loss
                + weights["classification"] * cls_loss
                + weights["contrastive"] * con_loss
                + weights["prototype"] * proto_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(class_logits) >= 0.5).float()
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_metrics = {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}
        valid_metrics = _eval_epoch(model, valid_loader, cls_loss_fn, proto_loss_fn, device, weights)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "valid_loss": valid_metrics["loss"],
            "valid_acc": valid_metrics["acc"],
        })

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.6f} Acc: {train_metrics['acc']:.4f} | "
            f"Valid Loss: {valid_metrics['loss']:.6f} Acc: {valid_metrics['acc']:.4f}"
        )

        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(SELECTED_FEATURES),
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "proto_temperature": proto_temperature,
            "selected_features": SELECTED_FEATURES,
            "window_size": WINDOW_SIZE,
            "scaler_path": str(scaler_path),
            "defender_id": defender_id,
            "normal_path": str(normal_path),
            "attack_paths": [str(p) for p in _resolve_csvs(attack_paths)],
        },
        model_path,
    )

    dump_json(threshold_path, {"classifier_threshold": 0.5, "prototype_threshold": 0.5})

    manifest = {
        "defender_id": defender_id,
        "model_type": "ReConHIDv2",
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "normal_path": str(normal_path),
        "attack_paths": [str(p) for p in _resolve_csvs(attack_paths)],
        "normal_count": normal_count,
        "attack_count": attack_count,
        "loss_weights": weights,
        "best_valid_loss": best_valid,
    }
    dump_json(manifest_path, manifest)
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")

    print(f"\n===== TRAIN {defender_id} DONE =====")
    print(f"Saved model   : {model_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved history : {history_path}")

    return {
        "defender_id": defender_id,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "history_path": str(history_path),
    }
