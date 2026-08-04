from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.core.dataset import SELECTED_FEATURES, WINDOW_SIZE
from src.core.hybrid_v2_model import ReConHIDv2
from src.framework.config import require_cfg, get_cfg, dump_json
from src.framework.artifacts import defender_dir, round_dir


def _resolve_csvs(path: str | Path) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            files = sorted(path.glob("**/*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files under: {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def _df_to_array(df: pd.DataFrame) -> np.ndarray:
    seqs = []
    for _, row in df.iterrows():
        seq = []
        for t in range(WINDOW_SIZE):
            step = []
            for f in SELECTED_FEATURES:
                value = row.get(f"t{t}_{f}", 0.0)
                if pd.isna(value):
                    value = 0.0
                step.append(float(value))
            seq.append(step)
        seqs.append(seq)
    return np.asarray(seqs, dtype=np.float32)


def _load_labeled(path: str | Path, label: int, group: str) -> pd.DataFrame:
    rows = []
    for csv_path in _resolve_csvs(path):
        df = pd.read_csv(csv_path)
        df["true_label"] = int(label)
        df["group"] = group
        df["source_file"] = csv_path.name
        df["source_path"] = str(csv_path)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _load_model(model_path: Path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = ReConHIDv2(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        proto_temperature=ckpt["proto_temperature"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _score(model, arr_scaled, batch_size, device) -> pd.DataFrame:
    loader = DataLoader(TensorDataset(torch.tensor(arr_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    mse = nn.MSELoss(reduction="none")
    rows = []
    idx = 0

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed, class_logits, latent, latent_norm, proto_logits, proto_distances = model(batch)

            recon = mse(reconstructed, batch).mean(dim=(1, 2)).cpu().numpy()
            cls_prob = torch.sigmoid(class_logits).cpu().numpy()
            proto_prob = F.softmax(proto_logits, dim=1)[:, 1].cpu().numpy()
            proto_dist_np = proto_distances.cpu().numpy()
            latent_norm_value = torch.norm(latent, dim=1).cpu().numpy()

            for i in range(batch.size(0)):
                rows.append({
                    "global_index": idx,
                    "reconstruction_error": float(recon[i]),
                    "classifier_attack_probability": float(cls_prob[i]),
                    "prototype_attack_probability": float(proto_prob[i]),
                    "distance_to_normal_prototype": float(proto_dist_np[i, 0]),
                    "distance_to_attack_prototype": float(proto_dist_np[i, 1]),
                    "latent_norm": float(latent_norm_value[i]),
                })
                idx += 1

    return pd.DataFrame(rows)


def _compute_thresholds(scored_df: pd.DataFrame) -> Dict[str, float]:
    normal = scored_df[scored_df["true_label"] == 0]
    if len(normal) == 0:
        raise ValueError("No normal rows for threshold computation.")

    return {
        "recon_90": float(normal["reconstruction_error"].quantile(0.90)),
        "recon_95": float(normal["reconstruction_error"].quantile(0.95)),
        "recon_99": float(normal["reconstruction_error"].quantile(0.99)),
        "latent_dist_90": float(normal["distance_to_normal_prototype"].quantile(0.90)),
        "latent_dist_95": float(normal["distance_to_normal_prototype"].quantile(0.95)),
        "latent_dist_99": float(normal["distance_to_normal_prototype"].quantile(0.99)),
        "classifier_threshold": 0.5,
        "prototype_threshold": 0.5,
    }


def _apply_predictions(df: pd.DataFrame, th: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["pred_classifier"] = (df["classifier_attack_probability"] >= th["classifier_threshold"]).astype(int)
    df["pred_prototype"] = (df["prototype_attack_probability"] >= th["prototype_threshold"]).astype(int)
    df["pred_recon_90"] = (df["reconstruction_error"] > th["recon_90"]).astype(int)
    df["pred_recon_95"] = (df["reconstruction_error"] > th["recon_95"]).astype(int)
    df["pred_recon_99"] = (df["reconstruction_error"] > th["recon_99"]).astype(int)
    df["pred_latent_dist_95"] = (df["distance_to_normal_prototype"] > th["latent_dist_95"]).astype(int)
    df["pred_or95"] = ((df["pred_classifier"] == 1) | (df["pred_prototype"] == 1) | (df["pred_recon_95"] == 1)).astype(int)
    df["pred_or90"] = ((df["pred_classifier"] == 1) | (df["pred_prototype"] == 1) | (df["pred_recon_90"] == 1)).astype(int)
    df["pred_or95_with_latent"] = ((df["pred_or95"] == 1) | (df["pred_latent_dist_95"] == 1)).astype(int)
    return df


def _metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, Any]:
    y_true = df["true_label"].to_numpy()
    y_pred = df[pred_col].to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def div(a, b):
        return float(a / b) if b else 0.0

    return {
        "method": pred_col,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": div(tp, tp + fp),
        "recall_attack_detection_rate": div(tp, tp + fn),
        "f1": div(2 * div(tp, tp + fp) * div(tp, tp + fn), div(tp, tp + fp) + div(tp, tp + fn)),
        "normal_false_positive_rate": div(fp, fp + tn),
    }


def evaluate_hybrid_defender(
    cfg: Dict[str, Any],
    defender_id: str = "D1",
    attack_path: str | Path | None = None,
    out_name: str | None = None,
) -> Dict[str, Any]:
    if attack_path is None:
        attack_path = Path(require_cfg(cfg, "paths.attack_generated_root")) / "round1_A1" / "candidates.csv"

    ddir = defender_dir(cfg, defender_id)
    rdir = round_dir(cfg, out_name or f"eval_{defender_id}")

    model_path = ddir / "model.pt"
    threshold_path = ddir / "thresholds.json"
    normal_path = Path(require_cfg(cfg, "paths.normal_window_csv"))

    if not model_path.exists():
        raise FileNotFoundError(f"{defender_id} model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(model_path, device)
    scaler = joblib.load(ckpt["scaler_path"])

    normal_df = _load_labeled(normal_path, label=0, group="normal")
    attack_df = _load_labeled(attack_path, label=1, group="attack")
    df = pd.concat([normal_df, attack_df], ignore_index=True)

    arr = _df_to_array(df)
    n, s, f = arr.shape
    arr_scaled = scaler.transform(arr.reshape(-1, f)).reshape(n, s, f).astype(np.float32)

    score_df = _score(model, arr_scaled, int(get_cfg(cfg, "training.batch_size", 128)), device)
    scored = pd.concat([df[["true_label", "group", "source_file", "source_path"]].reset_index(drop=True), score_df.drop(columns=["global_index"])], axis=1)
    scored.insert(0, "global_index", range(len(scored)))

    thresholds = _compute_thresholds(scored)
    pred_df = _apply_predictions(scored, thresholds)

    methods = ["pred_classifier", "pred_prototype", "pred_recon_95", "pred_or95", "pred_or90", "pred_or95_with_latent"]
    summary_df = pd.DataFrame([_metrics(pred_df, m) for m in methods])
    source_summary = pred_df.groupby(["group", "source_file"]).agg(
        count=("true_label", "count"),
        mean_reconstruction_error=("reconstruction_error", "mean"),
        mean_classifier_attack_probability=("classifier_attack_probability", "mean"),
        mean_prototype_attack_probability=("prototype_attack_probability", "mean"),
        oracle_or95_detection_rate=("pred_or95", "mean"),
        oracle_or90_detection_rate=("pred_or90", "mean"),
    ).reset_index()

    predictions_csv = rdir / f"{defender_id}_predictions.csv"
    summary_csv = rdir / f"{defender_id}_summary.csv"
    source_summary_csv = rdir / f"{defender_id}_source_summary.csv"

    pred_df.to_csv(predictions_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    source_summary.to_csv(source_summary_csv, index=False, encoding="utf-8")
    dump_json(threshold_path, thresholds)

    print(f"[OK] {defender_id} hybrid evaluation done")
    print(f"  - predictions: {predictions_csv}")
    print(f"  - summary    : {summary_csv}")
    print(f"  - thresholds : {threshold_path}")
    print(summary_df.to_string(index=False))

    return {
        "predictions_csv": str(predictions_csv),
        "summary_csv": str(summary_csv),
        "source_summary_csv": str(source_summary_csv),
        "threshold_path": str(threshold_path),
    }
