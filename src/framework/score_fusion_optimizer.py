from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import math

import numpy as np
import pandas as pd

from src.framework.config import get_cfg, require_cfg, dump_json
from src.framework.artifacts import defender_dir


PRED_COL = "pred_v7_policy"
SCORE_COL = "v7_policy_score"


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _robust_scale_to_attack_score(values: pd.Series, normal_values: pd.Series) -> pd.Series:
    """
    Convert any high-is-more-anomalous raw score to robust normal-reference scale.
    Higher output means more attack-like.
    """
    v = _as_numeric(values)
    n = _as_numeric(normal_values)

    med = float(n.median()) if len(n) else 0.0
    q1 = float(n.quantile(0.25)) if len(n) else 0.0
    q3 = float(n.quantile(0.75)) if len(n) else 1.0
    iqr = q3 - q1

    if abs(iqr) < 1e-12:
        std = float(n.std()) if len(n) else 1.0
        denom = std if std > 1e-12 else 1.0
    else:
        denom = iqr

    z = (v - med) / denom
    return z.clip(lower=-10.0, upper=50.0)


def _minmax01(values: pd.Series, normal_values: pd.Series | None = None) -> pd.Series:
    v = _as_numeric(values)
    ref = _as_numeric(normal_values) if normal_values is not None else v
    lo = float(ref.quantile(0.01)) if len(ref) else float(v.min())
    hi = float(ref.quantile(0.99)) if len(ref) else float(v.max())

    if abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(v)), index=v.index)

    return ((v - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)


def _build_score_candidates(df: pd.DataFrame) -> Dict[str, pd.Series]:
    if "true_label" not in df.columns:
        raise KeyError("predictions CSV must contain true_label column")

    normal_mask = _as_numeric(df["true_label"]).astype(int) == 0

    score_cols: Dict[str, pd.Series] = {}

    recon_col = _find_col(df, ["reconstruction_error", "recon_error", "mse", "ae_error"])
    cls_col = _find_col(df, ["classifier_attack_probability", "class_attack_probability", "attack_probability", "classifier_prob"])
    proto_col = _find_col(df, ["prototype_attack_probability", "proto_attack_probability", "prototype_prob"])
    latent_col = _find_col(df, ["latent_distance", "latent_anomaly_score", "prototype_distance", "min_prototype_distance"])

    normalized = {}

    if recon_col:
        normalized["recon"] = _robust_scale_to_attack_score(df[recon_col], df.loc[normal_mask, recon_col])
        score_cols["score_recon_robust"] = normalized["recon"]

    if cls_col:
        normalized["classifier"] = _minmax01(df[cls_col], df.loc[normal_mask, cls_col])
        score_cols["score_classifier_prob"] = normalized["classifier"]

    if proto_col:
        normalized["prototype"] = _minmax01(df[proto_col], df.loc[normal_mask, proto_col])
        score_cols["score_prototype_prob"] = normalized["prototype"]

    if latent_col:
        normalized["latent"] = _robust_scale_to_attack_score(df[latent_col], df.loc[normal_mask, latent_col])
        score_cols["score_latent_robust"] = normalized["latent"]

    keys = list(normalized.keys())

    def add_combo(name: str, parts: List[str], mode: str):
        if not all(p in normalized for p in parts):
            return
        arrs = [normalized[p] for p in parts]
        if mode == "mean":
            score_cols[name] = sum(arrs) / len(arrs)
        elif mode == "max":
            score_cols[name] = pd.concat(arrs, axis=1).max(axis=1)
        elif mode == "weighted_recon_cls":
            score_cols[name] = 0.55 * normalized["recon"] + 0.45 * normalized["classifier"]
        elif mode == "weighted_recon_cls_proto":
            score_cols[name] = 0.45 * normalized["recon"] + 0.35 * normalized["classifier"] + 0.20 * normalized["prototype"]
        elif mode == "weighted_all":
            weights = {
                "recon": 0.40,
                "classifier": 0.30,
                "prototype": 0.20,
                "latent": 0.10,
            }
            score = None
            denom = 0.0
            for p in parts:
                w = weights.get(p, 1.0)
                denom += w
                score = normalized[p] * w if score is None else score + normalized[p] * w
            score_cols[name] = score / max(denom, 1e-12)

    add_combo("score_mean_recon_classifier", ["recon", "classifier"], "mean")
    add_combo("score_max_recon_classifier", ["recon", "classifier"], "max")
    add_combo("score_weighted_recon_classifier", ["recon", "classifier"], "weighted_recon_cls")

    add_combo("score_mean_recon_proto", ["recon", "prototype"], "mean")
    add_combo("score_max_recon_proto", ["recon", "prototype"], "max")

    add_combo("score_mean_classifier_proto", ["classifier", "prototype"], "mean")
    add_combo("score_max_classifier_proto", ["classifier", "prototype"], "max")

    add_combo("score_mean_recon_classifier_proto", ["recon", "classifier", "prototype"], "mean")
    add_combo("score_max_recon_classifier_proto", ["recon", "classifier", "prototype"], "max")
    add_combo("score_weighted_recon_classifier_proto", ["recon", "classifier", "prototype"], "weighted_recon_cls_proto")

    add_combo("score_mean_all", keys, "mean")
    add_combo("score_max_all", keys, "max")
    add_combo("score_weighted_all", keys, "weighted_all")

    if not score_cols:
        # Absolute fallback from any pred-like column.
        pred_candidates = [c for c in df.columns if c.startswith("pred_")]
        if pred_candidates:
            score_cols["score_existing_pred_fallback"] = _as_numeric(df[pred_candidates[0]])
        else:
            score_cols["score_zero_fallback"] = pd.Series(np.zeros(len(df)), index=df.index)

    return score_cols


def _threshold_for_fpr(score: pd.Series, labels: pd.Series, max_fpr: float) -> Tuple[float, np.ndarray]:
    y = _as_numeric(labels).astype(int).to_numpy()
    s = _as_numeric(score).to_numpy(dtype=np.float64)

    normal_scores = s[y == 0]
    if len(normal_scores) == 0:
        thr = float(np.quantile(s, 0.95)) if len(s) else 0.0
    else:
        # Use strict greater-than prediction. 95% threshold gives roughly <= 5% FPR.
        q = max(0.0, min(1.0, 1.0 - max_fpr))
        thr = float(np.quantile(normal_scores, q))

    pred = (s > thr).astype(int)

    # If ties around threshold cause FPR too high, move threshold just above tied normal score.
    for _ in range(5):
        normal_pred = pred[y == 0]
        fpr = float(normal_pred.mean()) if len(normal_pred) else 0.0
        if fpr <= max_fpr + 0.001 + 1e-12:
            break
        offending = normal_scores[normal_scores <= np.max(normal_scores)]
        if len(offending):
            thr = float(np.nextafter(thr, np.inf))
            pred = (s > thr).astype(int)
        else:
            break

    return thr, pred


def _metrics(pred: np.ndarray, labels: pd.Series) -> Dict[str, float]:
    y = _as_numeric(labels).astype(int).to_numpy()

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    acc = (tp + tn) / max(len(y), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall_attack_detection_rate": recall,
        "normal_false_positive_rate": fpr,
        "f1": f1,
        "bypass_rate": 1.0 - recall,
    }


def optimize_decision_policy(
    cfg: Dict[str, Any],
    predictions_csv: str | Path,
    out_dir: str | Path,
    defender_id: str,
    stage_name: str,
) -> Dict[str, Any]:
    predictions_csv = Path(predictions_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_csv)
    if "true_label" not in df.columns:
        raise KeyError(f"predictions CSV has no true_label column: {predictions_csv}")

    max_fpr = float(get_cfg(
        cfg,
        "decision_policy.max_normal_false_positive_rate",
        get_cfg(cfg, "adaptive_loop.max_normal_false_positive_rate", 0.05),
    ))
    target_adr = float(get_cfg(
        cfg,
        "decision_policy.target_attack_detection_rate",
        get_cfg(cfg, "adaptive_loop.target_attack_detection_rate", 0.95),
    ))

    score_candidates = _build_score_candidates(df)
    labels = df["true_label"]

    rows = []
    candidate_payload = {}

    for name, score in score_candidates.items():
        threshold, pred = _threshold_for_fpr(score, labels, max_fpr)
        m = _metrics(pred, labels)

        row = {
            "policy_name": name,
            "threshold": threshold,
            **m,
        }
        rows.append(row)

        candidate_payload[name] = {
            "score": _as_numeric(score),
            "pred": pred,
            "threshold": threshold,
            "metrics": m,
        }

    summary_df = pd.DataFrame(rows)

    valid = summary_df[summary_df["normal_false_positive_rate"] <= max_fpr + 1e-12].copy()

    if len(valid) > 0:
        selected = valid.sort_values(
            by=["recall_attack_detection_rate", "f1", "precision", "normal_false_positive_rate"],
            ascending=[False, False, False, True],
        ).iloc[0]
        reason = f"selected highest ADR among {len(valid)} FPR-valid fusion policies"
    else:
        selected = summary_df.sort_values(
            by=["normal_false_positive_rate", "recall_attack_detection_rate", "f1"],
            ascending=[True, False, False],
        ).iloc[0]
        reason = "no FPR-valid fusion policy; selected lowest-FPR fallback"

    selected_name = str(selected["policy_name"])
    selected_payload = candidate_payload[selected_name]

    df[SCORE_COL] = selected_payload["score"]
    df[PRED_COL] = selected_payload["pred"].astype(int)

    policy_predictions_csv = out_dir / f"{stage_name}_policy_predictions.csv"
    policy_summary_csv = out_dir / f"{stage_name}_policy_summary.csv"
    policy_config_json = out_dir / f"{stage_name}_policy_config.json"

    df.to_csv(policy_predictions_csv, index=False, encoding="utf-8")
    summary_df.sort_values(
        by=["normal_false_positive_rate", "recall_attack_detection_rate"],
        ascending=[True, False],
    ).to_csv(policy_summary_csv, index=False, encoding="utf-8")

    selected_metrics = selected_payload["metrics"]

    policy = {
        "defender_id": defender_id,
        "stage_name": stage_name,
        "source_predictions_csv": str(predictions_csv),
        "policy_predictions_csv": str(policy_predictions_csv),
        "policy_summary_csv": str(policy_summary_csv),
        "policy_name": selected_name,
        "score_col": SCORE_COL,
        "pred_col": PRED_COL,
        "threshold": float(selected_payload["threshold"]),
        "max_normal_false_positive_rate": max_fpr,
        "target_attack_detection_rate": target_adr,
        "selection_reason": reason,
        "metrics": selected_metrics,
        "available_policies": sorted(score_candidates.keys()),
        "note": "Threshold optimized on current evaluation set for framework hardening analysis.",
    }

    dump_json(policy_config_json, policy)

    # Persist latest defender policy for export/runtime reference.
    ddir = defender_dir(cfg, defender_id)
    ddir.mkdir(parents=True, exist_ok=True)
    dump_json(ddir / "policy_config.json", policy)

    print("[OK] V7 score-fusion decision policy optimized")
    print(f"  - defender : {defender_id}")
    print(f"  - stage    : {stage_name}")
    print(f"  - policy   : {selected_name}")
    print(f"  - threshold: {float(selected_payload['threshold']):.6f}")
    print(f"  - ADR      : {selected_metrics['recall_attack_detection_rate']:.4f}")
    print(f"  - FPR      : {selected_metrics['normal_false_positive_rate']:.4f}")
    print(f"  - bypass   : {selected_metrics['bypass_rate']:.4f}")
    print(f"  - summary  : {policy_summary_csv}")

    return {
        "defender_id": defender_id,
        "stage_name": stage_name,
        "method": selected_name,
        "policy_name": selected_name,
        "pred_col": PRED_COL,
        "score_col": SCORE_COL,
        "threshold": float(selected_payload["threshold"]),
        "attack_detection_rate": float(selected_metrics["recall_attack_detection_rate"]),
        "normal_false_positive_rate": float(selected_metrics["normal_false_positive_rate"]),
        "bypass_rate": float(selected_metrics["bypass_rate"]),
        "precision": float(selected_metrics["precision"]),
        "f1": float(selected_metrics["f1"]),
        "selection_reason": reason,
        "predictions_csv": str(policy_predictions_csv),
        "summary_csv": str(policy_summary_csv),
        "policy_config_json": str(policy_config_json),
    }
