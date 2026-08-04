from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

import numpy as np
import pandas as pd

from src.framework.config import get_cfg, require_cfg, dump_json
from src.generators.synthetic_hid_generator import generate_synthetic_attacks
from src.evaluation.evaluate_autoencoder import evaluate_d0
from src.evaluation.evaluate_hybrid_defender import evaluate_hybrid_defender


FEATURES = [
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "overlap_ratio",
    "simultaneous_key_count",
    "modifier_count",
    "shortcut_flag",
    "correction_ratio",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
    "current_key_category_id",
]

CONTINUOUS_FEATURES = [
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "overlap_ratio",
    "correction_ratio",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
]

INTEGER_FEATURES = [
    "simultaneous_key_count",
    "modifier_count",
    "shortcut_flag",
    "current_key_category_id",
]

D0_PRED_METHODS = [
    "pred_v7_policy",
    "pred_primary",
    "pred_recon_95",
    "pred_recon_90",
]

HYBRID_PRED_METHODS = [
    "pred_v7_policy",
    "pred_classifier",
    "pred_prototype",
    "pred_recon_95",
    "pred_or95",
    "pred_or95_with_latent",
    "pred_or90",
]

SCORE_PRIORITY = [
    "v7_policy_score",
    "classifier_attack_probability",
    "attack_probability",
    "prototype_attack_probability",
    "latent_distance",
    "reconstruction_error",
    "recon_error",
    "mse",
]


def _feature_cols(window_size: int) -> List[str]:
    return [f"t{t}_{f}" for t in range(window_size) for f in FEATURES]


def _available_feature_cols(df: pd.DataFrame, window_size: int) -> List[str]:
    cols = _feature_cols(window_size)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"candidate CSV missing feature columns. missing_count={len(missing)} first={missing[:10]}"
        )
    return cols


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _read_csv_if_exists(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _extract_summary_path(eval_info: Dict[str, Any]) -> Path | None:
    for key in ["summary_csv", "summary", "summary_path"]:
        if key in eval_info and eval_info[key]:
            p = Path(str(eval_info[key]))
            if p.exists():
                return p
    # Fallback: infer beside predictions.
    pred = eval_info.get("predictions_csv")
    if pred:
        pp = Path(pred)
        for cand in pp.parent.glob("*summary*.csv"):
            return cand
    return None


def _choose_pred_col_with_fpr_policy(
    preds: pd.DataFrame,
    eval_info: Dict[str, Any],
    defender_id: str,
    cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    v8.1 change:
    Pick the blind-spot mining prediction column under the same FPR constraint
    used by the final policy, instead of blindly using OR methods with high FPR.
    """
    max_fpr = float(get_cfg(cfg, "decision_policy.max_normal_false_positive_rate", 0.05))
    tol = float(get_cfg(cfg, "decision_policy.fpr_tolerance", 0.001))
    prefer = list(get_cfg(cfg, "decision_policy.prefer_methods", []))
    if not prefer:
        prefer = HYBRID_PRED_METHODS if defender_id != "D0" else D0_PRED_METHODS

    summary_path = _extract_summary_path(eval_info)
    summary = _read_csv_if_exists(summary_path)

    allowed_cols = D0_PRED_METHODS if defender_id == "D0" else HYBRID_PRED_METHODS
    existing = [c for c in allowed_cols if c in preds.columns]
    if not existing:
        existing = [c for c in preds.columns if c.startswith("pred_")]
    if not existing:
        raise KeyError(f"No prediction column found. columns={list(preds.columns)[:40]}")

    # If summary exists, choose a method with FPR <= max_fpr + tol and best recall.
    if summary is not None and "method" in summary.columns:
        rows = summary[summary["method"].isin(existing)].copy()
        if "normal_false_positive_rate" in rows.columns:
            rows["normal_false_positive_rate"] = pd.to_numeric(rows["normal_false_positive_rate"], errors="coerce")
        else:
            rows["normal_false_positive_rate"] = np.nan

        recall_col = None
        for c in ["recall_attack_detection_rate", "attack_detect_recall", "recall", "ADR"]:
            if c in rows.columns:
                recall_col = c
                rows[c] = pd.to_numeric(rows[c], errors="coerce")
                break

        valid = rows[rows["normal_false_positive_rate"] <= max_fpr + tol].copy()
        if len(valid) > 0:
            if recall_col:
                valid["_recall_rank"] = valid[recall_col].fillna(-1.0)
            else:
                valid["_recall_rank"] = 0.0
            valid["_pref_rank"] = valid["method"].apply(lambda m: prefer.index(m) if m in prefer else len(prefer))
            valid = valid.sort_values(
                by=["_recall_rank", "normal_false_positive_rate", "_pref_rank"],
                ascending=[False, True, True],
            )
            method = str(valid.iloc[0]["method"])
            return method, {
                "selection_mode": "fpr_constrained_summary",
                "summary_path": str(summary_path) if summary_path else None,
                "max_fpr": max_fpr,
                "fpr_tolerance": tol,
                "method_fpr": float(valid.iloc[0]["normal_false_positive_rate"]),
                "method_recall": float(valid.iloc[0]["_recall_rank"]),
            }

        # fallback to lowest FPR method in summary
        if len(rows) > 0 and rows["normal_false_positive_rate"].notna().any():
            rows["_pref_rank"] = rows["method"].apply(lambda m: prefer.index(m) if m in prefer else len(prefer))
            rows = rows.sort_values(by=["normal_false_positive_rate", "_pref_rank"], ascending=[True, True])
            method = str(rows.iloc[0]["method"])
            return method, {
                "selection_mode": "lowest_fpr_summary_fallback",
                "summary_path": str(summary_path) if summary_path else None,
                "max_fpr": max_fpr,
                "fpr_tolerance": tol,
                "method_fpr": float(rows.iloc[0]["normal_false_positive_rate"]),
            }

    # Last fallback
    for c in prefer:
        if c in preds.columns:
            return c, {"selection_mode": "preference_fallback", "summary_path": str(summary_path) if summary_path else None}
    return existing[0], {"selection_mode": "first_prediction_fallback", "summary_path": str(summary_path) if summary_path else None}


def _choose_score_col(preds: pd.DataFrame) -> str | None:
    for c in SCORE_PRIORITY:
        if c in preds.columns:
            return c
    for c in preds.columns:
        low = c.lower()
        if ("score" in low or "prob" in low or "error" in low or "distance" in low) and c not in {"true_label", "label"}:
            return c
    return None


def _attack_part(preds: pd.DataFrame, n_candidates: int) -> pd.DataFrame:
    if "true_label" in preds.columns:
        y = _numeric(preds["true_label"]).astype(int)
        atk = preds[y == 1].copy()
        if len(atk) >= n_candidates:
            return atk.tail(n_candidates).reset_index(drop=True)
        if len(atk) > 0:
            return atk.reset_index(drop=True)

    if "label" in preds.columns:
        lab = preds["label"].astype(str).str.lower()
        atk = preds[lab.str.contains("attack")].copy()
        if len(atk) >= n_candidates:
            return atk.tail(n_candidates).reset_index(drop=True)
        if len(atk) > 0:
            return atk.reset_index(drop=True)

    return preds.tail(n_candidates).copy().reset_index(drop=True)


def _evaluate_pool(cfg: Dict[str, Any], defender_id: str, attack_csv: Path, out_name: str) -> Dict[str, Any]:
    if defender_id == "D0":
        return evaluate_d0(cfg, attack_path=attack_csv, out_name=out_name)
    return evaluate_hybrid_defender(cfg, defender_id=defender_id, attack_path=attack_csv, out_name=out_name)


def _read_predictions(eval_info: Dict[str, Any]) -> pd.DataFrame:
    p = eval_info.get("predictions_csv")
    if not p:
        raise KeyError(f"Evaluation result has no predictions_csv. keys={list(eval_info.keys())}")
    return pd.read_csv(p)


def _attach_hardness(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    eval_info: Dict[str, Any],
    defender_id: str,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cand = candidates.reset_index(drop=True).copy()
    atk_pred = _attack_part(predictions, len(cand))

    pred_col, pred_policy_info = _choose_pred_col_with_fpr_policy(atk_pred, eval_info, defender_id, cfg)
    pred = _numeric(atk_pred[pred_col]).astype(int)
    is_fn = (pred == 0).astype(int)

    score_col = _choose_score_col(atk_pred)
    if score_col is not None:
        raw_score = _numeric(atk_pred[score_col]).to_numpy(dtype=np.float64)
        finite = raw_score[np.isfinite(raw_score)]
        if len(finite) == 0:
            norm_score = np.zeros(len(raw_score))
        else:
            lo = np.quantile(finite, 0.01)
            hi = np.quantile(finite, 0.99)
            if abs(hi - lo) < 1e-12:
                norm_score = np.zeros(len(raw_score))
            else:
                norm_score = np.clip((raw_score - lo) / (hi - lo), 0.0, 1.0)
        # Lower attack score = more normal-like = harder.
        score_hard = 1.0 - norm_score
    else:
        score_hard = np.zeros(len(cand))

    cand["_v81_pred_col"] = pred_col
    cand["_v81_predicted_attack"] = pred.to_numpy()
    cand["_v81_is_fn"] = is_fn.to_numpy()
    cand["_v81_score_col"] = score_col or ""
    cand["_v81_score_hardness"] = score_hard
    cand["_v81_hard_score"] = cand["_v81_is_fn"].astype(float) * 10.0 + cand["_v81_score_hardness"].astype(float)

    info = {
        "pred_col": pred_col,
        "score_col": score_col,
        "candidate_count": int(len(cand)),
        "false_negative_count": int(cand["_v81_is_fn"].sum()),
        "detected_count": int((cand["_v81_predicted_attack"] == 1).sum()),
        "pred_policy": pred_policy_info,
    }
    return cand, info


def _normal_bounds(cfg: Dict[str, Any], window_size: int) -> Dict[str, Tuple[float, float, float, float]]:
    normal_csv = Path(require_cfg(cfg, "paths.normal_window_csv"))
    df = pd.read_csv(normal_csv)

    bounds = {}
    for f in FEATURES:
        cols = [f"t{t}_{f}" for t in range(window_size) if f"t{t}_{f}" in df.columns]
        if not cols:
            continue
        vals = pd.to_numeric(df[cols].stack(), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            continue
        bounds[f] = (
            float(vals.quantile(0.001)),
            float(vals.quantile(0.01)),
            float(vals.quantile(0.99)),
            float(vals.quantile(0.999)),
        )
    return bounds


def _clamp_feature_value(feature: str, value: float, bounds: Dict[str, Tuple[float, float, float, float]], realism_expand: float) -> float:
    if feature not in bounds:
        return float(value)

    q001, q01, q99, q999 = bounds[feature]
    lo = min(q001, q01)
    hi = max(q999, q99)

    if feature in ["hold_time", "flight_time", "press_to_press_time", "release_to_release_time", "pause_duration", "timing_variance"]:
        lo = max(0.0, lo)
        hi = max(hi * realism_expand, lo + 1e-6)
    elif feature in ["overlap_ratio", "correction_ratio", "burst_density"]:
        lo = max(0.0, lo)
        hi = min(1.0, max(hi * realism_expand, lo + 1e-6))
    elif feature in ["keys_per_second", "timing_entropy"]:
        lo = max(0.0, lo)
        hi = max(hi * realism_expand, lo + 1e-6)

    return float(np.clip(value, lo, hi))


def _mutate_parent(
    rng: np.random.Generator,
    parent: pd.Series,
    generation: int,
    child_idx: int,
    window_size: int,
    bounds: Dict[str, Tuple[float, float, float, float]],
    mutation_strength: float,
    realism_expand: float,
) -> pd.Series:
    child = parent.copy()
    local_strength = mutation_strength * (0.88 ** generation)

    strategy = rng.choice([
        "timing_nearby",
        "speed_shift",
        "variance_shift",
        "shortcut_shift",
        "pause_burst_mix",
        "mixed_local",
    ])

    if strategy == "timing_nearby":
        selected_features = ["hold_time", "flight_time", "press_to_press_time", "release_to_release_time", "pause_duration"]
    elif strategy == "speed_shift":
        selected_features = ["press_to_press_time", "release_to_release_time", "keys_per_second", "burst_density"]
    elif strategy == "variance_shift":
        selected_features = ["timing_variance", "timing_entropy", "flight_time", "pause_duration"]
    elif strategy == "shortcut_shift":
        selected_features = ["shortcut_flag", "modifier_count", "simultaneous_key_count", "press_to_press_time"]
    elif strategy == "pause_burst_mix":
        selected_features = ["pause_duration", "burst_density", "press_to_press_time", "timing_variance"]
    else:
        selected_features = list(rng.choice(CONTINUOUS_FEATURES, size=min(6, len(CONTINUOUS_FEATURES)), replace=False))

    for f in selected_features:
        if f in CONTINUOUS_FEATURES:
            scale = float(np.exp(rng.normal(0.0, local_strength)))
            additive = float(rng.normal(0.0, local_strength * 0.02))
            for t in range(window_size):
                c = f"t{t}_{f}"
                if c not in child.index:
                    continue
                v = pd.to_numeric(child[c], errors="coerce")
                v = 0.0 if not np.isfinite(v) else float(v)
                child[c] = _clamp_feature_value(f, v * scale + additive, bounds, realism_expand)

        elif f in INTEGER_FEATURES:
            for t in range(window_size):
                c = f"t{t}_{f}"
                if c not in child.index:
                    continue
                if f == "shortcut_flag":
                    if rng.random() < 0.25:
                        child[c] = 1.0 if float(child[c]) < 0.5 else 0.0
                elif f == "modifier_count":
                    if rng.random() < 0.35:
                        child[c] = float(rng.choice([0, 1, 2, 3], p=[0.45, 0.35, 0.17, 0.03]))
                elif f == "simultaneous_key_count":
                    if rng.random() < 0.35:
                        child[c] = float(rng.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.12, 0.03]))
                elif f == "current_key_category_id":
                    if rng.random() < 0.15:
                        child[c] = float(rng.integers(0, 8))

    # Segment-level burst mutation: closer to old adaptive mimic idea.
    if rng.random() < 0.45:
        start = int(rng.integers(0, max(1, window_size - 8)))
        length = int(rng.integers(3, min(12, window_size - start) + 1))
        factor = float(np.exp(rng.normal(0.0, local_strength * 0.8)))
        for t in range(start, start + length):
            for f in ["press_to_press_time", "release_to_release_time", "flight_time", "keys_per_second", "burst_density"]:
                c = f"t{t}_{f}"
                if c in child.index:
                    v = pd.to_numeric(child[c], errors="coerce")
                    v = 0.0 if not np.isfinite(v) else float(v)
                    child[c] = _clamp_feature_value(f, v * factor, bounds, realism_expand)

    child["label"] = "attack"
    child["input_source"] = "synthetic_hid"
    child["generator_version"] = "v8_1_fpr_constrained_blindspot_evolution"
    child["parent_attack_mode"] = str(parent.get("attack_mode", parent.get("scenario", "unknown")))
    child["attack_mode"] = f"v8_1_evo_{strategy}"
    child["scenario"] = child["attack_mode"]
    child["_v81_generation"] = generation + 1
    child["_v81_strategy"] = strategy
    child["_v81_child_idx"] = child_idx

    return child


def _make_children(
    cfg: Dict[str, Any],
    parents: pd.DataFrame,
    generation: int,
    round_name: str,
    bounds: Dict[str, Tuple[float, float, float, float]],
) -> pd.DataFrame:
    window_size = int(get_cfg(cfg, "window.window_size", 50))
    seed = int(get_cfg(cfg, "generation.seed", 42))
    rng = np.random.default_rng(seed + generation * 1013 + abs(hash(round_name)) % 100000)

    children_per_parent = int(get_cfg(cfg, "blindspot_evolution.children_per_parent", 4))
    mutation_strength = float(get_cfg(cfg, "blindspot_evolution.mutation_strength", 0.24))
    realism_expand = float(get_cfg(cfg, "blindspot_evolution.realism_expand", 1.40))

    rows = []
    for pidx, (_, parent) in enumerate(parents.iterrows()):
        for cidx in range(children_per_parent):
            rows.append(
                _mutate_parent(
                    rng=rng,
                    parent=parent,
                    generation=generation,
                    child_idx=pidx * children_per_parent + cidx,
                    window_size=window_size,
                    bounds=bounds,
                    mutation_strength=mutation_strength,
                    realism_expand=realism_expand,
                )
            )

    if not rows:
        return pd.DataFrame(columns=parents.columns)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_v81_") and c not in {"_v81_generation", "_v81_strategy"}], errors="ignore")
    out["window_id"] = [f"{round_name}_v81_g{generation+1}_{i:06d}" for i in range(len(out))]
    return out


def _drop_internal(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("_v81_") and c not in {"_v81_generation", "_v81_strategy"}], errors="ignore")


def _write_candidates(df: pd.DataFrame, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _drop_internal(df).to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def _select_hard(scored: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(scored) == 0:
        return scored
    x = scored.copy()
    if "attack_mode" not in x.columns:
        x["attack_mode"] = "unknown"

    x = x.sort_values(by=["_v81_is_fn", "_v81_hard_score"], ascending=[False, False])

    # Diversity-preserving head selection.
    parts = []
    modes = list(x["attack_mode"].dropna().unique())
    if modes and limit >= len(modes):
        per_mode = max(1, limit // min(len(modes), 10))
        for m in modes:
            parts.append(x[x["attack_mode"] == m].head(per_mode))
    diverse = pd.concat(parts, ignore_index=False) if parts else x.head(0)

    used_idx = set(diverse.index.tolist())
    rest = x[~x.index.isin(used_idx)]
    selected = pd.concat([diverse, rest], ignore_index=False).head(limit)

    return selected.reset_index(drop=True)


def generate_defender_aware_attacks(
    cfg: Dict[str, Any],
    defender_id: str,
    round_name: str = "round0_A0",
) -> Path:
    t0 = time.time()

    attack_root = Path(require_cfg(cfg, "paths.attack_generated_root"))
    selected_dir = attack_root / round_name
    selected_csv = selected_dir / "candidates.csv"

    enabled = bool(get_cfg(cfg, "blindspot_evolution.enabled", True))
    if not enabled:
        return generate_synthetic_attacks(cfg, round_name)

    generations = int(get_cfg(cfg, "blindspot_evolution.generations", 4))
    parents_per_generation = int(get_cfg(cfg, "blindspot_evolution.parents_per_generation", 100))
    selected_n = int(get_cfg(cfg, "defender_aware_generator.selected_candidates", 120))
    archive_limit = int(get_cfg(cfg, "blindspot_evolution.archive_limit", 8000))
    window_size = int(get_cfg(cfg, "window.window_size", 50))
    seed = int(get_cfg(cfg, "generation.seed", 42))

    print("[V8.1] FPR-constrained blind-spot evolutionary generator start", flush=True)
    print(f"  - defender : {defender_id}", flush=True)
    print(f"  - round    : {round_name}", flush=True)
    print(f"  - generations: {generations}", flush=True)
    print(f"  - parents/gen: {parents_per_generation}", flush=True)
    print(f"  - final selected: {selected_n}", flush=True)

    bounds = _normal_bounds(cfg, window_size=window_size)

    current_pool_csv = generate_synthetic_attacks(cfg, f"{round_name}_pool")
    current_pool = pd.read_csv(current_pool_csv)
    _available_feature_cols(current_pool, window_size=window_size)

    archive_scored = []
    generation_reports = []

    for g in range(generations):
        gen_name = f"{round_name}_v81_g{g}_screen_{defender_id}"
        eval_info = _evaluate_pool(cfg, defender_id, current_pool_csv, gen_name)
        preds = _read_predictions(eval_info)

        scored, info = _attach_hardness(
            candidates=current_pool,
            predictions=preds,
            eval_info=eval_info,
            defender_id=defender_id,
            cfg=cfg,
        )
        scored["_v81_generation"] = g
        archive_scored.append(scored)

        parents = _select_hard(scored, limit=min(parents_per_generation, len(scored)))

        report = {
            "generation": g,
            "pool_csv": str(current_pool_csv),
            "pool_size": int(len(current_pool)),
            "pred_col": info["pred_col"],
            "score_col": info["score_col"],
            "false_negative_count": info["false_negative_count"],
            "detected_count": info["detected_count"],
            "parents_selected": int(len(parents)),
            "pred_policy": info["pred_policy"],
        }
        generation_reports.append(report)

        print(
            f"[V8.1] gen={g} pool={len(current_pool)} "
            f"FN={info['false_negative_count']} detected={info['detected_count']} "
            f"pred_col={info['pred_col']} "
            f"policy={info['pred_policy'].get('selection_mode')} parents={len(parents)}",
            flush=True,
        )

        if g >= generations - 1:
            break

        children = _make_children(cfg, parents, g, round_name, bounds)

        next_pool = pd.concat([
            _drop_internal(parents),
            children,
        ], ignore_index=True)

        next_pool_limit = int(get_cfg(cfg, "blindspot_evolution.next_pool_limit", max(400, parents_per_generation * 5)))
        if len(next_pool) > next_pool_limit:
            # Keep parents, sample children if needed.
            parent_count = min(len(parents), next_pool_limit // 2)
            kept_parents = _drop_internal(parents).head(parent_count)
            remain_n = max(0, next_pool_limit - len(kept_parents))
            child_pool = children.sample(n=min(remain_n, len(children)), random_state=seed + g).reset_index(drop=True)
            next_pool = pd.concat([kept_parents, child_pool], ignore_index=True)

        next_dir = attack_root / f"{round_name}_v81_g{g+1}_pool"
        current_pool_csv = next_dir / "candidates.csv"
        _write_candidates(next_pool, current_pool_csv)
        current_pool = next_pool

    archive = pd.concat(archive_scored, ignore_index=True) if archive_scored else current_pool.copy()
    if "_v81_is_fn" not in archive.columns:
        archive["_v81_is_fn"] = 0
    if "_v81_hard_score" not in archive.columns:
        archive["_v81_hard_score"] = 0.0

    if len(archive) > archive_limit:
        archive = archive.sort_values(by=["_v81_is_fn", "_v81_hard_score"], ascending=[False, False]).head(archive_limit)

    selected = _select_hard(archive, limit=selected_n)
    selected["window_id"] = [f"{round_name}_v81_selected_{i:06d}" for i in range(len(selected))]
    selected["label"] = "attack"
    selected["input_source"] = "synthetic_hid"
    selected["generator_version"] = "v8_1_fpr_constrained_blindspot_evolution"

    _write_candidates(selected, selected_csv)

    archive_dir = attack_root / f"{round_name}_v81_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_csv = archive_dir / "scored_archive.csv"
    archive.to_csv(archive_csv, index=False, encoding="utf-8")

    selected_fn = int(selected["_v81_is_fn"].sum()) if "_v81_is_fn" in selected.columns else None
    summary = {
        "generator_version": "v8_1_fpr_constrained_blindspot_evolution",
        "defender": defender_id,
        "round": round_name,
        "selected": str(selected_csv),
        "selected_count": int(len(selected)),
        "selected_false_negative_count": selected_fn,
        "archive_csv": str(archive_csv),
        "archive_count": int(len(archive)),
        "generations": generation_reports,
        "elapsed_seconds": round(time.time() - t0, 3),
        "note": "Defensive feature-level FPR-constrained blind-spot candidate generation; no payload commands are generated.",
    }
    dump_json(selected_dir / "defender_aware_generation_summary.json", summary)

    print("[OK] V8.1 FPR-constrained blind-spot candidates selected", flush=True)
    print(f"  - defender : {defender_id}", flush=True)
    print(f"  - round    : {round_name}", flush=True)
    print(f"  - selected : {selected_csv}", flush=True)
    print(f"  - selected count: {len(selected)}", flush=True)
    if selected_fn is not None:
        print(f"  - selected FN   : {selected_fn}/{len(selected)}", flush=True)
    print(f"  - archive  : {archive_csv}", flush=True)
    print(f"  - elapsed  : {summary['elapsed_seconds']} sec", flush=True)

    return selected_csv
