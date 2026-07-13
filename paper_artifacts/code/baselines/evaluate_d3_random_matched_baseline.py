from pathlib import Path
import subprocess
import sys
import pandas as pd

FORCE = True

DETECTORS = {
    "D3_guided": {
        "model_path": Path("checkpoints/recon_hid_v2_round3.pt"),
        "description": "D3 trained with detector-guided V6/V7 hard negatives",
    },
    "D3_random_matched": {
        "model_path": Path("checkpoints/recon_hid_v2_round3_random_matched.pt"),
        "description": "D3 baseline trained with random-matched V6/V7 candidates",
    },
}

ATTACKS = [
    {
        "short": "V4",
        "attack_name": "FeatureGuideHumanMimic_v4",
        "attack_dir": Path("data/attack/FeatureGuideHumanMimic"),
    },
    {
        "short": "V5_test",
        "attack_name": "AdaptiveMimic_v5_constrained_test",
        "attack_dir": Path("data/attack/AdaptiveMimic_v5_constrained_test"),
    },
    {
        "short": "V6_test",
        "attack_name": "AdaptiveMimic_v6_D1_OR90_bypass_test",
        "attack_dir": Path("data/attack/AdaptiveMimic_v6_D1_OR90_bypass_test"),
    },
    {
        "short": "V7_test",
        "attack_name": "AdaptiveMimic_v7_D2_OR90_bypass_test",
        "attack_dir": Path("data/attack/AdaptiveMimic_v7_D2_OR90_bypass_test"),
    },
    {
        "short": "V8_candidate",
        "attack_name": "AdaptiveMimic_v8_candidate",
        "attack_dir": Path("data/attack/AdaptiveMimic_v8_candidate"),
    },
    {
        "short": "V8_hardest",
        "attack_name": "AdaptiveMimic_v8_D3_hardest_no_bypass",
        "attack_dir": Path("data/attack/AdaptiveMimic_v8_D3_hardest_no_bypass"),
    },
]

REPORT_ROOT = Path("reports/benchmark/d3_random_matched_baseline")
OUT_DIR = Path("paper_artifacts/tables/baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_oracle(detector_key, detector, attack, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = list(out_dir.rglob("*.csv"))
    if existing_csvs and not FORCE:
        print(f"[SKIP] {detector_key} {attack['short']} already exists: {out_dir}")
        return

    model_path = detector["model_path"]
    attack_dir = attack["attack_dir"]

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    if not attack_dir.exists():
        raise FileNotFoundError(f"Missing attack dir: {attack_dir}")

    cmd = [
        sys.executable,
        "-m",
        "src.detector_oracle_v2",
        "--model-path",
        str(model_path),
        "--target-path",
        str(attack_dir),
        "--target-group",
        attack["attack_name"],
        "--out-dir",
        str(out_dir),
    ]

    print()
    print(f"[RUN] {detector_key} vs {attack['short']}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def find_prediction_csv(out_dir):
    csvs = list(out_dir.rglob("*.csv"))
    if not csvs:
        return None

    prediction_like = [
        p for p in csvs
        if "prediction" in p.name.lower() or "predictions" in p.name.lower()
    ]
    if prediction_like:
        return max(prediction_like, key=lambda p: p.stat().st_size)

    return max(csvs, key=lambda p: p.stat().st_size)


def bool01(series):
    if series.dtype == bool:
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)

    lowered = series.astype(str).str.lower().str.strip()
    return lowered.isin(["1", "true", "yes", "attack", "detected", "malicious"]).astype(int)


def infer_masks(df):
    label_candidates = ["label", "true_label", "target", "y", "is_attack"]

    for col in label_candidates:
        if col in df.columns:
            vals = df[col]
            if pd.api.types.is_numeric_dtype(vals):
                attack_mask = vals.astype(float) > 0
                normal_mask = vals.astype(float) == 0
                if attack_mask.any():
                    return attack_mask, normal_mask

            lowered = vals.astype(str).str.lower().str.strip()
            attack_mask = lowered.isin(["1", "attack", "malicious", "true"])
            normal_mask = lowered.isin(["0", "normal", "benign", "false"])
            if attack_mask.any():
                return attack_mask, normal_mask

    for col in ["group", "dataset", "source_group", "target_group"]:
        if col in df.columns:
            lowered = df[col].astype(str).str.lower().str.strip()
            normal_mask = lowered == "normal"
            attack_mask = ~normal_mask
            if attack_mask.any():
                return attack_mask, normal_mask

    for col in ["source_file", "source", "file"]:
        if col in df.columns:
            lowered = df[col].astype(str).str.lower()
            normal_mask = lowered.str.contains("normal") | lowered.str.contains("typing")
            attack_mask = ~normal_mask
            if attack_mask.any():
                return attack_mask, normal_mask

    attack_mask = pd.Series([True] * len(df), index=df.index)
    normal_mask = pd.Series([False] * len(df), index=df.index)
    return attack_mask, normal_mask


def safe_rate(decision, mask):
    if mask.sum() == 0:
        return None
    return float(decision[mask].mean() * 100.0)


def summarize_predictions(detector_key, detector, attack, out_dir):
    pred_csv = find_prediction_csv(out_dir)

    if pred_csv is None:
        raise FileNotFoundError(f"No prediction CSV found in {out_dir}")

    df = pd.read_csv(pred_csv)
    attack_mask, normal_mask = infer_masks(df)

    row = {
        "detector": detector_key,
        "detector_description": detector["description"],
        "attack": attack["attack_name"],
        "attack_short": attack["short"],
        "prediction_csv": str(pred_csv),
        "attack_samples": int(attack_mask.sum()),
        "normal_samples": int(normal_mask.sum()),
    }

    decision_cols = {
        "or95": "pred_or95",
        "or90": "pred_or90",
        "classifier": "pred_classifier",
        "prototype": "pred_prototype",
        "recon90": "pred_recon_90",
        "recon95": "pred_recon_95",
        "latent90": "pred_latent_dist_90",
        "latent95": "pred_latent_dist_95",
        "or95_with_latent": "pred_or95_with_latent",
    }

    for name, col in decision_cols.items():
        if col not in df.columns:
            continue

        decision = bool01(df[col])
        det_rate = safe_rate(decision, attack_mask)
        fpr = safe_rate(decision, normal_mask)

        row[f"{name}_detection_rate"] = det_rate
        row[f"{name}_bypass_rate"] = None if det_rate is None else 100.0 - det_rate
        row[f"{name}_normal_fpr"] = fpr

    score_cols = [
        "reconstruction_error",
        "classifier_attack_probability",
        "prototype_attack_probability",
        "distance_to_normal_prototype",
        "distance_to_attack_prototype",
        "latent_norm",
    ]

    for col in score_cols:
        if col in df.columns:
            row[f"mean_{col}"] = float(pd.to_numeric(df.loc[attack_mask, col], errors="coerce").mean())

    return row


def write_md(df, path):
    tmp = df.copy().fillna("")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = []
    cols = list(tmp.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, r in tmp.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = []

    for attack in ATTACKS:
        for detector_key, detector in DETECTORS.items():
            out_dir = REPORT_ROOT / attack["short"] / f"oracle_{detector_key}_{attack['attack_name']}"
            run_oracle(detector_key, detector, attack, out_dir)
            row = summarize_predictions(detector_key, detector, attack, out_dir)
            rows.append(row)

    summary = pd.DataFrame(rows)

    summary_csv = OUT_DIR / "d3_guided_vs_random_matched_summary.csv"
    summary_md = OUT_DIR / "d3_guided_vs_random_matched_summary.md"
    summary.to_csv(summary_csv, index=False)
    write_md(summary, summary_md)

    print()
    print("=== Full summary ===")
    cols_to_show = [
        "detector",
        "attack_short",
        "attack_samples",
        "normal_samples",
        "or95_detection_rate",
        "or95_bypass_rate",
        "or90_detection_rate",
        "or90_bypass_rate",
        "classifier_detection_rate",
        "prototype_detection_rate",
        "recon95_detection_rate",
        "latent95_detection_rate",
    ]
    cols_to_show = [c for c in cols_to_show if c in summary.columns]
    print(summary[cols_to_show].round(2).to_string(index=False))

    print(f"[saved] {summary_csv}")
    print(f"[saved] {summary_md}")

    # Pivot 1: OR95 detection
    pivot_or95 = summary.pivot_table(
        index="detector",
        columns="attack_short",
        values="or95_detection_rate",
        aggfunc="mean",
    ).round(2)

    pivot_or95 = pivot_or95[
        [c for c in ["V4", "V5_test", "V6_test", "V7_test", "V8_candidate", "V8_hardest"] if c in pivot_or95.columns]
    ]

    pivot_csv = OUT_DIR / "d3_guided_vs_random_matched_or95_detection.csv"
    pivot_md = OUT_DIR / "d3_guided_vs_random_matched_or95_detection.md"
    pivot_or95.to_csv(pivot_csv)
    write_md(pivot_or95.reset_index(), pivot_md)

    print()
    print("=== OR95 detection pivot ===")
    print(pivot_or95.to_string())
    print(f"[saved] {pivot_csv}")
    print(f"[saved] {pivot_md}")

    # Pivot 2: OR90 bypass
    pivot_or90_bypass = summary.pivot_table(
        index="detector",
        columns="attack_short",
        values="or90_bypass_rate",
        aggfunc="mean",
    ).round(2)

    pivot_or90_bypass = pivot_or90_bypass[
        [c for c in ["V4", "V5_test", "V6_test", "V7_test", "V8_candidate", "V8_hardest"] if c in pivot_or90_bypass.columns]
    ]

    pivot_csv = OUT_DIR / "d3_guided_vs_random_matched_or90_bypass.csv"
    pivot_md = OUT_DIR / "d3_guided_vs_random_matched_or90_bypass.md"
    pivot_or90_bypass.to_csv(pivot_csv)
    write_md(pivot_or90_bypass.reset_index(), pivot_md)

    print()
    print("=== OR90 bypass pivot ===")
    print(pivot_or90_bypass.to_string())
    print(f"[saved] {pivot_csv}")
    print(f"[saved] {pivot_md}")


if __name__ == "__main__":
    main()
