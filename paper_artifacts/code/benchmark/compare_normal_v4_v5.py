import pandas as pd
from pathlib import Path

FILES = {
    "d0": {
        "v4": "reports/benchmark/featureguide_v4/oracle_d0_FeatureGuideHumanMimic_v4/oracle_predictions.csv",
        "v5": "reports/benchmark/adaptive_mimic_v5_test/oracle_d0_AdaptiveMimic_v5_test/oracle_predictions.csv",
    },
    "d1": {
        "v4": "reports/benchmark/featureguide_v4/oracle_d1_FeatureGuideHumanMimic_v4/oracle_predictions.csv",
        "v5": "reports/benchmark/adaptive_mimic_v5_test/oracle_d1_AdaptiveMimic_v5_test/oracle_predictions.csv",
    },
}

SCORE_COLS = [
    "reconstruction_error",
    "classifier_attack_probability",
    "prototype_attack_probability",
    "distance_to_normal_prototype",
    "distance_to_attack_prototype",
]

PRED_COLS = [
    "pred_classifier",
    "pred_prototype",
    "pred_recon_90",
    "pred_recon_95",
    "pred_recon_99",
    "pred_latent_dist_95",
    "pred_or95",
    "pred_or90",
]


def label_column(df: pd.DataFrame) -> str:
    if "true_label" in df.columns:
        return "true_label"
    if "label" in df.columns:
        return "label"
    raise KeyError("No label column found. Expected true_label or label.")


def summarize_subset(detector: str, dataset_name: str, df: pd.DataFrame) -> dict:
    row = {
        "detector": detector,
        "dataset": dataset_name,
        "samples": len(df),
    }

    for col in SCORE_COLS:
        if col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
            row[f"{col}_min"] = df[col].min()
            row[f"{col}_median"] = df[col].median()
            row[f"{col}_max"] = df[col].max()

    for col in PRED_COLS:
        if col in df.columns:
            row[f"{col}_rate"] = df[col].mean() * 100.0

    return row


def main():
    rows = []

    for detector, paths in FILES.items():
        v4 = pd.read_csv(paths["v4"])
        v5 = pd.read_csv(paths["v5"])

        label_col_v4 = label_column(v4)
        label_col_v5 = label_column(v5)

        normal = v4[v4[label_col_v4] == 0].copy()
        v4_attack = v4[v4[label_col_v4] == 1].copy()
        v5_attack = v5[v5[label_col_v5] == 1].copy()

        rows.append(summarize_subset(detector, "normal", normal))
        rows.append(summarize_subset(detector, "FeatureGuideHumanMimic_v4", v4_attack))
        rows.append(summarize_subset(detector, "AdaptiveMimic_v5_test", v5_attack))

    out = pd.DataFrame(rows)

    out_dir = Path("reports/benchmark/normal_v4_v5_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "normal_v4_v5_score_summary.csv"
    out.to_csv(out_path, index=False)

    # compact table for console
    compact_cols = [
        "detector",
        "dataset",
        "samples",
        "reconstruction_error_mean",
        "classifier_attack_probability_mean",
        "prototype_attack_probability_mean",
        "distance_to_normal_prototype_mean",
        "distance_to_attack_prototype_mean",
        "pred_classifier_rate",
        "pred_prototype_rate",
        "pred_or95_rate",
        "pred_or90_rate",
    ]

    compact_cols = [c for c in compact_cols if c in out.columns]
    compact = out[compact_cols]

    compact_path = out_dir / "normal_v4_v5_compact_summary.csv"
    compact.to_csv(compact_path, index=False)

    print("\n===== NORMAL / V4 / V5 COMPARISON =====")
    print(compact.to_string(index=False))

    print()
    print(f"saved: {out_path}")
    print(f"saved: {compact_path}")


if __name__ == "__main__":
    main()
