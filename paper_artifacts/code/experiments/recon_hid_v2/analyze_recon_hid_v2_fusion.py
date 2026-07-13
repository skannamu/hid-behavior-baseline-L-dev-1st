import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PRED_CSV = "results/recon_hid_v2/recon_hid_v2_predictions_featureguide_v4.csv"
OUT_DIR = "results/recon_hid_v2/fusion_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_metrics(df, pred_col, method):
    y_true = df["true_label"].to_numpy()
    y_pred = df[pred_col].to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)

    return {
        "method": method,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall_attack_detection_rate": recall,
        "f1": f1,
        "normal_false_positive_rate": fpr,
    }


def robust_score(series, normal_series):
    """
    normal 분포 기준 robust normalization.
    normal median을 0 근처,
    normal 95% quantile을 1 근처로 맞춘다.
    """
    median = normal_series.quantile(0.50)
    q95 = normal_series.quantile(0.95)
    denom = max(q95 - median, 1e-12)

    score = (series - median) / denom
    score = score.clip(lower=0.0, upper=10.0)

    return score


def main():
    df = pd.read_csv(PRED_CSV)

    normal = df[df["true_label"] == 0].copy()
    attack = df[df["true_label"] == 1].copy()

    # 기존 scalar score들
    df["score_recon"] = robust_score(
        df["reconstruction_error"],
        normal["reconstruction_error"],
    )

    df["score_latent"] = robust_score(
        df["distance_to_normal_prototype"],
        normal["distance_to_normal_prototype"],
    )

    # probability는 0~1이라 그대로 사용하되, score scale에 맞춰 2배 정도만 반영
    df["score_classifier"] = df["classifier_attack_probability"] * 2.0
    df["score_prototype"] = df["prototype_attack_probability"] * 2.0

    # 여러 fusion score 후보
    df["fusion_score_recon_cls"] = (
        1.0 * df["score_recon"]
        + 0.5 * df["score_classifier"]
    )

    df["fusion_score_recon_proto"] = (
        1.0 * df["score_recon"]
        + 0.5 * df["score_prototype"]
    )

    df["fusion_score_recon_cls_proto"] = (
        1.0 * df["score_recon"]
        + 0.5 * df["score_classifier"]
        + 0.5 * df["score_prototype"]
    )

    df["fusion_score_all"] = (
        1.0 * df["score_recon"]
        + 0.5 * df["score_classifier"]
        + 0.5 * df["score_prototype"]
        + 0.5 * df["score_latent"]
    )

    normal = df[df["true_label"] == 0].copy()

    summary_rows = []

    # 1) OR rule
    recon90 = normal["reconstruction_error"].quantile(0.90)
    recon95 = normal["reconstruction_error"].quantile(0.95)
    recon99 = normal["reconstruction_error"].quantile(0.99)

    df["pred_recon90"] = (df["reconstruction_error"] > recon90).astype(int)
    df["pred_recon95"] = (df["reconstruction_error"] > recon95).astype(int)
    df["pred_recon99"] = (df["reconstruction_error"] > recon99).astype(int)

    df["pred_cls"] = (df["classifier_attack_probability"] >= 0.5).astype(int)
    df["pred_proto"] = (df["prototype_attack_probability"] >= 0.5).astype(int)

    df["pred_or_cls_proto_recon95"] = (
        (df["pred_cls"] == 1)
        | (df["pred_proto"] == 1)
        | (df["pred_recon95"] == 1)
    ).astype(int)

    df["pred_or_cls_proto_recon90"] = (
        (df["pred_cls"] == 1)
        | (df["pred_proto"] == 1)
        | (df["pred_recon90"] == 1)
    ).astype(int)

    summary_rows.append(compute_metrics(df, "pred_recon90", "recon90_only"))
    summary_rows.append(compute_metrics(df, "pred_recon95", "recon95_only"))
    summary_rows.append(compute_metrics(df, "pred_recon99", "recon99_only"))
    summary_rows.append(compute_metrics(df, "pred_cls", "classifier_only"))
    summary_rows.append(compute_metrics(df, "pred_proto", "prototype_only"))
    summary_rows.append(compute_metrics(df, "pred_or_cls_proto_recon95", "or_cls_proto_recon95"))
    summary_rows.append(compute_metrics(df, "pred_or_cls_proto_recon90", "or_cls_proto_recon90"))

    # 2) fusion score quantile threshold
    fusion_cols = [
        "fusion_score_recon_cls",
        "fusion_score_recon_proto",
        "fusion_score_recon_cls_proto",
        "fusion_score_all",
    ]

    for col in fusion_cols:
        for q in [0.90, 0.95, 0.99]:
            threshold = normal[col].quantile(q)
            pred_col = f"pred_{col}_q{int(q * 100)}"
            df[pred_col] = (df[col] > threshold).astype(int)

            summary_rows.append(
                compute_metrics(
                    df,
                    pred_col,
                    f"{col}_normal_q{int(q * 100)}_{threshold:.6f}",
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUT_DIR, "fusion_summary.csv"),
        index=False,
        encoding="utf-8",
    )

    df.to_csv(
        os.path.join(OUT_DIR, "fusion_predictions.csv"),
        index=False,
        encoding="utf-8",
    )

    # source별 best 후보 요약
    source_summary = (
        df
        .groupby(["group", "source_file"])
        .agg(
            count=("true_label", "count"),
            mean_recon=("reconstruction_error", "mean"),
            mean_cls_prob=("classifier_attack_probability", "mean"),
            mean_proto_prob=("prototype_attack_probability", "mean"),
            mean_latent_dist=("distance_to_normal_prototype", "mean"),
            recon90_detection=("pred_recon90", "mean"),
            recon95_detection=("pred_recon95", "mean"),
            or_recon95_detection=("pred_or_cls_proto_recon95", "mean"),
            or_recon90_detection=("pred_or_cls_proto_recon90", "mean"),
        )
        .reset_index()
    )

    source_summary.to_csv(
        os.path.join(OUT_DIR, "fusion_source_summary.csv"),
        index=False,
        encoding="utf-8",
    )

    # 그래프: 주요 method 비교
    plot_df = summary_df[
        summary_df["method"].isin([
            "classifier_only",
            "prototype_only",
            "recon90_only",
            "recon95_only",
            "or_cls_proto_recon95",
            "or_cls_proto_recon90",
        ])
    ].copy()

    plot_df["recall_percent"] = plot_df["recall_attack_detection_rate"] * 100
    plot_df["fpr_percent"] = plot_df["normal_false_positive_rate"] * 100

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["method"], plot_df["recall_percent"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Attack detection rate (%)")
    plt.title("ReCon-HID v2 Fusion Detection Rate")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "01_fusion_detection_rate.png"),
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["method"], plot_df["fpr_percent"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Normal false positive rate (%)")
    plt.title("ReCon-HID v2 Fusion False Positive Rate")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "02_fusion_false_positive_rate.png"),
        dpi=300,
    )
    plt.close()

    print("\n===== ReCon-HID v2 Fusion Analysis Done =====")
    print(summary_df.to_string(index=False))
    print("\n===== Source Summary =====")
    print(source_summary.to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
