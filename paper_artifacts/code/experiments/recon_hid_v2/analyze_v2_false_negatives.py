import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import SELECTED_FEATURES, WINDOW_SIZE


DEFAULT_ORACLE_PRED_CSV = "results/recon_hid_v2/oracle/oracle_predictions.csv"
DEFAULT_TARGET_DIR = "data/attack/FeatureGuideHumanMimic"
DEFAULT_NORMAL_DIR = "data/processed"

DEFAULT_OUT_DIR = "results/recon_hid_v2/false_negative_analysis"

DEFAULT_FN_OR95_DIR = "data/attack/FeatureGuideHumanMimic_FN_or95"
DEFAULT_FN_OR90_DIR = "data/attack/FeatureGuideHumanMimic_FN_or90"


def resolve_csv_files(path):
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return files

    raise FileNotFoundError(f"Path not found: {path}")


def load_all_csv(path, exclude_filenames=None):
    exclude_filenames = exclude_filenames or set()
    rows = []

    files = resolve_csv_files(path)

    for file in files:
        base = os.path.basename(file)

        if base in exclude_filenames:
            continue

        df = pd.read_csv(file)
        df["source_file"] = base
        df["row_index_in_source"] = np.arange(len(df))
        rows.append(df)

    if not rows:
        raise ValueError(f"No CSV rows loaded from {path}")

    return pd.concat(rows, axis=0, ignore_index=True)


def assign_row_index_in_source(pred_df):
    """
    oracle_predictions.csv에는 원본 CSV row index가 없으므로,
    oracle 생성 순서와 동일하게 group/source_file별 누적 순서로 row index를 복구한다.
    """
    df = pred_df.copy()
    df = df.sort_values("global_index").reset_index(drop=True)

    df["row_index_in_source"] = (
        df
        .groupby(["group", "source_file"])
        .cumcount()
    )

    return df


def save_false_negative_feature_rows(
    fn_pred_df,
    target_dir,
    output_dir,
    combined_csv_path,
):
    """
    False negative prediction row에 대응하는 실제 feature-window row를 원본 CSV에서 뽑아 저장한다.
    이후 AdaptiveMimic_v5 생성기의 seed pool로 사용한다.
    """
    os.makedirs(output_dir, exist_ok=True)

    combined_rows = []

    for source_file, sub in fn_pred_df.groupby("source_file"):
        source_path = os.path.join(target_dir, source_file)

        if not os.path.exists(source_path):
            print(f"[WARN] Source file not found: {source_path}")
            continue

        source_df = pd.read_csv(source_path)
        selected_indices = sub["row_index_in_source"].astype(int).tolist()

        selected_df = source_df.iloc[selected_indices].copy()
        selected_df.insert(0, "source_file", source_file)
        selected_df.insert(1, "row_index_in_source", selected_indices)

        combined_rows.append(selected_df)

        # dataset loader가 그대로 읽을 수 있게 source별 파일도 저장
        # extra metadata columns가 있어도 현재 loader는 t*_feature만 읽기 때문에 문제 없음
        out_file = os.path.join(output_dir, source_file)
        selected_df.to_csv(out_file, index=False, encoding="utf-8")

    if combined_rows:
        combined_df = pd.concat(combined_rows, axis=0, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    combined_df.to_csv(combined_csv_path, index=False, encoding="utf-8")

    return combined_df


def window_feature_means(df):
    """
    각 window row에 대해 feature별 timestep 평균을 계산한다.
    output shape:
    row 단위로 hold_time_mean, flight_time_mean, ... 생성
    """
    rows = []

    for idx, row in df.iterrows():
        out = {
            "row_id": idx,
        }

        if "source_file" in df.columns:
            out["source_file"] = row["source_file"]

        if "row_index_in_source" in df.columns:
            out["row_index_in_source"] = row["row_index_in_source"]

        for feature in SELECTED_FEATURES:
            values = []

            for t in range(WINDOW_SIZE):
                col = f"t{t}_{feature}"

                if col in row.index:
                    value = row[col]
                    if pd.isna(value):
                        value = 0.0
                    values.append(float(value))

            if values:
                out[feature] = float(np.mean(values))
            else:
                out[feature] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)


def summarize_feature_means(feature_df, group_name):
    rows = []

    for feature in SELECTED_FEATURES:
        values = feature_df[feature].dropna()

        if len(values) == 0:
            continue

        rows.append(
            {
                "group": group_name,
                "feature": feature,
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p25": values.quantile(0.25),
                "p50": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "max": values.max(),
            }
        )

    return pd.DataFrame(rows)


def make_feature_comparison(normal_features, all_attack_features, fn_or95_features, detected_or95_features):
    normal_summary = summarize_feature_means(normal_features, "normal")
    all_attack_summary = summarize_feature_means(all_attack_features, "featureguide_v4_all")
    fn_summary = summarize_feature_means(fn_or95_features, "or95_false_negative")
    detected_summary = summarize_feature_means(detected_or95_features, "or95_detected")

    summary = pd.concat(
        [
            normal_summary,
            all_attack_summary,
            fn_summary,
            detected_summary,
        ],
        axis=0,
        ignore_index=True,
    )

    pivot = summary.pivot_table(
        index="feature",
        columns="group",
        values="mean",
        aggfunc="first",
    ).reset_index()

    # normal 대비 FN이 얼마나 다른지 보기 위한 단순 차이
    if "normal" in pivot.columns and "or95_false_negative" in pivot.columns:
        pivot["fn_minus_normal"] = (
            pivot["or95_false_negative"] - pivot["normal"]
        )

    if "normal" in pivot.columns and "or95_detected" in pivot.columns:
        pivot["detected_minus_normal"] = (
            pivot["or95_detected"] - pivot["normal"]
        )

    return summary, pivot


def plot_source_fn_summary(summary_df, out_dir):
    attack_summary = summary_df[
        summary_df["group"] == "FeatureGuideHumanMimic_v4"
    ].copy()

    if attack_summary.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(
        attack_summary["source_file"],
        attack_summary["or95_fn_count"],
    )
    plt.ylabel("False negative count")
    plt.title("OR95 False Negatives by Source File")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "01_or95_false_negative_count_by_source.png"),
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(
        attack_summary["source_file"],
        attack_summary["or95_fn_rate_percent"],
    )
    plt.ylabel("False negative rate (%)")
    plt.ylim(0, 100)
    plt.title("OR95 False Negative Rate by Source File")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "02_or95_false_negative_rate_by_source.png"),
        dpi=300,
    )
    plt.close()


def plot_feature_diff(feature_pivot, out_dir):
    if "fn_minus_normal" not in feature_pivot.columns:
        return

    plot_df = feature_pivot.copy()
    plot_df["abs_fn_minus_normal"] = plot_df["fn_minus_normal"].abs()
    plot_df = plot_df.sort_values("abs_fn_minus_normal", ascending=False)

    plt.figure(figsize=(12, 7))
    plt.bar(
        plot_df["feature"],
        plot_df["fn_minus_normal"],
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("False negative mean - normal mean")
    plt.title("Feature Mean Difference: OR95 False Negatives vs Normal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "03_fn_vs_normal_feature_mean_difference.png"),
        dpi=300,
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ReCon-HID v2 false negatives",
    )

    parser.add_argument(
        "--oracle-pred-csv",
        default=DEFAULT_ORACLE_PRED_CSV,
    )

    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
    )

    parser.add_argument(
        "--normal-dir",
        default=DEFAULT_NORMAL_DIR,
    )

    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--fn-or95-dir",
        default=DEFAULT_FN_OR95_DIR,
    )

    parser.add_argument(
        "--fn-or90-dir",
        default=DEFAULT_FN_OR90_DIR,
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pred_df = pd.read_csv(args.oracle_pred_csv)
    pred_df = assign_row_index_in_source(pred_df)

    # 공격 샘플만 대상으로 FN 분석
    attack_df = pred_df[pred_df["true_label"] == 1].copy()

    # OR95 / OR90 false negative
    fn_or95 = attack_df[attack_df["pred_or95"] == 0].copy()
    fn_or90 = attack_df[attack_df["pred_or90"] == 0].copy()

    detected_or95 = attack_df[attack_df["pred_or95"] == 1].copy()
    detected_or90 = attack_df[attack_df["pred_or90"] == 1].copy()

    # 저장: prediction 기준 FN 목록
    fn_or95_csv = os.path.join(args.out_dir, "false_negative_predictions_or95.csv")
    fn_or90_csv = os.path.join(args.out_dir, "false_negative_predictions_or90.csv")
    detected_or95_csv = os.path.join(args.out_dir, "detected_predictions_or95.csv")
    detected_or90_csv = os.path.join(args.out_dir, "detected_predictions_or90.csv")

    fn_or95.to_csv(fn_or95_csv, index=False, encoding="utf-8")
    fn_or90.to_csv(fn_or90_csv, index=False, encoding="utf-8")
    detected_or95.to_csv(detected_or95_csv, index=False, encoding="utf-8")
    detected_or90.to_csv(detected_or90_csv, index=False, encoding="utf-8")

    # 저장: 실제 feature-window row
    fn_or95_features_csv = os.path.join(args.out_dir, "false_negative_features_or95.csv")
    fn_or90_features_csv = os.path.join(args.out_dir, "false_negative_features_or90.csv")
    detected_or95_features_csv = os.path.join(args.out_dir, "detected_features_or95.csv")

    fn_or95_features = save_false_negative_feature_rows(
        fn_pred_df=fn_or95,
        target_dir=args.target_dir,
        output_dir=args.fn_or95_dir,
        combined_csv_path=fn_or95_features_csv,
    )

    fn_or90_features = save_false_negative_feature_rows(
        fn_pred_df=fn_or90,
        target_dir=args.target_dir,
        output_dir=args.fn_or90_dir,
        combined_csv_path=fn_or90_features_csv,
    )

    detected_or95_features = save_false_negative_feature_rows(
        fn_pred_df=detected_or95,
        target_dir=args.target_dir,
        output_dir=os.path.join(args.out_dir, "detected_or95_by_source"),
        combined_csv_path=detected_or95_features_csv,
    )

    # source별 FN 요약
    source_summary = (
        attack_df
        .groupby(["group", "source_file"])
        .agg(
            count=("true_label", "count"),
            or95_detected_count=("pred_or95", "sum"),
            or90_detected_count=("pred_or90", "sum"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            mean_classifier_attack_probability=("classifier_attack_probability", "mean"),
            mean_prototype_attack_probability=("prototype_attack_probability", "mean"),
            mean_distance_to_normal_prototype=("distance_to_normal_prototype", "mean"),
        )
        .reset_index()
    )

    source_summary["or95_fn_count"] = (
        source_summary["count"] - source_summary["or95_detected_count"]
    )
    source_summary["or90_fn_count"] = (
        source_summary["count"] - source_summary["or90_detected_count"]
    )

    source_summary["or95_detection_rate_percent"] = (
        source_summary["or95_detected_count"] / source_summary["count"] * 100
    )
    source_summary["or90_detection_rate_percent"] = (
        source_summary["or90_detected_count"] / source_summary["count"] * 100
    )

    source_summary["or95_fn_rate_percent"] = (
        source_summary["or95_fn_count"] / source_summary["count"] * 100
    )
    source_summary["or90_fn_rate_percent"] = (
        source_summary["or90_fn_count"] / source_summary["count"] * 100
    )

    source_summary_csv = os.path.join(args.out_dir, "false_negative_source_summary.csv")
    source_summary.to_csv(source_summary_csv, index=False, encoding="utf-8")

    # 전체 요약
    overall_rows = [
        {
            "method": "OR95 balanced",
            "total_attack": len(attack_df),
            "detected": len(detected_or95),
            "false_negative": len(fn_or95),
            "detection_rate_percent": len(detected_or95) / len(attack_df) * 100,
            "false_negative_rate_percent": len(fn_or95) / len(attack_df) * 100,
        },
        {
            "method": "OR90 aggressive",
            "total_attack": len(attack_df),
            "detected": len(detected_or90),
            "false_negative": len(fn_or90),
            "detection_rate_percent": len(detected_or90) / len(attack_df) * 100,
            "false_negative_rate_percent": len(fn_or90) / len(attack_df) * 100,
        },
    ]

    overall_summary = pd.DataFrame(overall_rows)
    overall_summary_csv = os.path.join(args.out_dir, "false_negative_overall_summary.csv")
    overall_summary.to_csv(overall_summary_csv, index=False, encoding="utf-8")

    # feature mean 비교
    normal_raw = load_all_csv(args.normal_dir)
    attack_raw = load_all_csv(args.target_dir, exclude_filenames={"window5.csv"})

    normal_feature_means = window_feature_means(normal_raw)
    attack_feature_means = window_feature_means(attack_raw)
    fn_or95_feature_means = window_feature_means(fn_or95_features)
    detected_or95_feature_means = window_feature_means(detected_or95_features)

    feature_summary, feature_pivot = make_feature_comparison(
        normal_feature_means,
        attack_feature_means,
        fn_or95_feature_means,
        detected_or95_feature_means,
    )

    feature_summary_csv = os.path.join(args.out_dir, "feature_mean_summary.csv")
    feature_pivot_csv = os.path.join(args.out_dir, "feature_mean_comparison_pivot.csv")

    feature_summary.to_csv(feature_summary_csv, index=False, encoding="utf-8")
    feature_pivot.to_csv(feature_pivot_csv, index=False, encoding="utf-8")

    # plots
    plot_source_fn_summary(source_summary, args.out_dir)
    plot_feature_diff(feature_pivot, args.out_dir)

    print("\n===== ReCon-HID v2 False Negative Analysis Done =====")

    print("\nOverall Summary:")
    print(overall_summary.to_string(index=False))

    print("\nSource Summary:")
    print(source_summary.to_string(index=False))

    print("\nFeature Mean Comparison:")
    print(feature_pivot.to_string(index=False))

    print("\nSaved files:")
    print(f"- {fn_or95_csv}")
    print(f"- {fn_or90_csv}")
    print(f"- {fn_or95_features_csv}")
    print(f"- {fn_or90_features_csv}")
    print(f"- {source_summary_csv}")
    print(f"- {overall_summary_csv}")
    print(f"- {feature_summary_csv}")
    print(f"- {feature_pivot_csv}")
    print(f"- FN OR95 feature dir: {args.fn_or95_dir}")
    print(f"- FN OR90 feature dir: {args.fn_or90_dir}")
    print(f"- Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()

