import os
import glob
import math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


NORMAL_DATA_PATH = "data/processed"
ATTACK_ROOT_PATH = "data/attack"

OUTPUT_DIR = "results/feature_distribution_compare_all_no_boxplot"

# 공식 결과 기준: window5는 없었던 것으로 처리
EXCLUDE_FILENAMES = {"window5.csv"}

SELECTED_FEATURES = [
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

FOCUS_FEATURES = [
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
    "correction_ratio",
]

RANDOM_STATE = 42
MAX_PLOT_SAMPLES_PER_GROUP = 50000


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_csv_files(path, exclude_filenames=None):
    exclude_filenames = exclude_filenames or set()

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    files = [
        f for f in files
        if os.path.basename(f) not in exclude_filenames
    ]

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    return files


def discover_attack_groups():
    root = Path(ATTACK_ROOT_PATH)

    if not root.exists():
        raise FileNotFoundError(f"Attack root not found: {ATTACK_ROOT_PATH}")

    groups = {}

    # data/attack 바로 아래 csv가 있으면 하나의 그룹으로 처리
    root_csvs = sorted(root.glob("*.csv"))
    root_csvs = [
        str(p) for p in root_csvs
        if p.name not in EXCLUDE_FILENAMES
    ]
    if root_csvs:
        groups["attack_root"] = root_csvs

    # data/attack/HumanMimic, StrongHumanMimic, FeatureGuideHumanMimic 등 자동 탐색
    for child in sorted(root.iterdir()):
        if child.is_dir():
            files = sorted(child.glob("*.csv"))
            files = [
                str(p) for p in files
                if p.name not in EXCLUDE_FILENAMES
            ]

            if files:
                groups[child.name] = files

    if not groups:
        raise FileNotFoundError(f"No attack CSV files found under: {ATTACK_ROOT_PATH}")

    return groups


def load_csv_group(group_name, files_or_path):
    if isinstance(files_or_path, list):
        files = files_or_path
    else:
        files = resolve_csv_files(files_or_path)

    dfs = []

    print(f"\n[{group_name}]")
    for f in files:
        print(f"  - {f}")
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        df["group"] = group_name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    print(f"  rows: {len(out)}")

    return out


def extract_feature_values(df, feature):
    # event/state CSV처럼 feature 컬럼이 직접 있는 경우
    if feature in df.columns:
        values = df[feature].to_numpy(dtype=float)
        return values[np.isfinite(values)]

    # window CSV: t0_hold_time, t1_hold_time, ..., t49_hold_time
    timestep_cols = []

    for col in df.columns:
        if col.endswith("_" + feature):
            prefix = col[:-(len(feature) + 1)]
            if prefix.startswith("t") and prefix[1:].isdigit():
                timestep_cols.append(col)

    timestep_cols = sorted(
        timestep_cols,
        key=lambda x: int(x.split("_", 1)[0][1:])
    )

    if not timestep_cols:
        raise ValueError(f"Feature columns not found: {feature}")

    values = df[timestep_cols].to_numpy(dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    return values


def build_long_feature_df(groups):
    records = []

    for group_name, df in groups.items():
        for feature in SELECTED_FEATURES:
            values = extract_feature_values(df, feature)

            tmp = pd.DataFrame({
                "group": group_name,
                "feature": feature,
                "value": values,
            })

            records.append(tmp)

    return pd.concat(records, ignore_index=True)


def summarize_feature_distribution(long_df):
    rows = []

    for (group, feature), sub in long_df.groupby(["group", "feature"]):
        x = sub["value"].dropna().to_numpy()

        q25 = np.quantile(x, 0.25)
        q50 = np.quantile(x, 0.50)
        q75 = np.quantile(x, 0.75)

        rows.append({
            "group": group,
            "feature": feature,
            "count": len(x),
            "mean": np.mean(x),
            "std": np.std(x, ddof=1),
            "min": np.min(x),
            "q25": q25,
            "median": q50,
            "q75": q75,
            "iqr": q75 - q25,
            "q90": np.quantile(x, 0.90),
            "q95": np.quantile(x, 0.95),
            "q99": np.quantile(x, 0.99),
            "max": np.max(x),
        })

    return pd.DataFrame(rows)


def ks_statistic(x, y):
    x = np.sort(np.asarray(x))
    y = np.sort(np.asarray(y))

    data_all = np.sort(np.concatenate([x, y]))

    cdf_x = np.searchsorted(x, data_all, side="right") / len(x)
    cdf_y = np.searchsorted(y, data_all, side="right") / len(y)

    return np.max(np.abs(cdf_x - cdf_y))


def compare_against_normal_including_normal(long_df, summary_df):
    rows = []

    normal_summary = summary_df[summary_df["group"] == "normal"].set_index("feature")
    all_groups = list(summary_df["group"].drop_duplicates())

    for group in all_groups:
        for feature in SELECTED_FEATURES:
            normal_values = long_df[
                (long_df["group"] == "normal") &
                (long_df["feature"] == feature)
            ]["value"].to_numpy()

            group_values = long_df[
                (long_df["group"] == group) &
                (long_df["feature"] == feature)
            ]["value"].to_numpy()

            n = normal_summary.loc[feature]
            g = summary_df[
                (summary_df["group"] == group) &
                (summary_df["feature"] == feature)
            ].iloc[0]

            normal_mean = n["mean"]
            normal_std = n["std"] if n["std"] != 0 else np.nan

            mean_diff = g["mean"] - normal_mean
            z_mean_diff = mean_diff / normal_std if normal_std and not math.isnan(normal_std) else np.nan

            if group == "normal":
                ks = 0.0
                z_mean_diff = 0.0
                std_ratio = 1.0
                iqr_ratio = 1.0
            else:
                ks = ks_statistic(normal_values, group_values)
                std_ratio = g["std"] / n["std"] if n["std"] != 0 else np.nan
                iqr_ratio = g["iqr"] / n["iqr"] if n["iqr"] != 0 else np.nan

            rows.append({
                "group": group,
                "feature": feature,
                "normal_mean": normal_mean,
                "group_mean": g["mean"],
                "mean_diff": mean_diff,
                "z_mean_diff": z_mean_diff,
                "normal_std": n["std"],
                "group_std": g["std"],
                "std_ratio_to_normal": std_ratio,
                "normal_iqr": n["iqr"],
                "group_iqr": g["iqr"],
                "iqr_ratio_to_normal": iqr_ratio,
                "ks_statistic": ks,
            })

    return pd.DataFrame(rows)


def sample_values(x, max_n=MAX_PLOT_SAMPLES_PER_GROUP):
    if len(x) <= max_n:
        return x

    rng = np.random.default_rng(RANDOM_STATE)
    return rng.choice(x, size=max_n, replace=False)


def plot_histograms(long_df, group_order):
    ensure_dir(OUTPUT_DIR)

    for feature in FOCUS_FEATURES:
        plt.figure(figsize=(11, 6))

        all_feature_values = long_df[
            long_df["feature"] == feature
        ]["value"].dropna().to_numpy()

        # 모든 그룹이 같은 x축 범위를 쓰도록 global 99% 사용
        global_upper = np.quantile(all_feature_values, 0.99)
        global_lower = np.quantile(all_feature_values, 0.01)

        if global_lower == global_upper:
            global_lower = np.min(all_feature_values)
            global_upper = np.max(all_feature_values)

        for group in group_order:
            x = long_df[
                (long_df["group"] == group) &
                (long_df["feature"] == feature)
            ]["value"].dropna().to_numpy()

            x = sample_values(x)
            x_plot = x[(x >= global_lower) & (x <= global_upper)]

            if len(x_plot) == 0:
                continue

            plt.hist(
                x_plot,
                bins=60,
                alpha=0.35,
                density=True,
                label=group,
            )

        plt.title(f"{feature} histogram: normal and all attack groups")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"hist_{feature}_all_groups.png")
        plt.savefig(out, dpi=220)
        plt.close()
        print(f"Saved: {out}")


def plot_mean_feature_comparison(summary_df, group_order):
    ensure_dir(OUTPUT_DIR)

    for feature in FOCUS_FEATURES:
        sub = summary_df[
            summary_df["feature"] == feature
        ].set_index("group").loc[group_order].reset_index()

        plt.figure(figsize=(10, 6))

        x = np.arange(len(sub))
        plt.bar(x, sub["mean"])

        for i, value in enumerate(sub["mean"]):
            plt.text(
                i,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xticks(x, sub["group"], rotation=30, ha="right")
        plt.ylabel(f"Mean {feature}")
        plt.title(f"Mean {feature}: normal and all attack groups")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"mean_{feature}_all_groups.png")
        plt.savefig(out, dpi=220)
        plt.close()
        print(f"Saved: {out}")


def plot_distance_bars(compare_df, group_order):
    ensure_dir(OUTPUT_DIR)

    metrics = [
        "ks_statistic",
        "z_mean_diff",
        "std_ratio_to_normal",
        "iqr_ratio_to_normal",
    ]

    for metric in metrics:
        pivot = compare_df.pivot(
            index="feature",
            columns="group",
            values=metric,
        )

        pivot = pivot.loc[FOCUS_FEATURES, group_order]

        plt.figure(figsize=(14, 7))

        x = np.arange(len(pivot.index))
        width = min(0.8 / len(group_order), 0.18)

        for i, group in enumerate(group_order):
            offset = (i - (len(group_order) - 1) / 2) * width
            plt.bar(
                x + offset,
                pivot[group],
                width=width,
                label=group,
            )

        if metric == "std_ratio_to_normal" or metric == "iqr_ratio_to_normal":
            plt.axhline(1.0, linestyle="--", linewidth=1.2)
        elif metric == "ks_statistic" or metric == "z_mean_diff":
            plt.axhline(0.0, linestyle="--", linewidth=1.2)

        plt.xticks(x, pivot.index, rotation=45, ha="right")
        plt.title(f"Feature distance from normal: {metric}")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"distance_{metric}_all_groups.png")
        plt.savefig(out, dpi=220)
        plt.close()
        print(f"Saved: {out}")


def main():
    ensure_dir(OUTPUT_DIR)

    attack_groups = discover_attack_groups()

    groups = {
        "normal": load_csv_group("normal", NORMAL_DATA_PATH),
    }

    for group_name, files in attack_groups.items():
        groups[group_name] = load_csv_group(group_name, files)

    group_order = list(groups.keys())

    print("\n===== GROUP ORDER =====")
    for g in group_order:
        print(f"- {g}")
    print("=======================\n")

    print("Building long feature dataframe...")
    long_df = build_long_feature_df(groups)

    long_sample_path = os.path.join(OUTPUT_DIR, "feature_values_long_sample.csv")

    sample_df = long_df.groupby(["group", "feature"], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 3000), random_state=RANDOM_STATE)
    )
    sample_df.to_csv(long_sample_path, index=False, encoding="utf-8")

    print("Summarizing feature distributions...")
    summary_df = summarize_feature_distribution(long_df)
    summary_path = os.path.join(OUTPUT_DIR, "feature_distribution_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("Comparing every group against normal...")
    compare_df = compare_against_normal_including_normal(long_df, summary_df)
    compare_path = os.path.join(OUTPUT_DIR, "feature_distance_from_normal_including_normal.csv")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8")

    print("Plotting histograms...")
    plot_histograms(long_df, group_order)

    print("Plotting mean feature comparisons...")
    plot_mean_feature_comparison(summary_df, group_order)

    print("Plotting distance bars...")
    plot_distance_bars(compare_df, group_order)

    print("\n===== SAVED =====")
    print(f"Summary CSV : {summary_path}")
    print(f"Distance CSV: {compare_path}")
    print(f"Sample long : {long_sample_path}")
    print(f"Figures     : {OUTPUT_DIR}")
    print("=================")


if __name__ == "__main__":
    main()