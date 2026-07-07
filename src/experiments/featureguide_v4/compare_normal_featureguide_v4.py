import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NORMAL_DATA_PATH = "data/processed"
FEATURE_GUIDE_V4_DATA_PATH = "data/attack/FeatureGuideHumanMimic"

OUTPUT_DIR = "results/feature_distribution_featureguide_v4"

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


def resolve_csv_files(path):
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")

        return files

    raise FileNotFoundError(f"Path not found: {path}")


def load_csv_group(group_name, path):
    files = resolve_csv_files(path)

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
    if feature in df.columns:
        values = df[feature].to_numpy().astype(float)
        return values[np.isfinite(values)]

    timestep_cols = []

    for col in df.columns:
        if col.endswith("_" + feature):
            prefix = col[:-(len(feature) + 1)]

            if prefix.startswith("t") and prefix[1:].isdigit():
                timestep_cols.append(col)

    timestep_cols = sorted(
        timestep_cols,
        key=lambda x: int(x.split("_", 1)[0][1:]),
    )

    if not timestep_cols:
        raise ValueError(f"Feature columns not found for feature: {feature}")

    values = df[timestep_cols].to_numpy().reshape(-1)
    values = values.astype(float)
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

        q01 = np.quantile(x, 0.01)
        q05 = np.quantile(x, 0.05)
        q25 = np.quantile(x, 0.25)
        q50 = np.quantile(x, 0.50)
        q75 = np.quantile(x, 0.75)
        q90 = np.quantile(x, 0.90)
        q95 = np.quantile(x, 0.95)
        q99 = np.quantile(x, 0.99)

        rows.append({
            "group": group,
            "feature": feature,
            "count": len(x),
            "mean": np.mean(x),
            "std": np.std(x, ddof=1),
            "min": np.min(x),
            "q01": q01,
            "q05": q05,
            "q25": q25,
            "median": q50,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "max": np.max(x),
            "iqr": q75 - q25,
        })

    return pd.DataFrame(rows)


def ks_statistic(x, y):
    x = np.sort(np.asarray(x))
    y = np.sort(np.asarray(y))

    data_all = np.sort(np.concatenate([x, y]))

    cdf_x = np.searchsorted(x, data_all, side="right") / len(x)
    cdf_y = np.searchsorted(y, data_all, side="right") / len(y)

    return np.max(np.abs(cdf_x - cdf_y))


def compare_featureguide_against_normal(long_df, summary_df):
    rows = []

    normal_summary = summary_df[summary_df["group"] == "normal"].set_index("feature")
    attack_summary = summary_df[summary_df["group"] == "FeatureGuide v4"].set_index("feature")

    for feature in SELECTED_FEATURES:
        normal_values = long_df[
            (long_df["group"] == "normal")
            & (long_df["feature"] == feature)
        ]["value"].to_numpy()

        attack_values = long_df[
            (long_df["group"] == "FeatureGuide v4")
            & (long_df["feature"] == feature)
        ]["value"].to_numpy()

        n = normal_summary.loc[feature]
        a = attack_summary.loc[feature]

        normal_mean = n["mean"]
        normal_std = n["std"] if n["std"] != 0 else np.nan

        mean_diff = a["mean"] - normal_mean
        z_mean_diff = mean_diff / normal_std if normal_std and not math.isnan(normal_std) else np.nan

        std_ratio = a["std"] / n["std"] if n["std"] != 0 else np.nan
        iqr_ratio = a["iqr"] / n["iqr"] if n["iqr"] != 0 else np.nan

        ks = ks_statistic(normal_values, attack_values)

        rows.append({
            "feature": feature,
            "normal_mean": normal_mean,
            "featureguide_v4_mean": a["mean"],
            "mean_diff": mean_diff,
            "z_mean_diff": z_mean_diff,
            "normal_std": n["std"],
            "featureguide_v4_std": a["std"],
            "std_ratio_v4_to_normal": std_ratio,
            "normal_iqr": n["iqr"],
            "featureguide_v4_iqr": a["iqr"],
            "iqr_ratio_v4_to_normal": iqr_ratio,
            "ks_statistic": ks,
        })

    return pd.DataFrame(rows)


def sample_values(x, max_size=50000):
    if len(x) > max_size:
        rng = np.random.default_rng(42)
        return rng.choice(x, size=max_size, replace=False)

    return x


def plot_boxplots(long_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for feature in FOCUS_FEATURES:
        data = []
        labels = []

        for group in ["normal", "FeatureGuide v4"]:
            x = long_df[
                (long_df["group"] == group)
                & (long_df["feature"] == feature)
            ]["value"].dropna().to_numpy()

            x = sample_values(x)

            data.append(x)
            labels.append(group)

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels, showfliers=False)

        plt.title(f"{feature}: Normal vs FeatureGuideHumanMimic v4")
        plt.ylabel(feature)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"boxplot_{feature}_normal_vs_v4.png")
        plt.savefig(out, dpi=250)
        plt.close()

        print(f"Saved: {out}")


def plot_histograms(long_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for feature in FOCUS_FEATURES:
        plt.figure(figsize=(9, 5))

        for group in ["normal", "FeatureGuide v4"]:
            x = long_df[
                (long_df["group"] == group)
                & (long_df["feature"] == feature)
            ]["value"].dropna().to_numpy()

            x = sample_values(x)

            upper = np.quantile(x, 0.99)
            lower = np.quantile(x, 0.01)

            x_plot = x[(x >= lower) & (x <= upper)]

            plt.hist(
                x_plot,
                bins=60,
                alpha=0.38,
                density=True,
                label=group,
            )

        plt.title(f"{feature} Histogram: Normal vs FeatureGuideHumanMimic v4")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"hist_{feature}_normal_vs_v4.png")
        plt.savefig(out, dpi=250)
        plt.close()

        print(f"Saved: {out}")


def plot_mean_comparison(summary_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    focus_summary = summary_df[summary_df["feature"].isin(FOCUS_FEATURES)].copy()

    pivot = focus_summary.pivot(
        index="feature",
        columns="group",
        values="mean",
    )

    pivot = pivot.loc[FOCUS_FEATURES]

    x = np.arange(len(pivot.index))
    width = 0.35

    plt.figure(figsize=(13, 6))

    plt.bar(
        x - width / 2,
        pivot["normal"],
        width,
        label="Normal",
    )

    plt.bar(
        x + width / 2,
        pivot["FeatureGuide v4"],
        width,
        label="FeatureGuide v4",
    )

    plt.xticks(x, pivot.index, rotation=45, ha="right")
    plt.ylabel("Feature Mean")
    plt.title("Feature Mean Comparison: Normal vs FeatureGuideHumanMimic v4")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "mean_feature_comparison_normal_vs_v4.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")


def plot_distance_bar(compare_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = [
        "ks_statistic",
        "z_mean_diff",
        "std_ratio_v4_to_normal",
        "iqr_ratio_v4_to_normal",
    ]

    for metric in metrics:
        target = compare_df[compare_df["feature"].isin(FOCUS_FEATURES)].copy()
        target = target.set_index("feature").loc[FOCUS_FEATURES]

        plt.figure(figsize=(12, 6))

        plt.bar(target.index, target[metric])

        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"Feature Distance from Normal: {metric}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"distance_{metric}_normal_vs_v4.png")
        plt.savefig(out, dpi=300)
        plt.close()

        print(f"Saved: {out}")


def plot_entropy_focus(summary_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entropy = summary_df[summary_df["feature"] == "timing_entropy"].copy()

    labels = entropy["group"].tolist()
    means = entropy["mean"].tolist()

    plt.figure(figsize=(7, 5))

    plt.bar(labels, means)

    plt.axhline(
        2.459148,
        linestyle="--",
        linewidth=2,
        label="Normal q01 = 2.459",
    )

    plt.axhline(
        2.815922,
        linestyle="--",
        linewidth=2,
        label="Normal q05 = 2.816",
    )

    plt.axhline(
        3.309915,
        linestyle="--",
        linewidth=2,
        label="Normal mean = 3.310",
    )

    for i, value in enumerate(means):
        plt.text(
            i,
            value + 0.05,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.ylabel("Timing Entropy")
    plt.title("Timing Entropy: Normal vs FeatureGuideHumanMimic v4")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "timing_entropy_focus_normal_vs_v4.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = {
        "normal": load_csv_group("normal", NORMAL_DATA_PATH),
        "FeatureGuide v4": load_csv_group("FeatureGuide v4", FEATURE_GUIDE_V4_DATA_PATH),
    }

    print("\nBuilding long feature dataframe...")
    long_df = build_long_feature_df(groups)

    long_path = os.path.join(OUTPUT_DIR, "feature_values_long_sample_normal_vs_v4.csv")

    sample_df = long_df.groupby(["group", "feature"], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 3000), random_state=42)
    )

    sample_df.to_csv(long_path, index=False, encoding="utf-8")

    print("Summarizing feature distributions...")
    summary_df = summarize_feature_distribution(long_df)

    summary_path = os.path.join(OUTPUT_DIR, "feature_distribution_summary_normal_vs_v4.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("Comparing FeatureGuide v4 against normal...")
    compare_df = compare_featureguide_against_normal(long_df, summary_df)

    compare_path = os.path.join(OUTPUT_DIR, "feature_distance_normal_vs_v4.csv")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8")

    print("Plotting boxplots...")
    plot_boxplots(long_df)

    print("Plotting histograms...")
    plot_histograms(long_df)

    print("Plotting mean comparison...")
    plot_mean_comparison(summary_df)

    print("Plotting distance bars...")
    plot_distance_bar(compare_df)

    print("Plotting timing entropy focus...")
    plot_entropy_focus(summary_df)

    print("\n===== SAVED =====")
    print(f"Summary CSV : {summary_path}")
    print(f"Distance CSV: {compare_path}")
    print(f"Sample long : {long_path}")
    print(f"Figures     : {OUTPUT_DIR}")
    print("=================")


if __name__ == "__main__":
    main()
