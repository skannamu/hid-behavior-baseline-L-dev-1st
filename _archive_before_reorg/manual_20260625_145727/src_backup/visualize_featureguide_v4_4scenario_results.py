import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


RESULT_CSV = "results/reconstruction_errors_all.csv"
OUTPUT_DIR = "results/figures_featureguide_v4_4scenario_no_window5"

NORMAL_DATA_DIR = Path("data/processed")
ATTACK_DATA_DIR = Path("data/attack/FeatureGuideHumanMimic")

NORMAL_LABEL = "Normal"
ATTACK_LABEL = "FeatureGuideHumanMimic v4\nwindow1-4"

RANDOM_STATE = 42
MAX_TSNE_SAMPLES_PER_CLASS = 500


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_result():
    df = pd.read_csv(RESULT_CSV)

    normal = df[df["label"] == "normal"]["reconstruction_error"]
    attack = df[df["label"] == "attack"]["reconstruction_error"]

    thresholds = {
        "90%": normal.quantile(0.90),
        "95%": normal.quantile(0.95),
        "99%": normal.quantile(0.99),
    }

    return df, normal, attack, thresholds


def plot_error_histogram(normal, attack, thresholds):
    plt.figure(figsize=(10, 6))

    plt.hist(
        normal,
        bins=80,
        alpha=0.55,
        density=True,
        label=NORMAL_LABEL,
    )

    plt.hist(
        attack,
        bins=50,
        alpha=0.65,
        density=True,
        label=ATTACK_LABEL,
    )

    for name, value in thresholds.items():
        plt.axvline(
            value,
            linestyle="--",
            linewidth=1.6,
            label=f"Normal {name} threshold = {value:.3f}",
        )

    plt.xlim(0, max(attack.max(), thresholds["99%"]) * 1.15)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distribution\nNormal vs FeatureGuideHumanMimic v4 window1-4")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("01_error_distribution_histogram_v4_4scenario.png")


def plot_error_boxplot(normal, attack):
    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [normal, attack],
        labels=[NORMAL_LABEL, ATTACK_LABEL],
        showfliers=False,
    )

    plt.ylabel("Reconstruction error")
    plt.title("Reconstruction Error Boxplot\nNormal vs FeatureGuideHumanMimic v4 window1-4")
    plt.grid(axis="y", alpha=0.25)

    savefig("02_error_boxplot_v4_4scenario.png")


def plot_detection_rate(normal, attack, thresholds):
    rows = []

    for name, threshold in thresholds.items():
        fpr = (normal > threshold).mean() * 100
        detection = (attack > threshold).mean() * 100

        rows.append({
            "threshold": f"Normal {name}",
            "threshold_value": threshold,
            "normal_false_positive_rate": fpr,
            "attack_detection_rate": detection,
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "03_threshold_detection_rate_v4_4scenario.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    plt.figure(figsize=(8, 6))

    x = np.arange(len(out))
    width = 0.35

    plt.bar(
        x - width / 2,
        out["normal_false_positive_rate"],
        width,
        label="Normal false positive rate",
    )

    plt.bar(
        x + width / 2,
        out["attack_detection_rate"],
        width,
        label="Attack detection rate",
    )

    for i, value in enumerate(out["attack_detection_rate"]):
        plt.text(
            i + width / 2,
            value + 0.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x, out["threshold"])
    plt.ylabel("Rate (%)")
    plt.title("Detection Rate by Normal Threshold\nFeatureGuideHumanMimic v4 window1-4")
    plt.ylim(0, max(15, out[["normal_false_positive_rate", "attack_detection_rate"]].max().max() + 5))
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    savefig("03_threshold_detection_rate_v4_4scenario.png")


def plot_mean_error_bar(normal, attack):
    labels = [NORMAL_LABEL, ATTACK_LABEL]
    means = [normal.mean(), attack.mean()]
    stds = [normal.std(), attack.std()]

    plt.figure(figsize=(8, 6))

    plt.bar(labels, means, yerr=stds, capsize=8)

    for i, value in enumerate(means):
        plt.text(
            i,
            value + 0.03,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel("Mean reconstruction error")
    plt.title("Mean Reconstruction Error\nNormal vs FeatureGuideHumanMimic v4 window1-4")
    plt.grid(axis="y", alpha=0.25)

    savefig("04_mean_error_bar_v4_4scenario.png")


def plot_source_file_error(df):
    attack_df = df[df["label"] == "attack"].copy()

    summary = (
        attack_df
        .groupby("source_file")["reconstruction_error"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("source_file")
    )

    out_path = os.path.join(OUTPUT_DIR, "05_source_file_error_summary_v4_4scenario.csv")
    summary.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    plt.figure(figsize=(9, 6))

    x = np.arange(len(summary))

    plt.bar(
        summary["source_file"],
        summary["mean"],
        yerr=summary["std"],
        capsize=6,
    )

    for i, row in summary.iterrows():
        plt.text(
            i,
            row["mean"] + 0.025,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel("Mean reconstruction error")
    plt.title("Mean Reconstruction Error by Attack Source File\nFeatureGuideHumanMimic v4 window1-4")
    plt.grid(axis="y", alpha=0.25)

    savefig("05_source_file_mean_error_v4_4scenario.png")


def save_quantile_table(normal, attack):
    rows = []

    for name, series in [
        ("normal", normal),
        ("attack_v4_window1_4", attack),
    ]:
        rows.append({
            "label": name,
            "count": len(series),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "q50": series.quantile(0.50),
            "q90": series.quantile(0.90),
            "q95": series.quantile(0.95),
            "q99": series.quantile(0.99),
            "max": series.max(),
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "06_error_quantile_table_v4_4scenario.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")


def find_feature_columns(df):
    feature_cols = []

    for col in df.columns:
        if "_" not in col:
            continue

        prefix, feature = col.split("_", 1)

        if prefix.startswith("t") and prefix[1:].isdigit():
            feature_cols.append(col)

    feature_cols = sorted(
        feature_cols,
        key=lambda c: (int(c.split("_", 1)[0][1:]), c.split("_", 1)[1])
    )

    return feature_cols


def load_window_data_for_tsne():
    normal_files = sorted(NORMAL_DATA_DIR.glob("*.csv"))
    attack_files = sorted(ATTACK_DATA_DIR.glob("window[1-4].csv"))

    rows = []

    for label, files in [("normal", normal_files), ("attack", attack_files)]:
        dfs = []

        for file in files:
            df = pd.read_csv(file)
            df["source_file"] = file.name
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)

        if len(merged) > MAX_TSNE_SAMPLES_PER_CLASS:
            merged = merged.sample(
                MAX_TSNE_SAMPLES_PER_CLASS,
                random_state=RANDOM_STATE,
            )

        merged["tsne_label"] = label
        rows.append(merged)

    all_df = pd.concat(rows, ignore_index=True)

    feature_cols = find_feature_columns(all_df)

    X = all_df[feature_cols].to_numpy(dtype=np.float32)
    y = all_df["tsne_label"].to_numpy()

    return X, y


def plot_tsne():
    if not SKLEARN_AVAILABLE:
        print("Skipped t-SNE: scikit-learn is not available.")
        return

    X, y = load_window_data_for_tsne()

    X = StandardScaler().fit_transform(X)

    perplexity = min(30, max(5, (len(X) - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )

    Z = tsne.fit_transform(X)

    out = pd.DataFrame({
        "tsne_1": Z[:, 0],
        "tsne_2": Z[:, 1],
        "label": y,
    })

    out_path = os.path.join(OUTPUT_DIR, "07_tsne_v4_4scenario.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    plt.figure(figsize=(8, 7))

    for label, display in [
        ("normal", NORMAL_LABEL),
        ("attack", ATTACK_LABEL),
    ]:
        sub = out[out["label"] == label]
        plt.scatter(
            sub["tsne_1"],
            sub["tsne_2"],
            s=14,
            alpha=0.65,
            label=display,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Projection\nNormal vs FeatureGuideHumanMimic v4 window1-4")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("07_tsne_v4_4scenario.png")


def main():
    ensure_dir(OUTPUT_DIR)

    df, normal, attack, thresholds = load_result()

    print("\n===== V4 4-SCENARIO VISUALIZATION =====")
    print(f"Normal count: {len(normal)}")
    print(f"Attack count: {len(attack)}")
    print(f"Normal mean : {normal.mean():.6f}")
    print(f"Attack mean : {attack.mean():.6f}")
    print("======================================\n")

    plot_error_histogram(normal, attack, thresholds)
    plot_error_boxplot(normal, attack)
    plot_detection_rate(normal, attack, thresholds)
    plot_mean_error_bar(normal, attack)
    plot_source_file_error(df)
    save_quantile_table(normal, attack)
    plot_tsne()

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
