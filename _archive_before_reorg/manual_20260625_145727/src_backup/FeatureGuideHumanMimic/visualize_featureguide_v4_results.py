import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.dataset import SELECTED_FEATURES, WINDOW_SIZE, resolve_csv_files


RESULT_CSV = "results/reconstruction_errors_all.csv"

NORMAL_DATA_PATH = "data/processed"
ATTACK_DATA_PATH = "data/attack/FeatureGuideHumanMimic"

SCALER_PATH = "checkpoints/scaler.pkl"

FIG_DIR = "results/figures_featureguide_v4"

THRESHOLD_90 = 0.804061
THRESHOLD_95 = 0.925666
THRESHOLD_99 = 1.365235

NORMAL_TSNE_SAMPLE = 1500
ATTACK_TSNE_SAMPLE = None

RANDOM_SEED = 42


def ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def save_error_distribution():
    df = pd.read_csv(RESULT_CSV)

    normal = df[df["label"] == "normal"]["reconstruction_error"]
    attack = df[df["label"] == "attack"]["reconstruction_error"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        normal,
        bins=100,
        alpha=0.60,
        label="Normal",
    )

    plt.hist(
        attack,
        bins=60,
        alpha=0.75,
        label="FeatureGuideHumanMimic v4",
    )

    plt.axvline(
        THRESHOLD_95,
        linestyle="--",
        linewidth=2,
        label=f"Normal 95% Threshold = {THRESHOLD_95:.3f}",
    )

    plt.xlim(0, 5)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Window Count")
    plt.title("Reconstruction Error Distribution: Normal vs FeatureGuideHumanMimic v4")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "01_error_distribution_histogram_v4.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def save_error_boxplot():
    df = pd.read_csv(RESULT_CSV)

    normal = df[df["label"] == "normal"]["reconstruction_error"]
    attack = df[df["label"] == "attack"]["reconstruction_error"]

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [normal, attack],
        labels=["Normal", "FeatureGuide v4"],
        showfliers=False,
    )

    plt.axhline(
        THRESHOLD_95,
        linestyle="--",
        linewidth=2,
        label=f"Normal 95% Threshold = {THRESHOLD_95:.3f}",
    )

    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error Boxplot: Normal vs FeatureGuideHumanMimic v4")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "02_error_boxplot_v4.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def save_threshold_detection_rate():
    thresholds = ["Normal 90%", "Normal 95%", "Normal 99%"]

    false_positive_rates = [10.01, 5.00, 1.01]

    # FeatureGuideHumanMimic v4 result
    attack_detection_rates = [48.89, 25.83, 13.89]

    x = np.arange(len(thresholds))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        false_positive_rates,
        width,
        label="Normal False Positive Rate",
    )

    plt.bar(
        x + width / 2,
        attack_detection_rates,
        width,
        label="FeatureGuide v4 Detection Rate",
    )

    for i, value in enumerate(false_positive_rates):
        plt.text(
            x[i] - width / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=9,
        )

    for i, value in enumerate(attack_detection_rates):
        plt.text(
            x[i] + width / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=9,
        )

    plt.xticks(x, thresholds)
    plt.ylabel("Rate (%)")
    plt.ylim(0, 110)
    plt.title("Detection Trade-off: Normal vs FeatureGuideHumanMimic v4")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "03_threshold_detection_rate_v4.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def save_error_quantile_table():
    df = pd.read_csv(RESULT_CSV)

    rows = []

    for label in ["normal", "attack"]:
        x = df[df["label"] == label]["reconstruction_error"]

        rows.append({
            "label": label,
            "count": len(x),
            "mean": x.mean(),
            "std": x.std(),
            "min": x.min(),
            "q50": x.quantile(0.50),
            "q90": x.quantile(0.90),
            "q95": x.quantile(0.95),
            "q99": x.quantile(0.99),
            "max": x.max(),
        })

    out = pd.DataFrame(rows)

    csv_path = os.path.join(FIG_DIR, "04_error_quantile_table_v4.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(out)


def save_mean_error_bar():
    df = pd.read_csv(RESULT_CSV)

    summary = (
        df.groupby("label")["reconstruction_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    label_map = {
        "normal": "Normal",
        "attack": "FeatureGuide v4",
    }

    labels = [label_map.get(x, x) for x in summary["label"]]
    values = summary["mean"]

    plt.figure(figsize=(8, 6))

    plt.bar(labels, values)

    plt.axhline(
        THRESHOLD_95,
        linestyle="--",
        linewidth=2,
        label=f"Normal 95% Threshold = {THRESHOLD_95:.3f}",
    )

    for i, value in enumerate(values):
        plt.text(
            i,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.ylabel("Mean Reconstruction Error")
    plt.title("Mean Reconstruction Error: Normal vs FeatureGuideHumanMimic v4")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "05_mean_error_bar_v4.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def save_normal_outlier_lineplot():
    df = pd.read_csv(RESULT_CSV)

    normal = df[df["label"] == "normal"].copy()

    start_idx = 5150
    end_idx = 5300

    target = normal[
        (normal["global_index"] >= start_idx)
        & (normal["global_index"] <= end_idx)
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        target["global_index"],
        target["reconstruction_error"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Normal Reconstruction Error",
    )

    plt.axhline(
        THRESHOLD_95,
        linestyle="--",
        linewidth=2,
        label=f"Normal 95% Threshold = {THRESHOLD_95:.3f}",
    )

    plt.xlabel("Window Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Normal Behavioral Outlier Around Long Pause Window")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "06_normal_outlier_lineplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def load_window_csvs_as_flat_features(data_path, label):
    csv_files = resolve_csv_files(data_path)

    rows = []
    source_files = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            sequence = []

            for t in range(WINDOW_SIZE):
                timestep = []

                for feature in SELECTED_FEATURES:
                    col_name = f"t{t}_{feature}"
                    value = row.get(col_name, 0.0)

                    if pd.isna(value):
                        value = 0.0

                    timestep.append(float(value))

                sequence.append(timestep)

            rows.append(sequence)
            source_files.append(os.path.basename(csv_file))

    arr = np.array(rows, dtype=np.float32)

    n_samples, seq_len, feat_dim = arr.shape

    flat = arr.reshape(-1, feat_dim)

    scaler = joblib.load(SCALER_PATH)
    flat_scaled = scaler.transform(flat)

    scaled = flat_scaled.reshape(n_samples, seq_len, feat_dim)
    flattened_windows = scaled.reshape(n_samples, seq_len * feat_dim)

    meta = pd.DataFrame({
        "label": label,
        "source_file": source_files,
    })

    return flattened_windows, meta


def save_tsne_normal_attack():
    np.random.seed(RANDOM_SEED)

    normal_x, normal_meta = load_window_csvs_as_flat_features(
        NORMAL_DATA_PATH,
        label="normal",
    )

    attack_x, attack_meta = load_window_csvs_as_flat_features(
        ATTACK_DATA_PATH,
        label="attack",
    )

    if NORMAL_TSNE_SAMPLE is not None and len(normal_x) > NORMAL_TSNE_SAMPLE:
        normal_indices = np.random.choice(
            len(normal_x),
            size=NORMAL_TSNE_SAMPLE,
            replace=False,
        )

        normal_x = normal_x[normal_indices]
        normal_meta = normal_meta.iloc[normal_indices].reset_index(drop=True)

    if ATTACK_TSNE_SAMPLE is not None and len(attack_x) > ATTACK_TSNE_SAMPLE:
        attack_indices = np.random.choice(
            len(attack_x),
            size=ATTACK_TSNE_SAMPLE,
            replace=False,
        )

        attack_x = attack_x[attack_indices]
        attack_meta = attack_meta.iloc[attack_indices].reset_index(drop=True)

    x = np.vstack([normal_x, attack_x])
    meta = pd.concat([normal_meta, attack_meta], ignore_index=True)

    print("\n===== t-SNE INPUT INFO =====")
    print(f"Normal samples: {len(normal_x)}")
    print(f"Attack samples: {len(attack_x)}")
    print(f"Input dim     : {x.shape[1]}")
    print("===========================\n")

    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    x_pca = pca.fit_transform(x)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_SEED,
    )

    x_tsne = tsne.fit_transform(x_pca)

    tsne_df = pd.DataFrame({
        "tsne_1": x_tsne[:, 0],
        "tsne_2": x_tsne[:, 1],
        "label": meta["label"].values,
        "source_file": meta["source_file"].values,
    })

    tsne_csv_path = os.path.join(FIG_DIR, "07_tsne_normal_featureguide_v4.csv")
    tsne_df.to_csv(tsne_csv_path, index=False, encoding="utf-8")

    plt.figure(figsize=(10, 7))

    normal = tsne_df[tsne_df["label"] == "normal"]
    attack = tsne_df[tsne_df["label"] == "attack"]

    plt.scatter(
        normal["tsne_1"],
        normal["tsne_2"],
        s=12,
        alpha=0.45,
        label="Normal",
    )

    plt.scatter(
        attack["tsne_1"],
        attack["tsne_2"],
        s=28,
        alpha=0.85,
        label="FeatureGuide v4",
        marker="x",
    )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE: Normal vs FeatureGuideHumanMimic v4 Windows")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "07_tsne_normal_featureguide_v4.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")
    print(f"Saved: {tsne_csv_path}")


def main():
    ensure_dir()

    save_error_distribution()
    save_error_boxplot()
    save_threshold_detection_rate()
    save_error_quantile_table()
    save_mean_error_bar()
    save_normal_outlier_lineplot()
    save_tsne_normal_attack()

    print("\n===== VISUALIZATION DONE =====")
    print(f"Figure directory: {FIG_DIR}")
    print("==============================\n")


if __name__ == "__main__":
    main()
