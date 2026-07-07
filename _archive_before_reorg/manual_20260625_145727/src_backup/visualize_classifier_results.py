import os
import pandas as pd
import matplotlib.pyplot as plt


PRED_CSV = "results/classifier_lstm/classifier_predictions_featureguide_v4.csv"
OUT_DIR = "results/classifier_lstm/figures"
THRESHOLD = 0.5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(PRED_CSV)

    normal = df[df["true_label"] == 0]
    attack = df[df["true_label"] == 1]

    # 1. Normal vs V4 attack probability histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        normal["attack_probability"],
        bins=80,
        alpha=0.6,
        density=True,
        label="Normal",
    )
    plt.hist(
        attack["attack_probability"],
        bins=80,
        alpha=0.6,
        density=True,
        label="FeatureGuideHumanMimic v4",
    )
    plt.axvline(
        THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {THRESHOLD}",
    )
    plt.xlabel("Attack probability")
    plt.ylabel("Density")
    plt.title("LSTM Classifier Attack Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "01_attack_probability_distribution.png"),
        dpi=300,
    )
    plt.close()

    # 2. Mean attack probability by source file
    source_summary = (
        attack
        .groupby("source_file")["attack_probability"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("source_file")
    )

    source_summary.to_csv(
        os.path.join(OUT_DIR, "02_source_probability_summary.csv"),
        index=False,
        encoding="utf-8",
    )

    plt.figure(figsize=(9, 6))
    plt.bar(
        source_summary["source_file"],
        source_summary["mean"],
    )
    plt.axhline(
        THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {THRESHOLD}",
    )
    plt.ylabel("Mean attack probability")
    plt.title("Mean Attack Probability by V4 Source File")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "02_mean_attack_probability_by_source.png"),
        dpi=300,
    )
    plt.close()

    # 3. Attack probability distribution by source
    plt.figure(figsize=(10, 6))
    for source, sub in attack.groupby("source_file"):
        plt.hist(
            sub["attack_probability"],
            bins=50,
            alpha=0.45,
            density=True,
            label=source,
        )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {THRESHOLD}",
    )
    plt.xlabel("Attack probability")
    plt.ylabel("Density")
    plt.title("Attack Probability Distribution by V4 Source File")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "03_attack_probability_by_source_histogram.png"),
        dpi=300,
    )
    plt.close()

    # 4. Detection rate by source
    attack = attack.copy()
    attack["detected"] = (
        attack["attack_probability"] >= THRESHOLD
    ).astype(int)

    detection_summary = (
        attack
        .groupby("source_file")["detected"]
        .mean()
        .reset_index()
        .sort_values("source_file")
    )
    detection_summary["detection_rate_percent"] = (
        detection_summary["detected"] * 100
    )

    detection_summary.to_csv(
        os.path.join(OUT_DIR, "04_source_detection_summary.csv"),
        index=False,
        encoding="utf-8",
    )

    plt.figure(figsize=(9, 6))
    plt.bar(
        detection_summary["source_file"],
        detection_summary["detection_rate_percent"],
    )
    plt.ylabel("Detection rate (%)")
    plt.ylim(0, 100)
    plt.title("LSTM Classifier Detection Rate by V4 Source File")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "04_detection_rate_by_source.png"),
        dpi=300,
    )
    plt.close()

    print("\n===== CLASSIFIER VISUALIZATION DONE =====")
    print(f"Output directory: {OUT_DIR}")
    print("\nSource probability summary:")
    print(source_summary.to_string(index=False))
    print("\nSource detection summary:")
    print(detection_summary.to_string(index=False))


if __name__ == "__main__":
    main()

