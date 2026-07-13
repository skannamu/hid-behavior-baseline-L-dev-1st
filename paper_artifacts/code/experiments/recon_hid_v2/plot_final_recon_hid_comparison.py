import os
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = "results/final_recon_hid_comparison"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    rows = [
        {
            "method": "LSTM AE 95%",
            "detection_rate": 4.43,
            "normal_fpr": 5.00,
            "type": "baseline",
        },
        {
            "method": "LSTM Classifier",
            "detection_rate": 28.39,
            "normal_fpr": 0.00,
            "type": "baseline",
        },
        {
            "method": "ReCon-HID v1 OR95",
            "detection_rate": 21.09,
            "normal_fpr": 5.00,
            "type": "ablation",
        },
        {
            "method": "ReCon-HID v2 recon95",
            "detection_rate": 36.46,
            "normal_fpr": 5.00,
            "type": "ablation",
        },
        {
            "method": "ReCon-HID v2 OR95",
            "detection_rate": 45.57,
            "normal_fpr": 5.00,
            "type": "proposed",
        },
        {
            "method": "ReCon-HID v2 OR90",
            "detection_rate": 73.44,
            "normal_fpr": 10.00,
            "type": "proposed",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUT_DIR, "final_detection_comparison.csv"),
        index=False,
        encoding="utf-8",
    )

    # 1. Detection rate comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df["method"], df["detection_rate"])
    plt.ylabel("Attack detection rate (%)")
    plt.ylim(0, 100)
    plt.title("Detection Rate against FeatureGuideHumanMimic v4")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "01_detection_rate_comparison.png"),
        dpi=300,
    )
    plt.close()

    # 2. False positive rate comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df["method"], df["normal_fpr"])
    plt.ylabel("Normal false positive rate (%)")
    plt.ylim(0, max(12, df["normal_fpr"].max() + 2))
    plt.title("Normal False Positive Rate")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "02_false_positive_rate_comparison.png"),
        dpi=300,
    )
    plt.close()

    # 3. Detection-FPR tradeoff scatter
    plt.figure(figsize=(8, 6))

    for _, row in df.iterrows():
        plt.scatter(row["normal_fpr"], row["detection_rate"], s=80)
        plt.text(
            row["normal_fpr"] + 0.15,
            row["detection_rate"] + 0.8,
            row["method"],
            fontsize=9,
        )

    plt.xlabel("Normal false positive rate (%)")
    plt.ylabel("Attack detection rate (%)")
    plt.title("Detection-FPR Tradeoff")
    plt.xlim(-0.5, 12)
    plt.ylim(0, 80)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "03_detection_fpr_tradeoff.png"),
        dpi=300,
    )
    plt.close()

    # 4. Source-level comparison for ReCon-HID v2
    source_rows = [
        {
            "source_file": "window1.csv",
            "recon95": 13.95,
            "or95": 13.95,
            "or90": 100.00,
        },
        {
            "source_file": "window2.csv",
            "recon95": 25.39,
            "or95": 25.91,
            "or90": 55.96,
        },
        {
            "source_file": "window3.csv",
            "recon95": 84.85,
            "or95": 84.85,
            "or90": 89.39,
        },
        {
            "source_file": "window4.csv",
            "recon95": 35.37,
            "or95": 76.83,
            "or90": 87.80,
        },
    ]

    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(
        os.path.join(OUT_DIR, "source_detection_comparison.csv"),
        index=False,
        encoding="utf-8",
    )

    x = range(len(source_df))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(
        [i - width for i in x],
        source_df["recon95"],
        width=width,
        label="recon95",
    )
    plt.bar(
        list(x),
        source_df["or95"],
        width=width,
        label="OR95",
    )
    plt.bar(
        [i + width for i in x],
        source_df["or90"],
        width=width,
        label="OR90",
    )

    plt.xticks(list(x), source_df["source_file"])
    plt.ylabel("Detection rate (%)")
    plt.ylim(0, 100)
    plt.title("ReCon-HID v2 Source-level Detection Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "04_source_level_detection_comparison.png"),
        dpi=300,
    )
    plt.close()

    print("\n===== Final Comparison Figures Done =====")
    print(df.to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
