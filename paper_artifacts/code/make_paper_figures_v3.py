from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "paper_artifacts"
TABLE_ROOT = PAPER_ROOT / "tables"
BASELINE_TABLE_ROOT = TABLE_ROOT / "baselines"
FIG_ROOT = PAPER_ROOT / "figures" / "v3"
FIG_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Global style: paper-like, cleaner, more formal
# ============================================================
mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
})


# ============================================================
# Helpers
# ============================================================
def save_figure(fig, name: str):
    png_path = FIG_ROOT / f"{name}.png"
    pdf_path = FIG_ROOT / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    plt.close(fig)


def display_label_attack(x: str) -> str:
    mapping = {
        "V4": "V4\nFeatureGuide",
        "V5": "V5",
        "V6": "V6",
        "V7": "V7",
        "V8_candidate": "V8\ncandidate",
        "V8_hardest": "V8\nhardest",
        "V5_test": "V5\ntest",
        "V6_test": "V6\ntest",
        "V7_test": "V7\ntest",
    }
    return mapping.get(x, x)


def display_label_detector(x: str) -> str:
    mapping = {
        "d0": "D0",
        "d1": "D1",
        "d2": "D2",
        "d3": "D3",
        "D3_guided": "D3\nguided",
        "D3_random_matched": "D3\nrandom-matched",
    }
    return mapping.get(x, x)


def pretty_matrix_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0)
    # detector rows x attack columns  -> transpose for paper-friendly layout
    df = df.T
    df.index = [display_label_attack(x) for x in df.index]
    df.columns = [display_label_detector(x) for x in df.columns]
    return df


def annotate_heatmap(ax, data, im, fmt="{:.1f}", threshold=50):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < threshold else "black"
            ax.text(
                j, i, fmt.format(val),
                ha="center", va="center",
                color=color, fontsize=10, fontweight="semibold"
            )


def add_box(ax, xy, w, h, text, fc="#f8f8f8", ec="black", fontsize=12, lw=1.4):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def add_arrow(ax, p1, p2, lw=1.7):
    arrow = FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=15,
        linewidth=lw, color="black"
    )
    ax.add_patch(arrow)


# ============================================================
# Figure 1: Framework overview
# ============================================================
def plot_framework_overview():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02, 0.96,
        "Detector-guided hard-negative hardening framework",
        fontsize=18, fontweight="bold", ha="left", va="top"
    )

    boxes = {}

    boxes["normal"] = add_box(ax, (0.03, 0.58), 0.14, 0.16, "Normal HID\nbehavior data")
    boxes["seed"]   = add_box(ax, (0.03, 0.28), 0.14, 0.16, "Initial attack\nseed data (V4)")
    boxes["d0"]     = add_box(ax, (0.25, 0.43), 0.16, 0.18, "Train detector\n$D_0$")
    boxes["gen"]    = add_box(ax, (0.49, 0.58), 0.20, 0.16, "Detector-guided\ncandidate generation")
    boxes["eval"]   = add_box(ax, (0.49, 0.28), 0.20, 0.16, "Oracle evaluation\n(OR90 / OR95)")
    boxes["select"] = add_box(ax, (0.77, 0.58), 0.18, 0.16, "Select bypass\nhard negatives")
    boxes["retrain"]= add_box(ax, (0.77, 0.28), 0.18, 0.16, "Retrain / harden\n$D_{r+1}$")

    add_arrow(ax, (0.17, 0.66), (0.25, 0.56))
    add_arrow(ax, (0.17, 0.36), (0.25, 0.48))
    add_arrow(ax, (0.41, 0.52), (0.49, 0.66))
    add_arrow(ax, (0.59, 0.58), (0.59, 0.44))
    add_arrow(ax, (0.69, 0.36), (0.77, 0.66))
    add_arrow(ax, (0.87, 0.58), (0.87, 0.44))
    add_arrow(ax, (0.77, 0.36), (0.69, 0.36))  # feedback-style look
    add_arrow(ax, (0.77, 0.36), (0.69, 0.66))  # feedback դեպի gen

    # extra annotation
    ax.text(
        0.02, 0.08,
        "Each round searches for detector blind spots, promotes successful bypass cases to hard negatives,\n"
        "and retrains the next detector using the augmented training set.",
        fontsize=11, ha="left", va="bottom"
    )

    save_figure(fig, "framework_overview_v3")


# ============================================================
# Figure 2: ReCon-HID v2 architecture
# ============================================================
def plot_reconhid_architecture():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02, 0.96,
        "ReCon-HID v2 hybrid detector architecture",
        fontsize=18, fontweight="bold", ha="left", va="top"
    )

    add_box(ax, (0.03, 0.42), 0.14, 0.18, "Input sequence\n$X \\in \\mathbb{R}^{50 \\times 15}$")
    add_box(ax, (0.23, 0.42), 0.14, 0.18, "LSTM encoder\n$f_{\\theta}$")
    add_box(ax, (0.43, 0.42), 0.14, 0.18, "Latent state\n$z$")

    add_box(ax, (0.66, 0.68), 0.16, 0.14, "Reconstruction\nhead")
    add_box(ax, (0.66, 0.42), 0.16, 0.14, "Classifier\nhead")
    add_box(ax, (0.66, 0.16), 0.16, 0.14, "Prototype / latent\n distance head")

    add_box(ax, (0.87, 0.42), 0.10, 0.18, "Fusion / Oracle\nOR90, OR95")

    add_arrow(ax, (0.17, 0.51), (0.23, 0.51))
    add_arrow(ax, (0.37, 0.51), (0.43, 0.51))

    add_arrow(ax, (0.57, 0.51), (0.66, 0.75))
    add_arrow(ax, (0.57, 0.51), (0.66, 0.49))
    add_arrow(ax, (0.57, 0.51), (0.66, 0.23))

    add_arrow(ax, (0.82, 0.75), (0.87, 0.51))
    add_arrow(ax, (0.82, 0.49), (0.87, 0.51))
    add_arrow(ax, (0.82, 0.23), (0.87, 0.51))

    ax.text(
        0.02, 0.08,
        "The detector jointly exploits reconstruction error, classifier confidence, and latent/prototype separation\n"
        "to reduce blind spots that may remain under a single-signal detector.",
        fontsize=11, ha="left", va="bottom"
    )

    save_figure(fig, "reconhid_v2_architecture_v3")


# ============================================================
# Figure 3: OR95 detection heatmap
# ============================================================
def plot_or95_heatmap():
    csv_path = BASELINE_TABLE_ROOT / "baseline_no_hardening_vs_guided_or95.csv"
    df = pretty_matrix_from_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    data = df.values.astype(float)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)

    ax.set_title("OR95 detection rate across detectors and adaptive attack rounds", pad=12, fontweight="bold")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Attack set")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Detection rate (%)")

    annotate_heatmap(ax, data, im, fmt="{:.1f}", threshold=55)

    save_figure(fig, "detector_attack_heatmap_or95_v3")


# ============================================================
# Figure 4: OR90 bypass heatmap
# ============================================================
def plot_or90_bypass_heatmap():
    csv_path = BASELINE_TABLE_ROOT / "baseline_no_hardening_vs_guided_or90_bypass.csv"
    df = pretty_matrix_from_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    data = df.values.astype(float)
    im = ax.imshow(data, cmap="magma", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)

    ax.set_title("OR90 bypass rate across detectors and adaptive attack rounds", pad=12, fontweight="bold")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Attack set")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bypass rate (%)")

    annotate_heatmap(ax, data, im, fmt="{:.1f}", threshold=55)

    save_figure(fig, "detector_attack_heatmap_or90_bypass_v3")


# ============================================================
# Figure 5: Candidate generation / bypass yield
# ============================================================
def plot_generation_yield():
    df = pd.read_csv(BASELINE_TABLE_ROOT / "baseline_generation_yield_summary.csv")

    x = np.arange(len(df))
    rounds = df["round"].tolist()
    generated = df["generated_candidates"].values
    selected = df["selected_or_bypass"].values
    yield_rate = df["yield_rate"].values

    fig, ax1 = plt.subplots(figsize=(10.5, 5.8))

    width = 0.34
    bars1 = ax1.bar(x - width/2, generated, width=width, color="#d9d9d9",
                    edgecolor="black", linewidth=1.0, label="Generated candidates")
    bars2 = ax1.bar(x + width/2, selected, width=width, color="#5c5c5c",
                    edgecolor="black", linewidth=1.0, label="Selected bypass hard negatives")

    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds)
    ax1.set_ylabel("Number of samples")
    ax1.set_xlabel("Detector-guided generation round")
    ax1.set_title("Bypass candidate yield across co-evolution rounds", fontweight="bold", pad=12)
    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(x, yield_rate, marker="o", linewidth=2.2, color="black", label="Yield rate")
    ax2.set_ylabel("Yield rate (%)")
    ax2.set_ylim(0, max(100, yield_rate.max() * 1.15))

    for xi, yi in zip(x, yield_rate):
        ax2.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=10, fontweight="semibold")

    # bar labels
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{int(h)}", (b.get_x() + b.get_width()/2, h),
                     textcoords="offset points", xytext=(0, 4),
                     ha="center", va="bottom", fontsize=9)

    for b in bars2:
        h = b.get_height()
        ax1.annotate(f"{int(h)}", (b.get_x() + b.get_width()/2, h),
                     textcoords="offset points", xytext=(0, 4),
                     ha="center", va="bottom", fontsize=9, color="black")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    save_figure(fig, "generation_yield_summary_v3")


# ============================================================
# Figure 6: Guided vs random-matched baseline
# ============================================================
def plot_guided_vs_random_baseline():
    df = pd.read_csv(BASELINE_TABLE_ROOT / "d3_guided_vs_random_matched_final.csv")

    attack_order = ["V4", "V5_test", "V6_test", "V7_test", "V8_candidate", "V8_hardest"]
    df["attack_set"] = pd.Categorical(df["attack_set"], categories=attack_order, ordered=True)
    df = df.sort_values("attack_set")

    labels = [display_label_attack(x) for x in df["attack_set"]]
    x = np.arange(len(df))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    # left: OR95 detection
    ax = axes[0]
    ax.bar(x - width/2, df["D3_guided_OR95_detection"], width=width,
           color="#4d4d4d", edgecolor="black", label="Guided hardening")
    ax.bar(x + width/2, df["D3_random_matched_OR95_detection"], width=width,
           color="#cfcfcf", edgecolor="black", label="Random-matched baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("OR95 detection rate (%)")
    ax.set_title("(a) Detection performance", fontweight="bold")
    ax.grid(axis="y")
    ax.legend(loc="lower left", frameon=True)

    for i, v in enumerate(df["D3_guided_OR95_detection"]):
        ax.text(i - width/2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(df["D3_random_matched_OR95_detection"]):
        ax.text(i + width/2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # right: OR90 bypass
    ax = axes[1]
    ax.bar(x - width/2, df["D3_guided_OR90_bypass"], width=width,
           color="#4d4d4d", edgecolor="black", label="Guided hardening")
    ax.bar(x + width/2, df["D3_random_matched_OR90_bypass"], width=width,
           color="#cfcfcf", edgecolor="black", label="Random-matched baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(30, df["D3_random_matched_OR90_bypass"].max() * 1.3))
    ax.set_ylabel("OR90 bypass rate (%)")
    ax.set_title("(b) Bypass vulnerability", fontweight="bold")
    ax.grid(axis="y")
    ax.legend(loc="upper right", frameon=True)

    for i, v in enumerate(df["D3_guided_OR90_bypass"]):
        ax.text(i - width/2, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(df["D3_random_matched_OR90_bypass"]):
        ax.text(i + width/2, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Detector-guided hardening vs random-matched hardening", fontsize=15, fontweight="bold")
    save_figure(fig, "guided_vs_random_matched_baseline_v3")


# ============================================================
# Figure 7: grouped bar for detector progression
# ============================================================
def plot_detector_progression_bar():
    csv_path = BASELINE_TABLE_ROOT / "baseline_no_hardening_vs_guided_or95.csv"
    df = pd.read_csv(csv_path, index_col=0)

    attack_order = ["V4", "V5", "V6", "V7"]
    df = df[attack_order]
    df.index = [display_label_detector(x) for x in df.index]

    x = np.arange(len(attack_order))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    detectors = list(df.index)

    grayscale = ["#d9d9d9", "#a6a6a6", "#737373", "#404040"]

    for i, det in enumerate(detectors):
        values = df.loc[det].values.astype(float)
        ax.bar(x + (i - 1.5) * width, values, width=width,
               color=grayscale[i], edgecolor="black", label=det)

        for xi, yi in zip(x + (i - 1.5) * width, values):
            ax.text(xi, yi + 1.5, f"{yi:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([display_label_attack(a) for a in attack_order])
    ax.set_ylim(0, 108)
    ax.set_ylabel("OR95 detection rate (%)")
    ax.set_xlabel("Attack set")
    ax.set_title("Detector performance progression across adaptive attack rounds", fontweight="bold", pad=12)
    ax.grid(axis="y")
    ax.legend(title="Detector", frameon=True)

    save_figure(fig, "detector_progression_grouped_bar_or95_v3")


# ============================================================
# Main
# ============================================================
def main():
    print("============================================================")
    print("Generating publication-style paper figures (v3)")
    print("============================================================")

    plot_framework_overview()
    plot_reconhid_architecture()
    plot_or95_heatmap()
    plot_or90_bypass_heatmap()
    plot_generation_yield()
    plot_guided_vs_random_baseline()
    plot_detector_progression_bar()

    print("============================================================")
    print("Done.")
    print(f"Output directory: {FIG_ROOT}")
    print("============================================================")


if __name__ == "__main__":
    main()
