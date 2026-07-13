from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path("paper_artifacts")
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures" / "v2"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

def savefig(name: str):
    path = FIGURES / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def short_attack_name(x: str) -> str:
    x = str(x)
    if "FeatureGuideHumanMimic" in x:
        return "V4\nFeatureGuide"
    if "AdaptiveMimic_v5" in x:
        return "V5"
    if "AdaptiveMimic_v6" in x:
        return "V6"
    if "AdaptiveMimic_v7" in x:
        return "V7"
    if "v8_candidate" in x or "AdaptiveMimic_v8_candidate" in x:
        return "V8\ncandidate"
    if "hardest_no_bypass" in x or "D3_hardest" in x:
        return "V8\nhardest"
    return x

def detector_order_key(x: str) -> int:
    x = str(x).lower()
    for i in range(10):
        if x == f"d{i}":
            return i
    return 999

def prepare_benchmark() -> pd.DataFrame:
    main = load_csv(TABLES / "benchmark_summary_v4_v5_v6_v7_d0_d1_d2_d3.csv")
    main["attack_short"] = main["attack"].map(short_attack_name)
    main["detector"] = main["detector"].astype(str).str.lower()

    frames = [main]

    v8_candidate = TABLES / "benchmark_v8_candidate_d0_d1_d2_d3.csv"
    if v8_candidate.exists():
        df = load_csv(v8_candidate)
        df["attack_short"] = "V8\ncandidate"
        df["detector"] = df["detector"].astype(str).str.lower()
        frames.append(df)

    v8_hardest = TABLES / "benchmark_v8_hardest_no_bypass_d0_d1_d2_d3.csv"
    if v8_hardest.exists():
        df = load_csv(v8_hardest)
        df["attack_short"] = "V8\nhardest"
        df["detector"] = df["detector"].astype(str).str.lower()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, sort=False)

    attack_order = [
        "V4\nFeatureGuide",
        "V5",
        "V6",
        "V7",
        "V8\ncandidate",
        "V8\nhardest",
    ]
    df["attack_short"] = pd.Categorical(df["attack_short"], categories=attack_order, ordered=True)
    df = df.sort_values(["attack_short", "detector"])
    return df

def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, filename: str, cmap: str):
    pivot = df.pivot_table(
        index="attack_short",
        columns="detector",
        values=value_col,
        aggfunc="mean",
        observed=False,
    )

    cols = sorted([c for c in pivot.columns if str(c).lower().startswith("d")], key=detector_order_key)
    pivot = pivot[cols].dropna(how="all")
    data = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=100, cmap=cmap)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rate (%)")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c).upper() for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))

    ax.set_xlabel("Detector")
    ax.set_ylabel("Attack / candidate set")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                label = "-"
                text_color = "black"
            else:
                label = f"{val:.1f}"
                # 낮은 값은 배경이 어둡고, 높은 값은 배경이 밝은 colormap 기준.
                text_color = "white" if val < 45 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=10, color=text_color)

    savefig(filename)

def plot_grouped_bar_or95(df: pd.DataFrame):
    main_attacks = ["V4\nFeatureGuide", "V5", "V6", "V7"]
    sub = df[df["attack_short"].isin(main_attacks)].copy()

    pivot = sub.pivot_table(
        index="attack_short",
        columns="detector",
        values="or95_detection_rate",
        aggfunc="mean",
        observed=False,
    )

    pivot = pivot.loc[main_attacks]
    cols = sorted([c for c in pivot.columns if str(c).lower().startswith("d")], key=detector_order_key)
    pivot = pivot[cols]

    x = np.arange(len(pivot.index))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for idx, col in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        values = pivot[col].values
        bars = ax.bar(x + offset, values, width, label=str(col).upper())

        for bar, val in zip(bars, values):
            label_y = val + 1.5 if val > 0 else 1.5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if val == 0 else 0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 112)
    ax.set_ylabel("OR95 detection rate (%)")
    ax.set_xlabel("Attack set")
    ax.set_title("Detector performance across adaptive attack rounds")
    ax.legend(title="Detector", loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    savefig("detector_attack_grouped_bar_or95_v2.png")

def plot_bypass_yield_by_round():
    rows = [
        {"round": "D1 → V6", "generated_candidates": 3600, "bypass_candidates": 2431, "bypass_yield": 67.53},
        {"round": "D2 → V7", "generated_candidates": 15000, "bypass_candidates": 252, "bypass_yield": 1.68},
        {"round": "D3 → V8", "generated_candidates": 15000, "bypass_candidates": 0, "bypass_yield": 0.00},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "bypass_yield_by_round.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(out["round"], out["bypass_yield"], marker="o", linewidth=2.4)

    ax.set_ylim(0, 75)
    ax.set_ylabel("Bypass yield (%)")
    ax.set_xlabel("Detector-guided generation round")
    ax.set_title("Bypass yield decreases across co-evolution rounds")
    ax.grid(axis="y", alpha=0.3)

    label_offsets = [2.0, 3.0, 2.0]
    for x, y, off in zip(out["round"], out["bypass_yield"], label_offsets):
        ax.text(x, y + off, f"{y:.2f}%", ha="center", va="bottom", fontsize=11)

    savefig("bypass_yield_by_round_v2.png")

def add_box(ax, x, y, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        linewidth=1.4,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box

def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.6),
    )

def plot_timeline_v2():
    fig, ax = plt.subplots(figsize=(12.5, 3.6))

    y = 0.55
    w = 1.15
    h = 0.44
    gap = 0.38

    labels = [
        "Normal\n+ V4 seed",
        "Train\nD0",
        "Generate\nV5",
        "Harden\nD1",
        "Generate\nV6",
        "Harden\nD2",
        "Generate\nV7",
        "Harden\nD3",
        "V8 attempt\n0 bypass",
    ]

    x = 0.05
    centers = []
    for label in labels:
        add_box(ax, x, y, w, h, label, fontsize=9)
        centers.append((x + w / 2, y + h / 2))
        x += w + gap

    for i in range(len(centers) - 1):
        x1 = centers[i][0] + w / 2 - 0.03
        x2 = centers[i + 1][0] - w / 2 + 0.03
        add_arrow(ax, x1, centers[i][1], x2, centers[i + 1][1])

    ax.text(0.05, 1.28, "Detector-guided co-evolution timeline", fontsize=15, ha="left", va="center")
    ax.text(0.05, 0.17, "Each round generates detector-guided hard negatives and retrains the next detector.", fontsize=10, ha="left")

    ax.set_xlim(-0.1, x - gap + 0.15)
    ax.set_ylim(0, 1.45)
    ax.axis("off")
    savefig("co_evolution_timeline_v2.png")

def plot_framework_loop_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    boxes = {
        "normal": (0.55, 4.5, 1.7, 0.65, "Normal HID\nbehavior data"),
        "seed": (0.55, 3.35, 1.7, 0.65, "Initial attack\nseed data"),
        "train": (3.0, 4.0, 1.9, 0.75, "Train detector\nD_r"),
        "generate": (5.7, 4.0, 2.2, 0.75, "Detector-guided\ncandidate generation"),
        "oracle": (5.7, 2.6, 2.2, 0.75, "Oracle evaluation\nOR90 / OR95"),
        "select": (3.0, 2.6, 1.9, 0.75, "Select bypass\nhard negatives"),
        "harden": (3.0, 1.25, 1.9, 0.75, "Retrain / harden\nD_{r+1}"),
        "stop": (7.95, 1.25, 1.8, 0.75, "No bypass found\n→ report"),
    }

    centers = {}
    for key, (x, y, w, h, text) in boxes.items():
        add_box(ax, x, y, w, h, text, fontsize=10)
        centers[key] = (x + w / 2, y + h / 2)

    add_arrow(ax, centers["normal"][0] + 0.85, centers["normal"][1], centers["train"][0] - 0.95, centers["train"][1] + 0.2)
    add_arrow(ax, centers["seed"][0] + 0.85, centers["seed"][1], centers["train"][0] - 0.95, centers["train"][1] - 0.2)
    add_arrow(ax, centers["train"][0] + 0.95, centers["train"][1], centers["generate"][0] - 1.1, centers["generate"][1])
    add_arrow(ax, centers["generate"][0], centers["generate"][1] - 0.38, centers["oracle"][0], centers["oracle"][1] + 0.38)
    add_arrow(ax, centers["oracle"][0] - 1.1, centers["oracle"][1], centers["select"][0] + 0.95, centers["select"][1])
    add_arrow(ax, centers["select"][0], centers["select"][1] - 0.38, centers["harden"][0], centers["harden"][1] + 0.38)

    # loop back to generation with hardened detector
    ax.annotate(
        "",
        xy=(centers["generate"][0] - 1.1, centers["generate"][1] - 0.05),
        xytext=(centers["harden"][0] + 0.95, centers["harden"][1]),
        arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=-0.28"),
    )

    # stop path
    add_arrow(ax, centers["oracle"][0] + 1.1, centers["oracle"][1] - 0.1, centers["stop"][0] - 0.9, centers["stop"][1] + 0.25)

    ax.text(0.55, 5.55, "Detector-guided hard-negative hardening framework", fontsize=15, ha="left")
    ax.text(0.55, 0.55, "Loop continues while bypass candidates are found under the configured search budget.", fontsize=10, ha="left")

    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.25, 5.9)
    ax.axis("off")
    savefig("framework_loop_diagram.png")

def plot_reconhid_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    boxes = {
        "input": (0.45, 2.65, 1.75, 0.75, "Input sequence\n50 × 15 features"),
        "encoder": (2.75, 2.65, 1.65, 0.75, "LSTM\nencoder"),
        "latent": (4.85, 2.65, 1.65, 0.75, "Latent\nrepresentation"),
        "recon": (7.05, 3.75, 1.9, 0.75, "Reconstruction\nhead"),
        "clf": (7.05, 2.65, 1.9, 0.75, "Classifier\nhead"),
        "proto": (7.05, 1.55, 1.9, 0.75, "Prototype /\nlatent distance"),
        "oracle": (9.55, 2.65, 1.7, 0.75, "Fusion / Oracle\nOR90, OR95"),
    }

    centers = {}
    for key, (x, y, w, h, text) in boxes.items():
        add_box(ax, x, y, w, h, text, fontsize=10)
        centers[key] = (x + w / 2, y + h / 2)

    add_arrow(ax, centers["input"][0] + 0.88, centers["input"][1], centers["encoder"][0] - 0.83, centers["encoder"][1])
    add_arrow(ax, centers["encoder"][0] + 0.83, centers["encoder"][1], centers["latent"][0] - 0.83, centers["latent"][1])

    add_arrow(ax, centers["latent"][0] + 0.83, centers["latent"][1] + 0.1, centers["recon"][0] - 0.95, centers["recon"][1])
    add_arrow(ax, centers["latent"][0] + 0.83, centers["latent"][1], centers["clf"][0] - 0.95, centers["clf"][1])
    add_arrow(ax, centers["latent"][0] + 0.83, centers["latent"][1] - 0.1, centers["proto"][0] - 0.95, centers["proto"][1])

    add_arrow(ax, centers["recon"][0] + 0.95, centers["recon"][1], centers["oracle"][0] - 0.85, centers["oracle"][1] + 0.22)
    add_arrow(ax, centers["clf"][0] + 0.95, centers["clf"][1], centers["oracle"][0] - 0.85, centers["oracle"][1])
    add_arrow(ax, centers["proto"][0] + 0.95, centers["proto"][1], centers["oracle"][0] - 0.85, centers["oracle"][1] - 0.22)

    ax.text(0.45, 4.9, "ReCon-HID v2 hybrid detector architecture", fontsize=15, ha="left")
    ax.text(0.45, 0.75, "The detector combines reconstruction, classifier, and prototype/latent signals for attack decision.", fontsize=10, ha="left")

    ax.set_xlim(0, 11.7)
    ax.set_ylim(0.45, 5.25)
    ax.axis("off")
    savefig("reconhid_v2_architecture.png")

def main():
    df = prepare_benchmark()

    plot_heatmap(
        df,
        value_col="or95_detection_rate",
        title="OR95 detection rate across detectors and attack rounds",
        filename="detector_attack_heatmap_or95_v2.png",
        cmap="viridis",
    )

    plot_heatmap(
        df,
        value_col="or90_bypass_rate",
        title="OR90 bypass rate across detectors and attack rounds",
        filename="detector_attack_heatmap_or90_bypass_v2.png",
        cmap="magma",
    )

    plot_grouped_bar_or95(df)
    plot_bypass_yield_by_round()
    plot_timeline_v2()
    plot_framework_loop_diagram()
    plot_reconhid_architecture()

if __name__ == "__main__":
    main()
