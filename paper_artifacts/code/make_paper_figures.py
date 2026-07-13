from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("paper_artifacts")
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

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
    for i in range(4):
        if x == f"d{i}":
            return i
    return 999

def prepare_main_benchmark() -> pd.DataFrame:
    main = load_csv(TABLES / "benchmark_summary_v4_v5_v6_v7_d0_d1_d2_d3.csv")
    main["attack_short"] = main["attack"].map(short_attack_name)
    main["detector"] = main["detector"].astype(str).str.lower()

    extra_frames = []

    v8_candidate_path = TABLES / "benchmark_v8_candidate_d0_d1_d2_d3.csv"
    if v8_candidate_path.exists():
        v8c = load_csv(v8_candidate_path)
        v8c["attack_short"] = "V8\ncandidate"
        v8c["detector"] = v8c["detector"].astype(str).str.lower()
        extra_frames.append(v8c)

    v8_hardest_path = TABLES / "benchmark_v8_hardest_no_bypass_d0_d1_d2_d3.csv"
    if v8_hardest_path.exists():
        v8h = load_csv(v8_hardest_path)
        v8h["attack_short"] = "V8\nhardest"
        v8h["detector"] = v8h["detector"].astype(str).str.lower()
        extra_frames.append(v8h)

    if extra_frames:
        df = pd.concat([main] + extra_frames, ignore_index=True, sort=False)
    else:
        df = main

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

def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, filename: str, cmap="viridis"):
    pivot = df.pivot_table(
        index="attack_short",
        columns="detector",
        values=value_col,
        aggfunc="mean",
        observed=False,
    )

    cols = sorted([c for c in pivot.columns if str(c).lower().startswith("d")], key=detector_order_key)
    pivot = pivot[cols]
    pivot = pivot.dropna(how="all")

    data = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(8.8, 5.2))
    im = plt.imshow(data, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.set_label("Rate (%)")

    plt.xticks(np.arange(len(pivot.columns)), [str(c).upper() for c in pivot.columns])
    plt.yticks(np.arange(len(pivot.index)), list(pivot.index))
    plt.title(title)
    plt.xlabel("Detector")
    plt.ylabel("Attack / candidate set")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                text = "-"
            else:
                text = f"{data[i, j]:.1f}"
            plt.text(j, i, text, ha="center", va="center", fontsize=9)

    savefig(filename)

def plot_grouped_bar_or95(df: pd.DataFrame):
    # Main V4~V7 only, because this is the cleanest detector-vs-attack comparison.
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

    plt.figure(figsize=(9.5, 5.2))
    for idx, col in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns)-1)/2) * width
        plt.bar(x + offset, pivot[col].values, width, label=str(col).upper())

    plt.xticks(x, pivot.index)
    plt.ylim(0, 110)
    plt.ylabel("OR95 detection rate (%)")
    plt.xlabel("Attack set")
    plt.title("Detector performance across adaptive attack rounds")
    plt.legend(title="Detector")
    plt.grid(axis="y", alpha=0.3)

    savefig("detector_attack_grouped_bar_or95.png")

def plot_bypass_yield_by_round():
    # These are the key co-evolution search yields from the completed experiment.
    # V6: 2431 / 3600 = 67.53%
    # V7: 252 / 15000 = 1.68%
    # V8: 0 / 15000 = 0.00%
    rows = [
        {"round": "D1 → V6", "target_detector": "D1", "generated_candidates": 3600, "bypass_candidates": 2431, "bypass_yield": 67.53},
        {"round": "D2 → V7", "target_detector": "D2", "generated_candidates": 15000, "bypass_candidates": 252, "bypass_yield": 1.68},
        {"round": "D3 → V8", "target_detector": "D3", "generated_candidates": 15000, "bypass_candidates": 0, "bypass_yield": 0.00},
    ]
    out = pd.DataFrame(rows)
    out_path = TABLES / "bypass_yield_by_round.csv"
    out.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(out["round"], out["bypass_yield"], marker="o", linewidth=2.2)
    plt.ylim(0, 75)
    plt.ylabel("Bypass yield (%)")
    plt.xlabel("Detector-guided generation round")
    plt.title("Bypass yield decreases across co-evolution rounds")
    plt.grid(axis="y", alpha=0.3)

    for x, y in zip(out["round"], out["bypass_yield"]):
        plt.text(x, y + 2.0, f"{y:.2f}%", ha="center", fontsize=10)

    savefig("bypass_yield_by_round.png")

def plot_co_evolution_timeline():
    steps = [
        ("Normal\n+ V4 seed", 0),
        ("Train\nD0", 1),
        ("Generate\nV5", 2),
        ("Harden\nD1", 3),
        ("Generate\nV6", 4),
        ("Harden\nD2", 5),
        ("Generate\nV7", 6),
        ("Harden\nD3", 7),
        ("V8 attempt\n0 bypass", 8),
    ]

    plt.figure(figsize=(13, 3.2))
    y = 0

    for label, x in steps:
        plt.scatter(x, y, s=900)
        plt.text(x, y, label, ha="center", va="center", fontsize=9)

    for i in range(len(steps)-1):
        x1 = steps[i][1]
        x2 = steps[i+1][1]
        plt.annotate(
            "",
            xy=(x2 - 0.35, y),
            xytext=(x1 + 0.35, y),
            arrowprops=dict(arrowstyle="->", lw=1.8),
        )

    plt.xlim(-0.7, 8.7)
    plt.ylim(-1, 1)
    plt.axis("off")
    plt.title("Detector-guided co-evolution timeline", pad=18)

    savefig("co_evolution_timeline.png")

def main():
    df = prepare_main_benchmark()

    required_cols = {
        "or95_detection_rate",
        "or90_bypass_rate",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    plot_heatmap(
        df,
        value_col="or95_detection_rate",
        title="OR95 detection rate across detectors and attack rounds",
        filename="detector_attack_heatmap_or95.png",
        cmap="viridis",
    )

    plot_heatmap(
        df,
        value_col="or90_bypass_rate",
        title="OR90 bypass rate across detectors and attack rounds",
        filename="detector_attack_heatmap_or90_bypass.png",
        cmap="magma",
    )

    plot_grouped_bar_or95(df)
    plot_bypass_yield_by_round()
    plot_co_evolution_timeline()

if __name__ == "__main__":
    main()
