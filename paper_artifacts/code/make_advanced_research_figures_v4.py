from __future__ import annotations

from pathlib import Path
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e


# ============================================================
# Paths
# ============================================================
# This script is expected at paper_artifacts/code/make_advanced_research_figures_v4.py
PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parent

DATA_ROOT = PAPER_ROOT / "datasets"
TABLE_ROOT = PAPER_ROOT / "tables"
BASELINE_ROOT = TABLE_ROOT / "baselines"
FIG_ROOT = PAPER_ROOT / "figures" / "v4"
FIG_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Selected HID features
# ============================================================
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


# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.35,
})


# ============================================================
# Helpers
# ============================================================
def savefig(fig: plt.Figure, name: str) -> None:
    png = FIG_ROOT / f"{name}.png"
    pdf = FIG_ROOT / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")
    plt.close(fig)


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {e}")
        return pd.DataFrame()


def concat_csvs(paths: List[Path], max_rows: int | None = None, random_state: int = 7) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = read_csv_safely(p)
        if not df.empty:
            df["_source_file"] = str(p)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state)
    return out.reset_index(drop=True)


def numeric_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Return numeric matrix. Prefer selected HID feature columns when available."""
    if df.empty:
        return np.empty((0, 0)), []

    if all(f in df.columns for f in SELECTED_FEATURES):
        use_cols = SELECTED_FEATURES
    else:
        drop_like = [
            "label", "session", "timestamp", "participant", "scenario",
            "source", "id", "device", "path", "file"
        ]
        numeric_cols = []
        for c in df.columns:
            if any(k in c.lower() for k in drop_like):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
        use_cols = numeric_cols

    if not use_cols:
        return np.empty((0, 0)), []

    x = df[use_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return x, use_cols


def build_group_frames(max_per_group: int = 900) -> Dict[str, pd.DataFrame]:
    groups = {}

    groups["Normal"] = concat_csvs(
        [DATA_ROOT / "normal" / "window_typing_10000.csv"],
        max_rows=max_per_group,
        random_state=1,
    )

    groups["V4"] = concat_csvs(
        sorted((DATA_ROOT / "attack" / "FeatureGuideHumanMimic").glob("*.csv")),
        max_rows=max_per_group,
        random_state=2,
    )

    groups["V5"] = concat_csvs(
        sorted((DATA_ROOT / "attack" / "AdaptiveMimic_v5_constrained_test").glob("*.csv")),
        max_rows=max_per_group,
        random_state=3,
    )

    groups["V6"] = concat_csvs(
        [DATA_ROOT / "attack" / "AdaptiveMimic_v6_D1_OR90_bypass_test" / "v6_or90_test.csv"],
        max_rows=max_per_group,
        random_state=4,
    )

    groups["V7"] = concat_csvs(
        [DATA_ROOT / "attack" / "AdaptiveMimic_v7_D2_OR90_bypass_test" / "v7_or90_test.csv"],
        max_rows=max_per_group,
        random_state=5,
    )

    groups["V8 candidate"] = concat_csvs(
        [DATA_ROOT / "attack" / "AdaptiveMimic_v8_candidate" / "v8_candidates.csv"],
        max_rows=max_per_group,
        random_state=6,
    )

    groups["V8 hardest"] = concat_csvs(
        [DATA_ROOT / "attack" / "AdaptiveMimic_v8_D3_hardest_no_bypass" / "AdaptiveMimic_v8_D3_hardest_no_bypass.csv"],
        max_rows=max_per_group,
        random_state=7,
    )

    # Remove empty groups
    return {k: v for k, v in groups.items() if not v.empty}


def build_embedding_matrix(max_per_group: int = 900) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    groups = build_group_frames(max_per_group=max_per_group)

    xs, ys = [], []
    cols_ref = None
    for label, df in groups.items():
        x, cols = numeric_matrix(df)
        if x.size == 0:
            continue
        if cols_ref is None:
            cols_ref = cols
        # Align dimensions if numeric columns differ
        if len(cols) != len(cols_ref):
            common = [c for c in cols_ref if c in df.columns]
            if not common:
                continue
            x = df[common].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            cols = common
        xs.append(x)
        ys.extend([label] * len(x))

    if not xs:
        raise RuntimeError("No dataset rows could be loaded. Check paper_artifacts/datasets.")

    X = np.vstack(xs)
    y = np.array(ys)
    return X, y, sorted(set(y.tolist())), cols_ref or []


# ============================================================
# Figure 1: t-SNE behavior space
# ============================================================
def plot_attack_space_tsne() -> None:
    X, y, labels, cols = build_embedding_matrix(max_per_group=700)
    Xs = StandardScaler().fit_transform(X)

    n = len(Xs)
    perplexity = min(35, max(5, n // 80))
    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    markers = ["o", "^", "s", "D", "P", "X", "*", "v"]
    for idx, label in enumerate(labels):
        mask = y == label
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=16,
            alpha=0.72,
            marker=markers[idx % len(markers)],
            label=f"{label} (n={mask.sum()})",
            linewidths=0.3,
        )

    ax.set_title("t-SNE projection of HID behavior windows", fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True)
    ax.legend(loc="best", frameon=True, ncol=2)
    ax.text(
        0.01, -0.12,
        "Purpose: visualize whether detector-guided hard negatives occupy normal-adjacent regions in behavior-feature space.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    savefig(fig, "attack_space_tsne_v4")


# ============================================================
# Figure 2: PCA behavior space
# ============================================================
def plot_attack_space_pca() -> None:
    X, y, labels, cols = build_embedding_matrix(max_per_group=1000)
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    markers = ["o", "^", "s", "D", "P", "X", "*", "v"]
    for idx, label in enumerate(labels):
        mask = y == label
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=14,
            alpha=0.7,
            marker=markers[idx % len(markers)],
            label=f"{label} (n={mask.sum()})",
            linewidths=0.3,
        )

    ax.set_title("PCA projection of HID behavior windows", fontweight="bold")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    ax.grid(True)
    ax.legend(loc="best", frameon=True, ncol=2)
    ax.text(
        0.01, -0.12,
        "Purpose: provide a deterministic low-dimensional view complementary to t-SNE.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    savefig(fig, "attack_space_pca_v4")


# ============================================================
# Figure 3: feature mean shift heatmap
# ============================================================
def plot_feature_shift_heatmap() -> None:
    groups = build_group_frames(max_per_group=None)
    if "Normal" not in groups:
        raise RuntimeError("Normal dataset is required for feature shift heatmap.")

    normal_x, cols = numeric_matrix(groups["Normal"])
    if not cols:
        raise RuntimeError("No numeric features found.")

    # Use selected features if possible, otherwise use first 20 numeric columns
    if all(f in groups["Normal"].columns for f in SELECTED_FEATURES):
        cols = SELECTED_FEATURES
    else:
        cols = cols[:20]

    normal = groups["Normal"][cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = normal.mean()
    sigma = normal.std().replace(0, 1.0)

    rows = []
    row_labels = []
    for label, df in groups.items():
        if label == "Normal":
            continue
        if not all(c in df.columns for c in cols):
            continue
        cur = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        z = ((cur.mean() - mu) / sigma).clip(-5, 5)
        rows.append(z.to_numpy())
        row_labels.append(label)

    if not rows:
        raise RuntimeError("No attack groups with matching numeric columns were found.")

    mat = np.vstack(rows)

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title("Feature-level mean shift from normal behavior", fontweight="bold")
    ax.set_xlabel("HID behavior feature")
    ax.set_ylabel("Attack group")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z-score shift from normal mean")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if abs(mat[i, j]) >= 1.5:
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7)

    ax.text(
        0.01, -0.24,
        "Purpose: identify which behavioral features make each generated attack family depart from normal input.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    savefig(fig, "feature_shift_heatmap_v4")


# ============================================================
# Figure 4: radar profile for important features
# ============================================================
def plot_feature_profile_radar() -> None:
    groups = build_group_frames(max_per_group=None)
    if "Normal" not in groups:
        raise RuntimeError("Normal dataset is required for radar profile.")

    focus = [
        "hold_time",
        "flight_time",
        "press_to_press_time",
        "keys_per_second",
        "timing_variance",
        "timing_entropy",
        "pause_duration",
        "correction_ratio",
    ]
    focus = [f for f in focus if f in groups["Normal"].columns]
    if len(focus) < 4:
        warnings.warn("Not enough named selected features found; radar plot skipped.")
        return

    normal = groups["Normal"][focus].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = normal.mean()
    sigma = normal.std().replace(0, 1.0)

    labels_to_plot = [g for g in ["V4", "V6", "V7", "V8 hardest"] if g in groups]
    if not labels_to_plot:
        warnings.warn("No attack groups found for radar plot.")
        return

    angles = np.linspace(0, 2 * np.pi, len(focus), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.2, 8.2))
    ax = plt.subplot(111, polar=True)

    for label in labels_to_plot:
        df = groups[label]
        if not all(f in df.columns for f in focus):
            continue
        cur = df[focus].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        z = ((cur.mean() - mu) / sigma).abs().clip(0, 4)
        values = z.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=label)
        ax.fill(angles, values, alpha=0.08)

    ax.set_title("Behavioral deviation profile of generated attack families", y=1.08, fontweight="bold")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(focus)
    ax.set_ylim(0, 4)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1σ", "2σ", "3σ", "4σ"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=True)
    savefig(fig, "feature_profile_radar_v4")


# ============================================================
# Figure 5: hard-negative funnel
# ============================================================
def plot_hard_negative_funnel() -> None:
    path = BASELINE_ROOT / "baseline_generation_yield_summary.csv"
    if not path.exists():
        path = TABLE_ROOT / "bypass_yield_by_round.csv"
    if not path.exists():
        warnings.warn("No generation yield table found; funnel skipped.")
        return

    df = pd.read_csv(path)
    if "round" not in df.columns:
        # Try common fallback columns
        df.columns = [c.strip() for c in df.columns]

    rounds = df["round"].astype(str).tolist()
    generated = df["generated_candidates"].astype(float).to_numpy()
    selected = df["selected_or_bypass"].astype(float).to_numpy()
    yield_rate = df["yield_rate"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.8))
    y = np.arange(len(rounds))

    ax.barh(y + 0.18, generated, height=0.32, label="Generated candidates")
    ax.barh(y - 0.18, selected, height=0.32, label="Bypass hard negatives")
    ax.set_yticks(y)
    ax.set_yticklabels(rounds)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    ax.set_title("Hard-negative mining funnel by detector-guided round", fontweight="bold")
    ax.grid(axis="x")
    ax.legend(frameon=True)

    for i, (g, s, r) in enumerate(zip(generated, selected, yield_rate)):
        ax.text(g + max(generated) * 0.01, i + 0.18, f"{int(g)}", va="center", fontsize=9)
        ax.text(max(s + max(generated) * 0.01, max(generated) * 0.01), i - 0.18,
                f"{int(s)} ({r:.2f}%)", va="center", fontsize=9)

    savefig(fig, "hard_negative_funnel_v4")


# ============================================================
# Figure 6: V8 hardest signal ablation
# ============================================================
def plot_v8_hardest_signal_ablation() -> None:
    path = BASELINE_ROOT / "baseline_v8_hardest_d3_full_signal_ablation.csv"
    if not path.exists():
        warnings.warn("V8 hardest ablation table not found; skipped.")
        return

    df = pd.read_csv(path)
    cols = [c for c in df.columns if "detection" in c.lower() or "rate" in c.lower()]
    # Expected columns may vary. Try to infer signal and detection rate.
    signal_col = "signal" if "signal" in df.columns else df.columns[0]
    det_col = None
    for candidate in ["detection_rate", "V8_hardest_detection", "detection", "detect_rate"]:
        if candidate in df.columns:
            det_col = candidate
            break
    if det_col is None:
        # choose last numeric column if exact name unknown
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            warnings.warn("Could not infer detection column for V8 ablation.")
            return
        det_col = numeric_cols[-1]

    signals = df[signal_col].astype(str).tolist()
    values = df[det_col].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(signals))
    ax.bar(x, values, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(signals, rotation=35, ha="right")
    ax.set_ylim(0, max(105, values.max() * 1.15))
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("D3 signal ablation on V8 hardest no-bypass set", fontweight="bold")
    ax.grid(axis="y")

    for xi, yi in zip(x, values):
        ax.text(xi, yi + 1.5, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)

    ax.text(
        0.01, -0.26,
        "Purpose: show why reconstruction-only detection is insufficient for the hardest adaptive mimics.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    savefig(fig, "v8_hardest_signal_ablation_v4")


# ============================================================
# Figure 7: code artifact map
# ============================================================
def plot_code_artifact_map() -> None:
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.95, "Code artifact map for the detector-guided hardening pipeline",
            fontsize=15, fontweight="bold", va="top")

    boxes = [
        (0.04, 0.70, 0.23, 0.14, "Data\nnormal + V4/V5/V6/V7/V8", "paper_artifacts/datasets/"),
        (0.36, 0.70, 0.23, 0.14, "Attack generation\nV5/V6/V7/V8 candidates", "code/experiments/attack_generation_*"),
        (0.68, 0.70, 0.23, 0.14, "Detector oracle\nOR90 / OR95 decision", "code/experiments/recon_hid_v2/detector_oracle_v2.py"),
        (0.20, 0.38, 0.23, 0.14, "Hardening training\nD0 / D1 / D2 / D3", "code/experiments/co_evolution_round*/train_*.py"),
        (0.52, 0.38, 0.23, 0.14, "Baselines\nrandom-matched + reports", "code/baselines/"),
        (0.36, 0.10, 0.23, 0.14, "Paper outputs\nfigures + tables + notes", "figures/  tables/  notes/"),
    ]

    for x, y, w, h, title, path in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            facecolor="white",
            edgecolor="black",
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w/2, y + h*0.25, path, ha="center", va="center", fontsize=8)

    def arrow(p1, p2):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", lw=1.5))

    arrow((0.27, 0.77), (0.36, 0.77))
    arrow((0.59, 0.77), (0.68, 0.77))
    arrow((0.78, 0.70), (0.38, 0.52))
    arrow((0.43, 0.38), (0.52, 0.45))
    arrow((0.64, 0.38), (0.50, 0.24))
    arrow((0.31, 0.38), (0.47, 0.24))

    ax.text(
        0.02, 0.02,
        "Use this figure when explaining that the seminar results are grounded in reproducible code artifacts, not only diagrams.",
        fontsize=9,
    )

    savefig(fig, "code_artifact_map_v4")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("============================================================")
    print("Generating advanced research figures v4")
    print(f"PAPER_ROOT = {PAPER_ROOT}")
    print(f"FIG_ROOT   = {FIG_ROOT}")
    print("============================================================")

    plot_attack_space_tsne()
    plot_attack_space_pca()
    plot_feature_shift_heatmap()
    plot_feature_profile_radar()
    plot_hard_negative_funnel()
    plot_v8_hardest_signal_ablation()
    plot_code_artifact_map()

    print("============================================================")
    print("Done.")
    print("============================================================")


if __name__ == "__main__":
    main()
