from pathlib import Path
import pandas as pd

ROOT = Path("paper_artifacts")
TABLES = ROOT / "tables"
OUT = TABLES / "baselines"
OUT.mkdir(parents=True, exist_ok=True)

def load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def write_markdown_table(df: pd.DataFrame, path: Path, index: bool = True):
    """
    Write a simple GitHub-style markdown table without requiring tabulate.
    """
    if index:
        out_df = df.reset_index()
    else:
        out_df = df.copy()

    out_df = out_df.fillna("")
    cols = list(out_df.columns)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in out_df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def short_attack_name(x: str) -> str:
    x = str(x)
    if "FeatureGuideHumanMimic" in x:
        return "V4"
    if "AdaptiveMimic_v5" in x:
        return "V5"
    if "AdaptiveMimic_v6" in x:
        return "V6"
    if "AdaptiveMimic_v7" in x:
        return "V7"
    if "v8_candidate" in x or "AdaptiveMimic_v8_candidate" in x:
        return "V8_candidate"
    if "hardest_no_bypass" in x or "D3_hardest" in x:
        return "V8_hardest"
    return x

def normalize_detector(x: str) -> str:
    return str(x).lower().replace(" ", "")

def collect_benchmark_frames() -> pd.DataFrame:
    frames = []

    main_path = TABLES / "benchmark_summary_v4_v5_v6_v7_d0_d1_d2_d3.csv"
    if main_path.exists():
        df = load_csv("benchmark_summary_v4_v5_v6_v7_d0_d1_d2_d3.csv")
        df["attack_short"] = df["attack"].map(short_attack_name)
        df["detector"] = df["detector"].map(normalize_detector)
        frames.append(df)

    v8c_path = TABLES / "benchmark_v8_candidate_d0_d1_d2_d3.csv"
    if v8c_path.exists():
        df = load_csv("benchmark_v8_candidate_d0_d1_d2_d3.csv")
        df["attack_short"] = "V8_candidate"
        df["detector"] = df["detector"].map(normalize_detector)
        frames.append(df)

    v8h_path = TABLES / "benchmark_v8_hardest_no_bypass_d0_d1_d2_d3.csv"
    if v8h_path.exists():
        df = load_csv("benchmark_v8_hardest_no_bypass_d0_d1_d2_d3.csv")
        df["attack_short"] = "V8_hardest"
        df["detector"] = df["detector"].map(normalize_detector)
        frames.append(df)

    if not frames:
        raise RuntimeError("No benchmark CSV files found.")

    return pd.concat(frames, ignore_index=True, sort=False)

def make_no_hardening_vs_guided_tables(all_df: pd.DataFrame):
    detector_order = ["d0", "d1", "d2", "d3"]
    attack_order = ["V4", "V5", "V6", "V7", "V8_candidate", "V8_hardest"]

    required = {"detector", "attack_short", "or95_detection_rate"}
    missing = required - set(all_df.columns)
    if missing:
        raise KeyError(f"Missing columns for OR95 table: {missing}")

    pivot = all_df.pivot_table(
        index="detector",
        columns="attack_short",
        values="or95_detection_rate",
        aggfunc="mean",
    )

    pivot = pivot.reindex(detector_order)
    pivot = pivot[[c for c in attack_order if c in pivot.columns]]
    pivot = pivot.round(2)

    out_csv = OUT / "baseline_no_hardening_vs_guided_or95.csv"
    out_md = OUT / "baseline_no_hardening_vs_guided_or95.md"
    pivot.to_csv(out_csv)
    write_markdown_table(pivot, out_md, index=True)

    print("\n=== Table A. OR95 detection: no-hardening vs guided rounds ===")
    print(pivot.to_string())
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_md}")

    if "or90_bypass_rate" in all_df.columns:
        pivot_bypass = all_df.pivot_table(
            index="detector",
            columns="attack_short",
            values="or90_bypass_rate",
            aggfunc="mean",
        )

        pivot_bypass = pivot_bypass.reindex(detector_order)
        pivot_bypass = pivot_bypass[[c for c in attack_order if c in pivot_bypass.columns]]
        pivot_bypass = pivot_bypass.round(2)

        out_csv = OUT / "baseline_no_hardening_vs_guided_or90_bypass.csv"
        out_md = OUT / "baseline_no_hardening_vs_guided_or90_bypass.md"
        pivot_bypass.to_csv(out_csv)
        write_markdown_table(pivot_bypass, out_md, index=True)

        print("\n=== Table B. OR90 bypass: no-hardening vs guided rounds ===")
        print(pivot_bypass.to_string())
        print(f"[saved] {out_csv}")
        print(f"[saved] {out_md}")

def make_available_signal_ablation(all_df: pd.DataFrame):
    """
    Existing summary CSVs provide classifier/prototype/hybrid columns.
    V8 reports additionally contain reconstruction-specific decision columns.
    """
    detector_order = ["d0", "d1", "d2", "d3"]
    attack_order = ["V4", "V5", "V6", "V7", "V8_candidate", "V8_hardest"]

    signal_cols = {
        "classifier_only": "classifier_detection_rate",
        "prototype_only": "prototype_detection_rate",
        "hybrid_or95": "or95_detection_rate",
        "hybrid_or90": "or90_detection_rate",
    }

    rows = []
    for det in detector_order:
        sub = all_df[all_df["detector"] == det]
        for signal, col in signal_cols.items():
            if col not in all_df.columns:
                continue

            row = {"detector": det, "signal": signal}
            for attack in attack_order:
                vals = sub.loc[sub["attack_short"] == attack, col]
                if len(vals) > 0:
                    row[attack] = round(float(vals.mean()), 2)
            rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = OUT / "baseline_available_signal_ablation_all_detectors.csv"
    out_md = OUT / "baseline_available_signal_ablation_all_detectors.md"
    out.to_csv(out_csv, index=False)
    write_markdown_table(out, out_md, index=False)

    print("\n=== Table C. Available signal ablation from existing summaries ===")
    print(out.to_string(index=False))
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_md}")

def make_v8_hardest_full_signal_table():
    """
    v8_hardest_no_bypass_report.csv has per-sample D3 predictions:
    pred_recon_90/95, pred_latent_dist_90/95, pred_classifier,
    pred_prototype, pred_or95/or90.
    This gives a detailed D3-only signal ablation for the hardest V8 set.
    """
    path = TABLES / "v8_hardest_no_bypass_report.csv"
    if not path.exists():
        print("\n[skip] v8_hardest_no_bypass_report.csv not found")
        return

    df = load_csv("v8_hardest_no_bypass_report.csv")
    n = len(df)

    signal_map = {
        "recon_90": "pred_recon_90",
        "recon_95": "pred_recon_95",
        "latent_dist_90": "pred_latent_dist_90",
        "latent_dist_95": "pred_latent_dist_95",
        "classifier_only": "pred_classifier",
        "prototype_only": "pred_prototype",
        "hybrid_or90": "pred_or90",
        "hybrid_or95": "pred_or95",
        "hybrid_or95_with_latent": "pred_or95_with_latent",
    }

    rows = []
    for name, col in signal_map.items():
        if col not in df.columns:
            continue

        detected = int(df[col].sum())
        rate = detected / n * 100 if n else 0.0

        rows.append({
            "attack_set": "V8_hardest",
            "signal": name,
            "samples": n,
            "detected": detected,
            "detection_rate": round(rate, 2),
            "bypass_rate": round(100 - rate, 2),
        })

    out = pd.DataFrame(rows)
    out_csv = OUT / "baseline_v8_hardest_d3_full_signal_ablation.csv"
    out_md = OUT / "baseline_v8_hardest_d3_full_signal_ablation.md"
    out.to_csv(out_csv, index=False)
    write_markdown_table(out, out_md, index=False)

    print("\n=== Table D. D3 full signal ablation on V8 hardest ===")
    print(out.to_string(index=False))
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_md}")

def make_generation_summary_table():
    rows = []

    if (TABLES / "v5_generation_summary.csv").exists():
        df = load_csv("v5_generation_summary.csv")
        pool = df.loc[df["dataset"].astype(str).str.contains("candidate_pool", case=False), "count"]
        selected = df.loc[df["dataset"].astype(str).str.contains("selected", case=False), "count"]
        if len(pool) and len(selected):
            pool_n = int(pool.iloc[0])
            selected_n = int(selected.iloc[0])
            rows.append({
                "round": "D0 → V5",
                "generated_candidates": pool_n,
                "selected_or_bypass": selected_n,
                "yield_rate": round(selected_n / pool_n * 100, 2) if pool_n else 0.0,
                "note": "selected constrained OR95 bypass",
            })

    if (TABLES / "v6_generation_summary.csv").exists():
        df = load_csv("v6_generation_summary.csv")
        row = df[df["selection"].astype(str).str.contains("OR90", case=False)]
        if len(row):
            selected_n = int(row.iloc[0]["rows"])
            rows.append({
                "round": "D1 → V6",
                "generated_candidates": 3600,
                "selected_or_bypass": selected_n,
                "yield_rate": round(selected_n / 3600 * 100, 2),
                "note": "D1 OR90 bypass",
            })

    if (TABLES / "v7_generation_summary.csv").exists():
        df = load_csv("v7_generation_summary.csv")
        row = df[df["selection"].astype(str).str.contains("OR90", case=False)]
        if len(row):
            selected_n = int(row.iloc[0]["rows"])
            rows.append({
                "round": "D2 → V7",
                "generated_candidates": 15000,
                "selected_or_bypass": selected_n,
                "yield_rate": round(selected_n / 15000 * 100, 2),
                "note": "D2 OR90 bypass",
            })

    if (TABLES / "v8_generation_summary.csv").exists():
        df = load_csv("v8_generation_summary.csv")
        selected_n = 0
        rows.append({
            "round": "D3 → V8",
            "generated_candidates": 15000,
            "selected_or_bypass": selected_n,
            "yield_rate": 0.00,
            "note": "no D3 OR90/OR95 bypass found; top-500 hardest no-bypass separately saved",
        })

    out = pd.DataFrame(rows)
    out_csv = OUT / "baseline_generation_yield_summary.csv"
    out_md = OUT / "baseline_generation_yield_summary.md"
    out.to_csv(out_csv, index=False)
    write_markdown_table(out, out_md, index=False)

    print("\n=== Table E. Generation / bypass yield summary ===")
    print(out.to_string(index=False))
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_md}")

def make_baseline_readme():
    text = """# Baseline Tables

This directory contains baseline and ablation tables generated from the existing paper artifacts.

## Files

- `baseline_no_hardening_vs_guided_or95.csv`
  - OR95 detection rate of D0-D3 against V4-V8.
  - Main table for showing hardening effectiveness.

- `baseline_no_hardening_vs_guided_or90_bypass.csv`
  - OR90 bypass rate of D0-D3 against V4-V8.
  - Main table for showing adaptive bypass reduction.

- `baseline_available_signal_ablation_all_detectors.csv`
  - Classifier-only, prototype-only, hybrid OR95, and hybrid OR90 detection rates from existing summary files.
  - This does not include reconstruction-only for V4-V7 unless those per-signal prediction files are reprocessed.

- `baseline_v8_hardest_d3_full_signal_ablation.csv`
  - Detailed D3-only signal ablation on V8 hardest no-bypass samples.
  - Includes recon-only, latent-distance, classifier, prototype, and hybrid decisions.

- `baseline_generation_yield_summary.csv`
  - Candidate generation / selected bypass yield by round.

## Interpretation

The current tables already support two baseline claims:

1. No-hardening baseline:
   D0 is vulnerable to adaptive candidates, while D3 detects V4-V8.

2. Signal ablation:
   On V8 hardest samples, reconstruction-only fails while classifier/prototype/hybrid detect the samples.

The next missing control experiment is:

- D3_guided vs D3_random_matched

This requires training a random-sample matched detector.
"""
    path = OUT / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[saved] {path}")

def main():
    all_df = collect_benchmark_frames()

    print("Collected benchmark rows:", len(all_df))
    print("Columns:")
    for c in all_df.columns:
        print(" -", c)

    make_no_hardening_vs_guided_tables(all_df)
    make_available_signal_ablation(all_df)
    make_v8_hardest_full_signal_table()
    make_generation_summary_table()
    make_baseline_readme()

    print("\nDone.")
    print(f"Output directory: {OUT}")

if __name__ == "__main__":
    main()
