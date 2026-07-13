from pathlib import Path
import pandas as pd

BASE = Path("paper_artifacts/tables/baselines")
BASE.mkdir(parents=True, exist_ok=True)

def write_md(df: pd.DataFrame, path: Path, index: bool = False):
    if index:
        df = df.reset_index()
    df = df.fillna("")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def make_guided_random_final():
    summary_path = BASE / "d3_guided_vs_random_matched_summary.csv"
    if not summary_path.exists():
        print("[skip] d3_guided_vs_random_matched_summary.csv not found")
        return

    df = pd.read_csv(summary_path)

    needed = {"detector", "attack_short", "or95_detection_rate", "or90_bypass_rate"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[skip] missing columns in guided/random summary: {missing}")
        return

    attack_order = ["V4", "V5_test", "V6_test", "V7_test", "V8_candidate", "V8_hardest"]

    pivot_or95 = df.pivot_table(
        index="detector",
        columns="attack_short",
        values="or95_detection_rate",
        aggfunc="mean",
    )

    pivot_or90_bypass = df.pivot_table(
        index="detector",
        columns="attack_short",
        values="or90_bypass_rate",
        aggfunc="mean",
    )

    rows = []
    for attack in attack_order:
        if attack not in pivot_or95.columns:
            continue

        guided = float(pivot_or95.loc["D3_guided", attack])
        random = float(pivot_or95.loc["D3_random_matched", attack])
        guided_bypass = float(pivot_or90_bypass.loc["D3_guided", attack])
        random_bypass = float(pivot_or90_bypass.loc["D3_random_matched", attack])

        rows.append({
            "attack_set": attack,
            "D3_guided_OR95_detection": round(guided, 2),
            "D3_random_matched_OR95_detection": round(random, 2),
            "OR95_detection_gap": round(guided - random, 2),
            "D3_guided_OR90_bypass": round(guided_bypass, 2),
            "D3_random_matched_OR90_bypass": round(random_bypass, 2),
            "OR90_bypass_gap": round(random_bypass - guided_bypass, 2),
        })

    final = pd.DataFrame(rows)

    out_csv = BASE / "d3_guided_vs_random_matched_final.csv"
    out_md = BASE / "d3_guided_vs_random_matched_final.md"

    final.to_csv(out_csv, index=False)
    write_md(final, out_md)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_md}")

def build_report():
    make_guided_random_final()

    sections = [
        (
            "Table 1. No-hardening vs Guided Hardening: OR95 Detection",
            "baseline_no_hardening_vs_guided_or95.md",
            "D0부터 D3까지 hardening round가 진행되면서 V4~V8 공격군 탐지율이 어떻게 변하는지 보여준다."
        ),
        (
            "Table 2. No-hardening vs Guided Hardening: OR90 Bypass",
            "baseline_no_hardening_vs_guided_or90_bypass.md",
            "공격자 관점의 우회율 표다. D0/D1/D2의 blind spot이 D3에서 제거되는지 확인한다."
        ),
        (
            "Table 3. Available Signal Ablation Across Detectors",
            "baseline_available_signal_ablation_all_detectors.md",
            "classifier-only, prototype-only, hybrid OR95, hybrid OR90 신호별 탐지율을 비교한다."
        ),
        (
            "Table 4. D3 Full Signal Ablation on V8 Hardest",
            "baseline_v8_hardest_d3_full_signal_ablation.md",
            "V8 hardest에서 reconstruction-only AE가 실패하고 latent/classifier/prototype signal이 성공하는지 보여준다."
        ),
        (
            "Table 5. Generation / Bypass Yield by Round",
            "baseline_generation_yield_summary.md",
            "D1→V6, D2→V7, D3→V8로 갈수록 bypass yield가 감소하는지 보여준다."
        ),
        (
            "Table 6. Random-Matched Dataset Summary",
            "random_matched_dataset_summary.csv",
            "D3_random_matched baseline이 guided baseline과 같은 수의 V6/V7 sample을 사용했는지 확인한다."
        ),
        (
            "Table 7. D3 Guided vs Random-Matched: Full Summary",
            "d3_guided_vs_random_matched_summary.md",
            "D3_guided와 D3_random_matched의 전체 상세 평가 결과다."
        ),
        (
            "Table 8. D3 Guided vs Random-Matched: OR95 Detection",
            "d3_guided_vs_random_matched_or95_detection.md",
            "D3_guided와 D3_random_matched의 OR95 탐지율 핵심 비교표다."
        ),
        (
            "Table 9. D3 Guided vs Random-Matched: OR90 Bypass",
            "d3_guided_vs_random_matched_or90_bypass.md",
            "D3_guided와 D3_random_matched의 OR90 우회율 비교표다."
        ),
        (
            "Table 10. D3 Guided vs Random-Matched: Compact Final Table",
            "d3_guided_vs_random_matched_final.md",
            "논문 본문에 바로 넣기 좋은 compact 최종 비교표다."
        ),
    ]

    report_lines = []
    report_lines.append("# All Baseline Comparison Tables\n")
    report_lines.append("This report collects all baseline and ablation tables for the detector-guided hard-negative hardening study.\n")

    for title, filename, desc in sections:
        path = BASE / filename
        report_lines.append(f"\n## {title}\n")
        report_lines.append(desc + "\n")

        if not path.exists():
            report_lines.append(f"\n**Missing file:** `{path}`\n")
            continue

        if path.suffix == ".csv":
            df = pd.read_csv(path)
            tmp_md = BASE / (path.stem + ".auto.md")
            write_md(df, tmp_md)
            report_lines.append(tmp_md.read_text(encoding="utf-8"))
        else:
            report_lines.append(path.read_text(encoding="utf-8"))

    out = BASE / "all_baseline_tables_report.md"
    out.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[saved] {out}")

if __name__ == "__main__":
    build_report()
