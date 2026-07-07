import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class DetectorConfig:
    name: str
    model_path: Path
    description: str


DEFAULT_DETECTORS: Dict[str, DetectorConfig] = {
    "d0": DetectorConfig(
        name="d0",
        model_path=Path("checkpoints/recon_hid_v2.pt"),
        description="Round 0 ReCon-HID v2 detector",
    ),
    "d1": DetectorConfig(
        name="d1",
        model_path=Path("checkpoints/recon_hid_v2_round1.pt"),
        description="Round 1 ReCon-HID v2 detector trained with V5 hard negatives",
    ),
}


def run_oracle(
    detector: DetectorConfig,
    attack_dir: Path,
    attack_name: str,
    out_dir: Path,
    force: bool = False,
) -> None:
    """
    Run the existing detector oracle script.

    This script intentionally reuses src.detector_oracle_v2 so that benchmark
    results stay consistent with previous D0/D1 oracle evaluations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = list(out_dir.glob("*.csv"))
    if existing_csvs and not force:
        print(f"[SKIP] {detector.name}: oracle output already exists at {out_dir}")
        return

    if not detector.model_path.exists():
        raise FileNotFoundError(f"Missing detector model: {detector.model_path}")

    if not attack_dir.exists():
        raise FileNotFoundError(f"Missing attack dataset directory: {attack_dir}")

    cmd = [
        sys.executable,
        "-m",
        "src.detector_oracle_v2",
        "--model-path",
        str(detector.model_path),
        "--target-path",
        str(attack_dir),
        "--target-group",
        attack_name,
        "--out-dir",
        str(out_dir),
    ]

    print()
    print(f"[RUN] {detector.name} oracle")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def find_prediction_csv(out_dir: Path) -> Optional[Path]:
    """
    Find the most likely per-sample prediction CSV produced by detector_oracle_v2.
    We prefer files containing 'prediction' in the filename.
    """
    csvs = list(out_dir.rglob("*.csv"))
    if not csvs:
        return None

    prediction_like = [
        p for p in csvs
        if "prediction" in p.name.lower() or "predictions" in p.name.lower()
    ]
    if prediction_like:
        return max(prediction_like, key=lambda p: p.stat().st_size)

    # fallback: choose largest csv because per-sample predictions are usually largest
    return max(csvs, key=lambda p: p.stat().st_size)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert common boolean/int/string decision columns to 0/1.
    """
    if series.dtype == bool:
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)

    lowered = series.astype(str).str.lower().str.strip()
    return lowered.isin(["1", "true", "yes", "attack", "detected", "malicious"]).astype(int)


def find_column_exact_or_contains(
    columns: List[str],
    exact_candidates: List[str],
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in columns}

    for cand in exact_candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

    if must_contain:
        must_not_contain = must_not_contain or []
        for c in columns:
            low = c.lower()
            if all(x.lower() in low for x in must_contain) and not any(
                x.lower() in low for x in must_not_contain
            ):
                return c

    return None


def infer_label_masks(df: pd.DataFrame, attack_name: str) -> Tuple[pd.Series, pd.Series]:
    """
    Return attack_mask, normal_mask.

    Preferred:
    - label == 1 for attack, label == 0 for normal
    Fallback:
    - group/source/target_group columns
    """
    columns = list(df.columns)

    label_col = find_column_exact_or_contains(
        columns,
        exact_candidates=["label", "y", "target", "is_attack"],
    )

    if label_col is not None:
        vals = df[label_col]
        if pd.api.types.is_numeric_dtype(vals):
            attack_mask = vals.astype(float) > 0
            normal_mask = vals.astype(float) == 0
            return attack_mask, normal_mask

        lowered = vals.astype(str).str.lower().str.strip()
        attack_mask = lowered.isin(["1", "attack", "malicious", "true", attack_name.lower()])
        normal_mask = lowered.isin(["0", "normal", "benign", "false"])
        if attack_mask.any() and normal_mask.any():
            return attack_mask, normal_mask

    group_col = find_column_exact_or_contains(
        columns,
        exact_candidates=["group", "dataset", "source_group", "target_group"],
        must_contain=["group"],
    )

    if group_col is not None:
        lowered = df[group_col].astype(str).str.lower().str.strip()
        attack_mask = lowered != "normal"
        normal_mask = lowered == "normal"
        return attack_mask, normal_mask

    source_col = find_column_exact_or_contains(
        columns,
        exact_candidates=["source", "source_file", "file"],
    )

    if source_col is not None:
        lowered = df[source_col].astype(str).str.lower()
        normal_mask = lowered.str.contains("normal") | lowered.str.contains("typing")
        attack_mask = ~normal_mask
        return attack_mask, normal_mask

    # Last fallback: assume all rows are attack.
    attack_mask = pd.Series([True] * len(df), index=df.index)
    normal_mask = pd.Series([False] * len(df), index=df.index)
    return attack_mask, normal_mask


def find_decision_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    columns = list(df.columns)

    or95_col = find_column_exact_or_contains(
        columns,
        exact_candidates=[
            "pred_or95",
            "oracle_or95_balanced",
            "oracle_OR95_balanced",
            "fusion_or95",
            "or95",
            "detected_or95",
            "is_attack_or95",
        ],
        must_contain=["or95"],
        must_not_contain=["rate", "threshold", "score", "prob", "error", "mean", "distance", "dist"],
    )

    or90_col = find_column_exact_or_contains(
        columns,
        exact_candidates=[
            "pred_or90",
            "oracle_or90_aggressive",
            "oracle_OR90_aggressive",
            "fusion_or90",
            "or90",
            "detected_or90",
            "is_attack_or90",
        ],
        must_contain=["or90"],
        must_not_contain=["rate", "threshold", "score", "prob", "error", "mean", "distance", "dist"],
    )

    classifier_col = find_column_exact_or_contains(
        columns,
        exact_candidates=[
            "pred_classifier",
            "classifier_decision",
            "classifier_pred",
            "cls_decision",
            "cls_pred",
        ],
        must_contain=["classifier"],
        must_not_contain=["prob", "score", "rate", "threshold", "distance", "dist"],
    )

    prototype_col = find_column_exact_or_contains(
        columns,
        exact_candidates=[
            "pred_prototype",
            "prototype_decision",
            "prototype_pred",
            "proto_decision",
            "proto_pred",
        ],
        must_contain=["prototype"],
        must_not_contain=["prob", "score", "rate", "threshold", "distance", "dist"],
    )

    return {
        "or95": or95_col,
        "or90": or90_col,
        "classifier": classifier_col,
        "prototype": prototype_col,
    }

def find_score_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    columns = list(df.columns)

    return {
        "mean_recon_error": find_column_exact_or_contains(
            columns,
            exact_candidates=[
                "reconstruction_error",
                "recon_error",
                "mean_recon",
                "reconstruction_loss",
            ],
            must_contain=["recon"],
            must_not_contain=["threshold", "decision", "pred"],
        ),
        "mean_classifier_prob": find_column_exact_or_contains(
            columns,
            exact_candidates=[
                "classifier_attack_prob",
                "classifier_prob",
                "cls_prob",
                "attack_prob",
            ],
            must_contain=["prob"],
            must_not_contain=["prototype"],
        ),
        "mean_prototype_prob": find_column_exact_or_contains(
            columns,
            exact_candidates=[
                "prototype_attack_prob",
                "prototype_prob",
                "proto_prob",
            ],
            must_contain=["proto"],
            must_not_contain=["decision", "pred"],
        ),
        "mean_latent_distance": find_column_exact_or_contains(
            columns,
            exact_candidates=[
                "latent_distance",
                "dist_normal",
                "distance_to_normal_prototype",
                "normal_distance",
            ],
            must_contain=["dist"],
            must_not_contain=["threshold", "decision", "pred"],
        ),
    }


def safe_rate(decision: pd.Series, mask: pd.Series) -> Optional[float]:
    if mask.sum() == 0:
        return None
    return float(decision[mask].mean() * 100.0)


def summarize_oracle_output(
    detector: DetectorConfig,
    attack_name: str,
    out_dir: Path,
) -> Dict[str, object]:
    prediction_csv = find_prediction_csv(out_dir)
    if prediction_csv is None:
        return {
            "detector": detector.name,
            "detector_description": detector.description,
            "attack": attack_name,
            "prediction_csv": "",
            "attack_samples": None,
            "normal_samples": None,
            "or95_detection_rate": None,
            "or95_bypass_rate": None,
            "or95_normal_fpr": None,
            "or90_detection_rate": None,
            "or90_bypass_rate": None,
            "or90_normal_fpr": None,
            "note": "No prediction CSV found",
        }

    df = pd.read_csv(prediction_csv)
    attack_mask, normal_mask = infer_label_masks(df, attack_name)
    decision_cols = find_decision_columns(df)
    score_cols = find_score_columns(df)

    row: Dict[str, object] = {
        "detector": detector.name,
        "detector_description": detector.description,
        "attack": attack_name,
        "prediction_csv": str(prediction_csv),
        "attack_samples": int(attack_mask.sum()),
        "normal_samples": int(normal_mask.sum()),
        "note": "",
    }

    for mode in ["or95", "or90", "classifier", "prototype"]:
        col = decision_cols.get(mode)
        if col is None:
            row[f"{mode}_decision_column"] = ""
            row[f"{mode}_detection_rate"] = None
            row[f"{mode}_bypass_rate"] = None
            row[f"{mode}_normal_fpr"] = None
            continue

        decision = normalize_bool_series(df[col])
        det_rate = safe_rate(decision, attack_mask)
        fpr = safe_rate(decision, normal_mask)

        row[f"{mode}_decision_column"] = col
        row[f"{mode}_detection_rate"] = det_rate
        row[f"{mode}_bypass_rate"] = None if det_rate is None else 100.0 - det_rate
        row[f"{mode}_normal_fpr"] = fpr

    for metric_name, col in score_cols.items():
        if col is None:
            row[metric_name] = None
            row[f"{metric_name}_column"] = ""
        else:
            row[metric_name] = float(pd.to_numeric(df.loc[attack_mask, col], errors="coerce").mean())
            row[f"{metric_name}_column"] = col

    return row


def write_summary_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # stable column order first, then any remaining columns
    preferred = [
        "detector",
        "detector_description",
        "attack",
        "attack_samples",
        "normal_samples",
        "or95_detection_rate",
        "or95_bypass_rate",
        "or95_normal_fpr",
        "or90_detection_rate",
        "or90_bypass_rate",
        "or90_normal_fpr",
        "classifier_detection_rate",
        "classifier_bypass_rate",
        "classifier_normal_fpr",
        "prototype_detection_rate",
        "prototype_bypass_rate",
        "prototype_normal_fpr",
        "mean_recon_error",
        "mean_classifier_prob",
        "mean_prototype_prob",
        "mean_latent_distance",
        "prediction_csv",
        "note",
    ]

    all_keys = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    fieldnames = preferred + [k for k in all_keys if k not in preferred]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_console_table(rows: List[Dict[str, object]]) -> None:
    print()
    print("===== BENCHMARK SUMMARY =====")

    for row in rows:
        detector = row.get("detector")
        attack = row.get("attack")
        attack_n = row.get("attack_samples")
        normal_n = row.get("normal_samples")

        print()
        print(f"[{detector}] attack={attack} attack_samples={attack_n} normal_samples={normal_n}")

        for mode in ["or95", "or90", "classifier", "prototype"]:
            det = row.get(f"{mode}_detection_rate")
            bypass = row.get(f"{mode}_bypass_rate")
            fpr = row.get(f"{mode}_normal_fpr")
            col = row.get(f"{mode}_decision_column")

            if det is None:
                print(f"  - {mode}: N/A")
            else:
                print(
                    f"  - {mode}: detection={det:.2f}% "
                    f"bypass={bypass:.2f}% normal_fpr={fpr:.2f}% "
                    f"(col={col})"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one HID attack dataset against benchmark detectors."
    )

    parser.add_argument(
        "--attack-dir",
        type=Path,
        default=Path("data/attack/FeatureGuideHumanMimic"),
        help="Attack dataset directory. Default: FeatureGuideHumanMimic v4",
    )

    parser.add_argument(
        "--attack-name",
        type=str,
        default="FeatureGuideHumanMimic_v4",
        help="Attack dataset name used in reports.",
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=sorted(DEFAULT_DETECTORS.keys()),
        default=["d0", "d1"],
        help="Detector keys to evaluate. Default: d0 d1",
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("reports/benchmark/featureguide_v4"),
        help="Output root directory.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run oracle even if output CSVs already exist.",
    )

    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Do not run oracle; only parse existing outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, object]] = []

    for detector_key in args.detectors:
        detector = DEFAULT_DETECTORS[detector_key]
        detector_out_dir = args.out_root / f"oracle_{detector.name}_{args.attack_name}"

        if not args.skip_oracle:
            run_oracle(
                detector=detector,
                attack_dir=args.attack_dir,
                attack_name=args.attack_name,
                out_dir=detector_out_dir,
                force=args.force,
            )

        row = summarize_oracle_output(
            detector=detector,
            attack_name=args.attack_name,
            out_dir=detector_out_dir,
        )
        rows.append(row)

    summary_path = args.out_root / "benchmark_summary.csv"
    write_summary_csv(rows, summary_path)
    print_console_table(rows)

    print()
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
