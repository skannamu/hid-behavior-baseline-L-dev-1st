import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MUTABLE_FEATURES = [
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "overlap_ratio",
    "simultaneous_key_count",
    "correction_ratio",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
]

# 공격 의미/키 시퀀스 보존 쪽에 가까운 feature는 기본 freeze
FROZEN_FEATURES = [
    "key_code",
    "make_code",
    "key_category_id",
    "current_key_category_id",
    "previous_key_category_id",
    "shift_active",
    "ctrl_active",
    "alt_active",
    "meta_active",
    "modifier_count",
    "shortcut_flag",
    "repeat_flag",
    "device_change_flag",
    "new_device_flag",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate AdaptiveMimic v6 candidates against D1 oracle."
    )

    p.add_argument(
        "--seed-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("data/attack/AdaptiveMimic_v5_constrained_test"),
            Path("data/attack/FeatureGuideHumanMimic"),
        ],
        help="Attack seed directories. Default: V5_test + V4.",
    )

    p.add_argument(
        "--normal-path",
        type=Path,
        default=Path("data/processed/window_typing_10000.csv"),
    )

    p.add_argument(
        "--detector-model",
        type=Path,
        default=Path("checkpoints/recon_hid_v2_round1.pt"),
        help="D1 detector checkpoint.",
    )

    p.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("data/attack/AdaptiveMimic_v6_candidate"),
    )

    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/adaptive_mimic_v6_against_d1"),
    )

    p.add_argument(
        "--oracle-dir",
        type=Path,
        default=Path("reports/benchmark/adaptive_mimic_v6_candidate/oracle_d1_AdaptiveMimic_v6_candidate"),
    )

    p.add_argument("--candidates-per-seed", type=int, default=12)
    p.add_argument("--seed-limit", type=int, default=0, help="0 means use all seeds.")
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--random-seed", type=int, default=20260707)
    p.add_argument("--force", action="store_true")

    return p.parse_args()


def load_csv_dir(d: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(d.glob("*.csv")):
        df = pd.read_csv(p)
        df["__seed_dir"] = str(d)
        df["__seed_file"] = p.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {d}")

    return pd.concat(frames, ignore_index=True)


def step_feature_columns(columns, features):
    out = []
    feature_set = set(features)

    for c in columns:
        if not c.startswith("t"):
            continue

        # expected form: t0_hold_time
        parts = c.split("_", 1)
        if len(parts) != 2:
            continue

        step_token, feat = parts
        if not step_token[1:].isdigit():
            continue

        if feat in feature_set:
            out.append(c)

    return out


def get_feature_name(col: str) -> str:
    return col.split("_", 1)[1]


def safe_numeric_stats(normal: pd.DataFrame, cols):
    stats = {}

    for c in cols:
        vals = pd.to_numeric(normal[c], errors="coerce").dropna()
        if len(vals) == 0:
            stats[c] = {
                "mean": 0.0,
                "std": 1.0,
                "q01": 0.0,
                "q99": 1.0,
            }
            continue

        std = float(vals.std())
        if not np.isfinite(std) or std == 0:
            std = 1e-6

        stats[c] = {
            "mean": float(vals.mean()),
            "std": std,
            "q01": float(vals.quantile(0.01)),
            "q99": float(vals.quantile(0.99)),
        }

    return stats


def clip_by_feature(value, feature, st):
    if not np.isfinite(value):
        value = st["mean"]

    # normal 분포의 넓은 범위 안으로 제한
    lo = st["q01"]
    hi = st["q99"]

    if hi <= lo:
        lo = st["mean"] - 3 * st["std"]
        hi = st["mean"] + 3 * st["std"]

    value = float(np.clip(value, lo, hi))

    # feature-specific physical constraints
    if feature in [
        "hold_time",
        "flight_time",
        "press_to_press_time",
        "release_to_release_time",
        "keys_per_second",
        "burst_density",
        "timing_variance",
        "timing_entropy",
        "pause_duration",
    ]:
        value = max(value, 0.0)

    if feature in ["overlap_ratio", "correction_ratio"]:
        value = float(np.clip(value, 0.0, 1.0))

    if feature == "simultaneous_key_count":
        value = float(np.clip(value, 0.0, 10.0))

    return value


def mutate_row(seed_row, normal_ref_row, mutable_cols, normal_stats, rng, strategy, alpha):
    cand = seed_row.copy()

    for c in mutable_cols:
        feature = get_feature_name(c)
        st = normal_stats[c]

        try:
            seed_v = float(seed_row[c])
        except Exception:
            seed_v = st["mean"]

        try:
            normal_v = float(normal_ref_row[c])
        except Exception:
            normal_v = st["mean"]

        std = st["std"]

        if strategy == "blend_ref":
            v = (1.0 - alpha) * seed_v + alpha * normal_v
            v += rng.normal(0.0, 0.08 * std)

        elif strategy == "blend_mean":
            v = (1.0 - alpha) * seed_v + alpha * st["mean"]
            v += rng.normal(0.0, 0.10 * std)

        elif strategy == "normal_jitter":
            v = st["mean"] + rng.normal(0.0, 0.30 * std)

        elif strategy == "soft_seed_jitter":
            v = seed_v + rng.normal(0.0, 0.35 * std)

        elif strategy == "normal_quantile_probe":
            q = rng.uniform(0.10, 0.90)
            # q01~q99 안에서 랜덤 probe
            v = st["q01"] + q * (st["q99"] - st["q01"])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        cand[c] = clip_by_feature(v, feature, st)

    # metadata update
    if "label" in cand.index:
        cand["label"] = 1
    if "scenario" in cand.index:
        cand["scenario"] = "adaptive_mimic_v6"
    if "input_source" in cand.index:
        cand["input_source"] = "AdaptiveMimic_v6"
    if "session_id" in cand.index:
        cand["session_id"] = "adaptive_mimic_v6"

    return cand


def run_oracle(args):
    if args.force and args.oracle_dir.exists():
        shutil.rmtree(args.oracle_dir)

    args.oracle_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.detector_oracle_v2",
        "--model-path",
        str(args.detector_model),
        "--target-path",
        str(args.candidate_dir),
        "--target-group",
        "AdaptiveMimic_v6_candidate",
        "--out-dir",
        str(args.oracle_dir),
    ]

    print()
    print("[RUN D1 ORACLE]")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def label_col(df):
    if "true_label" in df.columns:
        return "true_label"
    if "label" in df.columns:
        return "label"
    raise KeyError("No true_label/label column found in oracle predictions.")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    args.result_dir.mkdir(parents=True, exist_ok=True)

    print("===== LOAD SEEDS =====")
    seed_frames = []
    for d in args.seed_dirs:
        df = load_csv_dir(d)
        print(f"{d}: {len(df)} rows")
        seed_frames.append(df)

    seeds = pd.concat(seed_frames, ignore_index=True)

    if args.seed_limit and args.seed_limit > 0 and len(seeds) > args.seed_limit:
        seeds = seeds.sample(n=args.seed_limit, random_state=args.random_seed).reset_index(drop=True)
        print(f"seed limited to {len(seeds)} rows")

    print("\n===== LOAD NORMAL =====")
    normal = pd.read_csv(args.normal_path)
    print(f"normal: {normal.shape}")

    mutable_cols = step_feature_columns(seeds.columns, MUTABLE_FEATURES)
    mutable_cols = [c for c in mutable_cols if c in normal.columns]

    if not mutable_cols:
        raise RuntimeError("No mutable timestep feature columns found.")

    print(f"\nmutable cols: {len(mutable_cols)}")
    print("first mutable cols:", mutable_cols[:20])

    normal_stats = safe_numeric_stats(normal, mutable_cols)

    strategies = [
        "blend_ref",
        "blend_mean",
        "normal_jitter",
        "soft_seed_jitter",
        "normal_quantile_probe",
    ]

    alphas = [0.25, 0.40, 0.55, 0.70, 0.85, 0.95]

    candidates = []
    metadata_rows = []

    normal_indices = rng.integers(0, len(normal), size=len(seeds) * args.candidates_per_seed)

    print("\n===== GENERATE CANDIDATES =====")
    k = 0
    for seed_idx, (_, seed_row) in enumerate(seeds.iterrows()):
        for local_i in range(args.candidates_per_seed):
            normal_ref = normal.iloc[int(normal_indices[k])]
            strategy = strategies[k % len(strategies)]
            alpha = float(rng.choice(alphas))

            cand = mutate_row(
                seed_row=seed_row,
                normal_ref_row=normal_ref,
                mutable_cols=mutable_cols,
                normal_stats=normal_stats,
                rng=rng,
                strategy=strategy,
                alpha=alpha,
            )

            # lineage metadata는 별도 파일에도 저장하고, 기존 컬럼이 있으면 업데이트
            if "source_file" in cand.index:
                cand["source_file"] = "v6_candidates.csv"
            if "row_index_in_source" in cand.index:
                cand["row_index_in_source"] = k
            if "window_id" in cand.index:
                cand["window_id"] = k

            candidates.append(cand)

            metadata_rows.append({
                "candidate_id": k,
                "seed_idx": seed_idx,
                "seed_dir": seed_row.get("__seed_dir", ""),
                "seed_file": seed_row.get("__seed_file", ""),
                "strategy": strategy,
                "alpha": alpha,
            })

            k += 1

    cand_df = pd.DataFrame(candidates)

    # 내부용 seed metadata columns 제거
    for c in ["__seed_dir", "__seed_file"]:
        if c in cand_df.columns:
            cand_df = cand_df.drop(columns=[c])

    args.candidate_dir.mkdir(parents=True, exist_ok=True)

    # target dir에는 oracle이 읽을 csv 하나만 둔다
    if args.force:
        for old in args.candidate_dir.glob("*.csv"):
            old.unlink()

    candidate_csv = args.candidate_dir / "v6_candidates.csv"
    cand_df.to_csv(candidate_csv, index=False)

    meta = pd.DataFrame(metadata_rows)
    meta_path = args.result_dir / "v6_candidate_metadata.csv"
    meta.to_csv(meta_path, index=False)

    print(f"generated candidates: {len(cand_df)}")
    print(f"saved candidates: {candidate_csv}")
    print(f"saved metadata  : {meta_path}")

    run_oracle(args)

    pred_path = args.oracle_dir / "oracle_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing oracle predictions: {pred_path}")

    preds = pd.read_csv(pred_path)
    lc = label_col(preds)
    attack_preds = preds[preds[lc] == 1].reset_index(drop=True)

    if len(attack_preds) != len(cand_df):
        print()
        print("[WARN] candidate count and attack prediction count differ")
        print("candidates:", len(cand_df))
        print("attack preds:", len(attack_preds))

    n = min(len(cand_df), len(attack_preds))
    cand_df = cand_df.iloc[:n].reset_index(drop=True)
    attack_preds = attack_preds.iloc[:n].reset_index(drop=True)
    meta = meta.iloc[:n].reset_index(drop=True)

    joined = pd.concat(
        [
            meta,
            attack_preds.add_prefix("d1_"),
        ],
        axis=1,
    )

    joined_path = args.result_dir / "v6_candidate_oracle_joined.csv"
    joined.to_csv(joined_path, index=False)

    print(f"saved joined oracle: {joined_path}")

    # OR95 / OR90 bypass selection
    selected_or95 = attack_preds[attack_preds["pred_or95"] == 0].index.tolist()
    selected_or90 = attack_preds[attack_preds["pred_or90"] == 0].index.tolist()

    print()
    print("===== V6 AGAINST D1 RESULT =====")
    print(f"total candidates       : {n}")
    print(f"OR95 bypass candidates : {len(selected_or95)}")
    print(f"OR90 bypass candidates : {len(selected_or90)}")

    summary_rows = []

    def save_selected(indices, name):
        out_dir = Path("data/attack") / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.force:
            for old in out_dir.glob("*.csv"):
                old.unlink()

        selected = cand_df.iloc[indices].copy()
        out_csv = out_dir / f"{name}.csv"
        selected.to_csv(out_csv, index=False)

        print(f"saved {name}: {out_csv} rows={len(selected)}")

        summary_rows.append({
            "selection": name,
            "rows": len(selected),
            "path": str(out_csv),
        })

    if selected_or95:
        save_selected(selected_or95, "AdaptiveMimic_v6_D1_OR95_bypass")

    if selected_or90:
        save_selected(selected_or90, "AdaptiveMimic_v6_D1_OR90_bypass")

    # 우회가 하나도 없으면 가장 hard한 후보 top-k 저장
    # 낮은 classifier/prototype probability + 낮은 recon error 쪽을 우선
    if not selected_or95:
        tmp = attack_preds.copy()

        recon = tmp["reconstruction_error"]
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-9)

        tmp["d1_attack_score"] = (
            0.45 * tmp["classifier_attack_probability"]
            + 0.45 * tmp["prototype_attack_probability"]
            + 0.10 * recon_norm
        )

        hard_idx = tmp.sort_values("d1_attack_score", ascending=True).head(args.top_k).index.tolist()
        save_selected(hard_idx, "AdaptiveMimic_v6_D1_hardest_no_bypass")

        hard_report = pd.concat(
            [
                meta.iloc[hard_idx].reset_index(drop=True),
                tmp.iloc[hard_idx].reset_index(drop=True),
            ],
            axis=1,
        )
        hard_report_path = args.result_dir / "v6_hardest_no_bypass_report.csv"
        hard_report.to_csv(hard_report_path, index=False)
        print(f"saved hard report: {hard_report_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = args.result_dir / "v6_generation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
