import os
import glob
import shutil
import argparse
import random
import numpy as np
import pandas as pd

from src.dataset import SELECTED_FEATURES, WINDOW_SIZE
from src.detector_oracle_v2 import DetectorOracleV2


DEFAULT_SEED_DIR = "data/attack/FeatureGuideHumanMimic_FN_or95"
DEFAULT_NORMAL_DIR = "data/processed"

DEFAULT_WORK_DIR = "results/adaptive_mimic_v5_constrained"
DEFAULT_CANDIDATE_DIR = os.path.join(DEFAULT_WORK_DIR, "candidate_pool")
DEFAULT_SELECTED_DIR = "data/attack/AdaptiveMimic_v5_constrained"
DEFAULT_ORACLE_OUT_DIR = os.path.join(DEFAULT_WORK_DIR, "oracle_candidate_pool")

RANDOM_SEED = 42

CANDIDATES_PER_SEED = 8

# 기존보다 훨씬 약하게 normal 쪽으로 당김
ALPHA_MIN = 0.03
ALPHA_MAX = 0.25

NOISE_STD_FACTOR = 0.02

MAX_SELECTED = 1000


# 공격 구조 보존용: 거의 건드리지 않을 feature
FROZEN_FEATURES = {
    "current_key_category_id",
    "shortcut_flag",
    "modifier_count",
}

# 약하게만 바꿀 feature
WEAK_MUTATION_FEATURES = {
    "simultaneous_key_count",
    "correction_ratio",
    "overlap_ratio",
}

# 주로 조정할 timing / rhythm feature
TIMING_FEATURES = {
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
}

RATIO_FEATURES = {
    "overlap_ratio",
    "correction_ratio",
}

NON_NEGATIVE_FEATURES = {
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "overlap_ratio",
    "simultaneous_key_count",
    "modifier_count",
    "correction_ratio",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
    "current_key_category_id",
}

INTEGER_LIKE_FEATURES = {
    "simultaneous_key_count",
    "modifier_count",
    "shortcut_flag",
    "current_key_category_id",
}

BINARY_FEATURES = {
    "shortcut_flag",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def resolve_csv_files(path):
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return files

    raise FileNotFoundError(f"Path not found: {path}")


def load_csv_dir(path):
    rows = []

    for file in resolve_csv_files(path):
        df = pd.read_csv(file)
        df["seed_source_file"] = os.path.basename(file)
        df["seed_row_index"] = np.arange(len(df))
        rows.append(df)

    if not rows:
        raise ValueError(f"No data loaded from {path}")

    return pd.concat(rows, axis=0, ignore_index=True)


def load_normal_reference(normal_dir):
    normal_df = load_csv_dir(normal_dir)

    stats = {}

    for feature in SELECTED_FEATURES:
        for t in range(WINDOW_SIZE):
            col = f"t{t}_{feature}"

            if col not in normal_df.columns:
                continue

            values = pd.to_numeric(
                normal_df[col],
                errors="coerce",
            ).fillna(0.0)

            stats[col] = {
                "values": values.to_numpy(dtype=np.float32),
                "mean": float(values.mean()),
                "std": float(values.std() if values.std() > 1e-12 else 1e-6),
                "q01": float(values.quantile(0.01)),
                "q05": float(values.quantile(0.05)),
                "q95": float(values.quantile(0.95)),
                "q99": float(values.quantile(0.99)),
            }

    return stats


def sanitize_value(value, feature, q01, q99):
    if pd.isna(value):
        value = 0.0

    value = float(value)

    low = min(q01, q99)
    high = max(q01, q99)

    if abs(high - low) > 1e-12:
        value = float(np.clip(value, low, high))

    if feature in RATIO_FEATURES:
        value = float(np.clip(value, 0.0, 1.0))

    if feature in NON_NEGATIVE_FEATURES:
        value = max(0.0, value)

    if feature in INTEGER_LIKE_FEATURES:
        value = round(value)

    if feature in BINARY_FEATURES:
        value = 1 if value >= 0.5 else 0

    return value


def sample_normal_value(stat, rng):
    values = stat["values"]

    if len(values) == 0:
        return stat["mean"]

    idx = rng.integers(0, len(values))
    return float(values[idx])


def mutate_row_constrained(seed_row, normal_ref, rng, candidate_id):
    new_row = seed_row.copy()

    new_row["adaptive_candidate_id"] = candidate_id
    new_row["adaptive_generation"] = "AdaptiveMimic_v5_constrained"
    new_row["adaptive_seed_source_file"] = seed_row.get("seed_source_file", "")
    new_row["adaptive_seed_row_index"] = seed_row.get("seed_row_index", -1)

    alpha_global = rng.uniform(ALPHA_MIN, ALPHA_MAX)

    for feature in SELECTED_FEATURES:
        for t in range(WINDOW_SIZE):
            col = f"t{t}_{feature}"

            if col not in new_row.index:
                continue

            if col not in normal_ref:
                continue

            stat = normal_ref[col]

            seed_value = new_row[col]
            if pd.isna(seed_value):
                seed_value = 0.0
            seed_value = float(seed_value)

            # 1. category / shortcut / modifier는 공격 구조 보존을 위해 고정
            if feature in FROZEN_FEATURES:
                value = seed_value

            # 2. simultaneous / overlap / correction은 약하게만 조정
            elif feature in WEAK_MUTATION_FEATURES:
                alpha = np.clip(alpha_global * 0.35, 0.0, 0.12)
                normal_value = sample_normal_value(stat, rng)

                value = (
                    (1.0 - alpha) * seed_value
                    + alpha * normal_value
                    + rng.normal(0.0, stat["std"] * NOISE_STD_FACTOR)
                )

            # 3. timing / rhythm feature는 human-like하게 조정
            elif feature in TIMING_FEATURES:
                alpha = np.clip(
                    alpha_global + rng.normal(0.0, 0.03),
                    0.0,
                    0.35,
                )

                # 기존 v5처럼 normal mean만 쓰지 않고,
                # 실제 normal row의 값을 sample해서 분산을 살림
                normal_value = sample_normal_value(stat, rng)

                value = (
                    (1.0 - alpha) * seed_value
                    + alpha * normal_value
                    + rng.normal(0.0, stat["std"] * NOISE_STD_FACTOR)
                )

            else:
                value = seed_value

            value = sanitize_value(
                value=value,
                feature=feature,
                q01=stat["q01"],
                q99=stat["q99"],
            )

            new_row[col] = value

    return new_row


def generate_candidate_pool(seed_dir, normal_dir, candidate_dir):
    reset_dir(candidate_dir)

    rng = np.random.default_rng(RANDOM_SEED)

    seed_df = load_csv_dir(seed_dir)
    normal_ref = load_normal_reference(normal_dir)

    print("\n===== AdaptiveMimic v5 Constrained Candidate Generation =====")
    print(f"Seed samples          : {len(seed_df)}")
    print(f"Candidates per seed   : {CANDIDATES_PER_SEED}")
    print(f"Expected candidates   : {len(seed_df) * CANDIDATES_PER_SEED}")
    print(f"Candidate output dir  : {candidate_dir}")
    print("============================================================\n")

    rows = []
    candidate_id = 0

    for _, seed_row in seed_df.iterrows():
        for _ in range(CANDIDATES_PER_SEED):
            rows.append(
                mutate_row_constrained(
                    seed_row=seed_row,
                    normal_ref=normal_ref,
                    rng=rng,
                    candidate_id=candidate_id,
                )
            )
            candidate_id += 1

    candidate_df = pd.DataFrame(rows)

    for source_file, sub in candidate_df.groupby("seed_source_file"):
        if not isinstance(source_file, str) or source_file.strip() == "":
            source_file = "window_generated.csv"

        out_path = os.path.join(candidate_dir, source_file)
        sub.to_csv(out_path, index=False, encoding="utf-8")

    os.makedirs(DEFAULT_WORK_DIR, exist_ok=True)

    combined_path = os.path.join(
        DEFAULT_WORK_DIR,
        "adaptive_mimic_v5_constrained_candidate_pool_combined.csv",
    )

    candidate_df.to_csv(combined_path, index=False, encoding="utf-8")

    print(f"Saved combined candidate pool: {combined_path}")

    return candidate_df


def attach_row_index(pred_df):
    df = pred_df.copy()
    df["row_index_in_source"] = (
        df
        .groupby(["group", "source_file"])
        .cumcount()
    )
    return df


def save_selected_candidates(pred_df, candidate_dir, selected_dir):
    reset_dir(selected_dir)

    pred_df = attach_row_index(pred_df)

    normal = pred_df[pred_df["true_label"] == 0].copy()
    attack = pred_df[pred_df["true_label"] == 1].copy()

    normal_recon_q25 = normal["reconstruction_error"].quantile(0.25)
    normal_recon_q95 = normal["reconstruction_error"].quantile(0.95)

    normal_dist_q05 = normal["distance_to_normal_prototype"].quantile(0.05)
    normal_dist_q95 = normal["distance_to_normal_prototype"].quantile(0.95)

    # 핵심:
    # 1. OR95를 우회해야 함
    # 2. 너무 비정상적으로 낮은 recon error는 제외
    # 3. normal prototype distance도 normal 범위 안에 있어야 함
    selected = attack[
        (attack["pred_or95"] == 0)
        & (attack["reconstruction_error"] >= normal_recon_q25)
        & (attack["reconstruction_error"] <= normal_recon_q95)
        & (attack["distance_to_normal_prototype"] >= normal_dist_q05)
        & (attack["distance_to_normal_prototype"] <= normal_dist_q95)
        & (attack["classifier_attack_probability"] < 0.10)
        & (attack["prototype_attack_probability"] < 0.10)
    ].copy()

    # 너무 적게 뽑히면 조건 완화
    if len(selected) < 100:
        print("[WARN] Strict selected candidates are too few. Relaxing plausibility constraints.")

        selected = attack[
            (attack["pred_or95"] == 0)
            & (attack["reconstruction_error"] <= normal_recon_q95)
            & (attack["distance_to_normal_prototype"] <= normal_dist_q95)
            & (attack["classifier_attack_probability"] < 0.20)
            & (attack["prototype_attack_probability"] < 0.20)
        ].copy()

    selected["selection_score"] = (
        selected["reconstruction_error"]
        + selected["distance_to_normal_prototype"]
        + selected["classifier_attack_probability"]
        + selected["prototype_attack_probability"]
    )

    selected = selected.sort_values("selection_score", ascending=True)

    if len(selected) > MAX_SELECTED:
        selected = selected.head(MAX_SELECTED).copy()

    selected_meta_path = os.path.join(selected_dir, "_selected_metadata.csv")
    selected.to_csv(selected_meta_path, index=False, encoding="utf-8")

    selected_rows_all = []

    for source_file, sub in selected.groupby("source_file"):
        source_path = os.path.join(candidate_dir, source_file)

        if not os.path.exists(source_path):
            print(f"[WARN] candidate source not found: {source_path}")
            continue

        source_df = pd.read_csv(source_path)
        indices = sub["row_index_in_source"].astype(int).tolist()

        selected_source_df = source_df.iloc[indices].copy()

        selected_source_df.to_csv(
            os.path.join(selected_dir, source_file),
            index=False,
            encoding="utf-8",
        )

        selected_rows_all.append(selected_source_df)

    if selected_rows_all:
        combined = pd.concat(selected_rows_all, axis=0, ignore_index=True)
    else:
        combined = pd.DataFrame()

    combined.to_csv(
        os.path.join(selected_dir, "_combined.csv"),
        index=False,
        encoding="utf-8",
    )

    return selected, combined


def main():
    parser = argparse.ArgumentParser(
        description="Generate constrained feature-level AdaptiveMimic v5 candidates",
    )

    parser.add_argument("--seed-dir", default=DEFAULT_SEED_DIR)
    parser.add_argument("--normal-dir", default=DEFAULT_NORMAL_DIR)
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--selected-dir", default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--oracle-out-dir", default=DEFAULT_ORACLE_OUT_DIR)

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    generate_candidate_pool(
        seed_dir=args.seed_dir,
        normal_dir=args.normal_dir,
        candidate_dir=args.candidate_dir,
    )

    oracle = DetectorOracleV2()

    pred_df, summary_df, source_summary_df, thresholds = oracle.run_oracle(
        normal_path=args.normal_dir,
        target_path=args.candidate_dir,
        target_group="AdaptiveMimic_v5_constrained_candidate_pool",
        exclude_target_filenames=set(),
        out_dir=args.oracle_out_dir,
    )

    selected_meta, selected_rows = save_selected_candidates(
        pred_df=pred_df,
        candidate_dir=args.candidate_dir,
        selected_dir=args.selected_dir,
    )

    final_summary = pd.DataFrame(
        [
            {
                "dataset": "candidate_pool",
                "count": int((pred_df["true_label"] == 1).sum()),
            },
            {
                "dataset": "selected_constrained_or95_bypass",
                "count": len(selected_rows),
            },
        ]
    )

    os.makedirs(DEFAULT_WORK_DIR, exist_ok=True)

    final_summary_path = os.path.join(
        DEFAULT_WORK_DIR,
        "adaptive_mimic_v5_constrained_generation_summary.csv",
    )

    final_summary.to_csv(final_summary_path, index=False, encoding="utf-8")

    print("\n===== AdaptiveMimic v5 Constrained Generation Done =====")
    print(final_summary.to_string(index=False))
    print(f"\nCandidate dir     : {args.candidate_dir}")
    print(f"Selected dir      : {args.selected_dir}")
    print(f"Oracle out dir    : {args.oracle_out_dir}")
    print(f"Final summary     : {final_summary_path}")


if __name__ == "__main__":
    main()

