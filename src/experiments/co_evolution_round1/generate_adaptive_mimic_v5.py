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

DEFAULT_WORK_DIR = "results/adaptive_mimic_v5"
DEFAULT_CANDIDATE_DIR = os.path.join(DEFAULT_WORK_DIR, "candidate_pool")
DEFAULT_SELECTED_DIR = "data/attack/AdaptiveMimic_v5"
DEFAULT_SELECTED_OR90_DIR = "data/attack/AdaptiveMimic_v5_OR90_bypass"

DEFAULT_ORACLE_OUT_DIR = os.path.join(DEFAULT_WORK_DIR, "oracle_candidate_pool")

RANDOM_SEED = 42

# 너무 많이 만들면 느려지니까 처음에는 적당히 시작
CANDIDATES_PER_SEED = 6

# seed를 normal 쪽으로 얼마나 당길지
# 0이면 원본 seed 그대로, 1이면 normal 평균에 가까움
ALPHA_MIN = 0.10
ALPHA_MAX = 0.55

# normal std 기반 noise
NOISE_STD_FACTOR = 0.03

# 저장할 최대 개수
MAX_SELECTED_OR95 = 1000
MAX_SELECTED_OR90 = 1000


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


def compute_normal_column_stats(normal_dir):
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
                "mean": float(values.mean()),
                "std": float(values.std() if values.std() > 1e-12 else 1e-6),
                "q01": float(values.quantile(0.01)),
                "q99": float(values.quantile(0.99)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    return stats


def get_feature_name_from_col(col):
    # col format: t{idx}_{feature}
    # feature 이름 안에 underscore가 있으므로 첫 번째 "_" 뒤를 feature로 본다.
    if "_" not in col:
        return None
    return col.split("_", 1)[1]


def sanitize_value(value, feature, q01, q99):
    if pd.isna(value):
        value = 0.0

    value = float(value)

    # 정상 분포 극단값 안쪽으로 제한
    low = min(q01, q99)
    high = max(q01, q99)

    # q01 == q99이면 clip 범위가 너무 좁아질 수 있음
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


def mutate_row(seed_row, normal_stats, rng, candidate_id):
    """
    OR95 FN seed row를 normal distribution 쪽으로 당겨서 candidate 생성.
    실제 HID payload 생성이 아니라 feature-window level synthetic sample 생성.
    """
    new_row = seed_row.copy()

    new_row["adaptive_candidate_id"] = candidate_id
    new_row["adaptive_generation"] = "AdaptiveMimic_v5"
    new_row["adaptive_seed_source_file"] = seed_row.get("seed_source_file", "")
    new_row["adaptive_seed_row_index"] = seed_row.get("seed_row_index", -1)

    # candidate별로 전체적인 normal pull 강도를 하나 정함
    alpha_global = rng.uniform(ALPHA_MIN, ALPHA_MAX)

    for feature in SELECTED_FEATURES:
        for t in range(WINDOW_SIZE):
            col = f"t{t}_{feature}"

            if col not in new_row.index:
                continue

            if col not in normal_stats:
                continue

            stat = normal_stats[col]

            seed_value = new_row[col]
            if pd.isna(seed_value):
                seed_value = 0.0

            seed_value = float(seed_value)

            # timestep별로 약간 다르게 당김
            alpha = np.clip(
                alpha_global + rng.normal(0.0, 0.05),
                0.0,
                1.0,
            )

            normal_mean = stat["mean"]
            normal_std = stat["std"]

            # 카테고리성/정수성 feature는 너무 연속적으로 섞지 않기
            if feature in INTEGER_LIKE_FEATURES:
                if rng.random() < alpha:
                    value = normal_mean
                else:
                    value = seed_value
                value += rng.normal(0.0, normal_std * NOISE_STD_FACTOR)
            else:
                value = (
                    (1.0 - alpha) * seed_value
                    + alpha * normal_mean
                    + rng.normal(0.0, normal_std * NOISE_STD_FACTOR)
                )

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
    normal_stats = compute_normal_column_stats(normal_dir)

    print("\n===== AdaptiveMimic v5 Candidate Generation =====")
    print(f"Seed samples          : {len(seed_df)}")
    print(f"Candidates per seed   : {CANDIDATES_PER_SEED}")
    print(f"Expected candidates   : {len(seed_df) * CANDIDATES_PER_SEED}")
    print(f"Candidate output dir  : {candidate_dir}")

    candidate_rows = []
    candidate_id = 0

    for _, seed_row in seed_df.iterrows():
        for _ in range(CANDIDATES_PER_SEED):
            new_row = mutate_row(
                seed_row=seed_row,
                normal_stats=normal_stats,
                rng=rng,
                candidate_id=candidate_id,
            )
            candidate_rows.append(new_row)
            candidate_id += 1

    candidate_df = pd.DataFrame(candidate_rows)

    # source별로 다시 나눠 저장
    # 원래 seed_source_file 기준으로 window1.csv, window2.csv ... 형태 유지
    for source_file, sub in candidate_df.groupby("seed_source_file"):
        if not isinstance(source_file, str) or source_file.strip() == "":
            source_file = "window_generated.csv"

        out_path = os.path.join(candidate_dir, source_file)
        sub.to_csv(out_path, index=False, encoding="utf-8")

    combined_path = os.path.join(DEFAULT_WORK_DIR, "adaptive_mimic_v5_candidate_pool_combined.csv")
    os.makedirs(DEFAULT_WORK_DIR, exist_ok=True)
    candidate_df.to_csv(combined_path, index=False, encoding="utf-8")

    print(f"Saved combined candidate pool: {combined_path}")
    print("===============================================\n")

    return candidate_df


def save_selected_candidates(pred_df, candidate_dir, selected_dir, pred_col, max_selected):
    reset_dir(selected_dir)

    selected = pred_df[
        (pred_df["true_label"] == 1)
        & (pred_df[pred_col] == 0)
    ].copy()

    # detector score가 낮은 순서로 정렬
    selected["adaptive_score"] = (
        selected["reconstruction_error"]
        + selected["classifier_attack_probability"]
        + selected["prototype_attack_probability"]
        + selected["distance_to_normal_prototype"]
    )

    selected = selected.sort_values("adaptive_score", ascending=True)

    if len(selected) > max_selected:
        selected = selected.head(max_selected).copy()

    selected_meta_path = os.path.join(selected_dir, "_selected_metadata.csv")
    selected.to_csv(selected_meta_path, index=False, encoding="utf-8")

    # oracle prediction에는 원본 candidate row index가 없으므로 source_file별 순서로 복구
    selected["row_index_in_source"] = (
        selected
        .groupby("source_file")
        .cumcount()
    )

    # 문제:
    # selected 내부 cumcount는 selected subset 기준이라 원본 candidate row index와 다를 수 있음.
    # 따라서 source_file별 전체 prediction에도 row_index를 먼저 붙여서 merge해야 함.
    pred_with_idx = pred_df.copy()
    pred_with_idx["row_index_in_source"] = (
        pred_with_idx
        .groupby(["group", "source_file"])
        .cumcount()
    )

    selected_keys = pred_with_idx[
        (pred_with_idx["true_label"] == 1)
        & (pred_with_idx[pred_col] == 0)
    ].copy()

    selected_keys["adaptive_score"] = (
        selected_keys["reconstruction_error"]
        + selected_keys["classifier_attack_probability"]
        + selected_keys["prototype_attack_probability"]
        + selected_keys["distance_to_normal_prototype"]
    )

    selected_keys = selected_keys.sort_values("adaptive_score", ascending=True)

    if len(selected_keys) > max_selected:
        selected_keys = selected_keys.head(max_selected).copy()

    selected_keys.to_csv(selected_meta_path, index=False, encoding="utf-8")

    selected_rows_all = []

    for source_file, sub in selected_keys.groupby("source_file"):
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

    return selected_keys, combined


def main():
    parser = argparse.ArgumentParser(
        description="Generate feature-level AdaptiveMimic v5 candidates using ReCon-HID v2 oracle",
    )

    parser.add_argument(
        "--seed-dir",
        default=DEFAULT_SEED_DIR,
        help="False negative seed directory",
    )

    parser.add_argument(
        "--normal-dir",
        default=DEFAULT_NORMAL_DIR,
        help="Normal dataset directory",
    )

    parser.add_argument(
        "--candidate-dir",
        default=DEFAULT_CANDIDATE_DIR,
        help="Candidate pool output directory",
    )

    parser.add_argument(
        "--selected-dir",
        default=DEFAULT_SELECTED_DIR,
        help="Selected OR95-bypass AdaptiveMimic v5 output directory",
    )

    parser.add_argument(
        "--selected-or90-dir",
        default=DEFAULT_SELECTED_OR90_DIR,
        help="Selected OR90-bypass AdaptiveMimic v5 output directory",
    )

    parser.add_argument(
        "--oracle-out-dir",
        default=DEFAULT_ORACLE_OUT_DIR,
        help="Oracle output directory for candidate pool",
    )

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # 1. 후보 생성
    generate_candidate_pool(
        seed_dir=args.seed_dir,
        normal_dir=args.normal_dir,
        candidate_dir=args.candidate_dir,
    )

    # 2. Detector Oracle로 후보 평가
    oracle = DetectorOracleV2()

    pred_df, summary_df, source_summary_df, thresholds = oracle.run_oracle(
        normal_path=args.normal_dir,
        target_path=args.candidate_dir,
        target_group="AdaptiveMimic_v5_candidate_pool",
        exclude_target_filenames=set(),
        out_dir=args.oracle_out_dir,
    )

    # 3. OR95 우회 후보 저장
    selected_or95_meta, selected_or95_rows = save_selected_candidates(
        pred_df=pred_df,
        candidate_dir=args.candidate_dir,
        selected_dir=args.selected_dir,
        pred_col="pred_or95",
        max_selected=MAX_SELECTED_OR95,
    )

    # 4. OR90 우회 후보 저장
    selected_or90_meta, selected_or90_rows = save_selected_candidates(
        pred_df=pred_df,
        candidate_dir=args.candidate_dir,
        selected_dir=args.selected_or90_dir,
        pred_col="pred_or90",
        max_selected=MAX_SELECTED_OR90,
    )

    # 5. 최종 요약 저장
    final_summary = pd.DataFrame(
        [
            {
                "dataset": "candidate_pool",
                "count": int((pred_df["true_label"] == 1).sum()),
            },
            {
                "dataset": "selected_or95_bypass",
                "count": len(selected_or95_rows),
            },
            {
                "dataset": "selected_or90_bypass",
                "count": len(selected_or90_rows),
            },
        ]
    )

    os.makedirs(DEFAULT_WORK_DIR, exist_ok=True)

    final_summary_path = os.path.join(
        DEFAULT_WORK_DIR,
        "adaptive_mimic_v5_generation_summary.csv",
    )

    final_summary.to_csv(
        final_summary_path,
        index=False,
        encoding="utf-8",
    )

    print("\n===== AdaptiveMimic v5 Generation Done =====")
    print(final_summary.to_string(index=False))
    print(f"\nCandidate dir        : {args.candidate_dir}")
    print(f"Selected OR95 dir    : {args.selected_dir}")
    print(f"Selected OR90 dir    : {args.selected_or90_dir}")
    print(f"Oracle out dir       : {args.oracle_out_dir}")
    print(f"Final summary        : {final_summary_path}")


if __name__ == "__main__":
    main()
