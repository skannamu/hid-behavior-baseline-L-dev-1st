#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import re
import statistics

import numpy as np


ROOT = Path(
    "/home/js/hid_behavior_ai/experiments/"
    "seminar_n4_full_20260820_103101"
)

OUT = ROOT / "aggregated_results"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Feature Schema v2 — ACTUAL window.csv column names
# ============================================================

SEQ_BASE = [
    "hold_time_s",
    "press_interval_s",
    "signed_flight_time_s",
    "overlap_fraction",
    "concurrent_keys_at_press",
    "release_inversion_flag",
    "correction_key_flag",
    "repeat_flag",
    "shift_at_press",
    "ctrl_at_press",
    "alt_at_press",
    "meta_at_press",
]

CTX_COLUMNS = [
    "ctx_press_rate_hz",
    "ctx_active_press_interval_median_s",
    "ctx_active_press_interval_robust_cv",
    "ctx_pause_rate",
    "ctx_pause_time_fraction",
    "ctx_max_pause_interval_s",
    "ctx_hold_time_robust_cv",
    "ctx_max_burst_length_keys",
    "ctx_correction_rate",
    "ctx_command_shortcut_rate",
    "ctx_overlap_key_rate",
    "ctx_release_inversion_rate",
]


# ============================================================
# Basic helpers
# ============================================================

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(x):
    try:
        if x is None or str(x).strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def prediction_bool(row):
    v = str(row.get("predicted_attack", "")).strip().lower()

    if v in {"1", "true", "attack", "positive", "anomaly"}:
        return True

    if v in {"0", "false", "normal", "negative", "benign"}:
        return False

    try:
        return float(v) > 0
    except Exception:
        return None


def fmt(x, n=4):
    if x is None:
        return "-"
    try:
        if not math.isfinite(float(x)):
            return "-"
    except Exception:
        return "-"
    return f"{float(x):.{n}f}"


def pct(x, n=2):
    if x is None:
        return "-"
    return f"{100*float(x):.{n}f}%"


# ============================================================
# window.csv direct loader
#
# key = (absolute source_file, window_id)
#
# Only requested window IDs are retained in memory.
# ============================================================

FEATURE_CACHE = {}
SOURCE_SCAN_COUNT = defaultdict(int)
MISSING_WINDOWS = []


def load_requested_windows(source_file, wanted_ids):
    source_file = str(source_file)
    path = Path(source_file)

    wanted_ids = {str(x) for x in wanted_ids}

    if not path.exists():
        for wid in wanted_ids:
            MISSING_WINDOWS.append(
                (source_file, wid, "source_missing")
            )
        return

    missing = {
        wid for wid in wanted_ids
        if (source_file, wid) not in FEATURE_CACHE
    }

    if not missing:
        return

    SOURCE_SCAN_COUNT[source_file] += 1

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            return

        required = (
            ["window_id"]
            + [
                f"t{t}_{name}"
                for t in range(50)
                for name in SEQ_BASE
            ]
            + CTX_COLUMNS
        )

        absent = [
            c for c in required
            if c not in reader.fieldnames
        ]

        if absent:
            raise RuntimeError(
                f"\nFeature Schema mismatch:\n"
                f"file={path}\n"
                f"missing first columns={absent[:20]}"
            )

        found = set()

        for row in reader:
            wid = str(row["window_id"])

            if wid not in missing:
                continue

            FEATURE_CACHE[(source_file, wid)] = row
            found.add(wid)

            if found == missing:
                break

    for wid in missing - found:
        MISSING_WINDOWS.append(
            (source_file, wid, "window_id_missing")
        )


def preload_prediction_rows(prediction_rows):
    grouped = defaultdict(set)

    for r in prediction_rows:
        source = r.get("source_file")
        wid = r.get("window_id")

        if source is None or wid is None:
            continue

        grouped[str(source)].add(str(wid))

    for source, ids in grouped.items():
        load_requested_windows(source, ids)


def get_feature_row(prediction_row):
    source = str(prediction_row.get("source_file", ""))
    wid = str(prediction_row.get("window_id", ""))

    return FEATURE_CACHE.get((source, wid))


# ============================================================
# Convert wide Feature Schema v2 row
# to interpretable descriptors
#
# Each sequence feature:
# mean/std/p10/median/p90/min/max
#
# Context:
# raw value
# ============================================================

def describe_feature_row(row):
    out = {}

    for feature in SEQ_BASE:
        values = np.asarray(
            [
                safe_float(row[f"t{t}_{feature}"])
                for t in range(50)
            ],
            dtype=float,
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        out[f"seq_{feature}_mean"] = float(
            np.mean(values)
        )

        out[f"seq_{feature}_std"] = float(
            np.std(values)
        )

        out[f"seq_{feature}_p10"] = float(
            np.percentile(values, 10)
        )

        out[f"seq_{feature}_median"] = float(
            np.median(values)
        )

        out[f"seq_{feature}_p90"] = float(
            np.percentile(values, 90)
        )

        out[f"seq_{feature}_min"] = float(
            np.min(values)
        )

        out[f"seq_{feature}_max"] = float(
            np.max(values)
        )

    for c in CTX_COLUMNS:
        out[c] = safe_float(row[c])

    return out


# ============================================================
# Effect-size analysis
#
# Descriptive only.
# No p-values because stride-1 windows are correlated.
# ============================================================

def effect_table(A, B, label_a, label_b):
    if not A or not B:
        return []

    features = sorted(
        set(A[0].keys()) & set(B[0].keys())
    )

    results = []

    for feature in features:
        a = np.asarray(
            [x.get(feature, np.nan) for x in A],
            dtype=float,
        )

        b = np.asarray(
            [x.get(feature, np.nan) for x in B],
            dtype=float,
        )

        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) < 2 or len(b) < 2:
            continue

        ma = float(np.mean(a))
        mb = float(np.mean(b))

        meda = float(np.median(a))
        medb = float(np.median(b))

        sda = float(np.std(a, ddof=1))
        sdb = float(np.std(b, ddof=1))

        denom = len(a) + len(b) - 2

        if denom > 0:
            pooled_var = (
                (len(a) - 1) * sda ** 2
                + (len(b) - 1) * sdb ** 2
            ) / denom
        else:
            pooled_var = 0.0

        pooled_sd = math.sqrt(
            max(pooled_var, 0.0)
        )

        d = (
            (ma - mb) / pooled_sd
            if pooled_sd > 1e-12
            else 0.0
        )

        q1a, q3a = np.percentile(a, [25, 75])
        q1b, q3b = np.percentile(b, [25, 75])

        pooled_iqr = (
            (q3a - q1a)
            + (q3b - q1b)
        ) / 2.0

        robust_shift = (
            (meda - medb) / pooled_iqr
            if abs(pooled_iqr) > 1e-12
            else 0.0
        )

        results.append({
            "feature": feature,

            f"{label_a}_n": len(a),
            f"{label_b}_n": len(b),

            f"{label_a}_mean": ma,
            f"{label_b}_mean": mb,

            f"{label_a}_median": meda,
            f"{label_b}_median": medb,

            "mean_difference": ma - mb,
            "median_difference": meda - medb,

            "cohens_d": d,
            "abs_cohens_d": abs(d),

            "robust_median_shift": robust_shift,
            "abs_robust_median_shift":
                abs(robust_shift),
        })

    results.sort(
        key=lambda x: x["abs_cohens_d"],
        reverse=True,
    )

    return results


def save_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


def print_effect(title, rows, top=25):
    print()
    print("=" * 125)
    print(title)
    print("=" * 125)

    if not rows:
        print("NO DATA")
        return

    print(
        f"{'Feature':52s}"
        f"{'A median':>14s}"
        f"{'B median':>14s}"
        f"{'Cohen d':>12s}"
        f"{'Robust':>12s}"
    )

    for r in rows[:top]:
        med_keys = [
            k for k in r.keys()
            if k.endswith("_median")
            and k != "median_difference"
        ]

        a_med = (
            r[med_keys[0]]
            if len(med_keys) > 0
            else None
        )

        b_med = (
            r[med_keys[1]]
            if len(med_keys) > 1
            else None
        )

        print(
            f"{r['feature']:52s}"
            f"{fmt(a_med):>14s}"
            f"{fmt(b_med):>14s}"
            f"{fmt(r['cohens_d'],3):>12s}"
            f"{fmt(r['robust_median_shift'],3):>12s}"
        )


# ============================================================
# Load 12 ReCon jobs
# ============================================================

summary_files = sorted(
    ROOT.glob(
        "test_p*__cal_p*/"
        "recon_hid/experiment_summary.json"
    )
)

if len(summary_files) != 12:
    raise RuntimeError(
        f"Expected 12 ReCon jobs, "
        f"found {len(summary_files)}"
    )


jobs = []

for sp in summary_files:
    m = re.search(
        r"test_(p\d+)__cal_(p\d+)",
        str(sp)
    )

    test_p, cal_p = m.groups()

    vdir = (
        sp.parent
        / "final_evaluation"
        / "V_final"
    )

    jobs.append({
        "split":
            f"test_{test_p}__cal_{cal_p}",

        "test":
            test_p,

        "cal":
            cal_p,

        "attack":
            vdir
            / "A_final_window_predictions.csv",

        "normal":
            vdir
            / "normal_test_window_predictions.csv",
    })


# ============================================================
# QUICK SOURCE AUDIT
# ============================================================

print()
print("=" * 125)
print("SOURCE FEATURE AUDIT")
print("=" * 125)

audit_rows = []

for job in jobs[:2]:
    attack = read_csv(job["attack"])
    normal = read_csv(job["normal"])

    if attack:
        audit_rows.append(attack[0])

    if normal:
        audit_rows.append(normal[0])


preload_prediction_rows(audit_rows)

for r in audit_rows:
    source = r["source_file"]
    wid = r["window_id"]

    feature_row = get_feature_row(r)

    print()
    print(f"source    : {source}")
    print(f"window_id : {wid}")
    print(
        "loaded    : "
        + ("YES" if feature_row else "NO")
    )

    if feature_row:
        print(
            "schema    : "
            f"{feature_row.get('feature_schema_version')}"
        )

        print(
            "scenario  : "
            f"{feature_row.get('scenario')}"
        )


# ============================================================
# ANALYSIS 1
# LONG-HOLD:
# bypass vs detected
# ============================================================

long_bypass = []
long_detected = []

long_by_split = defaultdict(
    lambda: {
        "bypass": [],
        "detected": [],
    }
)


print()
print("=" * 125)
print("LOADING LONG-HOLD FEATURES")
print("=" * 125)


for job in jobs:
    rows = read_csv(job["attack"])

    rows = [
        r for r in rows
        if r.get("scenario_or_family")
        == "long_hold_mimic"
    ]

    preload_prediction_rows(rows)

    loaded = 0

    for r in rows:
        feature_row = get_feature_row(r)

        if feature_row is None:
            continue

        loaded += 1

        desc = describe_feature_row(
            feature_row
        )

        pred = prediction_bool(r)

        if pred is True:
            long_detected.append(desc)
            long_by_split[
                job["split"]
            ]["detected"].append(desc)

        elif pred is False:
            long_bypass.append(desc)
            long_by_split[
                job["split"]
            ]["bypass"].append(desc)

    print(
        f"{job['split']:28s} "
        f"requested={len(rows):4d} "
        f"loaded={loaded:4d}"
    )


print()
print(
    f"TOTAL detected long-hold = "
    f"{len(long_detected)}"
)

print(
    f"TOTAL bypassed long-hold = "
    f"{len(long_bypass)}"
)


if not long_bypass or not long_detected:
    raise RuntimeError(
        "Long-hold feature loading failed: "
        "one comparison group is empty."
    )


long_effect = effect_table(
    long_bypass,
    long_detected,
    "bypass",
    "detected",
)


save_csv(
    OUT
    / "feature_long_hold_"
      "bypass_vs_detected.csv",
    long_effect,
)


print_effect(
    "FEATURE ANALYSIS 1 — "
    "LONG-HOLD BYPASS vs DETECTED "
    "(A=BYPASS, B=DETECTED)",
    long_effect,
    30,
)


# ============================================================
# Within-split long-hold comparisons
#
# Avoid defender-to-defender confounding.
# ============================================================

eligible = []

for split, groups in long_by_split.items():
    nb = len(groups["bypass"])
    nd = len(groups["detected"])

    if nb >= 20 and nd >= 20:
        eligible.append(
            (
                min(nb, nd),
                split,
                groups,
            )
        )


eligible.sort(reverse=True)


for i, (_, split, groups) in enumerate(
    eligible[:5],
    1,
):
    table = effect_table(
        groups["bypass"],
        groups["detected"],
        "bypass",
        "detected",
    )

    save_csv(
        OUT
        / (
            f"feature_long_hold_"
            f"{split}_within_split.csv"
        ),
        table,
    )

    print_effect(
        f"FEATURE ANALYSIS 1.{i} — "
        f"{split} WITHIN-SPLIT "
        f"(bypass={len(groups['bypass'])}, "
        f"detected={len(groups['detected'])})",
        table,
        20,
    )


# ============================================================
# ANALYSIS 2
# FREE WRITING:
# same physical window under 3 calibration choices
# ============================================================

physical = defaultdict(list)


print()
print("=" * 125)
print("COLLECTING FREE-WRITING PREDICTIONS")
print("=" * 125)


for job in jobs:
    rows = read_csv(job["normal"])

    rows = [
        r for r in rows
        if r.get("scenario_or_family")
        == "free_writing"
    ]

    for r in rows:
        key = (
            r.get("participant_id"),
            r.get("session_id"),
            str(r.get("window_id")),
        )

        physical[key].append({
            "cal": job["cal"],
            "pred": prediction_bool(r),
            "row": r,
        })


# Preload each physical normal window ONCE.

representative_rows = [
    observations[0]["row"]
    for observations in physical.values()
]


preload_prediction_rows(
    representative_rows
)


stable_normal = []
repeated_fp = []
cal_sensitive = []

stability_export = []

participant_counts = defaultdict(
    lambda: {
        "stable_normal": 0,
        "cal_sensitive": 0,
        "repeated_fp": 0,
    }
)


for key, observations in physical.items():
    participant, session, window_id = key

    preds = [
        x["pred"]
        for x in observations
        if x["pred"] is not None
    ]

    if not preds:
        continue

    n = len(preds)
    fp_count = sum(x is True for x in preds)

    feature_row = get_feature_row(
        observations[0]["row"]
    )

    if feature_row is None:
        continue

    desc = describe_feature_row(
        feature_row
    )

    if fp_count == 0:
        category = "stable_normal"
        stable_normal.append(desc)

    elif fp_count >= math.ceil(n / 2):
        category = "repeated_fp"
        repeated_fp.append(desc)

    else:
        category = "cal_sensitive"
        cal_sensitive.append(desc)

    participant_counts[
        participant
    ][category] += 1

    stability_export.append({
        "participant": participant,
        "session_id": session,
        "window_id": window_id,
        "evaluations": n,
        "false_positive_count": fp_count,
        "false_positive_fraction":
            fp_count / n,
        "category": category,
        "predictions": ";".join(
            f"{x['cal']}="
            f"{'FP' if x['pred'] else 'TN'}"
            for x in sorted(
                observations,
                key=lambda z: z["cal"],
            )
        ),
    })


if not stability_export:
    raise RuntimeError(
        "No Free-Writing physical windows loaded."
    )


save_csv(
    OUT
    / "feature_free_writing_"
      "window_stability.csv",
    stability_export,
)


print()
print("FREE-WRITING PHYSICAL WINDOW STABILITY")
print("-" * 80)

print(
    f"Stable normal        : "
    f"{len(stable_normal)}"
)

print(
    f"Calibration-sensitive: "
    f"{len(cal_sensitive)}"
)

print(
    f"Repeated FP          : "
    f"{len(repeated_fp)}"
)

print()

print(
    f"{'Participant':14s}"
    f"{'Stable':>12s}"
    f"{'Sensitive':>14s}"
    f"{'Repeated FP':>15s}"
)


for p in ["p003", "p004", "p005", "p006"]:
    d = participant_counts[p]

    print(
        f"{p:14s}"
        f"{d['stable_normal']:12d}"
        f"{d['cal_sensitive']:14d}"
        f"{d['repeated_fp']:15d}"
    )


# ============================================================
# Repeated FP vs stable normal
# ============================================================

if not repeated_fp:
    raise RuntimeError(
        "No repeated Free-Writing FP windows."
    )


free_effect = effect_table(
    repeated_fp,
    stable_normal,
    "repeated_fp",
    "stable_normal",
)


save_csv(
    OUT
    / "feature_free_writing_"
      "repeated_fp_vs_stable_normal.csv",
    free_effect,
)


print_effect(
    "FEATURE ANALYSIS 2 — "
    "FREE-WRITING REPEATED FP "
    "vs STABLE NORMAL "
    "(A=REPEATED FP, B=STABLE NORMAL)",
    free_effect,
    35,
)


# ============================================================
# ANALYSIS 3
# Calibration-sensitive vs stable normal
# ============================================================

if cal_sensitive:
    sensitive_effect = effect_table(
        cal_sensitive,
        stable_normal,
        "sensitive",
        "stable_normal",
    )

    save_csv(
        OUT
        / "feature_free_writing_"
          "cal_sensitive_vs_stable_normal.csv",
        sensitive_effect,
    )

    print_effect(
        "FEATURE ANALYSIS 3 — "
        "FREE-WRITING CALIBRATION-SENSITIVE "
        "vs STABLE NORMAL",
        sensitive_effect,
        30,
    )


# ============================================================
# ANALYSIS 4
# Repeated FP vs Calibration-sensitive
#
# Separates intrinsic abnormal-looking normal behavior
# from threshold sensitivity.
# ============================================================

if repeated_fp and cal_sensitive:
    fp_vs_sensitive = effect_table(
        repeated_fp,
        cal_sensitive,
        "repeated_fp",
        "sensitive",
    )

    save_csv(
        OUT
        / "feature_free_writing_"
          "repeated_fp_vs_sensitive.csv",
        fp_vs_sensitive,
    )

    print_effect(
        "FEATURE ANALYSIS 4 — "
        "FREE-WRITING REPEATED FP "
        "vs CALIBRATION-SENSITIVE",
        fp_vs_sensitive,
        30,
    )


# ============================================================
# ANALYSIS 5
# p004+p006 vs p003+p005
#
# Deduplicated physical Free-Writing windows.
# ============================================================

high_people = []
low_people = []


for key, observations in physical.items():
    participant = key[0]

    feature_row = get_feature_row(
        observations[0]["row"]
    )

    if feature_row is None:
        continue

    desc = describe_feature_row(
        feature_row
    )

    if participant in {"p004", "p006"}:
        high_people.append(desc)

    elif participant in {"p003", "p005"}:
        low_people.append(desc)


participant_effect = effect_table(
    high_people,
    low_people,
    "p004_p006",
    "p003_p005",
)


save_csv(
    OUT
    / "feature_free_writing_"
      "p004p006_vs_p003p005.csv",
    participant_effect,
)


print_effect(
    "FEATURE ANALYSIS 5 — "
    "FREE-WRITING PARTICIPANT SHIFT "
    "(A=p004+p006, B=p003+p005)",
    participant_effect,
    35,
)


# ============================================================
# Collapse descriptor ranking back to original base features
# ============================================================

def base_name(descriptor):
    if descriptor.startswith("ctx_"):
        return descriptor

    if descriptor.startswith("seq_"):
        x = descriptor[4:]

        for suffix in [
            "_mean",
            "_std",
            "_p10",
            "_median",
            "_p90",
            "_min",
            "_max",
        ]:
            if x.endswith(suffix):
                return x[:-len(suffix)]

        return x

    return descriptor


def base_ranking(rows, top_n=40):
    groups = defaultdict(list)

    for rank, r in enumerate(
        rows[:top_n],
        1,
    ):
        groups[
            base_name(r["feature"])
        ].append({
            "rank": rank,
            "d": r["abs_cohens_d"],
            "feature": r["feature"],
        })

    result = []

    for base, vals in groups.items():
        result.append({
            "base_feature": base,
            "top_descriptor_count": len(vals),
            "best_rank":
                min(v["rank"] for v in vals),
            "max_abs_cohens_d":
                max(v["d"] for v in vals),
            "descriptors":
                "; ".join(
                    v["feature"]
                    for v in vals
                ),
        })

    result.sort(
        key=lambda x: (
            x["best_rank"],
            -x["max_abs_cohens_d"],
        )
    )

    return result


long_rank = base_ranking(
    long_effect
)

free_rank = base_ranking(
    free_effect
)


save_csv(
    OUT
    / "feature_long_hold_"
      "base_feature_ranking.csv",
    long_rank,
)

save_csv(
    OUT
    / "feature_free_writing_"
      "base_feature_ranking.csv",
    free_rank,
)


print()
print("=" * 125)
print("BASE FEATURE RANKING — LONG HOLD")
print("=" * 125)

print(
    f"{'Feature':46s}"
    f"{'Best rank':>12s}"
    f"{'Top count':>12s}"
    f"{'Max |d|':>12s}"
)


for r in long_rank[:15]:
    print(
        f"{r['base_feature']:46s}"
        f"{r['best_rank']:12d}"
        f"{r['top_descriptor_count']:12d}"
        f"{fmt(r['max_abs_cohens_d'],3):>12s}"
    )


print()
print("=" * 125)
print("BASE FEATURE RANKING — FREE WRITING")
print("=" * 125)

print(
    f"{'Feature':46s}"
    f"{'Best rank':>12s}"
    f"{'Top count':>12s}"
    f"{'Max |d|':>12s}"
)


for r in free_rank[:15]:
    print(
        f"{r['base_feature']:46s}"
        f"{r['best_rank']:12d}"
        f"{r['top_descriptor_count']:12d}"
        f"{fmt(r['max_abs_cohens_d'],3):>12s}"
    )


# ============================================================
# Final audit
# ============================================================

print()
print("=" * 125)
print("FINAL SOURCE AUDIT")
print("=" * 125)

print(
    f"Cached feature windows : "
    f"{len(FEATURE_CACHE)}"
)

print(
    f"Source CSV files read  : "
    f"{len(SOURCE_SCAN_COUNT)}"
)

print(
    f"Missing windows        : "
    f"{len(MISSING_WINDOWS)}"
)


if MISSING_WINDOWS:
    print()
    print("First missing examples:")

    for x in MISSING_WINDOWS[:20]:
        print(x)


print()
print("=" * 125)
print("OUTPUT FILES")
print("=" * 125)

for p in sorted(
    OUT.glob("feature_*.csv")
):
    print(p)


print()
print("[PASS] feature-level diagnosis complete")
