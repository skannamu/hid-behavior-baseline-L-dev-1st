#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import json
import re
import statistics
import math

ROOT = Path(
    "/home/js/hid_behavior_ai/experiments/"
    "seminar_n4_full_20260820_103101"
)

OUT = ROOT / "aggregated_results"
OUT.mkdir(parents=True, exist_ok=True)

METHOD = "recon_hid"


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.mean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def percentile(xs, p):
    xs = sorted(
        x for x in xs
        if x is not None and math.isfinite(x)
    )
    if not xs:
        return None

    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    frac = pos - lo
    return xs[lo] * (1-frac) + xs[hi] * frac


def pct(x, d=3):
    if x is None:
        return "-"
    return f"{100*x:.{d}f}%"


def num(x, d=4):
    if x is None:
        return "-"
    return f"{x:.{d}f}"


def find_col(columns, exact=(), contains=()):
    """
    Find an existing column:
    exact names first, then substring search.
    """
    lower_map = {c.lower(): c for c in columns}

    for name in exact:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for c in columns:
        lc = c.lower()
        for term in contains:
            if term.lower() in lc:
                return c

    return None


def get_prediction_bool(row, col):
    if col is None:
        return None

    v = str(row.get(col, "")).strip().lower()

    if v in {"1", "true", "attack", "positive", "anomaly"}:
        return True

    if v in {"0", "false", "normal", "negative", "benign"}:
        return False

    try:
        return float(v) > 0
    except Exception:
        return None


def score_summary(values):
    values = [
        x for x in values
        if x is not None and math.isfinite(x)
    ]

    if not values:
        return {
            "n": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }

    return {
        "n": len(values),
        "min": min(values),
        "p10": percentile(values, .10),
        "p25": percentile(values, .25),
        "median": percentile(values, .50),
        "p75": percentile(values, .75),
        "p90": percentile(values, .90),
        "max": max(values),
        "mean": mean(values),
    }


def print_score_summary(label, s, threshold=None):
    print(f"  {label}")
    print(
        f"    n={s['n']} | "
        f"min={num(s['min'])} | "
        f"p10={num(s['p10'])} | "
        f"p25={num(s['p25'])} | "
        f"median={num(s['median'])} | "
        f"p75={num(s['p75'])} | "
        f"p90={num(s['p90'])} | "
        f"max={num(s['max'])} | "
        f"mean={num(s['mean'])}"
    )

    if threshold is not None and s["median"] is not None:
        print(
            f"    median / threshold = "
            f"{num(s['median'] / threshold, 3)}x"
            if threshold != 0 else
            "    median / threshold = undefined"
        )


# ============================================================
# Load 12 ReCon jobs
# ============================================================

summary_files = sorted(
    ROOT.glob(
        "test_p*__cal_p*/recon_hid/"
        "experiment_summary.json"
    )
)

if len(summary_files) != 12:
    raise RuntimeError(
        f"Expected 12 ReCon jobs, found {len(summary_files)}"
    )


jobs = []

for sp in summary_files:
    s = load_json(sp)

    m = re.search(
        r"test_(p\d+)__cal_(p\d+)",
        str(sp)
    )

    if not m:
        raise RuntimeError(f"Cannot parse {sp}")

    test_p, cal_p = m.groups()

    vdir = sp.parent / "final_evaluation" / "V_final"

    final_metrics_path = vdir / "final_metrics.json"
    family_path = vdir / "A_final_family_metrics.csv"
    candidate_path = vdir / "A_final_candidate_metrics.csv"
    attack_window_path = vdir / "A_final_window_predictions.csv"
    normal_session_path = vdir / "normal_test_session_metrics.csv"
    normal_window_path = vdir / "normal_test_window_predictions.csv"

    fm = load_json(final_metrics_path)

    defender_id = fm.get(
        "defender", {}
    ).get("defender_id")

    threshold = fm.get(
        "defender", {}
    ).get("decision_threshold")

    jobs.append({
        "split": f"test_{test_p}__cal_{cal_p}",
        "test": test_p,
        "cal": cal_p,
        "dir": sp.parent,
        "defender_id": defender_id,
        "threshold": threshold,
        "family_path": family_path,
        "candidate_path": candidate_path,
        "attack_window_path": attack_window_path,
        "normal_session_path": normal_session_path,
        "normal_window_path": normal_window_path,
    })


# ============================================================
# PART A
# Long-hold failure location
# ============================================================

long_hold_rows = []

for job in jobs:
    rows = load_csv(job["family_path"])

    target = [
        r for r in rows
        if r["scenario_or_family"] == "long_hold_mimic"
    ]

    if len(target) != 1:
        raise RuntimeError(
            f"Unexpected long_hold rows in {job['family_path']}"
        )

    r = target[0]

    long_hold_rows.append({
        "split": job["split"],
        "test": job["test"],
        "cal": job["cal"],
        "defender": job["defender_id"],
        "threshold": job["threshold"],
        "windows": int(r["window_count"]),
        "tpr": safe_float(r["tpr"]),
        "bypass": safe_float(r["bypass_rate"]),
        "mean_decision_score":
            safe_float(r["mean_decision_score"]),
        "max_decision_score":
            safe_float(r["max_decision_score"]),
        "job": job,
    })


long_hold_rows.sort(
    key=lambda x: x["bypass"],
    reverse=True,
)


print()
print("=" * 120)
print("A. LONG_HOLD_MIMIC — ALL 12 RECON SPLITS")
print("=" * 120)

print(
    f"{'Split':28s}"
    f"{'Dfinal':>10s}"
    f"{'Threshold':>13s}"
    f"{'Windows':>10s}"
    f"{'TPR':>11s}"
    f"{'Bypass':>11s}"
    f"{'Mean score':>15s}"
)

for r in long_hold_rows:
    print(
        f"{r['split']:28s}"
        f"{str(r['defender']):>10s}"
        f"{num(r['threshold'], 4):>13s}"
        f"{r['windows']:10d}"
        f"{pct(r['tpr']):>11s}"
        f"{pct(r['bypass']):>11s}"
        f"{num(r['mean_decision_score'], 4):>15s}"
    )


with open(
    OUT / "diagnosis_long_hold_all_splits.csv",
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:
    export = [
        {k: v for k, v in r.items() if k != "job"}
        for r in long_hold_rows
    ]

    w = csv.DictWriter(
        f,
        fieldnames=list(export[0].keys())
    )
    w.writeheader()
    w.writerows(export)


# ============================================================
# Deep inspect worst long-hold splits
# ============================================================

worst_long_hold = [
    r for r in long_hold_rows
    if r["bypass"] > 0
][:5]


print()
print("=" * 120)
print("B. WORST LONG_HOLD SPLITS — WINDOW/CANDIDATE DIAGNOSIS")
print("=" * 120)


for rank, info in enumerate(worst_long_hold, 1):

    job = info["job"]
    threshold = info["threshold"]

    print()
    print("-" * 120)
    print(
        f"[LONG-HOLD #{rank}] {job['split']} "
        f"| {job['defender_id']} "
        f"| threshold={num(threshold)} "
        f"| bypass={pct(info['bypass'])}"
    )
    print("-" * 120)

    # Candidate level
    candidate_rows = load_csv(job["candidate_path"])

    if candidate_rows:
        cols = list(candidate_rows[0].keys())

        fam_col = find_col(
            cols,
            exact=[
                "scenario_or_family",
                "family",
                "attack_family",
            ],
            contains=["family"],
        )

        cid_col = find_col(
            cols,
            exact=[
                "candidate_id",
                "session_id",
                "attack_id",
            ],
            contains=["candidate"],
        )

        bypass_col = find_col(
            cols,
            exact=["bypass_rate"],
            contains=["bypass"],
        )

        tpr_col = find_col(
            cols,
            exact=["tpr", "window_tpr"],
            contains=["tpr"],
        )

        score_col = find_col(
            cols,
            exact=["mean_decision_score"],
            contains=["mean_decision"],
        )

        long_candidates = (
            [
                r for r in candidate_rows
                if r.get(fam_col) == "long_hold_mimic"
            ]
            if fam_col
            else []
        )

        if long_candidates:
            long_candidates.sort(
                key=lambda r: (
                    safe_float(r.get(bypass_col))
                    if bypass_col else 0
                ) or 0,
                reverse=True,
            )

            print("  Worst long-hold candidates:")

            for r in long_candidates[:10]:
                cid = r.get(cid_col, "?") if cid_col else "?"
                bp = (
                    safe_float(r.get(bypass_col))
                    if bypass_col else None
                )
                tp = (
                    safe_float(r.get(tpr_col))
                    if tpr_col else None
                )
                sc = (
                    safe_float(r.get(score_col))
                    if score_col else None
                )

                print(
                    f"    candidate={cid} "
                    f"| TPR={pct(tp)} "
                    f"| bypass={pct(bp)} "
                    f"| mean_score={num(sc)}"
                )
        else:
            print(
                "  Candidate CSV family column not found "
                "or no long-hold candidates."
            )
            print(f"  Columns: {cols}")

    # Window level
    attack_rows = load_csv(job["attack_window_path"])

    if attack_rows:
        cols = list(attack_rows[0].keys())

        fam_col = find_col(
            cols,
            exact=[
                "scenario_or_family",
                "family",
                "attack_family",
            ],
            contains=["family"],
        )

        score_col = find_col(
            cols,
            exact=[
                "decision_score",
                "final_decision_score",
                "score",
            ],
            contains=["decision_score"],
        )

        pred_col = find_col(
            cols,
            exact=[
                "predicted_attack",
                "prediction",
                "predicted_label",
                "is_attack",
            ],
            contains=["predicted_attack"],
        )

        if fam_col:
            lr = [
                r for r in attack_rows
                if r.get(fam_col) == "long_hold_mimic"
            ]
        else:
            lr = []

        print(f"  Attack prediction columns: {cols}")

        if lr and score_col:

            all_scores = [
                safe_float(r.get(score_col))
                for r in lr
            ]

            detected_scores = []
            bypass_scores = []

            for r in lr:
                score = safe_float(r.get(score_col))

                pred = get_prediction_bool(
                    r,
                    pred_col
                )

                # Fallback if prediction field absent:
                # decision score > threshold = attack.
                if pred is None and score is not None:
                    pred = score > threshold

                if pred is True:
                    detected_scores.append(score)
                elif pred is False:
                    bypass_scores.append(score)

            print_score_summary(
                "All long-hold windows",
                score_summary(all_scores),
                threshold,
            )

            print_score_summary(
                "Detected long-hold windows",
                score_summary(detected_scores),
                threshold,
            )

            print_score_summary(
                "BYPASSED long-hold windows",
                score_summary(bypass_scores),
                threshold,
            )

            print(
                f"  threshold={num(threshold)} | "
                f"detected={len(detected_scores)} | "
                f"bypassed={len(bypass_scores)}"
            )

            # Try component columns
            component_cols = []

            for c in cols:
                lc = c.lower()

                if any(
                    key in lc
                    for key in [
                        "reconstruction",
                        "prototype",
                        "classifier",
                        "logit",
                        "probability",
                    ]
                ):
                    component_cols.append(c)

            if component_cols:
                print("  Decision component diagnostics:")

                for c in component_cols:
                    detect_vals = []
                    bypass_vals = []

                    for r in lr:
                        v = safe_float(r.get(c))
                        if v is None:
                            continue

                        score = safe_float(r.get(score_col))
                        pred = get_prediction_bool(r, pred_col)

                        if pred is None and score is not None:
                            pred = score > threshold

                        if pred is True:
                            detect_vals.append(v)
                        elif pred is False:
                            bypass_vals.append(v)

                    print(
                        f"    {c:35s} | "
                        f"detected mean={num(mean(detect_vals))} | "
                        f"bypass mean={num(mean(bypass_vals))}"
                    )

            else:
                print(
                    "  No reconstruction/prototype/classifier "
                    "component columns found in prediction CSV."
                )

        else:
            print(
                "  Could not automatically filter long-hold "
                "windows or find decision score."
            )


# ============================================================
# PART C
# Free-writing behavior across splits
# ============================================================

free_rows = []

for job in jobs:

    sessions = load_csv(job["normal_session_path"])

    free = [
        r for r in sessions
        if r["scenario_or_family"] == "free_writing"
    ]

    total_windows = sum(
        int(r["window_count"])
        for r in free
    )

    total_fp = sum(
        int(r["predicted_attack_count"])
        for r in free
    )

    weighted_fpr = (
        total_fp / total_windows
        if total_windows else None
    )

    session_macro = mean(
        [safe_float(r["fpr"]) for r in free]
    )

    worst_session = max(
        [safe_float(r["fpr"]) for r in free],
        default=None,
    )

    free_rows.append({
        "split": job["split"],
        "test": job["test"],
        "cal": job["cal"],
        "defender": job["defender_id"],
        "threshold": job["threshold"],
        "free_windows": total_windows,
        "free_fp": total_fp,
        "free_fpr": weighted_fpr,
        "free_session_macro_fpr": session_macro,
        "free_worst_session_fpr": worst_session,
        "job": job,
    })


free_rows.sort(
    key=lambda x: x["free_fpr"],
    reverse=True,
)


print()
print("=" * 120)
print("C. FREE WRITING — CALIBRATION DEPENDENCY")
print("=" * 120)

print(
    f"{'Split':28s}"
    f"{'Dfinal':>9s}"
    f"{'Threshold':>13s}"
    f"{'Free FPR':>12s}"
    f"{'Session macro':>16s}"
    f"{'Worst session':>16s}"
)

for r in free_rows:
    print(
        f"{r['split']:28s}"
        f"{str(r['defender']):>9s}"
        f"{num(r['threshold'], 4):>13s}"
        f"{pct(r['free_fpr']):>12s}"
        f"{pct(r['free_session_macro_fpr']):>16s}"
        f"{pct(r['free_worst_session_fpr']):>16s}"
    )


print()
print("Per Test participant:")
print()

for p in ["p003", "p004", "p005", "p006"]:

    rr = [
        r for r in free_rows
        if r["test"] == p
    ]

    print(f"  Test={p}")

    for r in sorted(rr, key=lambda x: x["cal"]):
        print(
            f"    Cal={r['cal']} "
            f"| threshold={num(r['threshold'])} "
            f"| free FPR={pct(r['free_fpr'])}"
        )


with open(
    OUT / "diagnosis_free_writing_all_splits.csv",
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    export = [
        {k: v for k, v in r.items() if k != "job"}
        for r in free_rows
    ]

    w = csv.DictWriter(
        f,
        fieldnames=list(export[0].keys())
    )
    w.writeheader()
    w.writerows(export)


# ============================================================
# PART D
# Deep inspect worst free-writing jobs
# ============================================================

worst_free_jobs = free_rows[:6]


print()
print("=" * 120)
print("D. WORST FREE-WRITING JOBS — SCORE DISTRIBUTION")
print("=" * 120)


for rank, info in enumerate(worst_free_jobs, 1):

    job = info["job"]
    threshold = info["threshold"]

    print()
    print("-" * 120)
    print(
        f"[FREE #{rank}] {job['split']} "
        f"| {job['defender_id']} "
        f"| threshold={num(threshold)} "
        f"| Free FPR={pct(info['free_fpr'])}"
    )
    print("-" * 120)

    rows = load_csv(job["normal_window_path"])

    if not rows:
        print("  No rows.")
        continue

    cols = list(rows[0].keys())

    scenario_col = find_col(
        cols,
        exact=[
            "scenario_or_family",
            "scenario",
        ],
        contains=["scenario"],
    )

    session_col = find_col(
        cols,
        exact=["session_id"],
        contains=["session"],
    )

    score_col = find_col(
        cols,
        exact=[
            "decision_score",
            "final_decision_score",
            "score",
        ],
        contains=["decision_score"],
    )

    pred_col = find_col(
        cols,
        exact=[
            "predicted_attack",
            "prediction",
            "predicted_label",
            "is_attack",
        ],
        contains=["predicted_attack"],
    )

    print(f"  Normal prediction columns: {cols}")

    if scenario_col is None or score_col is None:
        print(
            "  Required scenario/decision score columns not found."
        )
        continue

    # Scenario summaries
    for scenario in [
        "coding_controlled",
        "english_typing",
        "free_writing",
        "korean_typing",
    ]:

        sr = [
            r for r in rows
            if r.get(scenario_col) == scenario
        ]

        if not sr:
            continue

        scores = []
        fp_scores = []
        normal_scores = []

        fp = 0

        for r in sr:
            score = safe_float(r.get(score_col))
            scores.append(score)

            pred = get_prediction_bool(
                r,
                pred_col
            )

            if pred is None and score is not None:
                pred = score > threshold

            if pred is True:
                fp += 1
                fp_scores.append(score)
            elif pred is False:
                normal_scores.append(score)

        print()
        print(
            f"  [{scenario}] "
            f"windows={len(sr)} "
            f"| FP={fp} "
            f"| FPR={pct(fp / len(sr))}"
        )

        print_score_summary(
            "All normal scores",
            score_summary(scores),
            threshold,
        )

        print_score_summary(
            "False-positive scores",
            score_summary(fp_scores),
            threshold,
        )

        print_score_summary(
            "Correct-normal scores",
            score_summary(normal_scores),
            threshold,
        )

    # Component analysis for FREE writing specifically
    free = [
        r for r in rows
        if r.get(scenario_col) == "free_writing"
    ]

    component_cols = []

    for c in cols:
        lc = c.lower()

        if any(
            key in lc
            for key in [
                "reconstruction",
                "prototype",
                "classifier",
                "logit",
                "probability",
            ]
        ):
            component_cols.append(c)

    if component_cols and free:
        print()
        print("  Free-writing decision component analysis:")

        for c in component_cols:

            fp_vals = []
            tn_vals = []

            for r in free:
                value = safe_float(r.get(c))

                if value is None:
                    continue

                score = safe_float(r.get(score_col))
                pred = get_prediction_bool(r, pred_col)

                if pred is None and score is not None:
                    pred = score > threshold

                if pred is True:
                    fp_vals.append(value)
                elif pred is False:
                    tn_vals.append(value)

            print(
                f"    {c:35s} | "
                f"FP mean={num(mean(fp_vals))} | "
                f"TN mean={num(mean(tn_vals))}"
            )

    else:
        print()
        print(
            "  No per-component score columns were found. "
            "If so, next step will inspect the actual Feature Schema "
            "window arrays for these sessions."
        )


# ============================================================
# PART E
# Physical free-writing session stability across calibration
# ============================================================

physical_sessions = defaultdict(list)

for job in jobs:
    session_rows = load_csv(job["normal_session_path"])

    for r in session_rows:
        if r["scenario_or_family"] != "free_writing":
            continue

        key = (
            r["participant_id"],
            r["session_id"],
        )

        physical_sessions[key].append({
            "cal": job["cal"],
            "threshold": job["threshold"],
            "fpr": safe_float(r["fpr"]),
            "mean_score":
                safe_float(r["mean_decision_score"]),
            "max_score":
                safe_float(r["max_decision_score"]),
        })


session_stability = []

for (participant, sid), vals in physical_sessions.items():

    session_stability.append({
        "participant": participant,
        "session_id": sid,
        "n_calibrations": len(vals),
        "mean_fpr": mean([v["fpr"] for v in vals]),
        "min_fpr": min(v["fpr"] for v in vals),
        "max_fpr": max(v["fpr"] for v in vals),
        "mean_decision_score":
            mean([v["mean_score"] for v in vals]),
        "calibration_detail": "; ".join(
            f"{v['cal']}:"
            f"T={num(v['threshold'],3)},"
            f"FPR={pct(v['fpr'],2)}"
            for v in sorted(vals, key=lambda x: x["cal"])
        ),
    })


session_stability.sort(
    key=lambda x: x["mean_fpr"],
    reverse=True,
)


print()
print("=" * 120)
print("E. FREE-WRITING PHYSICAL SESSION STABILITY")
print("=" * 120)

print(
    f"{'Participant':13s}"
    f"{'Session':28s}"
    f"{'Mean FPR':>12s}"
    f"{'Min':>10s}"
    f"{'Max':>10s}"
    f"  Calibration details"
)

for r in session_stability[:20]:
    print(
        f"{r['participant']:13s}"
        f"{r['session_id']:28s}"
        f"{pct(r['mean_fpr']):>12s}"
        f"{pct(r['min_fpr']):>10s}"
        f"{pct(r['max_fpr']):>10s}"
        f"  {r['calibration_detail']}"
    )


with open(
    OUT / "diagnosis_free_writing_session_stability.csv",
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(session_stability[0].keys())
    )
    w.writeheader()
    w.writerows(session_stability)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 120)
print("OUTPUT FILES")
print("=" * 120)

for path in [
    OUT / "diagnosis_long_hold_all_splits.csv",
    OUT / "diagnosis_free_writing_all_splits.csv",
    OUT / "diagnosis_free_writing_session_stability.csv",
]:
    print(path)

print()
print("[PASS] failure-mode diagnosis complete")
