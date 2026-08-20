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

STABILITY_CSV = (
    OUT / "feature_free_writing_window_stability.csv"
)

# ============================================================
# Helpers
# ============================================================

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def fmt(x, d=4):
    if x is None:
        return "-"
    return f"{float(x):.{d}f}"


def pct(x, d=2):
    if x is None:
        return "-"
    return f"{100*float(x):.{d}f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def find_col(columns, exact=(), contains=()):
    lower = {c.lower(): c for c in columns}

    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]

    for c in columns:
        lc = c.lower()

        if any(term.lower() in lc for term in contains):
            return c

    return None


def save_rows(path, rows):
    if not rows:
        return

    fields = []

    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def flatten(obj, prefix="", out=None):
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten(v, key, out)

    elif isinstance(obj, list):
        # scalar list only; otherwise recurse indexed
        if all(
            isinstance(x, (str, int, float, bool, type(None)))
            for x in obj
        ):
            out[prefix] = "|".join(str(x) for x in obj)
        else:
            for i, v in enumerate(obj):
                flatten(v, f"{prefix}[{i}]", out)

    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj

    return out


def object_contains(obj, needle):
    needle = str(needle)

    if isinstance(obj, dict):
        return any(
            object_contains(k, needle)
            or object_contains(v, needle)
            for k, v in obj.items()
        )

    if isinstance(obj, list):
        return any(
            object_contains(v, needle)
            for v in obj
        )

    return str(obj) == needle


def load_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


# ============================================================
# Time conversion helpers
# ============================================================

def time_scale(column_name, sample_value=None):
    name = (column_name or "").lower()

    if "nanosecond" in name or name.endswith("_ns") or "_ns_" in name:
        return 1e-9

    if "microsecond" in name or name.endswith("_us") or "_us_" in name:
        return 1e-6

    if "millisecond" in name or name.endswith("_ms") or "_ms_" in name:
        return 1e-3

    if name.endswith("_s") or "_s_" in name:
        return 1.0

    # Conservative magnitude fallback.
    if sample_value is not None:
        v = abs(float(sample_value))

        if v > 1e14:
            return 1e-9

        if v > 1e11:
            return 1e-3

    return 1.0


def seconds(value, column):
    v = safe_float(value)

    if v is None:
        return None

    return v * time_scale(column, v)


# ============================================================
# Find native raw/state files
# ============================================================

def find_native_file(session_root, kind):
    direct = [
        session_root / kind / f"{kind}.csv",
        session_root / f"{kind}.csv",
    ]

    for p in direct:
        if p.exists():
            return p

    matches = sorted(
        session_root.rglob(f"{kind}.csv")
    )

    return matches[0] if matches else None


# ============================================================
# Load exact window.csv row
# ============================================================

WINDOW_CACHE = {}


def get_window_row(source_file, window_id):
    source_file = str(source_file)
    key = (source_file, str(window_id))

    if key in WINDOW_CACHE:
        return WINDOW_CACHE[key]

    path = Path(source_file)

    if not path.exists():
        return None

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if str(row.get("window_id")) == str(window_id):
                WINDOW_CACHE[key] = row
                return row

    return None


# ============================================================
# Load ReCon jobs
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

    candidate_pool = (
        sp.parent
        / "final_evaluation"
        / "A_final"
        / "A_final"
        / "candidate_pool"
    )

    jobs.append({
        "split": f"test_{test_p}__cal_{cal_p}",
        "test": test_p,
        "cal": cal_p,
        "root": sp.parent,
        "normal_predictions":
            vdir / "normal_test_window_predictions.csv",
        "attack_predictions":
            vdir / "A_final_window_predictions.csv",
        "candidate_metrics":
            vdir / "A_final_candidate_metrics.csv",
        "manifest":
            candidate_pool / "attack_manifest.jsonl",
        "candidate_pool":
            candidate_pool,
    })


# ============================================================
# PART 1
# FREE-WRITING RAW/STATE BACKTRACE
# ============================================================

print()
print("=" * 130)
print("PART 1 — FREE-WRITING REPEATED-FP RAW/STATE BACKTRACE")
print("=" * 130)

if not STABILITY_CSV.exists():
    raise RuntimeError(
        f"Missing prior analysis: {STABILITY_CSV}"
    )

stability = read_csv(STABILITY_CSV)

repeated_keys = {
    (
        r["participant"],
        r["session_id"],
        str(r["window_id"]),
    )
    for r in stability
    if r.get("category") == "repeated_fp"
}

print(
    f"Repeated-FP physical windows from previous analysis: "
    f"{len(repeated_keys)}"
)

# Get one prediction record per physical window.
physical_prediction = {}

for job in jobs:
    rows = read_csv(job["normal_predictions"])

    for r in rows:
        if r.get("scenario_or_family") != "free_writing":
            continue

        key = (
            r.get("participant_id"),
            r.get("session_id"),
            str(r.get("window_id")),
        )

        if key in repeated_keys and key not in physical_prediction:
            physical_prediction[key] = r


# Rank repeated FPs by actual max pause in Feature Schema.
pause_candidates = []

for key, pred in physical_prediction.items():
    wr = get_window_row(
        pred["source_file"],
        pred["window_id"],
    )

    if wr is None:
        continue

    intervals = []

    for t in range(50):
        v = safe_float(
            wr.get(f"t{t}_press_interval_s")
        )

        if v is not None:
            intervals.append((v, t))

    if not intervals:
        continue

    max_interval, max_t = max(intervals)

    pause_candidates.append({
        "key": key,
        "prediction": pred,
        "window": wr,
        "max_interval_s": max_interval,
        "max_interval_t": max_t,
        "ctx_max_pause_s":
            safe_float(
                wr.get("ctx_max_pause_interval_s")
            ),
        "pause_fraction":
            safe_float(
                wr.get("ctx_pause_time_fraction")
            ),
        "press_rate":
            safe_float(
                wr.get("ctx_press_rate_hz")
            ),
    })


pause_candidates.sort(
    key=lambda x: x["max_interval_s"],
    reverse=True,
)

TOP_PAUSES = pause_candidates[:12]

print(
    f"Loaded repeated-FP windows with features: "
    f"{len(pause_candidates)}"
)

print()
print("Top extreme pauses:")

for i, x in enumerate(TOP_PAUSES, 1):
    p, sid, wid = x["key"]

    print(
        f"{i:2d}. {p} | {sid} | window={wid} "
        f"| max interval={fmt(x['max_interval_s'],3)}s "
        f"| ctx max pause={fmt(x['ctx_max_pause_s'],3)}s "
        f"| pause fraction={pct(x['pause_fraction'])} "
        f"| rate={fmt(x['press_rate'],2)} Hz"
    )


def find_state_index_column(columns):
    return find_col(
        columns,
        exact=[
            "keystroke_index",
            "press_index",
            "state_index",
            "key_index",
            "index",
        ],
        contains=[
            "keystroke_index",
            "press_index",
        ],
    )


def find_press_time_column(columns):
    # Avoid press interval; need absolute event time.
    candidates = [
        "press_time_ns",
        "press_timestamp_ns",
        "keydown_time_ns",
        "key_down_time_ns",
        "press_time_s",
        "press_timestamp_s",
        "press_time",
        "press_timestamp",
    ]

    c = find_col(columns, exact=candidates)

    if c:
        return c

    for col in columns:
        lc = col.lower()

        if (
            "press" in lc
            and "time" in lc
            and "interval" not in lc
            and "hold" not in lc
        ):
            return col

    return None


def find_release_time_column(columns):
    candidates = [
        "release_time_ns",
        "release_timestamp_ns",
        "keyup_time_ns",
        "key_up_time_ns",
        "release_time_s",
        "release_timestamp_s",
        "release_time",
        "release_timestamp",
    ]

    c = find_col(columns, exact=candidates)

    if c:
        return c

    for col in columns:
        lc = col.lower()

        if (
            "release" in lc
            and "time" in lc
            and "interval" not in lc
        ):
            return col

    return None


def state_row_for_index(rows, index_col, target_index):
    if index_col:
        for row in rows:
            if safe_int(row.get(index_col)) == target_index:
                return row

    # Feature window start indices are 1-based in Schema v2.
    pos = target_index - 1

    if 0 <= pos < len(rows):
        return rows[pos]

    return None


def find_raw_timestamp_column(columns):
    return find_col(
        columns,
        exact=[
            "timestamp_ns",
            "event_time_ns",
            "time_ns",
            "timestamp_s",
            "event_time_s",
            "timestamp",
            "event_time",
            "time",
        ],
        contains=[
            "timestamp",
            "event_time",
        ],
    )


def raw_event_brief(row, columns):
    interesting = []

    for c in columns:
        lc = c.lower()

        if any(
            token in lc
            for token in [
                "timestamp",
                "time_ns",
                "action",
                "event",
                "type",
                "vkey",
                "key",
                "make_code",
                "scan",
                "message",
            ]
        ):
            v = row.get(c)

            if v not in (None, ""):
                interesting.append(
                    f"{c}={v}"
                )

    return ", ".join(interesting[:12])


trace_rows = []
trace_text = []


for rank, item in enumerate(TOP_PAUSES, 1):
    participant, session_id, window_id = item["key"]
    pred = item["prediction"]
    wr = item["window"]

    source = Path(pred["source_file"])
    session_root = source.parent.parent

    state_path = find_native_file(
        session_root,
        "state"
    )

    raw_path = find_native_file(
        session_root,
        "raw"
    )

    start_index = safe_int(
        wr.get("start_keystroke_index")
    )

    t = item["max_interval_t"]

    # tX press interval = current press - previous press.
    current_key_index = (
        start_index + t
        if start_index is not None
        else None
    )

    previous_key_index = (
        current_key_index - 1
        if current_key_index is not None
        else None
    )

    result = {
        "rank": rank,
        "participant": participant,
        "session_id": session_id,
        "window_id": window_id,
        "source_window_csv": str(source),
        "max_interval_t": t,
        "feature_press_interval_s":
            item["max_interval_s"],
        "ctx_max_pause_s":
            item["ctx_max_pause_s"],
        "pause_fraction":
            item["pause_fraction"],
        "press_rate_hz":
            item["press_rate"],
        "current_keystroke_index":
            current_key_index,
        "previous_keystroke_index":
            previous_key_index,
        "state_csv":
            str(state_path) if state_path else "",
        "raw_csv":
            str(raw_path) if raw_path else "",
    }

    trace_text.append("")
    trace_text.append("=" * 120)
    trace_text.append(
        f"[FREE PAUSE #{rank}] "
        f"{participant} / {session_id} / window={window_id}"
    )
    trace_text.append("=" * 120)
    trace_text.append(
        f"Feature max interval = "
        f"{item['max_interval_s']:.6f}s at t{t}"
    )
    trace_text.append(
        f"Context max pause    = "
        f"{fmt(item['ctx_max_pause_s'],6)}s"
    )

    if not state_path:
        result["trace_status"] = "STATE_MISSING"
        trace_rows.append(result)
        continue

    state_rows = read_csv(state_path)

    if not state_rows:
        result["trace_status"] = "STATE_EMPTY"
        trace_rows.append(result)
        continue

    state_cols = list(state_rows[0].keys())

    idx_col = find_state_index_column(
        state_cols
    )

    press_col = find_press_time_column(
        state_cols
    )

    release_col = find_release_time_column(
        state_cols
    )

    result["state_index_column"] = idx_col or ""
    result["state_press_time_column"] = press_col or ""
    result["state_release_time_column"] = release_col or ""

    prev_state = state_row_for_index(
        state_rows,
        idx_col,
        previous_key_index,
    )

    curr_state = state_row_for_index(
        state_rows,
        idx_col,
        current_key_index,
    )

    if prev_state is None or curr_state is None:
        result["trace_status"] = "STATE_INDEX_NOT_FOUND"
        trace_rows.append(result)

        trace_text.append(
            f"State columns: {state_cols}"
        )
        continue

    prev_press_s = (
        seconds(
            prev_state.get(press_col),
            press_col,
        )
        if press_col else None
    )

    curr_press_s = (
        seconds(
            curr_state.get(press_col),
            press_col,
        )
        if press_col else None
    )

    prev_release_s = (
        seconds(
            prev_state.get(release_col),
            release_col,
        )
        if release_col else None
    )

    state_press_gap = (
        curr_press_s - prev_press_s
        if (
            curr_press_s is not None
            and prev_press_s is not None
        )
        else None
    )

    idle_after_release = (
        curr_press_s - prev_release_s
        if (
            curr_press_s is not None
            and prev_release_s is not None
        )
        else None
    )

    result["state_press_gap_s"] = state_press_gap
    result["idle_after_previous_release_s"] = (
        idle_after_release
    )

    trace_text.append(
        f"State press gap       = "
        f"{fmt(state_press_gap,6)}s"
    )

    trace_text.append(
        f"Idle after prev release = "
        f"{fmt(idle_after_release,6)}s"
    )

    raw_between_press = None
    raw_after_release = None
    raw_between_rows = []
    raw_idle_rows = []

    if raw_path:
        raw_rows = read_csv(raw_path)

        if raw_rows:
            raw_cols = list(
                raw_rows[0].keys()
            )

            raw_time_col = (
                find_raw_timestamp_column(
                    raw_cols
                )
            )

            result[
                "raw_timestamp_column"
            ] = raw_time_col or ""

            if (
                raw_time_col
                and prev_press_s is not None
                and curr_press_s is not None
            ):
                for rr in raw_rows:
                    ts = seconds(
                        rr.get(raw_time_col),
                        raw_time_col,
                    )

                    if ts is None:
                        continue

                    if (
                        prev_press_s < ts < curr_press_s
                    ):
                        raw_between_rows.append(
                            rr
                        )

                    if (
                        prev_release_s is not None
                        and prev_release_s < ts < curr_press_s
                    ):
                        raw_idle_rows.append(
                            rr
                        )

                raw_between_press = len(
                    raw_between_rows
                )

                raw_after_release = len(
                    raw_idle_rows
                )

                result[
                    "raw_events_between_presses"
                ] = raw_between_press

                result[
                    "raw_events_after_prev_release_before_next_press"
                ] = raw_after_release

                trace_text.append(
                    f"Raw events between presses = "
                    f"{raw_between_press}"
                )

                trace_text.append(
                    f"Raw events after previous release "
                    f"before next press = "
                    f"{raw_after_release}"
                )

                if raw_between_rows:
                    trace_text.append(
                        "Raw events inside press gap:"
                    )

                    preview = (
                        raw_between_rows[:4]
                        + (
                            raw_between_rows[-4:]
                            if len(raw_between_rows) > 4
                            else []
                        )
                    )

                    for rr in preview:
                        trace_text.append(
                            "  "
                            + raw_event_brief(
                                rr,
                                raw_cols,
                            )
                        )

    # Strongest conservative criterion:
    # long idle after previous key was released,
    # with no raw keyboard event during the idle.
    verified_idle = (
        idle_after_release is not None
        and idle_after_release >= 5.0
        and raw_after_release == 0
    )

    result["verified_keyboard_idle"] = (
        int(verified_idle)
    )

    if verified_idle:
        result["trace_status"] = (
            "VERIFIED_LONG_KEYBOARD_IDLE"
        )
    elif state_press_gap is not None:
        result["trace_status"] = (
            "LONG_GAP_FOUND_BUT_RAW_NOT_CLEAN"
        )
    else:
        result["trace_status"] = (
            "PARTIAL_TRACE"
        )

    trace_rows.append(result)


save_rows(
    OUT / "trace_free_writing_extreme_pauses.csv",
    trace_rows,
)

with open(
    OUT / "trace_free_writing_extreme_pauses.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(trace_text))


print()
print("FREE-WRITING RAW TRACE SUMMARY")
print("-" * 130)

print(
    f"{'Rank':>4s} "
    f"{'Participant':12s} "
    f"{'Feature gap':>12s} "
    f"{'State gap':>12s} "
    f"{'Idle after release':>19s} "
    f"{'Raw idle events':>16s} "
    f"Status"
)

for r in trace_rows:
    print(
        f"{r['rank']:4d} "
        f"{r['participant']:12s} "
        f"{fmt(r.get('feature_press_interval_s'),3):>12s} "
        f"{fmt(r.get('state_press_gap_s'),3):>12s} "
        f"{fmt(r.get('idle_after_previous_release_s'),3):>19s} "
        f"{str(r.get('raw_events_after_prev_release_before_next_press','-')):>16s} "
        f"{r.get('trace_status','')}"
    )


verified_count = sum(
    r.get("verified_keyboard_idle") == 1
    for r in trace_rows
)

print()
print(
    f"Verified long keyboard-idle examples: "
    f"{verified_count}/{len(trace_rows)}"
)


# ============================================================
# PART 2
# LONG-HOLD GENERATOR PARAMETER BACKTRACE
# ============================================================

print()
print()
print("=" * 130)
print("PART 2 — LONG-HOLD GENERATOR PARAMETER BACKTRACE")
print("=" * 130)


def candidate_id_column(columns):
    return find_col(
        columns,
        exact=[
            "candidate_id",
            "session_id",
            "attack_id",
        ],
        contains=[
            "candidate_id",
            "candidate",
        ],
    )


def family_column(columns):
    return find_col(
        columns,
        exact=[
            "scenario_or_family",
            "family",
            "attack_family",
        ],
        contains=["family"],
    )


def find_manifest_record(records, candidate_id):
    for record in records:
        if object_contains(
            record,
            candidate_id,
        ):
            return record

    return None


def find_candidate_source(
    prediction_rows,
    candidate_id,
):
    for r in prediction_rows:
        if (
            r.get("scenario_or_family")
            != "long_hold_mimic"
        ):
            continue

        haystack = " ".join(
            str(r.get(k, ""))
            for k in [
                "participant_id",
                "session_id",
                "source_file",
            ]
        )

        if str(candidate_id) in haystack:
            return r.get("source_file")

    return None


def load_candidate_local_metadata(
    candidate_root
):
    merged = {}

    if not candidate_root.exists():
        return merged

    # Small JSON metadata files only.
    files = []

    for p in candidate_root.rglob("*.json"):
        try:
            if p.stat().st_size <= 2_000_000:
                files.append(p)
        except Exception:
            pass

    for p in sorted(files)[:30]:
        try:
            with open(
                p,
                encoding="utf-8",
            ) as f:
                obj = json.load(f)

            flat = flatten(
                obj,
                prefix=f"local.{p.name}",
            )

            merged.update(flat)

        except Exception:
            pass

    return merged


all_candidate_records = []


for job in jobs:
    candidate_rows = read_csv(
        job["candidate_metrics"]
    )

    attack_predictions = read_csv(
        job["attack_predictions"]
    )

    manifest_records = load_jsonl(
        job["manifest"]
    )

    if not candidate_rows:
        continue

    cols = list(
        candidate_rows[0].keys()
    )

    cid_col = candidate_id_column(cols)
    fam_col = family_column(cols)

    tpr_col = find_col(
        cols,
        exact=[
            "tpr",
            "window_tpr",
            "positive_rate",
        ],
        contains=["tpr"],
    )

    bypass_col = find_col(
        cols,
        exact=["bypass_rate"],
        contains=["bypass"],
    )

    mean_score_col = find_col(
        cols,
        exact=["mean_decision_score"],
        contains=["mean_decision"],
    )

    if cid_col is None:
        raise RuntimeError(
            f"Candidate ID column not found: "
            f"{job['candidate_metrics']}\n"
            f"columns={cols}"
        )

    for row in candidate_rows:
        family = (
            row.get(fam_col)
            if fam_col
            else ""
        )

        if family != "long_hold_mimic":
            continue

        cid = str(row[cid_col])

        manifest_record = (
            find_manifest_record(
                manifest_records,
                cid,
            )
        )

        flat_manifest = (
            flatten(
                manifest_record,
                prefix="manifest",
            )
            if manifest_record
            else {}
        )

        source = find_candidate_source(
            attack_predictions,
            cid,
        )

        local_flat = {}

        candidate_root = None

        if source:
            # .../<candidate>/window/window.csv
            candidate_root = (
                Path(source).parent.parent
            )

            local_flat = (
                load_candidate_local_metadata(
                    candidate_root
                )
            )

        combined = {
            "split": job["split"],
            "test": job["test"],
            "cal": job["cal"],
            "candidate_id": cid,
            "family": family,
            "tpr":
                safe_float(
                    row.get(tpr_col)
                ) if tpr_col else None,
            "bypass_rate":
                safe_float(
                    row.get(bypass_col)
                ) if bypass_col else None,
            "mean_decision_score":
                safe_float(
                    row.get(mean_score_col)
                ) if mean_score_col else None,
            "source_file": source or "",
            "candidate_root":
                str(candidate_root)
                if candidate_root
                else "",
            "manifest_found":
                int(
                    manifest_record is not None
                ),
        }

        combined.update(flat_manifest)
        combined.update(local_flat)

        all_candidate_records.append(
            combined
        )


print(
    f"Long-hold candidate-evaluation records: "
    f"{len(all_candidate_records)}"
)

print(
    f"Manifest matched: "
    f"{sum(r['manifest_found'] for r in all_candidate_records)}"
    f"/{len(all_candidate_records)}"
)


save_rows(
    OUT / "trace_long_hold_candidate_params_all.csv",
    all_candidate_records,
)


# ============================================================
# Identify likely generator parameter fields
# ============================================================

EXCLUDE_KEYWORDS = [
    "candidate_id",
    "session_id",
    "participant",
    "source",
    "path",
    "sha",
    "hash",
    "status",
    "window_count",
    "detected",
    "bypass",
    "tpr",
    "score",
    "prediction",
    "label",
    "scenario",
    "family",
    "dataset",
    "method",
    "format",
    "version",
    "generation",
    "seed",
    "row",
]

PARAM_HINTS = [
    "hold",
    "interval",
    "flight",
    "pause",
    "jitter",
    "overlap",
    "correction",
    "shortcut",
    "burst",
    "repeat",
    "modifier",
    "concurrent",
    "rate",
    "prob",
    "duration",
    "latency",
    "speed",
    "timing",
    "sigma",
    "mean",
    "std",
    "min",
    "max",
]


def numeric_values(records, key):
    vals = []

    for r in records:
        v = safe_float(r.get(key))

        if v is not None:
            vals.append(v)

    return vals


all_keys = sorted({
    k
    for r in all_candidate_records
    for k in r.keys()
})

candidate_param_keys = []

for key in all_keys:
    lk = key.lower()

    if any(x in lk for x in EXCLUDE_KEYWORDS):
        continue

    vals = numeric_values(
        all_candidate_records,
        key,
    )

    if len(vals) < 4:
        continue

    rounded = {
        round(v, 12)
        for v in vals
    }

    if len(rounded) <= 1:
        continue

    candidate_param_keys.append(
        key
    )


# Prefer fields that actually look like policy parameters.
hinted = [
    k for k in candidate_param_keys
    if any(
        hint in k.lower()
        for hint in PARAM_HINTS
    )
]

if hinted:
    candidate_param_keys = hinted


print()
print(
    f"Varying numeric generator-like fields found: "
    f"{len(candidate_param_keys)}"
)

for k in candidate_param_keys[:40]:
    vals = numeric_values(
        all_candidate_records,
        k,
    )

    print(
        f"  {k}: "
        f"min={fmt(min(vals))} "
        f"max={fmt(max(vals))}"
    )


# ============================================================
# Global descriptive parameter contrast
#
# Strong bypass = >= 50% bypass
# Strong detected = >= 90% TPR
#
# This is descriptive; repeated candidates across defenders
# are NOT independent samples.
# ============================================================

strong_bypass = [
    r for r in all_candidate_records
    if (
        r.get("bypass_rate") is not None
        and r["bypass_rate"] >= 0.50
    )
]

strong_detected = [
    r for r in all_candidate_records
    if (
        r.get("tpr") is not None
        and r["tpr"] >= 0.90
    )
]


param_contrast = []

for key in candidate_param_keys:
    a = [
        safe_float(r.get(key))
        for r in strong_bypass
    ]

    b = [
        safe_float(r.get(key))
        for r in strong_detected
    ]

    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]

    if len(a) < 2 or len(b) < 2:
        continue

    ma = mean(a)
    mb = mean(b)

    meda = median(a)
    medb = median(b)

    sda = (
        statistics.stdev(a)
        if len(a) >= 2
        else 0
    )

    sdb = (
        statistics.stdev(b)
        if len(b) >= 2
        else 0
    )

    denom = len(a) + len(b) - 2

    pooled_var = (
        (
            (len(a)-1)*sda*sda
            + (len(b)-1)*sdb*sdb
        )
        / denom
        if denom > 0
        else 0
    )

    pooled_sd = math.sqrt(
        max(pooled_var, 0)
    )

    d = (
        (ma - mb) / pooled_sd
        if pooled_sd > 1e-12
        else 0
    )

    param_contrast.append({
        "parameter": key,
        "strong_bypass_n": len(a),
        "strong_detected_n": len(b),
        "strong_bypass_mean": ma,
        "strong_detected_mean": mb,
        "strong_bypass_median": meda,
        "strong_detected_median": medb,
        "difference": ma - mb,
        "cohens_d_descriptive": d,
        "abs_cohens_d":
            abs(d),
    })


param_contrast.sort(
    key=lambda r: r["abs_cohens_d"],
    reverse=True,
)

save_rows(
    OUT / "trace_long_hold_parameter_contrast.csv",
    param_contrast,
)


print()
print("=" * 130)
print(
    "LONG-HOLD PARAMETER CONTRAST "
    "— STRONG BYPASS vs STRONG DETECTED"
)
print("=" * 130)

print(
    f"Strong-bypass records : "
    f"{len(strong_bypass)}"
)

print(
    f"Strong-detected records: "
    f"{len(strong_detected)}"
)

print()

print(
    f"{'Parameter':60s}"
    f"{'Bypass med':>13s}"
    f"{'Detect med':>13s}"
    f"{'d':>10s}"
)

for r in param_contrast[:30]:
    print(
        f"{r['parameter'][:60]:60s}"
        f"{fmt(r['strong_bypass_median']):>13s}"
        f"{fmt(r['strong_detected_median']):>13s}"
        f"{fmt(r['cohens_d_descriptive'],3):>10s}"
    )


# ============================================================
# Representative within-split contrasts
# ============================================================

by_split = defaultdict(list)

for r in all_candidate_records:
    by_split[
        r["split"]
    ].append(r)


split_rank = []

for split, rows in by_split.items():
    max_bypass = max(
        (
            r["bypass_rate"]
            for r in rows
            if r.get("bypass_rate") is not None
        ),
        default=0,
    )

    split_rank.append(
        (max_bypass, split, rows)
    )


split_rank.sort(reverse=True)

rep_lines = []


for order, (
    max_bypass,
    split,
    rows,
) in enumerate(
    split_rank[:6],
    1,
):
    worst = max(
        rows,
        key=lambda r:
            r.get("bypass_rate")
            if r.get("bypass_rate") is not None
            else -1
    )

    # pick highly detected candidate different from worst
    detected_sorted = sorted(
        rows,
        key=lambda r:
            r.get("tpr")
            if r.get("tpr") is not None
            else -1,
        reverse=True,
    )

    best = next(
        (
            r for r in detected_sorted
            if r["candidate_id"]
            != worst["candidate_id"]
        ),
        detected_sorted[0],
    )

    rep_lines.append("")
    rep_lines.append("=" * 130)
    rep_lines.append(
        f"[SPLIT #{order}] {split}"
    )
    rep_lines.append("=" * 130)

    rep_lines.append(
        f"BYPASS representative: "
        f"{worst['candidate_id']} "
        f"| bypass={pct(worst.get('bypass_rate'))} "
        f"| TPR={pct(worst.get('tpr'))} "
        f"| score={fmt(worst.get('mean_decision_score'))}"
    )

    rep_lines.append(
        f"DETECTED representative: "
        f"{best['candidate_id']} "
        f"| bypass={pct(best.get('bypass_rate'))} "
        f"| TPR={pct(best.get('tpr'))} "
        f"| score={fmt(best.get('mean_decision_score'))}"
    )

    differences = []

    for key in candidate_param_keys:
        a = safe_float(
            worst.get(key)
        )

        b = safe_float(
            best.get(key)
        )

        if a is None or b is None:
            continue

        if math.isclose(
            a,
            b,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            continue

        scale = (
            abs(a) + abs(b) + 1e-9
        )

        rel = abs(a-b) / scale

        differences.append(
            (
                rel,
                key,
                a,
                b,
            )
        )

    differences.sort(reverse=True)

    rep_lines.append(
        "Differing generator parameters:"
    )

    if not differences:
        rep_lines.append(
            "  [No varying numeric policy fields found "
            "in manifest/local JSON metadata]"
        )
    else:
        for _, key, a, b in differences[:35]:
            rep_lines.append(
                f"  {key}: "
                f"BYPASS={fmt(a,6)} | "
                f"DETECTED={fmt(b,6)}"
            )


with open(
    OUT / "trace_long_hold_representatives.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(rep_lines))


print()
print("=" * 130)
print("LONG-HOLD REPRESENTATIVE CANDIDATES")
print("=" * 130)

for line in rep_lines:
    print(line)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print()
print("=" * 130)
print("FINAL OUTPUT")
print("=" * 130)

files = [
    OUT / "trace_free_writing_extreme_pauses.csv",
    OUT / "trace_free_writing_extreme_pauses.txt",
    OUT / "trace_long_hold_candidate_params_all.csv",
    OUT / "trace_long_hold_parameter_contrast.csv",
    OUT / "trace_long_hold_representatives.txt",
]

for p in files:
    print(p)

print()
print("[PASS] raw/policy backtrace complete")
