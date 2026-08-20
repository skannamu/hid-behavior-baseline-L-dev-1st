#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import math
import re
import statistics


ROOT = Path(
    "/home/js/hid_behavior_ai/experiments/"
    "seminar_n4_full_20260820_103101"
)

OUT = ROOT / "aggregated_results"
OUT.mkdir(parents=True, exist_ok=True)

STABILITY_CSV = (
    OUT / "feature_free_writing_window_stability.csv"
)

LONGHOLD_ALL_CSV = (
    OUT / "trace_long_hold_candidate_params_all.csv"
)

TARGET_PARTICIPANTS = {"p004", "p006"}

# 검증할 서로 다른 physical pause 수 / participant
TOP_UNIQUE_PAUSES_PER_PARTICIPANT = 6

# 논문용 representative 기준
STRONG_BYPASS = 0.50
STRONG_DETECTED_TPR = 0.90


# ============================================================
# Helpers
# ============================================================

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        return

    fields = []

    for r in rows:
        for k in r:
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


def fmt(x, d=3):
    if x is None:
        return "-"
    return f"{float(x):.{d}f}"


def pct(x, d=2):
    if x is None:
        return "-"
    return f"{100*float(x):.{d}f}%"


def find_col(columns, exact=(), contains=()):
    lower = {
        c.lower(): c
        for c in columns
    }

    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]

    for c in columns:
        lc = c.lower()

        for token in contains:
            if token.lower() in lc:
                return c

    return None


def time_scale(column, sample_value=None):
    name = (column or "").lower()

    if "nanosecond" in name or name.endswith("_ns") or "_ns_" in name:
        return 1e-9

    if "microsecond" in name or name.endswith("_us") or "_us_" in name:
        return 1e-6

    if "millisecond" in name or name.endswith("_ms") or "_ms_" in name:
        return 1e-3

    if name.endswith("_s") or "_s_" in name:
        return 1.0

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

    return v * time_scale(
        column,
        v,
    )


# ============================================================
# Load 12 recon jobs
# ============================================================

summary_files = sorted(
    ROOT.glob(
        "test_p*__cal_p*/"
        "recon_hid/experiment_summary.json"
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

    jobs.append({
        "split":
            f"test_{test_p}__cal_{cal_p}",

        "test":
            test_p,

        "cal":
            cal_p,

        "normal_predictions":
            vdir
            / "normal_test_window_predictions.csv",
    })


# ============================================================
# Window CSV direct lookup
# ============================================================

WINDOW_INDEX_CACHE = {}


def get_window_row(path, window_id):
    path = str(path)

    if path not in WINDOW_INDEX_CACHE:
        index = {}

        with open(
            path,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                index[
                    str(row["window_id"])
                ] = row

        WINDOW_INDEX_CACHE[path] = index

    return WINDOW_INDEX_CACHE[
        path
    ].get(str(window_id))


# ============================================================
# Native state/raw helpers
# ============================================================

def find_native_file(session_root, kind):
    candidates = [
        session_root
        / kind
        / f"{kind}.csv",

        session_root
        / f"{kind}.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = sorted(
        session_root.rglob(
            f"{kind}.csv"
        )
    )

    return (
        matches[0]
        if matches
        else None
    )


def find_state_index_col(cols):
    return find_col(
        cols,
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


def find_press_col(cols):
    preferred = [
        "press_time_ns",
        "press_timestamp_ns",
        "keydown_time_ns",
        "key_down_time_ns",
        "press_time_s",
        "press_timestamp_s",
        "press_time",
        "press_timestamp",
    ]

    c = find_col(
        cols,
        exact=preferred,
    )

    if c:
        return c

    for col in cols:
        lc = col.lower()

        if (
            "press" in lc
            and "time" in lc
            and "interval" not in lc
            and "hold" not in lc
        ):
            return col

    return None


def find_release_col(cols):
    preferred = [
        "release_time_ns",
        "release_timestamp_ns",
        "keyup_time_ns",
        "key_up_time_ns",
        "release_time_s",
        "release_timestamp_s",
        "release_time",
        "release_timestamp",
    ]

    c = find_col(
        cols,
        exact=preferred,
    )

    if c:
        return c

    for col in cols:
        lc = col.lower()

        if (
            "release" in lc
            and "time" in lc
            and "interval" not in lc
        ):
            return col

    return None


def find_raw_time_col(cols):
    return find_col(
        cols,
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


def state_row_for_index(
    rows,
    index_col,
    target,
):
    if target is None:
        return None

    if index_col:
        for row in rows:
            if safe_int(
                row.get(index_col)
            ) == target:
                return row

    # Schema index is 1-based.
    pos = target - 1

    if 0 <= pos < len(rows):
        return rows[pos]

    return None


# ============================================================
# PART 1
# UNIQUE PHYSICAL FREE-WRITING PAUSES
# ============================================================

print()
print("=" * 125)
print(
    "PART 1 — UNIQUE PHYSICAL FREE-WRITING PAUSES"
)
print("=" * 125)


stability = read_csv(
    STABILITY_CSV
)

repeated_keys = {
    (
        r["participant"],
        r["session_id"],
        str(r["window_id"]),
    )
    for r in stability
    if (
        r.get("category")
        == "repeated_fp"
        and r.get("participant")
        in TARGET_PARTICIPANTS
    )
}


# 하나의 실제 physical window에 대응하는 prediction 하나만 확보
prediction_by_key = {}

for job in jobs:
    if job["test"] not in TARGET_PARTICIPANTS:
        continue

    rows = read_csv(
        job["normal_predictions"]
    )

    for r in rows:
        if (
            r.get("scenario_or_family")
            != "free_writing"
        ):
            continue

        key = (
            r.get("participant_id"),
            r.get("session_id"),
            str(r.get("window_id")),
        )

        if (
            key in repeated_keys
            and key not in prediction_by_key
        ):
            prediction_by_key[
                key
            ] = r


# ------------------------------------------------------------
# 각 repeated-FP window에서 "가장 긴 press interval"을 찾고
# 그것이 실제 어떤 keystroke transition인지 계산.
#
# 동일 participant/session/current_keystroke_index이면
# overlapping window가 달라도 같은 physical pause.
# ------------------------------------------------------------

physical_pause_groups = defaultdict(list)


for key, pred in prediction_by_key.items():
    participant, session_id, window_id = key

    wr = get_window_row(
        pred["source_file"],
        pred["window_id"],
    )

    if wr is None:
        continue

    start_idx = safe_int(
        wr.get(
            "start_keystroke_index"
        )
    )

    if start_idx is None:
        continue

    intervals = []

    for t in range(50):
        v = safe_float(
            wr.get(
                f"t{t}_press_interval_s"
            )
        )

        if v is not None:
            intervals.append(
                (v, t)
            )

    if not intervals:
        continue

    gap_s, t = max(
        intervals
    )

    current_key_idx = (
        start_idx + t
    )

    previous_key_idx = (
        current_key_idx - 1
    )

    # 이것이 dedup key.
    physical_key = (
        participant,
        session_id,
        current_key_idx,
    )

    physical_pause_groups[
        physical_key
    ].append({
        "participant":
            participant,

        "session_id":
            session_id,

        "window_id":
            window_id,

        "source_file":
            pred["source_file"],

        "feature_gap_s":
            gap_s,

        "t":
            t,

        "current_key_idx":
            current_key_idx,

        "previous_key_idx":
            previous_key_idx,

        "ctx_max_pause_s":
            safe_float(
                wr.get(
                    "ctx_max_pause_interval_s"
                )
            ),

        "pause_fraction":
            safe_float(
                wr.get(
                    "ctx_pause_time_fraction"
                )
            ),

        "press_rate_hz":
            safe_float(
                wr.get(
                    "ctx_press_rate_hz"
                )
            ),
    })


print(
    f"Repeated-FP windows "
    f"(p004+p006): {len(prediction_by_key)}"
)

print(
    f"Unique physical pauses after dedup: "
    f"{len(physical_pause_groups)}"
)


# 대표 window는 그 physical pause를 포함하는 것 중
# max pause context가 가장 큰 하나.
unique_pauses = []

for physical_key, members in physical_pause_groups.items():
    representative = max(
        members,
        key=lambda r:
            (
                r["ctx_max_pause_s"]
                if r["ctx_max_pause_s"] is not None
                else -1
            )
    )

    x = dict(
        representative
    )

    x[
        "overlapping_repeated_fp_windows"
    ] = len(members)

    unique_pauses.append(x)


# participant별 서로 다른 top pause 선택
selected_pauses = []

for p in sorted(
    TARGET_PARTICIPANTS
):
    pp = [
        x
        for x in unique_pauses
        if x["participant"] == p
    ]

    pp.sort(
        key=lambda x:
            x["feature_gap_s"],
        reverse=True,
    )

    selected_pauses.extend(
        pp[
            :TOP_UNIQUE_PAUSES_PER_PARTICIPANT
        ]
    )


# ============================================================
# Raw/state verification
# ============================================================

verified_rows = []


for x in selected_pauses:
    source = Path(
        x["source_file"]
    )

    session_root = (
        source.parent.parent
    )

    state_path = find_native_file(
        session_root,
        "state",
    )

    raw_path = find_native_file(
        session_root,
        "raw",
    )

    out = dict(x)

    out["state_csv"] = (
        str(state_path)
        if state_path
        else ""
    )

    out["raw_csv"] = (
        str(raw_path)
        if raw_path
        else ""
    )

    if not state_path:
        out["status"] = (
            "STATE_MISSING"
        )
        verified_rows.append(
            out
        )
        continue

    states = read_csv(
        state_path
    )

    if not states:
        out["status"] = (
            "STATE_EMPTY"
        )
        verified_rows.append(
            out
        )
        continue

    state_cols = list(
        states[0].keys()
    )

    idx_col = (
        find_state_index_col(
            state_cols
        )
    )

    press_col = (
        find_press_col(
            state_cols
        )
    )

    release_col = (
        find_release_col(
            state_cols
        )
    )

    previous = (
        state_row_for_index(
            states,
            idx_col,
            x["previous_key_idx"],
        )
    )

    current = (
        state_row_for_index(
            states,
            idx_col,
            x["current_key_idx"],
        )
    )

    if (
        previous is None
        or current is None
        or press_col is None
    ):
        out["status"] = (
            "STATE_MAPPING_FAILED"
        )
        verified_rows.append(
            out
        )
        continue

    previous_press = seconds(
        previous.get(
            press_col
        ),
        press_col,
    )

    current_press = seconds(
        current.get(
            press_col
        ),
        press_col,
    )

    previous_release = (
        seconds(
            previous.get(
                release_col
            ),
            release_col,
        )
        if release_col
        else None
    )

    state_press_gap = (
        current_press
        - previous_press
        if (
            current_press is not None
            and previous_press is not None
        )
        else None
    )

    idle_after_release = (
        current_press
        - previous_release
        if (
            current_press is not None
            and previous_release is not None
        )
        else None
    )

    out[
        "state_press_gap_s"
    ] = state_press_gap

    out[
        "idle_after_previous_release_s"
    ] = idle_after_release


    # Raw keyboard activity inside actual idle.
    raw_idle_events = None

    if (
        raw_path
        and previous_release is not None
        and current_press is not None
    ):
        raw = read_csv(
            raw_path
        )

        if raw:
            raw_cols = list(
                raw[0].keys()
            )

            time_col = (
                find_raw_time_col(
                    raw_cols
                )
            )

            if time_col:
                count = 0

                for rr in raw:
                    ts = seconds(
                        rr.get(
                            time_col
                        ),
                        time_col,
                    )

                    if ts is None:
                        continue

                    if (
                        previous_release
                        < ts
                        < current_press
                    ):
                        count += 1

                raw_idle_events = (
                    count
                )

    out[
        "raw_events_during_idle"
    ] = raw_idle_events


    feature_state_diff = (
        abs(
            x["feature_gap_s"]
            - state_press_gap
        )
        if state_press_gap is not None
        else None
    )

    out[
        "feature_vs_state_gap_abs_diff_s"
    ] = feature_state_diff


    verified = (
        state_press_gap is not None
        and idle_after_release is not None
        and feature_state_diff is not None
        and feature_state_diff < 0.01
        and idle_after_release >= 5.0
        and raw_idle_events == 0
    )

    out[
        "verified_physical_keyboard_idle"
    ] = int(
        verified
    )

    out["status"] = (
        "VERIFIED_PHYSICAL_IDLE"
        if verified
        else "NOT_FULLY_VERIFIED"
    )

    verified_rows.append(
        out
    )


verified_rows.sort(
    key=lambda x:
        (
            x["participant"],
            -x["feature_gap_s"],
        )
)


save_csv(
    OUT
    / "final_free_writing_unique_pause_verification.csv",
    verified_rows,
)


print()
print(
    "UNIQUE FREE-WRITING PAUSE VERIFICATION"
)
print("-" * 125)

print(
    f"{'P':5s}"
    f"{'Session':28s}"
    f"{'Key idx':>9s}"
    f"{'Gap':>9s}"
    f"{'State':>9s}"
    f"{'Idle':>9s}"
    f"{'RawEv':>8s}"
    f"{'OverlapW':>10s}"
    f"  Status"
)


for r in verified_rows:
    print(
        f"{r['participant']:5s}"
        f"{r['session_id']:28s}"
        f"{r['current_key_idx']:9d}"
        f"{fmt(r['feature_gap_s'],2):>9s}"
        f"{fmt(r.get('state_press_gap_s'),2):>9s}"
        f"{fmt(r.get('idle_after_previous_release_s'),2):>9s}"
        f"{str(r.get('raw_events_during_idle','-')):>8s}"
        f"{r['overlapping_repeated_fp_windows']:10d}"
        f"  {r['status']}"
    )


print()
print(
    "Verified by participant:"
)

for p in sorted(
    TARGET_PARTICIPANTS
):
    rows = [
        r
        for r in verified_rows
        if r["participant"] == p
    ]

    passed = sum(
        r.get(
            "verified_physical_keyboard_idle"
        ) == 1
        for r in rows
    )

    print(
        f"  {p}: "
        f"{passed}/{len(rows)}"
    )


# ============================================================
# PART 2
# CLEAN LONG-HOLD REPRESENTATIVE PAIRS
# ============================================================

print()
print()
print("=" * 125)
print(
    "PART 2 — CLEAN LONG-HOLD REPRESENTATIVE PAIRS"
)
print("=" * 125)


records = read_csv(
    LONGHOLD_ALL_CSV
)

if not records:
    raise RuntimeError(
        "Long-hold candidate table is empty."
    )


# ------------------------------------------------------------
# 중복된 local.metadata.* 제거.
# manifest.policy.* 만 공식 parameter source로 사용.
# ------------------------------------------------------------

policy_fields = sorted({
    k
    for r in records
    for k in r.keys()
    if k.startswith(
        "manifest.policy."
    )
})


# seed/generation 등의 parameter는 제외하고
# 실제 behavioral policy만 남김.
wanted_policy_suffixes = [
    "correction_probability",
    "hold_jitter_ms",
    "hold_mean_ms",
    "inter_key_jitter_ms",
    "inter_key_mean_ms",
    "modifier_probability",
    "overlap_probability",
    "pause_mean_ms",
    "pause_probability",
    "repeat_probability",
]


clean_policy_fields = []

for suffix in wanted_policy_suffixes:
    target = (
        "manifest.policy."
        + suffix
    )

    if target in policy_fields:
        clean_policy_fields.append(
            target
        )


print(
    f"Unique policy parameters retained: "
    f"{len(clean_policy_fields)}"
)

for p in clean_policy_fields:
    print(
        "  "
        + p.replace(
            "manifest.policy.",
            ""
        )
    )


# numeric normalize
normalized = []

for r in records:
    x = {
        "split":
            r["split"],

        "test":
            r.get("test", ""),

        "cal":
            r.get("cal", ""),

        "candidate_id":
            r["candidate_id"],

        "tpr":
            safe_float(
                r.get("tpr")
            ),

        "bypass_rate":
            safe_float(
                r.get("bypass_rate")
            ),

        "mean_decision_score":
            safe_float(
                r.get(
                    "mean_decision_score"
                )
            ),
    }

    for f in clean_policy_fields:
        short = f.replace(
            "manifest.policy.",
            ""
        )

        x[short] = (
            safe_float(
                r.get(f)
            )
        )

    normalized.append(
        x
    )


save_csv(
    OUT
    / "final_long_hold_candidate_policy_clean.csv",
    normalized,
)


# ============================================================
# Only eligible split:
#
# at least one candidate bypass >= 50%
# AND
# at least one candidate TPR >= 90%
# ============================================================

by_split = defaultdict(list)

for r in normalized:
    by_split[
        r["split"]
    ].append(r)


eligible_pairs = []
excluded_splits = []


for split, rows in sorted(
    by_split.items()
):
    bypass_candidates = [
        r
        for r in rows
        if (
            r["bypass_rate"]
            is not None
            and r["bypass_rate"]
            >= STRONG_BYPASS
        )
    ]

    detected_candidates = [
        r
        for r in rows
        if (
            r["tpr"]
            is not None
            and r["tpr"]
            >= STRONG_DETECTED_TPR
        )
    ]


    if (
        not bypass_candidates
        or not detected_candidates
    ):
        excluded_splits.append({
            "split": split,

            "has_bypass_ge_50":
                int(
                    bool(
                        bypass_candidates
                    )
                ),

            "has_tpr_ge_90":
                int(
                    bool(
                        detected_candidates
                    )
                ),

            "max_bypass":
                max(
                    (
                        r["bypass_rate"]
                        for r in rows
                        if r["bypass_rate"]
                        is not None
                    ),
                    default=None,
                ),

            "max_tpr":
                max(
                    (
                        r["tpr"]
                        for r in rows
                        if r["tpr"]
                        is not None
                    ),
                    default=None,
                ),
        })

        continue


    # strongest bypass:
    # bypass 최대,
    # tie라면 decision score 낮은 쪽
    bypass_rep = sorted(
        bypass_candidates,
        key=lambda r: (
            -r["bypass_rate"],
            (
                r["mean_decision_score"]
                if r["mean_decision_score"]
                is not None
                else float("inf")
            ),
        )
    )[0]


    # strongest detected:
    # TPR 최대,
    # tie라면 bypass 낮고 score 높은 쪽
    detected_rep = sorted(
        detected_candidates,
        key=lambda r: (
            -r["tpr"],
            (
                r["bypass_rate"]
                if r["bypass_rate"]
                is not None
                else float("inf")
            ),
            -(
                r["mean_decision_score"]
                if r["mean_decision_score"]
                is not None
                else -float("inf")
            ),
        )
    )[0]


    pair = {
        "split":
            split,

        "bypass_candidate":
            bypass_rep[
                "candidate_id"
            ],

        "bypass_rate":
            bypass_rep[
                "bypass_rate"
            ],

        "bypass_tpr":
            bypass_rep[
                "tpr"
            ],

        "bypass_mean_score":
            bypass_rep[
                "mean_decision_score"
            ],

        "detected_candidate":
            detected_rep[
                "candidate_id"
            ],

        "detected_tpr":
            detected_rep[
                "tpr"
            ],

        "detected_bypass_rate":
            detected_rep[
                "bypass_rate"
            ],

        "detected_mean_score":
            detected_rep[
                "mean_decision_score"
            ],
    }


    for suffix in wanted_policy_suffixes:
        a = bypass_rep.get(
            suffix
        )

        b = detected_rep.get(
            suffix
        )

        pair[
            f"bypass_{suffix}"
        ] = a

        pair[
            f"detected_{suffix}"
        ] = b

        pair[
            f"delta_{suffix}"
        ] = (
            a - b
            if (
                a is not None
                and b is not None
            )
            else None
        )


    eligible_pairs.append(
        pair
    )


save_csv(
    OUT
    / "final_long_hold_eligible_representative_pairs.csv",
    eligible_pairs,
)

save_csv(
    OUT
    / "final_long_hold_excluded_splits.csv",
    excluded_splits,
)


# ============================================================
# Pretty print eligible pairs
# ============================================================

print()
print(
    f"Eligible splits: "
    f"{len(eligible_pairs)}/{len(by_split)}"
)

print(
    f"Excluded splits: "
    f"{len(excluded_splits)}/{len(by_split)}"
)


print()
print(
    "ELIGIBLE LONG-HOLD PAIRS"
)
print("-" * 125)


text_lines = []


for i, pair in enumerate(
    eligible_pairs,
    1,
):
    text_lines.append(
        "=" * 110
    )

    text_lines.append(
        f"[PAIR #{i}] "
        f"{pair['split']}"
    )

    text_lines.append(
        "=" * 110
    )

    text_lines.append(
        "BYPASS   : "
        f"{pair['bypass_candidate']} "
        f"| bypass={pct(pair['bypass_rate'])} "
        f"| TPR={pct(pair['bypass_tpr'])} "
        f"| score={fmt(pair['bypass_mean_score'])}"
    )

    text_lines.append(
        "DETECTED : "
        f"{pair['detected_candidate']} "
        f"| TPR={pct(pair['detected_tpr'])} "
        f"| bypass={pct(pair['detected_bypass_rate'])} "
        f"| score={fmt(pair['detected_mean_score'])}"
    )

    text_lines.append("")
    text_lines.append(
        "Policy parameter comparison:"
    )

    for suffix in wanted_policy_suffixes:
        a = pair.get(
            f"bypass_{suffix}"
        )

        b = pair.get(
            f"detected_{suffix}"
        )

        d = pair.get(
            f"delta_{suffix}"
        )

        text_lines.append(
            f"  {suffix:28s}"
            f"  BYPASS={fmt(a,6):>11s}"
            f"  DETECT={fmt(b,6):>11s}"
            f"  Δ={fmt(d,6):>11s}"
        )

    text_lines.append("")


for line in text_lines:
    print(line)


with open(
    OUT
    / "final_long_hold_eligible_representative_pairs.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "\n".join(
            text_lines
        )
    )


# ============================================================
# Cross-pair direction consistency
#
# parameter가 eligible split들에서 얼마나 같은 방향으로
# 움직이는지 descriptive summary.
# ============================================================

direction_rows = []


for suffix in wanted_policy_suffixes:
    deltas = [
        pair.get(
            f"delta_{suffix}"
        )
        for pair in eligible_pairs
    ]

    deltas = [
        d
        for d in deltas
        if d is not None
    ]

    if not deltas:
        continue

    positive = sum(
        d > 0
        for d in deltas
    )

    negative = sum(
        d < 0
        for d in deltas
    )

    zero = sum(
        math.isclose(
            d,
            0.0,
            abs_tol=1e-12,
        )
        for d in deltas
    )

    direction_rows.append({
        "parameter":
            suffix,

        "eligible_pair_count":
            len(deltas),

        "bypass_greater_count":
            positive,

        "bypass_lower_count":
            negative,

        "equal_count":
            zero,

        "median_delta":
            statistics.median(
                deltas
            ),

        "mean_delta":
            statistics.mean(
                deltas
            ),
    })


save_csv(
    OUT
    / "final_long_hold_parameter_direction_consistency.csv",
    direction_rows,
)


print()
print(
    "PARAMETER DIRECTION CONSISTENCY"
)
print("-" * 100)

print(
    f"{'Parameter':30s}"
    f"{'Pairs':>7s}"
    f"{'BYP>DET':>10s}"
    f"{'BYP<DET':>10s}"
    f"{'Median Δ':>14s}"
)


for r in direction_rows:
    print(
        f"{r['parameter']:30s}"
        f"{r['eligible_pair_count']:7d}"
        f"{r['bypass_greater_count']:10d}"
        f"{r['bypass_lower_count']:10d}"
        f"{fmt(r['median_delta'],6):>14s}"
    )


# ============================================================
# Final outputs
# ============================================================

print()
print("=" * 125)
print("OUTPUT FILES")
print("=" * 125)

for p in [
    OUT
    / "final_free_writing_unique_pause_verification.csv",

    OUT
    / "final_long_hold_candidate_policy_clean.csv",

    OUT
    / "final_long_hold_eligible_representative_pairs.csv",

    OUT
    / "final_long_hold_eligible_representative_pairs.txt",

    OUT
    / "final_long_hold_excluded_splits.csv",

    OUT
    / "final_long_hold_parameter_direction_consistency.csv",
]:
    print(p)


print()
print(
    "[PASS] final failure-analysis cleanup complete"
)
