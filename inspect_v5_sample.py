import re
import pandas as pd
from pathlib import Path

FEATURES = [
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "overlap_ratio",
    "simultaneous_key_count",
    "modifier_count",
    "shortcut_flag",
    "correction_ratio",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
    "current_key_category_id",
]

p = Path("data/attack/AdaptiveMimic_v5_constrained_test/window1.csv")
df = pd.read_csv(p)

print("file:", p)
print("shape:", df.shape)
print("sample index: 0")

row = df.iloc[0]

# 컬럼명이 t0_hold_time / hold_time_0 / step0_hold_time 등 어떤 방식이어도 최대한 찾기
def find_value(row, step, feature):
    candidates = [
        f"t{step}_{feature}",
        f"{feature}_{step}",
        f"step{step}_{feature}",
        f"{step}_{feature}",
        f"{feature}_t{step}",
    ]
    for c in candidates:
        if c in row.index:
            return row[c]

    # fallback: feature 이름과 step 숫자가 같이 들어간 컬럼 탐색
    for c in row.index:
        low = c.lower()
        if feature.lower() in low and re.search(rf"(^|[^0-9]){step}([^0-9]|$)", low):
            return row[c]

    return None

records = []
for t in range(50):
    rec = {"step": t}
    found_any = False
    for feat in FEATURES:
        v = find_value(row, t, feat)
        if v is not None:
            rec[feat] = v
            found_any = True
    if found_any:
        records.append(rec)

if not records:
    print("\n[WARN] Could not infer timestep-feature layout from column names.")
    print("First 100 columns:")
    print(df.columns[:100].tolist())
else:
    out = pd.DataFrame(records)

    show_cols = [
        "step",
        "hold_time",
        "flight_time",
        "press_to_press_time",
        "release_to_release_time",
        "keys_per_second",
        "burst_density",
        "pause_duration",
        "correction_ratio",
        "shortcut_flag",
        "modifier_count",
        "current_key_category_id",
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    print("\n===== V5 sample as 50-step behavior sequence =====")
    print(out[show_cols].to_string(index=False))

    print("\n===== rough summary =====")
    for c in show_cols:
        if c != "step" and pd.api.types.is_numeric_dtype(out[c]):
            print(f"{c:30s} mean={out[c].mean():.6f} min={out[c].min():.6f} max={out[c].max():.6f}")
