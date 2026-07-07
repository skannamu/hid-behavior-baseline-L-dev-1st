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

# category id는 실제 키가 아니라 카테고리라서 임시 키로 치환
CATEGORY_KEYS = {
    0: "a",
    1: "A",
    2: "1",
    3: "Enter",
    4: "Backspace",
    5: "Ctrl",
    6: "Space",
    7: ".",
}

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

    for c in row.index:
        low = c.lower()
        if feature.lower() in low and re.search(rf"(^|[^0-9]){step}([^0-9]|$)", low):
            return row[c]

    return None

def get_float(row, step, feature, default=0.0):
    v = find_value(row, step, feature)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

p = Path("data/attack/AdaptiveMimic_v5_constrained_test/window1.csv")
df = pd.read_csv(p)
row = df.iloc[0]

current_time = 0.0
lines = []

for t in range(50):
    hold = get_float(row, t, "hold_time", 0.02)
    p2p = get_float(row, t, "press_to_press_time", 0.07)
    pause = get_float(row, t, "pause_duration", 0.0)
    kps = get_float(row, t, "keys_per_second", 0.0)
    corr = get_float(row, t, "correction_ratio", 0.0)
    shortcut = get_float(row, t, "shortcut_flag", 0.0)
    mod = get_float(row, t, "modifier_count", 0.0)
    cat = int(round(get_float(row, t, "current_key_category_id", 0.0)))

    key = CATEGORY_KEYS.get(cat, f"CAT{cat}")

    if shortcut >= 0.5 or mod >= 1:
        key_repr = f"MOD+{key}"
    elif corr > 0.1 and t % 7 == 0:
        key_repr = "Backspace"
    else:
        key_repr = key

    if pause > 0.05:
        lines.append(f"[{current_time:7.3f}s] pause {pause:.3f}s")
        current_time += pause

    lines.append(
        f"[{current_time:7.3f}s] press {key_repr:10s} "
        f"hold={hold:.4f}s p2p={p2p:.4f}s kps={kps:.2f}"
    )

    current_time += max(p2p, 0.001)

print("===== Approximate V5 typing timeline =====")
print("NOTE: keys are placeholders reconstructed from key category IDs, not original characters.")
print()
print("\n".join(lines))

print()
print(f"Approx duration: {current_time:.3f}s for 50 events")
print(f"Approx KPS     : {50 / current_time:.2f} keys/sec")
