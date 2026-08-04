from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import time

import numpy as np
import pandas as pd

from src.framework.config import get_cfg, require_cfg, dump_json

FEATURES = [
    "hold_time", "flight_time", "press_to_press_time", "release_to_release_time",
    "overlap_ratio", "simultaneous_key_count", "modifier_count", "shortcut_flag",
    "correction_ratio", "keys_per_second", "burst_density", "timing_variance",
    "timing_entropy", "pause_duration", "current_key_category_id",
]

DEFAULT_MODES = [
    "machine_fast",
    "burst_stop_burst",
    "shortcut_heavy",
    "human_mimic_low_variance",
    "paste_like",
]


def _cols(feature: str, window_size: int) -> List[str]:
    return [f"t{t}_{feature}" for t in range(window_size)]


def _all_feature_cols(window_size: int) -> List[str]:
    return [f"t{t}_{f}" for t in range(window_size) for f in FEATURES]


def _scale(out: pd.DataFrame, idx, feature: str, factor: float, window_size: int, lo=None, hi=None):
    cs = _cols(feature, window_size)
    vals = out.loc[idx, cs].astype(float) * factor
    if lo is not None:
        vals = vals.clip(lower=lo)
    if hi is not None:
        vals = vals.clip(upper=hi)
    out.loc[idx, cs] = vals


def _set_random(out: pd.DataFrame, idx, feature: str, low: float, high: float, rng: np.random.Generator, window_size: int):
    cs = _cols(feature, window_size)
    n = len(out.loc[idx])
    out.loc[idx, cs] = rng.uniform(low, high, size=(n, len(cs)))


def _set_const(out: pd.DataFrame, idx, feature: str, value: float, window_size: int):
    out.loc[idx, _cols(feature, window_size)] = value


def _apply_mode(out: pd.DataFrame, mode: str, idx, rng: np.random.Generator, window_size: int):
    if len(out.loc[idx]) == 0:
        return

    if mode == "machine_fast":
        _scale(out, idx, "hold_time", 0.25, window_size, lo=0.004)
        _scale(out, idx, "flight_time", 0.12, window_size, lo=0.0)
        _scale(out, idx, "press_to_press_time", 0.30, window_size, lo=0.004)
        _scale(out, idx, "release_to_release_time", 0.30, window_size, lo=0.004)
        _scale(out, idx, "keys_per_second", 2.8, window_size, hi=80.0)
        _scale(out, idx, "timing_variance", 0.12, window_size, lo=0.0)
        _scale(out, idx, "timing_entropy", 0.55, window_size, lo=0.0)
        _set_random(out, idx, "pause_duration", 0.0, 0.015, rng, window_size)

    elif mode == "burst_stop_burst":
        for t in range(window_size):
            burst = (t % 17) < 12
            for f in ["press_to_press_time", "release_to_release_time", "flight_time"]:
                c = f"t{t}_{f}"
                out.loc[idx, c] = rng.uniform(0.01, 0.07, len(out.loc[idx])) if burst else rng.uniform(0.35, 0.95, len(out.loc[idx]))
            out.loc[idx, f"t{t}_pause_duration"] = rng.uniform(0.0, 0.03, len(out.loc[idx])) if burst else rng.uniform(0.45, 1.2, len(out.loc[idx]))
            out.loc[idx, f"t{t}_burst_density"] = rng.uniform(0.75, 1.0, len(out.loc[idx])) if burst else rng.uniform(0.05, 0.25, len(out.loc[idx]))
        _scale(out, idx, "timing_variance", 3.0, window_size, lo=0.0)

    elif mode == "shortcut_heavy":
        _set_const(out, idx, "shortcut_flag", 1.0, window_size)
        _set_random(out, idx, "modifier_count", 1.0, 3.0, rng, window_size)
        _set_random(out, idx, "simultaneous_key_count", 2.0, 4.0, rng, window_size)
        _scale(out, idx, "press_to_press_time", 0.45, window_size, lo=0.004)
        _scale(out, idx, "keys_per_second", 2.1, window_size, hi=80.0)

    elif mode == "human_mimic_low_variance":
        _scale(out, idx, "hold_time", 0.55, window_size, lo=0.005)
        _scale(out, idx, "flight_time", 0.55, window_size, lo=0.0)
        _scale(out, idx, "press_to_press_time", 0.70, window_size, lo=0.004)
        _scale(out, idx, "release_to_release_time", 0.70, window_size, lo=0.004)
        _scale(out, idx, "timing_variance", 0.18, window_size, lo=0.0)
        _scale(out, idx, "timing_entropy", 0.70, window_size, lo=0.0)

    elif mode == "paste_like":
        _set_random(out, idx, "hold_time", 0.004, 0.018, rng, window_size)
        _set_random(out, idx, "flight_time", 0.000, 0.008, rng, window_size)
        _set_random(out, idx, "press_to_press_time", 0.004, 0.020, rng, window_size)
        _set_random(out, idx, "release_to_release_time", 0.004, 0.020, rng, window_size)
        _set_random(out, idx, "keys_per_second", 25.0, 80.0, rng, window_size)
        _set_random(out, idx, "burst_density", 0.90, 1.0, rng, window_size)
        _set_random(out, idx, "timing_variance", 0.0, 0.01, rng, window_size)
        _set_random(out, idx, "timing_entropy", 0.0, 0.8, rng, window_size)
        _set_random(out, idx, "pause_duration", 0.0, 0.01, rng, window_size)


def _candidate_total(cfg: Dict[str, Any], round_name: str, modes: List[str]) -> int:
    selected = int(get_cfg(cfg, "defender_aware_generator.selected_candidates", 90))
    multiplier = int(get_cfg(cfg, "defender_aware_generator.pool_multiplier", 3))
    per_mode = int(get_cfg(cfg, "generation.candidates_per_mode", 30))
    if "_pool" in round_name:
        return max(selected * multiplier, len(modes))
    return max(per_mode * len(modes), len(modes))


def generate_synthetic_attacks(cfg: Dict[str, Any], round_name: str = "round0_A0") -> Path:
    start = time.time()
    normal_csv = Path(require_cfg(cfg, "paths.normal_window_csv"))
    attack_root = Path(require_cfg(cfg, "paths.attack_generated_root"))
    out_dir = attack_root / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "candidates.csv"
    out_summary = out_dir / "generation_summary.json"

    seed = int(get_cfg(cfg, "generation.seed", 42)) + (abs(hash(round_name)) % 100000)
    window_size = int(get_cfg(cfg, "window.window_size", 50))
    rng = np.random.default_rng(seed)

    print(f"[GEN v7.1] loading normal csv: {normal_csv}", flush=True)
    df = pd.read_csv(normal_csv)

    feature_cols = _all_feature_cols(window_size)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {len(missing)} first={missing[:10]}")

    modes = get_cfg(cfg, "generation.modes", DEFAULT_MODES)
    if not isinstance(modes, list) or not modes:
        modes = DEFAULT_MODES

    total = _candidate_total(cfg, round_name, modes)
    meta_cols = [c for c in ["window_id", "participant_id", "session_id", "scenario", "input_source", "label"] if c in df.columns]
    base_cols = meta_cols + feature_cols

    sample_idx = rng.integers(0, len(df), size=total)
    out = df.iloc[sample_idx][base_cols].reset_index(drop=True).copy()

    mode_values = np.array([modes[i % len(modes)] for i in range(total)], dtype=object)
    rng.shuffle(mode_values)
    out["attack_mode"] = mode_values

    print(f"[GEN v7.1] generating {total} candidates, modes={modes}", flush=True)

    for mode in modes:
        idx = out.index[out["attack_mode"] == mode]
        _apply_mode(out, mode, idx, rng, window_size)
        print(f"[GEN v7.1] mode={mode} count={len(idx)}", flush=True)

    out["window_id"] = [f"{round_name}_{i:06d}" for i in range(len(out))]
    out["participant_id"] = "attack_synthetic"
    out["session_id"] = round_name
    out["input_source"] = "synthetic_hid"
    out["label"] = "attack"
    out["generator_version"] = "v7_1_fast"
    out["scenario"] = out["attack_mode"]

    out.to_csv(out_csv, index=False, encoding="utf-8")

    elapsed = round(time.time() - start, 3)
    summary = {
        "generator_version": "v7_1_fast",
        "round_name": round_name,
        "normal_csv": str(normal_csv),
        "output": str(out_csv),
        "count": int(len(out)),
        "modes": list(map(str, modes)),
        "elapsed_seconds": elapsed,
    }
    dump_json(out_summary, summary)

    print("[OK] Synthetic HID attack candidates generated", flush=True)
    print(f"  - output: {out_csv}", flush=True)
    print(f"  - count : {len(out)}", flush=True)
    print(f"  - elapsed: {elapsed} sec", flush=True)

    return out_csv
