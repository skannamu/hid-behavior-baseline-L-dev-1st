"""Physically consistent raw-event attack candidate generation for Stable v9.

The generator never edits model feature columns. It creates down/up event streams,
then passes them through the exact stable Feature Schema v2 extractor and window
builder used by the collector/offline pipeline.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.features.extractor import FeatureExtractorV2
from src.features.models import RawKeyEvent
from src.features.schema import EXPECTED_SCHEMA_SHA256, FEATURE_SCHEMA_VERSION
from src.features.validators import validate_raw_events, validate_states, validate_windows
from src.features.window_builder import WindowBuilder, write_windows_csv

from .common import atomic_json, sha256_file, write_jsonl


GENERATOR_VERSION = "raw_attack_generator_v9_1"
SUPPORTED_FAMILIES = (
    "constant_fast",
    "jittered_mimic",
    "burst_pause",
    "overlap_dense",
    "shortcut_heavy",
    "correction_heavy",
)


@dataclass(frozen=True)
class RawAttackPolicy:
    family: str
    inter_key_mean_ms: float
    inter_key_jitter_ms: float
    hold_mean_ms: float
    hold_jitter_ms: float
    pause_probability: float
    pause_mean_ms: float
    correction_probability: float
    modifier_probability: float
    repeat_probability: float
    overlap_probability: float

    def validate(self) -> None:
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported family={self.family!r}; expected one of "
                f"{SUPPORTED_FAMILIES}"
            )
        positive = {
            "inter_key_mean_ms": self.inter_key_mean_ms,
            "hold_mean_ms": self.hold_mean_ms,
            "pause_mean_ms": self.pause_mean_ms,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.inter_key_jitter_ms < 0 or self.hold_jitter_ms < 0:
            raise ValueError("jitter values must be non-negative")
        for name in (
            "pause_probability",
            "correction_probability",
            "modifier_probability",
            "repeat_probability",
            "overlap_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawAttackGenerationConfig:
    candidates: int = 24
    keystrokes_per_candidate: int = 72
    seed: int = 20260807
    generation: int = 0
    mutation_strength: float = 0.18
    parent_policy_path: str | None = None

    def validate(self) -> None:
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.keystrokes_per_candidate < 52:
            raise ValueError(
                "keystrokes_per_candidate must be at least 52 to produce "
                "a stable 50-keystroke window after transition filtering"
            )
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.mutation_strength < 0:
            raise ValueError("mutation_strength must be non-negative")


@dataclass(frozen=True)
class GeneratedAttackSession:
    candidate_id: str
    session_id: str
    family: str
    seed: int
    generation: int
    session_dir: str
    raw_path: str
    state_path: str
    window_path: str
    window_count: int
    raw_event_count: int
    state_count: int
    raw_sha256: str
    window_sha256: str
    policy: dict[str, Any]
    parent_candidate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _base_policy(family: str) -> RawAttackPolicy:
    policies = {
        "constant_fast": RawAttackPolicy(
            family=family,
            inter_key_mean_ms=18.0,
            inter_key_jitter_ms=1.0,
            hold_mean_ms=6.0,
            hold_jitter_ms=1.0,
            pause_probability=0.0,
            pause_mean_ms=400.0,
            correction_probability=0.0,
            modifier_probability=0.0,
            repeat_probability=0.0,
            overlap_probability=0.0,
        ),
        "jittered_mimic": RawAttackPolicy(
            family=family,
            inter_key_mean_ms=95.0,
            inter_key_jitter_ms=28.0,
            hold_mean_ms=68.0,
            hold_jitter_ms=18.0,
            pause_probability=0.025,
            pause_mean_ms=520.0,
            correction_probability=0.025,
            modifier_probability=0.025,
            repeat_probability=0.008,
            overlap_probability=0.10,
        ),
        "burst_pause": RawAttackPolicy(
            family=family,
            inter_key_mean_ms=48.0,
            inter_key_jitter_ms=12.0,
            hold_mean_ms=34.0,
            hold_jitter_ms=8.0,
            pause_probability=0.075,
            pause_mean_ms=760.0,
            correction_probability=0.015,
            modifier_probability=0.01,
            repeat_probability=0.004,
            overlap_probability=0.08,
        ),
        "overlap_dense": RawAttackPolicy(
            family=family,
            inter_key_mean_ms=38.0,
            inter_key_jitter_ms=9.0,
            hold_mean_ms=82.0,
            hold_jitter_ms=16.0,
            pause_probability=0.01,
            pause_mean_ms=450.0,
            correction_probability=0.01,
            modifier_probability=0.01,
            repeat_probability=0.005,
            overlap_probability=0.72,
        ),
        "shortcut_heavy": RawAttackPolicy(
            family=family,
            inter_key_mean_ms=105.0,
            inter_key_jitter_ms=20.0,
            hold_mean_ms=52.0,
            hold_jitter_ms=11.0,
            pause_probability=0.02,
            pause_mean_ms=480.0,
            correction_probability=0.015,
            modifier_probability=0.24,
            repeat_probability=0.004,
            overlap_probability=0.06,
        ),
        "correction_heavy": RawAttackPolicy(
            family=family,
            inter_key_mean_ms=88.0,
            inter_key_jitter_ms=24.0,
            hold_mean_ms=61.0,
            hold_jitter_ms=15.0,
            pause_probability=0.035,
            pause_mean_ms=610.0,
            correction_probability=0.18,
            modifier_probability=0.015,
            repeat_probability=0.01,
            overlap_probability=0.08,
        ),
    }
    return policies[family]


def _clamp_policy(policy: RawAttackPolicy) -> RawAttackPolicy:
    result = replace(
        policy,
        inter_key_mean_ms=float(np.clip(policy.inter_key_mean_ms, 6.0, 350.0)),
        inter_key_jitter_ms=float(np.clip(policy.inter_key_jitter_ms, 0.0, 140.0)),
        hold_mean_ms=float(np.clip(policy.hold_mean_ms, 2.0, 260.0)),
        hold_jitter_ms=float(np.clip(policy.hold_jitter_ms, 0.0, 100.0)),
        pause_probability=float(np.clip(policy.pause_probability, 0.0, 0.35)),
        pause_mean_ms=float(np.clip(policy.pause_mean_ms, 250.0, 2500.0)),
        correction_probability=float(np.clip(policy.correction_probability, 0.0, 0.40)),
        modifier_probability=float(np.clip(policy.modifier_probability, 0.0, 0.45)),
        repeat_probability=float(np.clip(policy.repeat_probability, 0.0, 0.20)),
        overlap_probability=float(np.clip(policy.overlap_probability, 0.0, 0.95)),
    )
    result.validate()
    return result


def mutate_policy(
    policy: RawAttackPolicy,
    *,
    rng: np.random.Generator,
    strength: float,
) -> RawAttackPolicy:
    if strength <= 0:
        return policy

    def multiplicative(value: float, local: float = 1.0) -> float:
        return float(value * np.exp(rng.normal(0.0, strength * local)))

    def probability(value: float, local: float = 0.35) -> float:
        return float(value + rng.normal(0.0, strength * local))

    return _clamp_policy(replace(
        policy,
        inter_key_mean_ms=multiplicative(policy.inter_key_mean_ms),
        inter_key_jitter_ms=multiplicative(max(policy.inter_key_jitter_ms, 0.5)),
        hold_mean_ms=multiplicative(policy.hold_mean_ms),
        hold_jitter_ms=multiplicative(max(policy.hold_jitter_ms, 0.5)),
        pause_probability=probability(policy.pause_probability, 0.18),
        pause_mean_ms=multiplicative(policy.pause_mean_ms, 0.65),
        correction_probability=probability(policy.correction_probability, 0.22),
        modifier_probability=probability(policy.modifier_probability, 0.22),
        repeat_probability=probability(policy.repeat_probability, 0.12),
        overlap_probability=probability(policy.overlap_probability, 0.25),
    ))


def _load_parent_policies(path: str | Path | None) -> list[tuple[str, RawAttackPolicy]]:
    if path is None:
        return []
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    parents: list[tuple[str, RawAttackPolicy]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            policy_payload = payload.get("policy")
            if not isinstance(policy_payload, dict):
                raise ValueError(
                    f"{source}:{line_number}: parent entry has no policy object"
                )
            policy = RawAttackPolicy(**policy_payload)
            policy.validate()
            parents.append((str(payload.get("candidate_id", "unknown")), policy))
    if not parents:
        raise ValueError(f"Parent policy file is empty: {source}")
    return parents


def _sample_positive_ms(
    rng: np.random.Generator,
    mean: float,
    jitter: float,
    *,
    minimum: float,
) -> float:
    return float(max(minimum, rng.normal(mean, jitter)))


def _make_event(
    *,
    session_id: str,
    timestamp_ns: int,
    vkey: int,
    action: str,
    device_id: str = "attack_kbd_001",
) -> RawKeyEvent:
    return RawKeyEvent(
        session_id=session_id,
        raw_event_index=-1,
        timestamp_ns=int(timestamp_ns),
        device_id_session_local=device_id,
        make_code=int(vkey),
        extended_flag=0,
        vkey=int(vkey),
        message=0,
        action=action,
    )


def _generate_raw_events(
    *,
    session_id: str,
    policy: RawAttackPolicy,
    logical_keystrokes: int,
    seed: int,
) -> tuple[RawKeyEvent, ...]:
    policy.validate()
    rng = np.random.default_rng(seed)
    # A broad non-payload key alphabet. It creates timing/overlap/modifier
    # behavior only and never encodes a command string.
    key_alphabet = list(range(65, 91)) + list(range(48, 58)) + [32, 13]
    events: list[RawKeyEvent] = []
    cursor_ns = 1_000_000_000
    recent_keys: list[int] = []
    active_until_ns: dict[int, int] = {}

    for index in range(logical_keystrokes):
        if rng.random() < policy.pause_probability:
            pause_ms = _sample_positive_ms(
                rng,
                policy.pause_mean_ms,
                policy.pause_mean_ms * 0.25,
                minimum=250.0,
            )
            cursor_ns += int(pause_ms * 1_000_000)

        active_until_ns = {
            key: release_ns
            for key, release_ns in active_until_ns.items()
            if release_ns > cursor_ns
        }
        correction_allowed = active_until_ns.get(8, 0) <= cursor_ns
        if rng.random() < policy.correction_probability and correction_allowed:
            vkey = 8
        else:
            choices = [
                key
                for key in key_alphabet
                if key not in recent_keys[-4:]
                and active_until_ns.get(key, 0) <= cursor_ns
            ]
            if not choices:
                choices = [
                    key for key in key_alphabet
                    if active_until_ns.get(key, 0) <= cursor_ns
                ]
            if not choices:
                cursor_ns = max(active_until_ns.values()) + 1_000_000
                active_until_ns.clear()
                choices = list(key_alphabet)
            vkey = int(rng.choice(choices))

        interval_ms = _sample_positive_ms(
            rng,
            policy.inter_key_mean_ms,
            policy.inter_key_jitter_ms,
            minimum=4.0,
        )
        hold_ms = _sample_positive_ms(
            rng,
            policy.hold_mean_ms,
            policy.hold_jitter_ms,
            minimum=2.0,
        )
        if rng.random() < policy.overlap_probability:
            hold_ms = max(hold_ms, interval_ms * float(rng.uniform(1.15, 2.5)))
        elif hold_ms >= interval_ms:
            hold_ms = max(2.0, interval_ms * float(rng.uniform(0.35, 0.85)))

        press_ns = cursor_ns
        hold_ns = int(hold_ms * 1_000_000)

        if rng.random() < policy.modifier_probability and vkey not in {8, 13, 32}:
            ctrl_vkey = 162
            modifier_down = max(0, press_ns - 8_000_000)
            target_down = press_ns
            target_up = target_down + hold_ns
            modifier_up = target_up + int(rng.uniform(5.0, 18.0) * 1_000_000)
            events.extend([
                _make_event(
                    session_id=session_id,
                    timestamp_ns=modifier_down,
                    vkey=ctrl_vkey,
                    action="down",
                ),
                _make_event(
                    session_id=session_id,
                    timestamp_ns=target_down,
                    vkey=vkey,
                    action="down",
                ),
                _make_event(
                    session_id=session_id,
                    timestamp_ns=target_up,
                    vkey=vkey,
                    action="up",
                ),
                _make_event(
                    session_id=session_id,
                    timestamp_ns=modifier_up,
                    vkey=ctrl_vkey,
                    action="up",
                ),
            ])
            active_until_ns[vkey] = target_up
            active_until_ns[ctrl_vkey] = modifier_up
            # Chords are completed before the next logical action to avoid
            # impossible duplicate modifier downs.
            cursor_ns = modifier_up + int(interval_ms * 1_000_000)
        else:
            events.append(_make_event(
                session_id=session_id,
                timestamp_ns=press_ns,
                vkey=vkey,
                action="down",
            ))
            if rng.random() < policy.repeat_probability:
                repeat_time = press_ns + max(1_000_000, hold_ns // 2)
                events.append(_make_event(
                    session_id=session_id,
                    timestamp_ns=repeat_time,
                    vkey=vkey,
                    action="down",
                ))
            release_ns = press_ns + hold_ns
            events.append(_make_event(
                session_id=session_id,
                timestamp_ns=release_ns,
                vkey=vkey,
                action="up",
            ))
            active_until_ns[vkey] = release_ns
            cursor_ns += int(interval_ms * 1_000_000)

        recent_keys.append(vkey)

    ordered = sorted(
        events,
        key=lambda event: (
            event.timestamp_ns,
            0 if event.action == "up" else 1,
            event.vkey,
        ),
    )
    return tuple(
        replace(event, raw_event_index=index)
        for index, event in enumerate(ordered)
    )


def _write_raw_csv(events: Iterable[RawKeyEvent], path: Path) -> Path:
    fieldnames = list(RawKeyEvent.__dataclass_fields__)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))
    return path


def _write_state_csv(states: Iterable[Any], path: Path) -> Path:
    rows = []
    for state in states:
        row = state.to_dict()
        row["quality_flags"] = json.dumps(row.get("quality_flags", []))
        rows.append(row)
    if not rows:
        raise ValueError("Cannot write zero attack states")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _generate_one_session(
    *,
    output_root: Path,
    round_name: str,
    candidate_index: int,
    policy: RawAttackPolicy,
    seed: int,
    generation: int,
    logical_keystrokes: int,
    parent_candidate_id: str | None,
) -> GeneratedAttackSession:
    candidate_id = f"{round_name}_g{generation}_c{candidate_index:04d}"
    session_id = candidate_id
    session_dir = output_root / candidate_id
    raw_path = session_dir / "raw" / "raw.csv"
    state_path = session_dir / "state" / "state.csv"
    window_path = session_dir / "window" / "window.csv"

    events = _generate_raw_events(
        session_id=session_id,
        policy=policy,
        logical_keystrokes=logical_keystrokes,
        seed=seed,
    )
    extractor = FeatureExtractorV2()
    result = extractor.extract_states(
        events,
        participant_id="attack_synthetic",
        scenario=policy.family,
        input_source="raw_synthetic_hid",
        label="attack",
    )
    windows = WindowBuilder().build(result.states)

    raw_report = validate_raw_events(events)
    state_report = validate_states(result.states)
    window_report = validate_windows(windows)
    if not raw_report.ok:
        raise ValueError(f"Generated raw events failed validation: {raw_report.issues}")
    if not state_report.ok:
        raise ValueError(f"Generated states failed validation: {state_report.issues}")
    if not window_report.ok:
        raise ValueError(f"Generated windows failed validation: {window_report.issues}")
    if not windows:
        raise ValueError(
            f"Generated candidate {candidate_id} produced zero model windows"
        )

    _write_raw_csv(events, raw_path)
    _write_state_csv(result.states, state_path)
    write_windows_csv(windows, window_path)

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "generation_level": "raw_event",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "round_name": round_name,
        "candidate_id": candidate_id,
        "session_id": session_id,
        "family": policy.family,
        "seed": seed,
        "generation": generation,
        "parent_candidate_id": parent_candidate_id,
        "policy": policy.to_dict(),
        "raw_event_count": len(events),
        "state_count": len(result.states),
        "window_count": len(windows),
        "quality_event_count": len(result.quality_events),
        "incomplete_key_count": result.incomplete_key_count,
        "trailing_completed_keystrokes_trimmed": (
            result.trailing_completed_keystrokes_trimmed
        ),
    }
    atomic_json(session_dir / "metadata.json", metadata)
    write_jsonl(
        session_dir / "quality_events.jsonl",
        (event.to_dict() for event in result.quality_events),
    )

    return GeneratedAttackSession(
        candidate_id=candidate_id,
        session_id=session_id,
        family=policy.family,
        seed=seed,
        generation=generation,
        session_dir=str(session_dir),
        raw_path=str(raw_path),
        state_path=str(state_path),
        window_path=str(window_path),
        window_count=len(windows),
        raw_event_count=len(events),
        state_count=len(result.states),
        raw_sha256=sha256_file(raw_path),
        window_sha256=sha256_file(window_path),
        policy=policy.to_dict(),
        parent_candidate_id=parent_candidate_id,
    )


def generate_raw_attack_pool(
    *,
    output_root: str | Path,
    round_name: str,
    config: RawAttackGenerationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RawAttackGenerationConfig()
    cfg.validate()
    root = Path(output_root) / round_name / "candidate_pool"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Attack candidate directory already exists and is not empty: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    parents = _load_parent_policies(cfg.parent_policy_path)
    generated: list[GeneratedAttackSession] = []

    for index in range(cfg.candidates):
        parent_candidate_id: str | None = None
        if parents:
            parent_candidate_id, base = parents[index % len(parents)]
            policy = mutate_policy(
                base,
                rng=rng,
                strength=cfg.mutation_strength,
            )
        else:
            family = SUPPORTED_FAMILIES[index % len(SUPPORTED_FAMILIES)]
            policy = mutate_policy(
                _base_policy(family),
                rng=rng,
                strength=cfg.mutation_strength,
            )

        candidate_seed = int(cfg.seed + cfg.generation * 100_003 + index * 997)
        generated.append(_generate_one_session(
            output_root=root,
            round_name=round_name,
            candidate_index=index,
            policy=policy,
            seed=candidate_seed,
            generation=cfg.generation,
            logical_keystrokes=cfg.keystrokes_per_candidate,
            parent_candidate_id=parent_candidate_id,
        ))

    manifest_path = root / "attack_manifest.jsonl"
    write_jsonl(manifest_path, (entry.to_dict() for entry in generated))
    summary = {
        "generator_version": GENERATOR_VERSION,
        "generation_level": "raw_event",
        "round_name": round_name,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "config": asdict(cfg),
        "candidate_count": len(generated),
        "total_windows": sum(entry.window_count for entry in generated),
        "families": sorted({entry.family for entry in generated}),
        "manifest_path": str(manifest_path),
    }
    summary_path = atomic_json(root / "generation_summary.json", summary)
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "candidate_count": len(generated),
        "total_windows": summary["total_windows"],
        "window_paths": [entry.window_path for entry in generated],
    }
