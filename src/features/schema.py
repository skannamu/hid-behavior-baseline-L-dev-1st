"""Single source of truth for Feature Schema v2 constants."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Iterable

FEATURE_SCHEMA_VERSION: Final[str] = "2.0.0"
EXPECTED_SCHEMA_SHA256: Final[str] = (
    "665802343c1e4fc4b7e94974a36879483fe98733c66ea82afd13f132817cd0f2"
)

WINDOW_SIZE: Final[int] = 50
STRIDE: Final[int] = 1
PAUSE_THRESHOLD_SECONDS: Final[float] = 0.5
ROBUST_CV_EPSILON: Final[float] = 1.0e-6

SEQUENCE_FEATURES: Final[tuple[str, ...]] = (
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
)

WINDOW_CONTEXT_FEATURES: Final[tuple[str, ...]] = (
    "press_rate_hz",
    "active_press_interval_median_s",
    "active_press_interval_robust_cv",
    "pause_rate",
    "pause_time_fraction",
    "max_pause_interval_s",
    "hold_time_robust_cv",
    "max_burst_length_keys",
    "correction_rate",
    "command_shortcut_rate",
    "overlap_key_rate",
    "release_inversion_rate",
)

WINDOW_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "feature_schema_version",
    "schema_hash",
    "participant_id",
    "session_id",
    "window_id",
    "start_keystroke_index",
    "end_keystroke_index",
    "label",
    "scenario",
    "input_source",
)

BINARY_SEQUENCE_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "release_inversion_flag",
        "correction_key_flag",
        "repeat_flag",
        "shift_at_press",
        "ctrl_at_press",
        "alt_at_press",
        "meta_at_press",
    }
)

UNIT_INTERVAL_SEQUENCE_FEATURES: Final[frozenset[str]] = frozenset(
    {"overlap_fraction"}
)

UNIT_INTERVAL_CONTEXT_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "pause_rate",
        "pause_time_fraction",
        "correction_rate",
        "command_shortcut_rate",
        "overlap_key_rate",
        "release_inversion_rate",
    }
)


def compute_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file."""
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_schema_file(path: str | Path) -> str:
    """Fail fast when the installed YAML differs from this implementation."""
    actual = compute_sha256(path)
    if actual != EXPECTED_SCHEMA_SHA256:
        raise ValueError(
            "Feature schema hash mismatch: "
            f"expected={EXPECTED_SCHEMA_SHA256}, actual={actual}, path={path}"
        )
    return actual


def sequence_column_names(window_size: int = WINDOW_SIZE) -> list[str]:
    return [
        f"t{t}_{feature}"
        for t in range(window_size)
        for feature in SEQUENCE_FEATURES
    ]


def context_column_names() -> list[str]:
    return [f"ctx_{feature}" for feature in WINDOW_CONTEXT_FEATURES]


def expected_window_columns(window_size: int = WINDOW_SIZE) -> list[str]:
    return [
        *WINDOW_METADATA_FIELDS,
        *sequence_column_names(window_size),
        *context_column_names(),
    ]


def require_exact_feature_order(features: Iterable[str]) -> None:
    supplied = tuple(features)
    if supplied != SEQUENCE_FEATURES:
        raise ValueError(
            "Sequence feature order mismatch. "
            f"expected={SEQUENCE_FEATURES}, supplied={supplied}"
        )
