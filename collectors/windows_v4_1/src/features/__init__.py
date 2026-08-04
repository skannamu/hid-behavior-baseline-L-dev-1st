"""Feature Schema v2 core package."""

from .schema import (
    FEATURE_SCHEMA_VERSION,
    EXPECTED_SCHEMA_SHA256,
    PAUSE_THRESHOLD_SECONDS,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
    STRIDE,
)
from .models import RawKeyEvent, KeystrokeState, QualityEvent, BuildResult, WindowRecord
from .keystroke_builder import KeystrokeBuilder
from .window_builder import WindowBuilder, write_windows_csv
from .validators import (
    ValidationIssue,
    ValidationReport,
    validate_raw_events,
    validate_states,
    validate_windows,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "EXPECTED_SCHEMA_SHA256",
    "PAUSE_THRESHOLD_SECONDS",
    "SEQUENCE_FEATURES",
    "WINDOW_CONTEXT_FEATURES",
    "WINDOW_SIZE",
    "STRIDE",
    "RawKeyEvent",
    "KeystrokeState",
    "QualityEvent",
    "BuildResult",
    "WindowRecord",
    "KeystrokeBuilder",
    "WindowBuilder",
    "write_windows_csv",
    "ValidationIssue",
    "ValidationReport",
    "validate_raw_events",
    "validate_states",
    "validate_windows",
]
