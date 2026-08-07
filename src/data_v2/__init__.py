"""Framework v9 data layer for Feature Schema v2."""

from .window_dataset import (
    FeatureV2WindowDataset,
    WindowSample,
    WindowFileReport,
    load_window_csv,
    discover_window_csvs,
)
from .split_manager import (
    SUPPORTED_GROUP_MODES,
    GroupSplit,
    GroupSplitConfig,
    group_key_for_mode,
    split_by_group,
)
from .balanced_sampler import (
    SUPPORTED_BALANCE_MODES,
    BalancedSamplingConfig,
    BalancedSamplingResult,
    balanced_subsample,
    participant_scenario_key,
)
from .dataset_manifest import (
    MANIFEST_FORMAT_VERSION,
    DatasetManifest,
    ManifestEntry,
    build_dataset_manifest,
    load_manifest_entries,
    participant_core_completion,
    resolve_manifest_window_paths,
    write_dataset_manifest,
)
from .torch_dataset import TorchFeatureV2Dataset
from .session_validator import (
    CORE_SCENARIOS,
    SCENARIO_PROTOCOL,
    DatasetValidationReport,
    SessionValidationConfig,
    SessionValidationResult,
    ValidationIssue,
    discover_session_dirs,
    validate_dataset_root,
    validate_session_dir,
)

__all__ = [
    "FeatureV2WindowDataset",
    "WindowSample",
    "WindowFileReport",
    "load_window_csv",
    "discover_window_csvs",
    "SUPPORTED_GROUP_MODES",
    "GroupSplit",
    "GroupSplitConfig",
    "group_key_for_mode",
    "split_by_group",
    "SUPPORTED_BALANCE_MODES",
    "BalancedSamplingConfig",
    "BalancedSamplingResult",
    "balanced_subsample",
    "participant_scenario_key",
    "MANIFEST_FORMAT_VERSION",
    "DatasetManifest",
    "ManifestEntry",
    "build_dataset_manifest",
    "load_manifest_entries",
    "participant_core_completion",
    "resolve_manifest_window_paths",
    "write_dataset_manifest",
    "TorchFeatureV2Dataset",
    "CORE_SCENARIOS",
    "SCENARIO_PROTOCOL",
    "DatasetValidationReport",
    "SessionValidationConfig",
    "SessionValidationResult",
    "ValidationIssue",
    "discover_session_dirs",
    "validate_dataset_root",
    "validate_session_dir",
]
