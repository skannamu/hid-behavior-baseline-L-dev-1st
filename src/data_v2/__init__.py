"""Framework v9 data layer for Feature Schema v2."""

from .window_dataset import (
    FeatureV2WindowDataset,
    WindowSample,
    WindowFileReport,
    load_window_csv,
    discover_window_csvs,
)
from .split_manager import (
    GroupSplit,
    GroupSplitConfig,
    split_by_group,
)
from .torch_dataset import TorchFeatureV2Dataset

__all__ = [
    "FeatureV2WindowDataset",
    "WindowSample",
    "WindowFileReport",
    "load_window_csv",
    "discover_window_csvs",
    "GroupSplit",
    "GroupSplitConfig",
    "split_by_group",
    "TorchFeatureV2Dataset",
]
