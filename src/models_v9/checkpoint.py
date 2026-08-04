"""Schema-bound checkpoint contract for ReCon-HID v9."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_FEATURES,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
)
from .recon_hid_v9 import ReConHIDV9, ReConHIDV9Config


CHECKPOINT_FORMAT_VERSION = "recon_hid_v9_checkpoint_v1"


def checkpoint_contract(config: ReConHIDV9Config) -> dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "window_size": WINDOW_SIZE,
        "sequence_features": list(SEQUENCE_FEATURES),
        "window_context_features": list(WINDOW_CONTEXT_FEATURES),
        "model_config": config.to_dict(),
    }


def validate_checkpoint_contract(payload: Mapping[str, Any]) -> None:
    expected = checkpoint_contract(ReConHIDV9Config())

    scalar_fields = (
        "checkpoint_format_version",
        "feature_schema_version",
        "schema_hash",
        "window_size",
    )
    for name in scalar_fields:
        if payload.get(name) != expected[name]:
            raise ValueError(
                f"Checkpoint contract mismatch for {name}: "
                f"expected={expected[name]!r}, actual={payload.get(name)!r}"
            )

    if list(payload.get("sequence_features", [])) != list(SEQUENCE_FEATURES):
        raise ValueError("Checkpoint sequence feature order mismatch")
    if list(payload.get("window_context_features", [])) != list(
        WINDOW_CONTEXT_FEATURES
    ):
        raise ValueError("Checkpoint context feature order mismatch")


def save_v9_checkpoint(
    path: str | Path,
    *,
    model: ReConHIDV9,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        **checkpoint_contract(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "epoch": epoch,
        "extra": dict(extra or {}),
    }
    torch.save(payload, output)
    return output


def load_v9_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ReConHIDV9, dict[str, Any]]:
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")

    validate_checkpoint_contract(payload)

    config = ReConHIDV9Config(**payload["model_config"])
    model = ReConHIDV9(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    if optimizer is not None and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return model, payload
