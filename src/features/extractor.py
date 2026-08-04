"""Convenience façade for raw-event to state extraction."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .keystroke_builder import KeystrokeBuilder
from .models import BuildResult, RawKeyEvent


class FeatureExtractorV2:
    """Thin façade kept stable for collectors, generators, and offline jobs."""

    def __init__(self, builder: KeystrokeBuilder | None = None) -> None:
        self.builder = builder or KeystrokeBuilder()

    def extract_states(
        self,
        events: Iterable[RawKeyEvent | Mapping[str, Any]],
        *,
        participant_id: str = "",
        scenario: str = "",
        input_source: str = "",
        label: str = "",
    ) -> BuildResult:
        return self.builder.build(
            events,
            participant_id=participant_id,
            scenario=scenario,
            input_source=input_source,
            label=label,
        )
