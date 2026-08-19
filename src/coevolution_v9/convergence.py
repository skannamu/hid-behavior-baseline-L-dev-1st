"""Convergence policy for iterative ReCon-HID defender hardening."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ConvergenceConfig:
    """Explicit convergence policy.

    No research-protocol thresholds have defaults intentionally.
    Values must be predeclared by the experiment configuration.
    """

    min_rounds: int
    max_rounds: int
    global_bypass_threshold: float
    family_bypass_threshold: float
    patience: int
    required_families: tuple[str, ...]

    def validate(self) -> None:
        if self.min_rounds <= 0:
            raise ValueError("min_rounds must be positive")

        if self.max_rounds < self.min_rounds:
            raise ValueError(
                "max_rounds must be >= min_rounds"
            )

        if self.patience <= 0:
            raise ValueError("patience must be positive")

        for name in (
            "global_bypass_threshold",
            "family_bypass_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1]"
                )

        if not self.required_families:
            raise ValueError(
                "required_families must not be empty"
            )

        if len(set(self.required_families)) != len(
            self.required_families
        ):
            raise ValueError(
                "required_families must not contain duplicates"
            )


@dataclass(frozen=True)
class ConvergenceDecision:
    probe_index: int
    completed_rounds: int

    global_bypass_rate: float
    worst_family_bypass_rate: float

    observed_families: tuple[str, ...]
    missing_required_families: tuple[str, ...]

    global_condition_met: bool
    family_condition_met: bool
    coverage_condition_met: bool
    robustness_condition_met: bool

    consecutive_passes: int

    should_stop: bool
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConvergenceTracker:
    """Stateful patience tracker over A_k probes."""

    config: ConvergenceConfig
    consecutive_passes: int = 0

    def __post_init__(self) -> None:
        self.config.validate()

    def evaluate(
        self,
        *,
        probe_index: int,
        weakness_summary: dict[str, Any],
    ) -> ConvergenceDecision:
        if probe_index < 0:
            raise ValueError(
                "probe_index must be non-negative"
            )

        completed_rounds = probe_index + 1

        global_rate = float(
            weakness_summary["bypass_rate"]
        )

        family_payload = weakness_summary.get(
            "family_bypass"
        )
        if not isinstance(family_payload, dict):
            raise ValueError(
                "weakness summary is missing family_bypass"
            )

        observed = tuple(
            sorted(str(name) for name in family_payload)
        )

        required = set(self.config.required_families)
        observed_set = set(observed)

        missing = tuple(sorted(required - observed_set))

        family_rates = {
            str(family): float(stats["bypass_rate"])
            for family, stats in family_payload.items()
        }

        required_rates = [
            family_rates[family]
            for family in self.config.required_families
            if family in family_rates
        ]

        worst_family_rate = (
            max(required_rates)
            if required_rates
            else 1.0
        )

        global_ok = (
            global_rate
            <= self.config.global_bypass_threshold
        )

        coverage_ok = not missing

        family_ok = (
            coverage_ok
            and all(
                family_rates[family]
                <= self.config.family_bypass_threshold
                for family in self.config.required_families
            )
        )

        robustness_ok = (
            global_ok
            and coverage_ok
            and family_ok
        )

        if robustness_ok:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        enough_rounds = (
            completed_rounds
            >= self.config.min_rounds
        )

        patience_met = (
            self.consecutive_passes
            >= self.config.patience
        )

        converged = (
            enough_rounds
            and patience_met
        )

        reached_cap = (
            completed_rounds
            >= self.config.max_rounds
        )

        if converged:
            should_stop = True
            stop_reason = "converged"
        elif reached_cap:
            should_stop = True
            stop_reason = "max_rounds_reached"
        else:
            should_stop = False
            stop_reason = None

        return ConvergenceDecision(
            probe_index=probe_index,
            completed_rounds=completed_rounds,
            global_bypass_rate=global_rate,
            worst_family_bypass_rate=worst_family_rate,
            observed_families=observed,
            missing_required_families=missing,
            global_condition_met=global_ok,
            family_condition_met=family_ok,
            coverage_condition_met=coverage_ok,
            robustness_condition_met=robustness_ok,
            consecutive_passes=self.consecutive_passes,
            should_stop=should_stop,
            stop_reason=stop_reason,
        )


def summarize_convergence_history(
    decisions: Sequence[ConvergenceDecision],
) -> dict[str, Any]:
    if not decisions:
        raise ValueError(
            "convergence history must not be empty"
        )

    final = decisions[-1]

    return {
        "probe_count": len(decisions),
        "final_decision": final.to_dict(),
        "converged": (
            final.stop_reason == "converged"
        ),
        "stop_reason": final.stop_reason,
        "history": [
            decision.to_dict()
            for decision in decisions
        ],
    }
