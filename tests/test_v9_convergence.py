from __future__ import annotations

import pytest

from src.coevolution_v9.convergence import (
    ConvergenceConfig,
    ConvergenceTracker,
)


FAMILIES = (
    "constant_fast",
    "jittered_mimic",
    "burst_pause",
    "overlap_dense",
    "shortcut_heavy",
    "correction_heavy",
)


def config(
    *,
    min_rounds=2,
    max_rounds=10,
    global_threshold=0.05,
    family_threshold=0.10,
    patience=2,
):
    return ConvergenceConfig(
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        global_bypass_threshold=global_threshold,
        family_bypass_threshold=family_threshold,
        patience=patience,
        required_families=FAMILIES,
    )


def summary(
    global_rate: float,
    *,
    family_rate: float,
    omit_family: str | None = None,
):
    payload = {}

    for family in FAMILIES:
        if family == omit_family:
            continue

        payload[family] = {
            "window_count": 100,
            "bypass_window_count": int(
                family_rate * 100
            ),
            "bypass_rate": family_rate,
        }

    return {
        "bypass_rate": global_rate,
        "family_bypass": payload,
    }


def test_requires_consecutive_passes():
    tracker = ConvergenceTracker(config())

    first = tracker.evaluate(
        probe_index=0,
        weakness_summary=summary(
            0.03,
            family_rate=0.05,
        ),
    )

    assert first.robustness_condition_met
    assert first.consecutive_passes == 1
    assert not first.should_stop

    second = tracker.evaluate(
        probe_index=1,
        weakness_summary=summary(
            0.04,
            family_rate=0.07,
        ),
    )

    assert second.consecutive_passes == 2
    assert second.should_stop
    assert second.stop_reason == "converged"


def test_failed_probe_resets_patience():
    tracker = ConvergenceTracker(config())

    tracker.evaluate(
        probe_index=0,
        weakness_summary=summary(
            0.03,
            family_rate=0.05,
        ),
    )

    failed = tracker.evaluate(
        probe_index=1,
        weakness_summary=summary(
            0.20,
            family_rate=0.05,
        ),
    )

    assert failed.consecutive_passes == 0
    assert not failed.should_stop


def test_family_failure_blocks_global_success():
    tracker = ConvergenceTracker(
        config(patience=1, min_rounds=1)
    )

    decision = tracker.evaluate(
        probe_index=0,
        weakness_summary=summary(
            0.01,
            family_rate=0.20,
        ),
    )

    assert decision.global_condition_met
    assert not decision.family_condition_met
    assert not decision.robustness_condition_met
    assert not decision.should_stop


def test_missing_family_blocks_convergence():
    tracker = ConvergenceTracker(
        config(patience=1, min_rounds=1)
    )

    decision = tracker.evaluate(
        probe_index=0,
        weakness_summary=summary(
            0.01,
            family_rate=0.01,
            omit_family="correction_heavy",
        ),
    )

    assert not decision.coverage_condition_met
    assert (
        decision.missing_required_families
        == ("correction_heavy",)
    )
    assert not decision.should_stop


def test_max_round_cap_stops_without_convergence():
    tracker = ConvergenceTracker(
        config(
            min_rounds=1,
            max_rounds=3,
            patience=2,
        )
    )

    for probe_index in range(2):
        decision = tracker.evaluate(
            probe_index=probe_index,
            weakness_summary=summary(
                0.50,
                family_rate=0.50,
            ),
        )
        assert not decision.should_stop

    final = tracker.evaluate(
        probe_index=2,
        weakness_summary=summary(
            0.50,
            family_rate=0.50,
        ),
    )

    assert final.should_stop
    assert final.stop_reason == "max_rounds_reached"


def test_thresholds_are_inclusive():
    tracker = ConvergenceTracker(
        config(
            min_rounds=1,
            patience=1,
        )
    )

    decision = tracker.evaluate(
        probe_index=0,
        weakness_summary=summary(
            0.05,
            family_rate=0.10,
        ),
    )

    assert decision.should_stop
    assert decision.stop_reason == "converged"


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ConvergenceConfig(
            min_rounds=2,
            max_rounds=1,
            global_bypass_threshold=0.05,
            family_bypass_threshold=0.10,
            patience=2,
            required_families=FAMILIES,
        ).validate()
