from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import src.coevolution_v9.experiment_loop as loop
from src.coevolution_v9.convergence import (
    ConvergenceConfig,
)
from src.coevolution_v9.round_runner import (
    CoevolutionRoundConfig,
)
from src.coevolution_v9.weakness_miner import (
    WeaknessMiningConfig,
)


FAMILIES = (
    "constant_fast",
    "jittered_mimic",
    "burst_pause",
    "overlap_dense",
    "shortcut_heavy",
    "correction_heavy",
)


def family_payload(rate: float):
    return {
        family: {
            "window_count": 10,
            "bypass_window_count": int(
                rate * 10
            ),
            "bypass_rate": rate,
        }
        for family in FAMILIES
    }


def base_config():
    return CoevolutionRoundConfig(
        method="recon_hid",
        weakness_mining=WeaknessMiningConfig(
            selection_mode="guided",
            parent_selection_mode="guided",
        ),
    )


def convergence(
    *,
    min_rounds=1,
    max_rounds=5,
    patience=1,
):
    return ConvergenceConfig(
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        global_bypass_threshold=0.05,
        family_bypass_threshold=0.10,
        patience=patience,
        required_families=FAMILIES,
    )


def fake_probe(
    *,
    round_index,
    parent_defender_run_dir,
    output_root,
    bypass_rate,
):
    root = (
        Path(output_root)
        / f"round_{round_index:02d}"
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "experiment_method": "recon_hid",
        "round_index": round_index,
        "round_dir": str(root),
        "probe_manifest_path": str(
            root / "probe_manifest.json"
        ),
        "parent_defender_run_dir": str(
            parent_defender_run_dir
        ),
        "normal_dataset_root": "normal",
        "normal_manifest_path": "manifest",
        "normal_manifest_sha256": "x",
        "parent_policy_path": None,
        "config": {},
        "attack_generation": {},
        "methodology": {},
        "next_parent_policy_path": str(
            root / "parents.jsonl"
        ),
        "weakness_mining": {
            "attack_window_count": 600,
            "bypass_rate": bypass_rate,
            "family_bypass": (
                family_payload(
                    bypass_rate
                )
            ),
            "worst_family_bypass_rate": (
                bypass_rate
            ),
            "hard_negative_count": 192,
            "selected_parent_session_count": 8,
            "parent_policy_path": str(
                root / "parents.jsonl"
            ),
            "hard_window_path": str(
                root / "hard.csv"
            ),
        },
    }


def test_convergence_can_freeze_d0_without_hardening(
    tmp_path,
    monkeypatch,
):
    harden_calls = []

    def probe(**kwargs):
        return fake_probe(
            round_index=kwargs[
                "round_index"
            ],
            parent_defender_run_dir=kwargs[
                "parent_defender_run_dir"
            ],
            output_root=kwargs[
                "output_root"
            ],
            bypass_rate=0.01,
        )

    def harden(**kwargs):
        harden_calls.append(kwargs)
        raise AssertionError(
            "converged defender must not harden"
        )

    monkeypatch.setattr(
        loop,
        "probe_coevolution_round",
        probe,
    )
    monkeypatch.setattr(
        loop,
        "harden_coevolution_probe",
        harden,
    )

    result = loop.run_experiment_loop(
        normal_dataset_root="normal",
        normal_manifest_path="manifest",
        d0_run_dir=tmp_path / "D0",
        output_root=tmp_path / "out",
        base_round_config=base_config(),
        loop_config=loop.ExperimentLoopConfig(
            stopping_mode="convergence",
            seed=1,
            convergence=convergence(),
        ),
    )

    assert result["converged"] is True
    assert result["stop_reason"] == "converged"
    assert (
        result["final_defender_run_dir"]
        == str(
            (tmp_path / "D0").resolve()
        )
    )
    assert len(harden_calls) == 0


def test_failed_probe_hardens_then_next_defender_can_converge(
    tmp_path,
    monkeypatch,
):
    rates = iter([0.50, 0.01])
    harden_count = 0

    def probe(**kwargs):
        return fake_probe(
            round_index=kwargs[
                "round_index"
            ],
            parent_defender_run_dir=kwargs[
                "parent_defender_run_dir"
            ],
            output_root=kwargs[
                "output_root"
            ],
            bypass_rate=next(rates),
        )

    def harden(**kwargs):
        nonlocal harden_count
        harden_count += 1

        root = Path(
            kwargs[
                "probe_result"
            ]["round_dir"]
        )

        return {
            "round_manifest_path": str(
                root / "round_manifest.json"
            ),
            "next_defender_run_dir": str(
                tmp_path / "D1"
            ),
            "next_parent_policy_path": str(
                root / "parents.jsonl"
            ),
        }

    monkeypatch.setattr(
        loop,
        "probe_coevolution_round",
        probe,
    )
    monkeypatch.setattr(
        loop,
        "harden_coevolution_probe",
        harden,
    )

    result = loop.run_experiment_loop(
        normal_dataset_root="normal",
        normal_manifest_path="manifest",
        d0_run_dir=tmp_path / "D0",
        output_root=tmp_path / "out",
        base_round_config=base_config(),
        loop_config=loop.ExperimentLoopConfig(
            stopping_mode="convergence",
            seed=1,
            convergence=convergence(),
        ),
    )

    assert harden_count == 1
    assert result["converged"] is True

    assert (
        result["final_defender_run_dir"]
        == str(
            (tmp_path / "D1").resolve()
        )
    )


def test_max_round_cap_does_not_claim_dfinal(
    tmp_path,
    monkeypatch,
):
    harden_count = 0

    def probe(**kwargs):
        return fake_probe(
            round_index=kwargs[
                "round_index"
            ],
            parent_defender_run_dir=kwargs[
                "parent_defender_run_dir"
            ],
            output_root=kwargs[
                "output_root"
            ],
            bypass_rate=0.50,
        )

    def harden(**kwargs):
        nonlocal harden_count
        harden_count += 1

        root = Path(
            kwargs[
                "probe_result"
            ]["round_dir"]
        )

        return {
            "round_manifest_path": str(
                root / "round_manifest.json"
            ),
            "next_defender_run_dir": str(
                tmp_path
                / f"D{harden_count}"
            ),
            "next_parent_policy_path": str(
                root / "parents.jsonl"
            ),
        }

    monkeypatch.setattr(
        loop,
        "probe_coevolution_round",
        probe,
    )
    monkeypatch.setattr(
        loop,
        "harden_coevolution_probe",
        harden,
    )

    result = loop.run_experiment_loop(
        normal_dataset_root="normal",
        normal_manifest_path="manifest",
        d0_run_dir=tmp_path / "D0",
        output_root=tmp_path / "out",
        base_round_config=base_config(),
        loop_config=loop.ExperimentLoopConfig(
            stopping_mode="convergence",
            seed=1,
            convergence=convergence(
                max_rounds=2,
            ),
        ),
    )

    # Probe D0 -> harden D1 -> probe D1 -> cap.
    assert harden_count == 1

    assert result["converged"] is False

    assert (
        result["stop_reason"]
        == "max_rounds_reached"
    )

    assert (
        result["final_defender_run_dir"]
        is None
    )


def test_method_profile_mismatch_rejected(
    tmp_path,
):
    bad = CoevolutionRoundConfig(
        method="random_iterative",
        weakness_mining=(
            WeaknessMiningConfig(
                selection_mode="guided",
                parent_selection_mode="guided",
            )
        ),
    )

    import pytest

    with pytest.raises(ValueError):
        loop.run_experiment_loop(
            normal_dataset_root="normal",
            normal_manifest_path="manifest",
            d0_run_dir=tmp_path / "D0",
            output_root=tmp_path / "out",
            base_round_config=bad,
            loop_config=(
                loop.ExperimentLoopConfig(
                    stopping_mode="fixed",
                    fixed_rounds=1,
                    seed=1,
                )
            ),
        )
