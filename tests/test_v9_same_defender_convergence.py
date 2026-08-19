from __future__ import annotations

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


def fake_probe_result(
    *,
    tmp_path,
    round_index,
    parent_defender_run_dir,
    rate,
    probe_label,
):
    root = (
        tmp_path
        / f"round_{round_index:02d}_{probe_label}"
    )

    return {
        "round_index": round_index,
        "round_dir": str(root),
        "probe_manifest_path": str(
            root / "probe_manifest.json"
        ),
        "weakness_mining": {
            "attack_window_count": 60,
            "bypass_rate": rate,
            "family_bypass": family_payload(
                rate
            ),
            "worst_family_bypass_rate": rate,
            "hard_negative_count": 24,
            "selected_parent_session_count": 6,
        },
        "parent_defender_run_dir": str(
            parent_defender_run_dir
        ),
    }


def test_patience_probes_same_defender_before_freeze(
    tmp_path,
    monkeypatch,
):
    seen_defenders = []
    harden_calls = []

    def fake_probe(**kwargs):
        seen_defenders.append(
            str(
                Path(
                    kwargs[
                        "parent_defender_run_dir"
                    ]
                ).resolve()
            )
        )

        return fake_probe_result(
            tmp_path=tmp_path,
            round_index=kwargs["round_index"],
            parent_defender_run_dir=kwargs[
                "parent_defender_run_dir"
            ],
            rate=0.01,
            probe_label=kwargs[
                "probe_label"
            ],
        )

    def fake_harden(**kwargs):
        harden_calls.append(kwargs)
        raise AssertionError(
            "A defender that passes both fresh probes "
            "must not be hardened"
        )

    monkeypatch.setattr(
        loop,
        "probe_coevolution_round",
        fake_probe,
    )
    monkeypatch.setattr(
        loop,
        "harden_coevolution_probe",
        fake_harden,
    )

    d0 = (tmp_path / "D0").resolve()

    result = loop.run_experiment_loop(
        normal_dataset_root="normal",
        normal_manifest_path="manifest",
        d0_run_dir=d0,
        output_root=tmp_path / "out",
        base_round_config=base_config(),
        loop_config=loop.ExperimentLoopConfig(
            stopping_mode="convergence",
            seed=1,
            convergence=ConvergenceConfig(
                min_rounds=1,
                max_rounds=3,
                global_bypass_threshold=0.05,
                family_bypass_threshold=0.10,
                patience=2,
                required_families=FAMILIES,
            ),
        ),
    )

    assert result["converged"] is True
    assert result["final_defender_run_dir"] == str(
        d0
    )

    assert seen_defenders == [
        str(d0),
        str(d0),
    ]

    assert len(harden_calls) == 0

    assert [
        row["verification_index"]
        for row in result["timeline"]
    ] == [0, 1]


def test_failed_verification_hardens_then_resets_patience(
    tmp_path,
    monkeypatch,
):
    rates = iter([
        0.01,  # D0 primary PASS
        0.50,  # same D0 verification FAIL
        0.01,  # D1 primary PASS
        0.01,  # same D1 verification PASS
    ])

    seen_defenders = []
    harden_count = 0

    def fake_probe(**kwargs):
        rate = next(rates)

        seen_defenders.append(
            Path(
                kwargs[
                    "parent_defender_run_dir"
                ]
            ).name
        )

        return fake_probe_result(
            tmp_path=tmp_path,
            round_index=kwargs["round_index"],
            parent_defender_run_dir=kwargs[
                "parent_defender_run_dir"
            ],
            rate=rate,
            probe_label=kwargs[
                "probe_label"
            ],
        )

    def fake_harden(**kwargs):
        nonlocal harden_count
        harden_count += 1

        return {
            "round_manifest_path": str(
                tmp_path
                / "round_manifest.json"
            ),
            "next_defender_run_dir": str(
                tmp_path / "D1"
            ),
            "next_parent_policy_path": str(
                tmp_path / "parents.jsonl"
            ),
        }

    monkeypatch.setattr(
        loop,
        "probe_coevolution_round",
        fake_probe,
    )
    monkeypatch.setattr(
        loop,
        "harden_coevolution_probe",
        fake_harden,
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
            convergence=ConvergenceConfig(
                min_rounds=1,
                max_rounds=3,
                global_bypass_threshold=0.05,
                family_bypass_threshold=0.10,
                patience=2,
                required_families=FAMILIES,
            ),
        ),
    )

    assert seen_defenders == [
        "D0",
        "D0",
        "D1",
        "D1",
    ]

    assert harden_count == 1
    assert result["converged"] is True

    assert Path(
        result["final_defender_run_dir"]
    ).name == "D1"

    assert [
        row["verification_index"]
        for row in result["timeline"]
    ] == [0, 1, 0, 1]
