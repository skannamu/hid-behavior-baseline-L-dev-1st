from __future__ import annotations

from dataclasses import dataclass

import src.coevolution_v9.round_runner as rr


@dataclass
class DummyTrainingResult:
    run_dir: str
    defender_id: str = "D1"


def config():
    return rr.CoevolutionRoundConfig(
        method="recon_hid",
    )


def test_probe_phase_never_trains(
    tmp_path,
    monkeypatch,
):
    normal_manifest = (
        tmp_path / "normal_manifest.jsonl"
    )
    normal_manifest.write_text(
        "{}\n",
        encoding="utf-8",
    )

    called = {
        "train": 0,
    }

    def fake_generate(**kwargs):
        return {
            "manifest_path": str(
                tmp_path / "attack.jsonl"
            ),
            "candidate_count": 1,
            "total_windows": 10,
        }

    def fake_mine(**kwargs):
        return {
            "hard_window_path": str(
                tmp_path / "hard.csv"
            ),
            "parent_policy_path": str(
                tmp_path / "parents.jsonl"
            ),
            "bypass_rate": 0.1,
        }

    def fake_train(**kwargs):
        called["train"] += 1
        raise AssertionError(
            "probe must not train a defender"
        )

    monkeypatch.setattr(
        rr,
        "generate_raw_attack_pool",
        fake_generate,
    )
    monkeypatch.setattr(
        rr,
        "mine_v9_weaknesses",
        fake_mine,
    )
    monkeypatch.setattr(
        rr,
        "train_hardened_defender_v9",
        fake_train,
    )

    result = rr.probe_coevolution_round(
        round_index=0,
        parent_defender_run_dir=(
            tmp_path / "D0"
        ),
        normal_dataset_root=(
            tmp_path / "normal"
        ),
        normal_manifest_path=normal_manifest,
        output_root=(
            tmp_path / "rounds"
        ),
        config=config(),
    )

    assert called["train"] == 0
    assert result["round_index"] == 0
    assert (
        result["methodology"][
            "defender_hardened"
        ]
        is False
    )


def test_harden_phase_uses_existing_probe(
    tmp_path,
    monkeypatch,
):
    normal_manifest = (
        tmp_path / "normal_manifest.jsonl"
    )
    normal_manifest.write_text(
        "{}\n",
        encoding="utf-8",
    )

    def fake_generate(**kwargs):
        return {
            "manifest_path": str(
                tmp_path / "attack.jsonl"
            ),
            "candidate_count": 1,
            "total_windows": 10,
        }

    def fake_mine(**kwargs):
        return {
            "hard_window_path": str(
                tmp_path / "hard.csv"
            ),
            "parent_policy_path": str(
                tmp_path / "parents.jsonl"
            ),
            "bypass_rate": 0.1,
        }

    train_calls = []

    def fake_train(**kwargs):
        train_calls.append(kwargs)

        return DummyTrainingResult(
            run_dir=str(
                tmp_path / "D1"
            ),
        )

    monkeypatch.setattr(
        rr,
        "generate_raw_attack_pool",
        fake_generate,
    )
    monkeypatch.setattr(
        rr,
        "mine_v9_weaknesses",
        fake_mine,
    )
    monkeypatch.setattr(
        rr,
        "train_hardened_defender_v9",
        fake_train,
    )

    probe = rr.probe_coevolution_round(
        round_index=0,
        parent_defender_run_dir=(
            tmp_path / "D0"
        ),
        normal_dataset_root=(
            tmp_path / "normal"
        ),
        normal_manifest_path=normal_manifest,
        output_root=(
            tmp_path / "rounds"
        ),
        config=config(),
    )

    result = rr.harden_coevolution_probe(
        probe_result=probe,
        config=config(),
    )

    assert len(train_calls) == 1

    assert (
        train_calls[0]["attack_paths"]
        == probe["weakness_mining"][
            "hard_window_path"
        ]
    )

    assert (
        result["methodology"][
            "defender_hardened"
        ]
        is True
    )

    assert (
        result["next_defender_run_dir"]
        == str(tmp_path / "D1")
    )
