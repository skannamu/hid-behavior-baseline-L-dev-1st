from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.coevolution_v9.final_evaluator as fe


def write_json(path: Path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_attack(
    path: Path,
    *,
    candidate_id: str,
    seed: int,
    family: str,
    window_hash: str,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps({
            "candidate_id": candidate_id,
            "seed": seed,
            "family": family,
            "window_sha256": window_hash,
        }) + "\n",
        encoding="utf-8",
    )


def test_convergence_lineage_includes_terminal_probe(
    tmp_path,
):
    attack0 = tmp_path / "A0.jsonl"
    attack1 = tmp_path / "A1.jsonl"

    write_attack(
        attack0,
        candidate_id="A0_c0",
        seed=100,
        family="constant_fast",
        window_hash="hash_A0",
    )

    write_attack(
        attack1,
        candidate_id="A1_c0",
        seed=200,
        family="jittered_mimic",
        window_hash="hash_A1",
    )

    probe0 = tmp_path / "round_00" / "probe_manifest.json"
    probe1 = tmp_path / "round_01" / "probe_manifest.json"

    round0 = tmp_path / "round_00" / "round_manifest.json"

    write_json(
        probe0,
        {
            "attack_generation": {
                "manifest_path": str(
                    attack0
                ),
            },
        },
    )

    write_json(
        probe1,
        {
            "attack_generation": {
                "manifest_path": str(
                    attack1
                ),
            },
        },
    )

    write_json(
        round0,
        {
            "attack_generation": {
                "manifest_path": str(
                    attack0
                ),
            },
        },
    )

    d1 = tmp_path / "D1"
    d1.mkdir()

    timeline = (
        tmp_path
        / "coevolution_timeline.json"
    )

    write_json(
        timeline,
        {
            "status": "PASS",
            "stopping_mode": "convergence",
            "converged": True,
            "final_defender_run_dir": str(
                d1
            ),
            "timeline": [
                {
                    "round_index": 0,
                    "probe_manifest_path": str(
                        probe0
                    ),
                    "round_manifest_path": str(
                        round0
                    ),
                },
                {
                    "round_index": 1,
                    "probe_manifest_path": str(
                        probe1
                    ),
                    "round_manifest_path": None,
                },
            ],
        },
    )

    result = fe._collect_training_lineage(
        timeline
    )

    assert result["round_count"] == 2

    assert len(
        result["probe_manifest_paths"]
    ) == 2

    assert len(
        result["round_manifest_paths"]
    ) == 1

    # Crucial: terminal A1 influenced convergence,
    # therefore it belongs to the excluded lineage.
    assert result["candidate_ids"] == [
        "A0_c0",
        "A1_c0",
    ]

    assert result["seeds"] == [
        100,
        200,
    ]

    assert result["families"] == [
        "constant_fast",
        "jittered_mimic",
    ]

    assert result["window_hashes"] == [
        "hash_A0",
        "hash_A1",
    ]

    assert (
        Path(
            result[
                "latest_defender_run_dir"
            ]
        ).resolve()
        == d1.resolve()
    )


def test_unconverged_timeline_cannot_supply_dfinal(
    tmp_path,
):
    attack = tmp_path / "A0.jsonl"

    write_attack(
        attack,
        candidate_id="A0_c0",
        seed=100,
        family="constant_fast",
        window_hash="hash",
    )

    probe = (
        tmp_path
        / "round_00"
        / "probe_manifest.json"
    )

    write_json(
        probe,
        {
            "attack_generation": {
                "manifest_path": str(
                    attack
                ),
            },
        },
    )

    timeline = (
        tmp_path
        / "coevolution_timeline.json"
    )

    write_json(
        timeline,
        {
            "status": "PASS",
            "stopping_mode": "convergence",
            "converged": False,
            "stop_reason": "max_rounds_reached",
            "final_defender_run_dir": None,
            "terminal_defender_run_dir": str(
                tmp_path / "D9"
            ),
            "timeline": [
                {
                    "round_index": 0,
                    "probe_manifest_path": str(
                        probe
                    ),
                    "round_manifest_path": None,
                },
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="no converged final defender",
    ):
        fe._collect_training_lineage(
            timeline
        )
