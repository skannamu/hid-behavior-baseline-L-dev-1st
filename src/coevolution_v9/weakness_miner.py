"""Mine Stable-v9 hard negatives without modifying feature columns."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data_v2.window_dataset import FeatureV2WindowDataset, load_window_csv
from src.features.schema import WINDOW_SIZE, expected_window_columns

from .bundle import DefenderBundle, load_defender_bundle
from .common import atomic_json, sha256_file, write_csv_rows, write_jsonl
from .selection_strategies import (
    random_diverse_select,
    random_parent_entries,
)


@dataclass(frozen=True)
class WeaknessMiningConfig:
    max_hard_windows: int = 192
    max_parent_sessions: int = 8
    include_detected_fillers: bool = True

    selection_mode: str = "guided"
    parent_selection_mode: str = "guided"
    selection_seed: int = 20260807
    require_full_budget: bool = False

    batch_size: int = 256
    device: str = "cpu"

    def validate(self) -> None:
        if self.max_hard_windows <= 0:
            raise ValueError("max_hard_windows must be positive")
        if self.max_parent_sessions <= 0:
            raise ValueError("max_parent_sessions must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.selection_mode not in {"guided", "random"}:
            raise ValueError(
                "selection_mode must be one of {'guided', 'random'}"
            )

        if self.parent_selection_mode not in {
            "guided",
            "random",
            "none",
        }:
            raise ValueError(
                "parent_selection_mode must be one of "
                "{'guided', 'random', 'none'}"
            )


def _load_attack_manifest(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number}: entry must be an object")
            candidate_id = str(payload.get("candidate_id", ""))
            if not candidate_id:
                raise ValueError(f"{source}:{line_number}: missing candidate_id")
            if candidate_id in seen:
                raise ValueError(f"Duplicate candidate_id in manifest: {candidate_id}")
            seen.add(candidate_id)
            window_path = Path(str(payload.get("window_path", "")))
            if not window_path.is_file():
                raise FileNotFoundError(window_path)
            expected_hash = str(payload.get("window_sha256", ""))
            if expected_hash and sha256_file(window_path) != expected_hash:
                raise ValueError(f"Attack window hash mismatch: {window_path}")
            entries.append(payload)
    if not entries:
        raise ValueError(f"Attack manifest is empty: {source}")
    return entries


def _read_exact_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = expected_window_columns(WINDOW_SIZE)
        if (reader.fieldnames or []) != expected:
            raise ValueError(f"Stable window header mismatch: {path}")
        return list(reader)


def _score_manifest_entries(
    bundle: DefenderBundle,
    entries: list[dict[str, Any]],
    *,
    config: WeaknessMiningConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prediction_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []

    for entry in entries:
        path = Path(str(entry["window_path"]))
        samples, report = load_window_csv(path)
        dataset = FeatureV2WindowDataset(samples)
        scores = bundle.score_dataset(
            dataset,
            device=config.device,
            batch_size=config.batch_size,
        )
        rows = _read_exact_rows(path)
        if len(rows) != report.row_count or len(rows) != len(scores):
            raise ValueError(f"Attack row/score count mismatch: {path}")

        for index, (sample, row) in enumerate(zip(samples, rows, strict=True)):
            global_index = len(source_rows)
            source_rows.append(row)
            prediction_rows.append({
                "global_index": global_index,
                "candidate_id": entry["candidate_id"],
                "family": entry.get("family", sample.scenario),
                "generation": entry.get("generation", 0),
                "parent_candidate_id": entry.get("parent_candidate_id"),
                "session_id": sample.session_id,
                "window_id": sample.window_id,
                "source_window_path": str(path),
                "source_row_number": sample.row_number,
                "sequence_reconstruction": float(
                    scores.sequence_reconstruction[index]
                ),
                "context_reconstruction": float(
                    scores.context_reconstruction[index]
                ),
                "reconstruction_combined": float(
                    scores.reconstruction_combined[index]
                ),
                "classifier_probability": float(
                    scores.classifier_probability[index]
                ),
                "prototype_distance": float(scores.prototype_distance[index]),
                "decision_score": float(scores.decision_score[index]),
                "predicted_attack": int(scores.predicted_attack[index]),
                "is_bypass": int(scores.predicted_attack[index] == 0),
            })

    return prediction_rows, source_rows


def _diverse_select(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    include_detected_fillers: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row["is_bypass"]),
            float(row["decision_score"]),
            str(row["family"]),
            str(row["candidate_id"]),
            int(row["window_id"]),
        ),
    )
    bypasses = [row for row in ordered if int(row["is_bypass"]) == 1]
    candidate_pool = bypasses
    if include_detected_fillers and len(candidate_pool) < limit:
        candidate_pool = ordered

    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_pool:
        by_session.setdefault(str(row["candidate_id"]), []).append(row)

    selected: list[dict[str, Any]] = []
    positions = {key: 0 for key in by_session}
    session_order = sorted(
        by_session,
        key=lambda key: (
            -sum(int(row["is_bypass"]) for row in by_session[key]),
            np.mean([float(row["decision_score"]) for row in by_session[key]]),
            key,
        ),
    )
    while len(selected) < min(limit, len(candidate_pool)):
        progressed = False
        for session in session_order:
            position = positions[session]
            if position >= len(by_session[session]):
                continue
            selected.append(by_session[session][position])
            positions[session] = position + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _select_parent_entries(
    entries: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
    entry_by_candidate = {str(entry["candidate_id"]): entry for entry in entries}

    ranked = []
    for candidate_id, rows in by_candidate.items():
        bypass_rate = float(np.mean([int(row["is_bypass"]) for row in rows]))
        mean_score = float(np.mean([float(row["decision_score"]) for row in rows]))
        minimum_score = float(min(float(row["decision_score"]) for row in rows))
        ranked.append((
            -bypass_rate,
            mean_score,
            minimum_score,
            candidate_id,
            {
                **entry_by_candidate[candidate_id],
                "bypass_rate": bypass_rate,
                "mean_decision_score": mean_score,
                "minimum_decision_score": minimum_score,
            },
        ))
    ranked.sort(key=lambda item: item[:4])
    return [item[4] for item in ranked[:limit]]


def _family_bypass_statistics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate bypass statistics without changing selection behavior."""
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        family = str(row["family"])
        grouped.setdefault(family, []).append(row)

    statistics: dict[str, dict[str, float | int]] = {}

    for family in sorted(grouped):
        family_rows = grouped[family]
        window_count = len(family_rows)
        bypass_count = sum(
            int(row["is_bypass"])
            for row in family_rows
        )

        statistics[family] = {
            "window_count": window_count,
            "bypass_window_count": bypass_count,
            "bypass_rate": (
                bypass_count / window_count
                if window_count
                else 0.0
            ),
        }

    return statistics


def mine_v9_weaknesses(
    *,
    defender_run_dir: str | Path,
    attack_manifest_path: str | Path,
    output_dir: str | Path,
    config: WeaknessMiningConfig | None = None,
) -> dict[str, Any]:
    cfg = config or WeaknessMiningConfig()
    cfg.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    bundle = load_defender_bundle(
        defender_run_dir,
        map_location=cfg.device,
    )
    entries = _load_attack_manifest(attack_manifest_path)
    prediction_rows, source_rows = _score_manifest_entries(
        bundle,
        entries,
        config=cfg,
    )

    family_bypass = _family_bypass_statistics(
        prediction_rows
    )
    worst_family_bypass_rate = max(
        (
            float(stats["bypass_rate"])
            for stats in family_bypass.values()
        ),
        default=0.0,
    )
    if cfg.selection_mode == "guided":
        selected = _diverse_select(
            prediction_rows,
            limit=cfg.max_hard_windows,
            include_detected_fillers=cfg.include_detected_fillers,
        )
    else:
        selected = random_diverse_select(
            prediction_rows,
            limit=cfg.max_hard_windows,
            seed=cfg.selection_seed,
        )

    if not selected:
        raise ValueError("Weakness mining selected zero windows")

    if (
        cfg.require_full_budget
        and len(selected) != cfg.max_hard_windows
    ):
        raise ValueError(
            "Hard-negative budget was not satisfied: "
            f"selected={len(selected)}, "
            f"required={cfg.max_hard_windows}"
        )

    predictions_path = write_csv_rows(
        output / "attack_predictions.csv",
        fieldnames=list(prediction_rows[0]),
        rows=prediction_rows,
    )
    selected_source_rows = [source_rows[int(row["global_index"])] for row in selected]
    hard_window_path = write_csv_rows(
        output / "hard_negatives" / "window.csv",
        fieldnames=expected_window_columns(WINDOW_SIZE),
        rows=selected_source_rows,
    )
    lineage_path = write_jsonl(
        output / "hard_negatives" / "lineage.jsonl",
        (
            {
                **row,
                "hard_negative_rank": rank,
            }
            for rank, row in enumerate(selected)
        ),
    )
    if cfg.parent_selection_mode == "guided":
        parent_entries = _select_parent_entries(
            entries,
            prediction_rows,
            limit=cfg.max_parent_sessions,
        )
    elif cfg.parent_selection_mode == "random":
        parent_entries = random_parent_entries(
            entries,
            limit=cfg.max_parent_sessions,
            seed=cfg.selection_seed + 1,
        )
    else:
        parent_entries = []

    if (
        cfg.require_full_budget
        and cfg.parent_selection_mode != "none"
        and len(parent_entries) != cfg.max_parent_sessions
    ):
        raise ValueError(
            "Parent-session budget was not satisfied: "
            f"selected={len(parent_entries)}, "
            f"required={cfg.max_parent_sessions}"
        )
    parent_path = write_jsonl(
        output / "selected_parent_policies.jsonl",
        parent_entries,
    )

    bypass_count = sum(int(row["is_bypass"]) for row in prediction_rows)
    selected_bypass_count = sum(int(row["is_bypass"]) for row in selected)
    summary = {
        "defender_run_dir": str(Path(defender_run_dir)),
        "attack_manifest_path": str(Path(attack_manifest_path)),
        "config": asdict(cfg),
        "candidate_session_count": len(entries),
        "attack_window_count": len(prediction_rows),
        "bypass_window_count": bypass_count,
        "bypass_rate": bypass_count / len(prediction_rows),
        "family_bypass": family_bypass,
        "observed_families": sorted(family_bypass),
        "worst_family_bypass_rate": (
            worst_family_bypass_rate
        ),
        "hard_negative_count": len(selected),
        "hard_negative_bypass_count": selected_bypass_count,
        "selected_parent_session_count": len(parent_entries),
        "selection_mode": cfg.selection_mode,
        "parent_selection_mode": cfg.parent_selection_mode,
        "selection_depends_on_defender_scores": (
            cfg.selection_mode == "guided"
        ),
        "parent_selection_depends_on_defender_scores": (
            cfg.parent_selection_mode == "guided"
        ),
        "scores_computed_for_audit": True,
        "full_budget_required": cfg.require_full_budget,
        "predictions_path": str(predictions_path),
        "hard_window_path": str(hard_window_path),
        "hard_window_sha256": sha256_file(hard_window_path),
        "lineage_path": str(lineage_path),
        "parent_policy_path": str(parent_path),
        "decision_policy_components": list(bundle.policy_components),
        "decision_threshold": bundle._decision_threshold(),
    }
    summary_path = atomic_json(output / "weakness_summary.json", summary)
    return {**summary, "summary_path": str(summary_path)}
