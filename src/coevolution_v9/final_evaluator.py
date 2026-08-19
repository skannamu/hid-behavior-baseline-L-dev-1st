"""One-shot held-out evaluation for a frozen Stable-v9 defender.

This module intentionally performs no training, mining, threshold selection, or
model selection.  It copies and hashes D_final, resolves the already-reserved
normal test split, generates an independent raw-event A_final pool, audits
lineage disjointness, and evaluates both sets exactly once into a new output
folder.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data_v2 import (
    FeatureV2WindowDataset,
    group_key_for_mode,
    load_manifest_entries,
    resolve_manifest_window_paths,
)
from src.features.schema import EXPECTED_SCHEMA_SHA256, FEATURE_SCHEMA_VERSION

from .bundle import DefenderBundle, DefenderScoreSet, load_defender_bundle
from .common import atomic_json, read_json, sha256_file, write_csv_rows
from .raw_attack_generator import (
    FINAL_HOLDOUT_FAMILIES,
    RawAttackGenerationConfig,
    generate_raw_attack_pool,
)


FINAL_EVALUATION_FORMAT_VERSION = "recon_hid_v9_final_evaluation_v1"
FROZEN_ARTIFACT_NAMES = (
    "best_model.pt",
    "normalizer.json",
    "calibration.json",
    "split_manifest.json",
    "config.json",
)


@dataclass(frozen=True)
class FinalEvaluationConfig:
    seed: int = 920260807
    candidates: int = 24
    keystrokes_per_candidate: int = 84
    mutation_strength: float = 0.10
    families: tuple[str, ...] = FINAL_HOLDOUT_FAMILIES
    batch_size: int = 256
    device: str = "cpu"
    verify_normal_manifest_hashes: bool = True
    require_timeline_defender_match: bool = True
    require_disjoint_seeds: bool = True
    require_disjoint_families: bool = True
    require_null_parent_lineage: bool = True

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.keystrokes_per_candidate < 52:
            raise ValueError("keystrokes_per_candidate must be at least 52")
        if self.mutation_strength < 0:
            raise ValueError("mutation_strength must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.families:
            raise ValueError("families must not be empty")
        if len(set(self.families)) != len(self.families):
            raise ValueError("families must not contain duplicates")


def _resolve_recorded_path(value: str | Path, *, anchor: Path) -> Path:
    """Resolve artifact paths recorded in manifests/timelines robustly.

    Historical artifacts may contain:
      * absolute paths,
      * paths relative to the process working directory/repository root,
      * paths relative to the manifest/timeline directory.

    Try existing candidates from the current working directory and every
    ancestor of the anchor before falling back to anchor-relative resolution.
    """
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    bases = [Path.cwd(), anchor, *anchor.parents]
    seen: set[str] = set()

    for base in bases:
        candidate = (base / path).resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        if candidate.exists():
            return candidate

    return (anchor / path).resolve()


def _prepare_empty_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(
            f"Final evaluation output already exists; one-shot results are never "
            f"overwritten: {output}"
        )
    output.mkdir(parents=True)
    atomic_json(output / "evaluation_started.json", {
        "status": "started",
        "format_version": FINAL_EVALUATION_FORMAT_VERSION,
    })
    return output


def freeze_defender_artifacts(
    defender_run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(defender_run_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    frozen = Path(output_dir)
    frozen.mkdir(parents=True, exist_ok=False)

    artifacts: list[dict[str, Any]] = []
    for name in FROZEN_ARTIFACT_NAMES:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = frozen / name
        shutil.copy2(src, dst)
        source_hash = sha256_file(src)
        frozen_hash = sha256_file(dst)
        if source_hash != frozen_hash:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")
        artifacts.append({
            "name": name,
            "source_path": str(src),
            "frozen_path": str(dst),
            "sha256": frozen_hash,
            "size_bytes": dst.stat().st_size,
        })

    manifest = {
        "format_version": "recon_hid_v9_frozen_defender_v1",
        "source_defender_run_dir": str(source),
        "frozen_defender_run_dir": str(frozen.resolve()),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "artifacts": artifacts,
    }
    manifest_path = atomic_json(frozen / "freeze_manifest.json", manifest)
    return {**manifest, "freeze_manifest_path": str(manifest_path)}


def _load_frozen_normal_test(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    split_manifest: dict[str, Any],
    verify_hashes: bool,
) -> tuple[FeatureV2WindowDataset, dict[str, Any]]:
    entries = load_manifest_entries(
        manifest_path,
        dataset_root=dataset_root,
        verify_files=True,
        verify_hashes=verify_hashes,
    )
    paths = resolve_manifest_window_paths(entries, dataset_root=dataset_root)
    dataset, reports = FeatureV2WindowDataset.from_paths(paths)

    group_mode = str(split_manifest.get("group_mode", "participant"))
    train_groups = {str(value) for value in split_manifest.get("train_groups", [])}
    calibration_groups = {
        str(value) for value in split_manifest.get("calibration_groups", [])
    }
    test_groups = {str(value) for value in split_manifest.get("test_groups", [])}
    if not train_groups or not calibration_groups or not test_groups:
        raise ValueError("Frozen split manifest has empty train/calibration/test groups")
    if train_groups & calibration_groups or train_groups & test_groups or calibration_groups & test_groups:
        raise ValueError("Frozen split manifest contains group leakage")

    test_indices: list[int] = []
    observed_groups: set[str] = set()
    for index, sample in enumerate(dataset):
        group = group_key_for_mode(sample, group_mode)
        if group in test_groups:
            test_indices.append(index)
            observed_groups.add(group)
        elif group not in train_groups and group not in calibration_groups:
            raise ValueError(
                f"Normal sample belongs to no frozen split group: {group}"
            )
    if observed_groups != test_groups:
        raise ValueError(
            "Frozen normal test groups do not match manifest data; "
            f"expected={sorted(test_groups)}, observed={sorted(observed_groups)}"
        )
    if not test_indices:
        raise ValueError("Frozen normal test split contains zero windows")

    test_dataset = dataset.subset(test_indices)
    return test_dataset, {
        "group_mode": group_mode,
        "train_groups": sorted(train_groups),
        "calibration_groups": sorted(calibration_groups),
        "test_groups": sorted(test_groups),
        "test_window_count": len(test_dataset),
        "test_participants": sorted({s.participant_id for s in test_dataset}),
        "test_sessions": sorted({s.session_id for s in test_dataset}),
        "source_file_count": len(reports),
        "source_manifest": str(Path(manifest_path)),
        "source_manifest_sha256": sha256_file(manifest_path),
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_no}: expected object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"JSONL contains no rows: {source}")
    return rows


def _collect_training_lineage(
    timeline_path: str | Path,
) -> dict[str, Any]:
    timeline_source = Path(timeline_path).resolve()
    timeline = read_json(timeline_source)
    rounds = timeline.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Co-evolution timeline contains no rounds")

    candidate_ids: set[str] = set()
    seeds: set[int] = set()
    families: set[str] = set()
    window_hashes: set[str] = set()
    round_manifests: list[str] = []
    attack_manifests: list[str] = []

    for item in rounds:
        if not isinstance(item, dict) or not item.get("round_manifest_path"):
            raise ValueError("Timeline round has no round_manifest_path")
        round_path = _resolve_recorded_path(
            str(item["round_manifest_path"]),
            anchor=timeline_source.parent,
        )
        round_manifest = read_json(round_path)
        round_manifests.append(str(round_path))
        generation = round_manifest.get("attack_generation")
        if not isinstance(generation, dict) or not generation.get("manifest_path"):
            raise ValueError(f"Round manifest has no attack manifest: {round_path}")
        attack_path = _resolve_recorded_path(
            str(generation["manifest_path"]),
            anchor=round_path.parent,
        )
        attack_manifests.append(str(attack_path))
        for entry in _load_jsonl(attack_path):
            candidate_ids.add(str(entry["candidate_id"]))
            seeds.add(int(entry["seed"]))
            families.add(str(entry["family"]))
            if entry.get("window_sha256"):
                window_hashes.add(str(entry["window_sha256"]))

    latest = timeline.get("latest_defender_run_dir")
    if not latest:
        latest = rounds[-1].get("defender_run_dir")
    if not latest:
        raise ValueError("Timeline has no latest defender run directory")
    latest_path = _resolve_recorded_path(str(latest), anchor=timeline_source.parent)

    return {
        "timeline_path": str(timeline_source),
        "timeline_sha256": sha256_file(timeline_source),
        "latest_defender_run_dir": str(latest_path),
        "round_count": len(rounds),
        "round_manifest_paths": round_manifests,
        "attack_manifest_paths": attack_manifests,
        "candidate_ids": sorted(candidate_ids),
        "seeds": sorted(seeds),
        "families": sorted(families),
        "window_hashes": sorted(window_hashes),
    }


def audit_final_lineage(
    *,
    final_entries: Iterable[dict[str, Any]],
    training_lineage: dict[str, Any],
    final_defender_source_dir: str | Path,
    config: FinalEvaluationConfig,
) -> dict[str, Any]:
    entries = list(final_entries)
    final_candidates = {str(entry["candidate_id"]) for entry in entries}
    final_seeds = {int(entry["seed"]) for entry in entries}
    final_families = {str(entry["family"]) for entry in entries}
    final_hashes = {str(entry.get("window_sha256", "")) for entry in entries}
    final_parents = {
        str(entry["parent_candidate_id"])
        for entry in entries
        if entry.get("parent_candidate_id") is not None
    }

    train_candidates = set(training_lineage["candidate_ids"])
    train_seeds = {int(value) for value in training_lineage["seeds"]}
    train_families = set(training_lineage["families"])
    train_hashes = set(training_lineage["window_hashes"])

    overlaps = {
        "candidate_ids": sorted(final_candidates & train_candidates),
        "seeds": sorted(final_seeds & train_seeds),
        "families": sorted(final_families & train_families),
        "window_hashes": sorted((final_hashes - {""}) & train_hashes),
        "non_null_parent_candidate_ids": sorted(final_parents),
    }
    defender_match = (
        Path(final_defender_source_dir).resolve()
        == Path(training_lineage["latest_defender_run_dir"]).resolve()
    )

    failures: list[str] = []
    if overlaps["candidate_ids"]:
        failures.append("candidate_id_overlap")
    if overlaps["window_hashes"]:
        failures.append("window_hash_overlap")
    if config.require_disjoint_seeds and overlaps["seeds"]:
        failures.append("seed_overlap")
    if config.require_disjoint_families and overlaps["families"]:
        failures.append("family_overlap")
    if config.require_null_parent_lineage and overlaps["non_null_parent_candidate_ids"]:
        failures.append("non_null_parent_lineage")
    if config.require_timeline_defender_match and not defender_match:
        failures.append("d_final_does_not_match_timeline_latest")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "defender_matches_timeline_latest": defender_match,
        "training_round_count": training_lineage["round_count"],
        "training_candidate_count": len(train_candidates),
        "final_candidate_count": len(final_candidates),
        "training_families": sorted(train_families),
        "final_families": sorted(final_families),
        "training_seed_count": len(train_seeds),
        "final_seed_count": len(final_seeds),
        "overlaps": overlaps,
        "requirements": {
            "require_timeline_defender_match": config.require_timeline_defender_match,
            "require_disjoint_seeds": config.require_disjoint_seeds,
            "require_disjoint_families": config.require_disjoint_families,
            "require_null_parent_lineage": config.require_null_parent_lineage,
        },
    }
    if failures:
        raise ValueError(f"A_final lineage audit failed: {failures}")
    return report


def _prediction_rows(
    dataset: FeatureV2WindowDataset,
    scores: DefenderScoreSet,
    *,
    dataset_type: str,
) -> list[dict[str, Any]]:
    if len(dataset) != len(scores):
        raise ValueError("Dataset/score length mismatch")
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset):
        rows.append({
            "dataset_type": dataset_type,
            "participant_id": sample.participant_id,
            "session_id": sample.session_id,
            "scenario_or_family": sample.scenario,
            "window_id": sample.window_id,
            "source_file": sample.source_file,
            "row_number": sample.row_number,
            "sequence_reconstruction": float(scores.sequence_reconstruction[index]),
            "context_reconstruction": float(scores.context_reconstruction[index]),
            "reconstruction_combined": float(scores.reconstruction_combined[index]),
            "classifier_probability": float(scores.classifier_probability[index]),
            "prototype_distance": float(scores.prototype_distance[index]),
            "decision_score": float(scores.decision_score[index]),
            "predicted_attack": int(scores.predicted_attack[index]),
        })
    return rows


def _group_metrics(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    truth_attack: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        predicted = np.asarray([int(row["predicted_attack"]) for row in group])
        positive_rate = float(predicted.mean())
        item = {name: value for name, value in zip(keys, key, strict=True)}
        item.update({
            "window_count": len(group),
            "predicted_attack_count": int(predicted.sum()),
            "positive_rate": positive_rate,
            "fpr": positive_rate if not truth_attack else "",
            "tpr": positive_rate if truth_attack else "",
            "bypass_rate": (1.0 - positive_rate) if truth_attack else "",
            "any_attack_prediction": int(bool(predicted.any())),
            "majority_attack_prediction": int(positive_rate >= 0.5),
            "mean_decision_score": float(np.mean([
                float(row["decision_score"]) for row in group
            ])),
            "max_decision_score": float(np.max([
                float(row["decision_score"]) for row in group
            ])),
        })
        result.append(item)
    return result


def _verify_frozen_hashes(freeze_manifest: dict[str, Any]) -> None:
    for artifact in freeze_manifest["artifacts"]:
        path = Path(str(artifact["frozen_path"]))
        if sha256_file(path) != str(artifact["sha256"]):
            raise RuntimeError(f"Frozen defender changed during evaluation: {path}")


def run_final_evaluation(
    *,
    final_defender_run_dir: str | Path,
    normal_dataset_root: str | Path,
    normal_manifest_path: str | Path,
    coevolution_timeline_path: str | Path,
    output_dir: str | Path,
    config: FinalEvaluationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FinalEvaluationConfig()
    cfg.validate()
    output = _prepare_empty_output(output_dir)

    freeze = freeze_defender_artifacts(
        final_defender_run_dir,
        output / "frozen_defender",
    )
    frozen_dir = Path(freeze["frozen_defender_run_dir"])
    bundle = load_defender_bundle(frozen_dir, map_location=cfg.device)

    normal_test, normal_split = _load_frozen_normal_test(
        dataset_root=normal_dataset_root,
        manifest_path=normal_manifest_path,
        split_manifest=bundle.split_manifest,
        verify_hashes=cfg.verify_normal_manifest_hashes,
    )

    generation_cfg = RawAttackGenerationConfig(
        candidates=cfg.candidates,
        keystrokes_per_candidate=cfg.keystrokes_per_candidate,
        seed=cfg.seed,
        generation=10_000,
        mutation_strength=cfg.mutation_strength,
        parent_policy_path=None,
        families=cfg.families,
    )
    attack_info = generate_raw_attack_pool(
        output_root=output / "A_final",
        round_name="A_final",
        config=generation_cfg,
    )
    final_entries = _load_jsonl(attack_info["manifest_path"])
    training_lineage = _collect_training_lineage(coevolution_timeline_path)
    lineage_audit = audit_final_lineage(
        final_entries=final_entries,
        training_lineage=training_lineage,
        final_defender_source_dir=final_defender_run_dir,
        config=cfg,
    )
    lineage_path = atomic_json(output / "V_final" / "lineage_audit.json", lineage_audit)

    attack_paths = [Path(str(entry["window_path"])) for entry in final_entries]
    attack_dataset, _ = FeatureV2WindowDataset.from_paths(attack_paths)

    normal_scores = bundle.score_dataset(
        normal_test,
        device=cfg.device,
        batch_size=cfg.batch_size,
    )
    attack_scores = bundle.score_dataset(
        attack_dataset,
        device=cfg.device,
        batch_size=cfg.batch_size,
    )

    normal_rows = _prediction_rows(normal_test, normal_scores, dataset_type="normal_test")
    attack_rows = _prediction_rows(attack_dataset, attack_scores, dataset_type="A_final")
    prediction_fields = list(normal_rows[0])
    normal_predictions_path = write_csv_rows(
        output / "V_final" / "normal_test_window_predictions.csv",
        fieldnames=prediction_fields,
        rows=normal_rows,
    )
    attack_predictions_path = write_csv_rows(
        output / "V_final" / "A_final_window_predictions.csv",
        fieldnames=prediction_fields,
        rows=attack_rows,
    )

    normal_session = _group_metrics(
        normal_rows,
        keys=("participant_id", "session_id", "scenario_or_family"),
        truth_attack=False,
    )
    attack_session = _group_metrics(
        attack_rows,
        keys=("session_id", "scenario_or_family"),
        truth_attack=True,
    )
    family_metrics = _group_metrics(
        attack_rows,
        keys=("scenario_or_family",),
        truth_attack=True,
    )
    metric_fields = list(normal_session[0])
    normal_session_path = write_csv_rows(
        output / "V_final" / "normal_test_session_metrics.csv",
        fieldnames=metric_fields,
        rows=normal_session,
    )
    attack_session_path = write_csv_rows(
        output / "V_final" / "A_final_candidate_metrics.csv",
        fieldnames=list(attack_session[0]),
        rows=attack_session,
    )
    family_path = write_csv_rows(
        output / "V_final" / "A_final_family_metrics.csv",
        fieldnames=list(family_metrics[0]),
        rows=family_metrics,
    )

    normal_pred = normal_scores.predicted_attack.astype(np.int64)
    attack_pred = attack_scores.predicted_attack.astype(np.int64)
    normal_session_fprs = [float(row["positive_rate"]) for row in normal_session]
    attack_candidate_tprs = [float(row["positive_rate"]) for row in attack_session]
    family_tprs = [float(row["positive_rate"]) for row in family_metrics]

    metrics = {
        "format_version": FINAL_EVALUATION_FORMAT_VERSION,
        "status": "PASS",
        "methodology": {
            "defender_frozen_before_scoring": True,
            "threshold_recalibrated_on_final_data": False,
            "model_selected_on_final_data": False,
            "A_final_used_for_training": False,
            "A_final_parent_policy_used": False,
            "normal_test_source": "frozen_untouched_split",
            "attack_generation_level": "raw_event",
            "window_metrics_are_correlated_due_to_stride_1": True,
        },
        "defender": {
            "source_run_dir": str(Path(final_defender_run_dir).resolve()),
            "frozen_run_dir": str(frozen_dir),
            "defender_id": bundle.defender_id,
            "classifier_trained": bundle.classifier_trained,
            "decision_policy_components": list(bundle.policy_components),
            "decision_threshold": bundle._decision_threshold(),
            "freeze_manifest_path": freeze["freeze_manifest_path"],
        },
        "normal_test": {
            **normal_split,
            "false_positive_windows": int(normal_pred.sum()),
            "window_fpr": float(normal_pred.mean()),
            "session_count": len(normal_session),
            "macro_session_fpr": float(np.mean(normal_session_fprs)),
            "max_session_fpr": float(np.max(normal_session_fprs)),
        },
        "A_final": {
            "generation_config": asdict(generation_cfg),
            "candidate_count": len(final_entries),
            "window_count": len(attack_dataset),
            "detected_windows": int(attack_pred.sum()),
            "bypass_windows": int((1 - attack_pred).sum()),
            "window_tpr": float(attack_pred.mean()),
            "window_bypass_rate": float(1.0 - attack_pred.mean()),
            "macro_candidate_tpr": float(np.mean(attack_candidate_tprs)),
            "minimum_candidate_tpr": float(np.min(attack_candidate_tprs)),
            "macro_family_tpr": float(np.mean(family_tprs)),
            "minimum_family_tpr": float(np.min(family_tprs)),
            "candidate_any_detection_rate": float(np.mean([
                int(row["any_attack_prediction"]) for row in attack_session
            ])),
            "candidate_majority_detection_rate": float(np.mean([
                int(row["majority_attack_prediction"]) for row in attack_session
            ])),
            "manifest_path": attack_info["manifest_path"],
            "manifest_sha256": sha256_file(attack_info["manifest_path"]),
        },
        "lineage_audit": {
            "status": lineage_audit["status"],
            "path": str(lineage_path),
            "training_timeline_path": training_lineage["timeline_path"],
            "training_timeline_sha256": training_lineage["timeline_sha256"],
        },
        "outputs": {
            "normal_window_predictions": str(normal_predictions_path),
            "attack_window_predictions": str(attack_predictions_path),
            "normal_session_metrics": str(normal_session_path),
            "attack_candidate_metrics": str(attack_session_path),
            "attack_family_metrics": str(family_path),
        },
    }
    metrics_path = atomic_json(output / "V_final" / "final_metrics.json", metrics)
    _verify_frozen_hashes(freeze)

    final_manifest = {
        "format_version": FINAL_EVALUATION_FORMAT_VERSION,
        "status": "COMPLETE",
        "config": asdict(cfg),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": EXPECTED_SCHEMA_SHA256,
        "freeze": freeze,
        "normal_manifest_path": str(Path(normal_manifest_path)),
        "normal_manifest_sha256": sha256_file(normal_manifest_path),
        "coevolution_timeline_path": training_lineage["timeline_path"],
        "coevolution_timeline_sha256": training_lineage["timeline_sha256"],
        "A_final_manifest_path": attack_info["manifest_path"],
        "A_final_manifest_sha256": sha256_file(attack_info["manifest_path"]),
        "lineage_audit_path": str(lineage_path),
        "metrics_path": str(metrics_path),
    }
    final_manifest_path = atomic_json(output / "final_evaluation_manifest.json", final_manifest)
    atomic_json(output / "evaluation_complete.json", {
        "status": "COMPLETE",
        "final_evaluation_manifest_path": str(final_manifest_path),
        "metrics_path": str(metrics_path),
    })
    return {
        "status": "PASS",
        "output_dir": str(output),
        "final_evaluation_manifest_path": str(final_manifest_path),
        "metrics_path": str(metrics_path),
        "normal_window_fpr": metrics["normal_test"]["window_fpr"],
        "attack_window_tpr": metrics["A_final"]["window_tpr"],
        "attack_window_bypass_rate": metrics["A_final"]["window_bypass_rate"],
        "lineage_audit_status": lineage_audit["status"],
    }
