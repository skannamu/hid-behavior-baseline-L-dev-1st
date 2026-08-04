from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.framework.config import load_config, require_cfg, get_cfg, dump_json
from src.framework.artifacts import init_framework_dirs, round_dir
from src.pipeline.build_normal_dataset import build_normal_dataset
from src.pipeline.validate_normal_dataset import validate_normal_dataset
from src.framework.train_defender_d0 import train_d0
from src.generators.synthetic_hid_generator import generate_synthetic_attacks
from src.generators.defender_aware_generator import generate_defender_aware_attacks
from src.evaluation.evaluate_autoencoder import evaluate_d0
from src.hardening.weakness_miner import mine_weaknesses
from src.hardening.train_hardened_defender import train_hardened_defender
from src.evaluation.evaluate_hybrid_defender import evaluate_hybrid_defender
from src.runtime.export_final_detector import export_final_detector
from src.framework.score_fusion_optimizer import optimize_decision_policy


def _adaptive_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(get_cfg(cfg, "adaptive_loop.enabled", False))


def _defender_aware_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(get_cfg(cfg, "defender_aware_generator.enabled", False))


def _score_fusion_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(get_cfg(cfg, "score_fusion_optimizer.enabled", True))


def _max_rounds(cfg: Dict[str, Any]) -> int:
    if _adaptive_enabled(cfg):
        return int(get_cfg(cfg, "adaptive_loop.max_rounds", get_cfg(cfg, "framework.rounds", 3)))
    return int(get_cfg(cfg, "framework.rounds", 3))


def print_plan(cfg: Dict[str, Any]) -> None:
    rounds = _max_rounds(cfg)
    adaptive = _adaptive_enabled(cfg)
    defender_aware = _defender_aware_enabled(cfg)
    score_fusion = _score_fusion_enabled(cfg)

    print("\n===== FRAMEWORK PLAN =====")
    print(f"Experiment : {get_cfg(cfg, 'experiment_name', 'unknown')}")
    print(f"Normal root: {require_cfg(cfg, 'paths.normal_raw_root')}")
    print(f"Normal CSV : {require_cfg(cfg, 'paths.normal_window_csv')}")
    print(f"Attack root: {require_cfg(cfg, 'paths.attack_generated_root')}")
    print(f"Checkpoints: {require_cfg(cfg, 'paths.checkpoints_root')}")
    print(f"Loop mode  : {'adaptive' if adaptive else 'fixed'}")
    print(f"Generator  : {'defender-aware' if defender_aware else 'synthetic'}")
    print(f"Decision   : {'score-fusion-optimizer' if score_fusion else get_cfg(cfg, 'decision_policy.mode', 'fpr_constrained')}")
    print(f"Max rounds : {rounds}")
    if adaptive:
        print(f"Min rounds : {get_cfg(cfg, 'adaptive_loop.min_rounds', 1)}")
        print(f"Target ADR : {get_cfg(cfg, 'adaptive_loop.target_attack_detection_rate', 0.95)}")
        print(f"Max FPR    : {get_cfg(cfg, 'adaptive_loop.max_normal_false_positive_rate', 0.05)}")
        print(f"Max bypass : {get_cfg(cfg, 'adaptive_loop.max_bypass_rate', 0.05)}")
        print(f"Patience   : {get_cfg(cfg, 'adaptive_loop.patience', 3)}")
    print("==========================\n")

    print("[PLAN] 1. Build normal window CSV from raw sessions")
    print("[PLAN] 2. Validate normal dataset")
    print("[PLAN] 3. Train D0 initial normal-only defender")
    print("[PLAN] 4. Repeat until stop criteria or max_rounds:")
    print("        pool_k = generate large candidate pool")
    print("        A_k = select hard candidates using current D_k")
    print("        Eval_k = evaluate D_k vs A_k")
    print("        Policy_k = optimize fusion score under FPR constraint")
    print("        W_k = mine weakness using optimized policy")
    print("        stop if ADR/FPR/bypass criteria are satisfied")
    print("        otherwise train D_{k+1} using normal + A0..Ak")
    print("[PLAN] 8. Export D_final from selected best/latest defender")
    print("[PLAN] 9. Generate A_final held-out validation candidates")
    print("[PLAN] 10. Optimize/evaluate D_final against A_final")
    print("")


def _collect_attack_paths(cfg: Dict[str, Any], upto_round: int) -> List[str]:
    attack_root = Path(require_cfg(cfg, "paths.attack_generated_root"))
    paths = []
    for k in range(upto_round + 1):
        p = attack_root / f"round{k}_A{k}" / "candidates.csv"
        if p.exists():
            paths.append(str(p))
    return paths


def _generate_attack_candidates(cfg: Dict[str, Any], defender_id: str, round_name: str) -> Path:
    if _defender_aware_enabled(cfg):
        return generate_defender_aware_attacks(
            cfg=cfg,
            defender_id=defender_id,
            round_name=round_name,
        )

    return generate_synthetic_attacks(cfg, round_name=round_name)


def _evaluate_defender(cfg: Dict[str, Any], defender_id: str, attack_path: Path, out_name: str) -> Dict[str, Any]:
    if defender_id == "D0":
        return evaluate_d0(cfg, attack_path=attack_path, out_name=out_name)

    return evaluate_hybrid_defender(
        cfg,
        defender_id=defender_id,
        attack_path=attack_path,
        out_name=out_name,
    )


def _prediction_csv_from_eval(info: Dict[str, Any]) -> str:
    if "predictions_csv" in info:
        return info["predictions_csv"]
    raise KeyError(f"Evaluation output does not include predictions_csv. Keys: {list(info.keys())}")


def _policy_metrics(
    cfg: Dict[str, Any],
    eval_info: Dict[str, Any],
    defender_id: str,
    stage_name: str,
    out_dir: Path,
) -> Dict[str, Any]:
    if _score_fusion_enabled(cfg):
        return optimize_decision_policy(
            cfg=cfg,
            predictions_csv=_prediction_csv_from_eval(eval_info),
            out_dir=out_dir / "decision_policy",
            defender_id=defender_id,
            stage_name=stage_name,
        )

    # Fallback: legacy behavior
    if defender_id == "D0" and isinstance(eval_info.get("summary"), dict):
        s = eval_info["summary"]
        adr = float(s.get("recall_attack_detection_rate", 0.0))
        fpr = float(s.get("normal_false_positive_rate", 1.0))
        return {
            "attack_detection_rate": adr,
            "normal_false_positive_rate": fpr,
            "bypass_rate": 1.0 - adr,
            "method": "pred_primary",
            "pred_col": "pred_primary",
            "selection_reason": "legacy D0 primary",
            "predictions_csv": _prediction_csv_from_eval(eval_info),
        }

    return {
        "attack_detection_rate": 0.0,
        "normal_false_positive_rate": 1.0,
        "bypass_rate": 1.0,
        "method": "pred_or95",
        "pred_col": "pred_or95",
        "selection_reason": "legacy fallback",
        "predictions_csv": _prediction_csv_from_eval(eval_info),
    }


def _should_stop_adaptive(
    cfg: Dict[str, Any],
    completed_rounds: int,
    metrics: Dict[str, Any],
    state: Dict[str, Any],
) -> Optional[str]:
    if not _adaptive_enabled(cfg):
        return None

    min_rounds = int(get_cfg(cfg, "adaptive_loop.min_rounds", 1))
    max_rounds = int(get_cfg(cfg, "adaptive_loop.max_rounds", 20))
    target_adr = float(get_cfg(cfg, "adaptive_loop.target_attack_detection_rate", 0.95))
    max_fpr = float(get_cfg(cfg, "adaptive_loop.max_normal_false_positive_rate", 0.05))
    max_bypass = float(get_cfg(cfg, "adaptive_loop.max_bypass_rate", 0.05))
    patience = int(get_cfg(cfg, "adaptive_loop.patience", 3))
    min_improvement = float(get_cfg(cfg, "adaptive_loop.min_improvement", 0.005))

    adr = float(metrics["attack_detection_rate"])
    fpr = float(metrics["normal_false_positive_rate"])
    bypass = float(metrics["bypass_rate"])

    fpr_valid = fpr <= max_fpr
    key = "best_valid_bypass" if fpr_valid else "best_any_bypass"
    best_bypass = state.get(key)

    if best_bypass is None or bypass < best_bypass - min_improvement:
        state[key] = bypass
        state["no_improve_count"] = 0
    else:
        state["no_improve_count"] = int(state.get("no_improve_count", 0)) + 1

    if completed_rounds < min_rounds:
        return None

    if adr >= target_adr and fpr <= max_fpr and bypass <= max_bypass:
        return (
            f"target reached: ADR={adr:.4f} >= {target_adr}, "
            f"FPR={fpr:.4f} <= {max_fpr}, bypass={bypass:.4f} <= {max_bypass}"
        )

    if state.get("no_improve_count", 0) >= patience:
        return (
            f"early stop by patience: no improvement for {state['no_improve_count']} rounds "
            f"(current ADR={adr:.4f}, FPR={fpr:.4f}, bypass={bypass:.4f})"
        )

    if completed_rounds >= max_rounds:
        return f"max_rounds reached: {completed_rounds}"

    return None


def _train_next_defender(cfg: Dict[str, Any], next_id: str, attack_paths: List[str], round_name: str) -> Dict[str, Any]:
    normal_path = require_cfg(cfg, "paths.normal_window_csv")

    return train_hardened_defender(
        cfg,
        defender_id=next_id,
        normal_path=normal_path,
        attack_paths=attack_paths,
        out_name=round_name,
    )


def run(cfg: Dict[str, Any], dry_run: bool = False, allow_empty: bool = False) -> None:
    init_framework_dirs(cfg)
    print_plan(cfg)

    normal_root = Path(require_cfg(cfg, "paths.normal_raw_root"))

    if dry_run:
        print("[DRY-RUN] No training/generation will be executed.")
        print(f"[DRY-RUN] Normal data exists: {normal_root.exists()} ({normal_root})")
        return

    normal_csv = build_normal_dataset(cfg, allow_empty=allow_empty)
    if normal_csv is None:
        print("[STOP] Normal data is not ready yet. Framework skeleton is ready.")
        return

    validate_normal_dataset(cfg)

    max_rounds = _max_rounds(cfg)
    timeline = []
    adaptive_state = {
        "best_valid_bypass": None,
        "best_any_bypass": None,
        "no_improve_count": 0,
    }

    d0_info = train_d0(cfg)
    current_defender = "D0"
    timeline.append({"stage": "train", "defender": "D0", **d0_info})

    stop_reason = None

    for k in range(max_rounds):
        attack_name = f"round{k}_A{k}"
        eval_name = f"round{k}_eval_D{k}_vs_A{k}"
        weakness_name = f"round{k}_weakness_D{k}"

        attack_csv = _generate_attack_candidates(cfg, current_defender, attack_name)

        eval_info = _evaluate_defender(
            cfg,
            defender_id=current_defender,
            attack_path=attack_csv,
            out_name=eval_name,
        )

        rdir = round_dir(cfg, eval_name)
        metrics = _policy_metrics(
            cfg=cfg,
            eval_info=eval_info,
            defender_id=current_defender,
            stage_name=eval_name,
            out_dir=rdir,
        )

        weakness_report = mine_weaknesses(
            predictions_csv=metrics["predictions_csv"],
            out_dir=round_dir(cfg, weakness_name),
            pred_col=metrics["pred_col"],
        )

        completed_rounds = k + 1

        print(
            f"[ROUND {k}] defender={current_defender} attack=A{k} "
            f"ADR={metrics['attack_detection_rate']:.4f} "
            f"FPR={metrics['normal_false_positive_rate']:.4f} "
            f"bypass={metrics['bypass_rate']:.4f} "
            f"policy={metrics['method']} "
            f"reason={metrics['selection_reason']}"
        )

        stop_reason = _should_stop_adaptive(
            cfg,
            completed_rounds=completed_rounds,
            metrics=metrics,
            state=adaptive_state,
        )

        timeline.append({
            "round": k,
            "completed_rounds": completed_rounds,
            "defender": current_defender,
            "attack": f"A{k}",
            "attack_csv": str(attack_csv),
            "eval": eval_info,
            "policy": metrics,
            "weakness": weakness_report,
            "stop_reason_after_round": stop_reason,
        })

        if stop_reason:
            print(f"[STOP] Adaptive loop stopped: {stop_reason}")
            break

        if k >= max_rounds - 1:
            stop_reason = f"max_rounds reached: {max_rounds}"
            print(f"[STOP] {stop_reason}")
            break

        next_defender = f"D{k+1}"
        attack_paths = _collect_attack_paths(cfg, upto_round=k)
        train_info = _train_next_defender(
            cfg,
            next_id=next_defender,
            attack_paths=attack_paths,
            round_name=f"round{k}_train_{next_defender}",
        )
        timeline.append({"stage": "train", "defender": next_defender, **train_info})
        current_defender = next_defender

    final_info = export_final_detector(
        cfg,
        source_defender_id=current_defender,
        final_id="D_final",
    )

    afinal_csv = _generate_attack_candidates(
        cfg,
        "D_final" if current_defender != "D0" else "D0",
        "final_A_final",
    )

    if current_defender == "D0":
        final_eval = evaluate_d0(
            cfg,
            attack_path=afinal_csv,
            out_name="final_eval_D_final_vs_A_final",
        )
    else:
        final_eval = evaluate_hybrid_defender(
            cfg,
            defender_id="D_final",
            attack_path=afinal_csv,
            out_name="final_eval_D_final_vs_A_final",
        )

    final_dir = round_dir(cfg, "final_eval_D_final_vs_A_final")
    final_metrics = _policy_metrics(
        cfg=cfg,
        eval_info=final_eval,
        defender_id="D_final",
        stage_name="final_eval_D_final_vs_A_final",
        out_dir=final_dir,
    )

    timeline.append({
        "stage": "final_validation",
        "defender": "D_final",
        "source_defender": current_defender,
        "attack": "A_final",
        "attack_csv": str(afinal_csv),
        "eval": final_eval,
        "policy": final_metrics,
        "export": final_info,
        "stop_reason": stop_reason,
    })

    out = Path(require_cfg(cfg, "paths.experiments_root")) / "framework_timeline.json"
    dump_json(out, timeline)

    print("\n===== FRAMEWORK V8.1 BLIND-SPOT EVOLUTION LOOP DONE =====")
    print("[OK] D0/A0 -> FPR-constrained blind-spot evolution -> D_final/A_final completed.")
    print(f"Loop mode     : {'adaptive' if _adaptive_enabled(cfg) else 'fixed'}")
    print(f"Generator     : {'defender-aware' if _defender_aware_enabled(cfg) else 'synthetic'}")
    print(f"Stop reason   : {stop_reason}")
    print(f"Final source  : {current_defender}")
    print("D_final       : checkpoints/D_final/model.pt")
    print(f"Final policy  : {final_metrics['method']}")
    print(f"Final ADR     : {final_metrics['attack_detection_rate']:.4f}")
    print(f"Final FPR     : {final_metrics['normal_false_positive_rate']:.4f}")
    print(f"Final bypass  : {final_metrics['bypass_rate']:.4f}")
    print(f"Timeline      : {out}")
    print("==============================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive HID attack-defense framework")
    parser.add_argument("--config", required=True, help="Path to framework YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--allow-empty", action="store_true", help="Do not fail when normal data is not collected yet")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg, dry_run=args.dry_run, allow_empty=args.allow_empty)


if __name__ == "__main__":
    main()
