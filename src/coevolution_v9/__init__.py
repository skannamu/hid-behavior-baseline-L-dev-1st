"""Stable Feature-v2 attack/defense co-evolution components."""

from .bundle import DefenderBundle, load_defender_bundle
from .raw_attack_generator import (
    RawAttackGenerationConfig,
    RawAttackPolicy,
    generate_raw_attack_pool,
)
from .weakness_miner import WeaknessMiningConfig, mine_v9_weaknesses
from .hardened_trainer import (
    HardenedTrainingConfig,
    HardenedTrainingResult,
    train_hardened_defender_v9,
)
from .round_runner import CoevolutionRoundConfig, run_coevolution_round
from .convergence import (
    ConvergenceConfig,
    ConvergenceDecision,
    ConvergenceTracker,
    summarize_convergence_history,
)
from .final_evaluator import (
    FinalEvaluationConfig,
    audit_final_lineage,
    freeze_defender_artifacts,
    run_final_evaluation,
)

__all__ = [
    "DefenderBundle",
    "load_defender_bundle",
    "RawAttackGenerationConfig",
    "RawAttackPolicy",
    "generate_raw_attack_pool",
    "WeaknessMiningConfig",
    "mine_v9_weaknesses",
    "HardenedTrainingConfig",
    "HardenedTrainingResult",
    "train_hardened_defender_v9",
    "CoevolutionRoundConfig",
    "run_coevolution_round",
    "ConvergenceConfig",
    "ConvergenceDecision",
    "ConvergenceTracker",
    "summarize_convergence_history",
    "FinalEvaluationConfig",
    "audit_final_lineage",
    "freeze_defender_artifacts",
    "run_final_evaluation",
]
