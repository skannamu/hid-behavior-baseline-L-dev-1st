# ReCon-HID Paper Scope Freeze

Frozen: 2026-08-20

## Primary Research Goal

Evaluate whether a user-agnostic behavioral HID detector initialized
from normal-only data can be hardened through adaptive,
physically-realizable co-evolution such that it generalizes better to
held-out HID injection attacks than matched-budget non-guided baselines,
without unacceptable degradation of heterogeneous human normality.

## Primary Claim

ReCon-HID is an adaptive hardening framework.

The paper does NOT claim universal HID attack detection or perfect
normal-user acceptance.

The central empirical question is whether defender-guided weakness
mining provides a robustness/generalization advantage beyond simply
performing iterative hardening.

## Core Contributions

1. Physically-realizable adaptive HID co-evolution
   - attacks originate from raw KeyDown/KeyUp behavior
   - attacks and normal data pass through the same Feature Schema
   - the current defender determines which attack behaviors constitute
     useful hard negatives

2. Controlled empirical evaluation
   - normal-only D0
   - participant-disjoint Train / Calibration / Test
   - ReCon-HID vs random iterative vs static mixed
   - convergence-defined evaluation
   - fixed-round matched-budget evaluation
   - lineage-disjoint held-out A_final

3. Failure/generalization characterization
   - normal-side failure:
     prolonged real keyboard inactivity can produce correlated
     free-writing false-positive bursts
   - attack-side failure:
     residual held-out long-hold bypasses occupy defender-dependent
     policy regions rather than a single universal parameter threshold

## Explicit Non-Claims

This paper does NOT claim:

- detection of every possible HID attack
- zero false positives for every human behavior
- universal cross-device or cross-OS robustness
- that residual long-hold failures have one proven causal parameter
- that overlapping windows are independent statistical samples

## Primary Evaluation Questions

RQ1:
How vulnerable is the normal-only D0 detector to adaptive,
physically-realizable HID injection?

RQ2:
Does iterative hardening reduce adaptive bypass while preserving normal
behavior?

RQ3:
Under equal adaptation budgets, does defender-guided ReCon-HID
outperform random/static hardening?

RQ4:
Does the hardened detector generalize to lineage-disjoint,
held-out A_final attack families?

RQ5:
How well does the frozen detector generalize to external human typing?

## Experiment Stop Rule

The primary experimental study is considered complete after:

1. protocol/leakage/lineage validation
2. convergence-based native experiment
3. fixed-round matched-budget D1/D2/D3/D5 experiment
4. participant-aware statistical aggregation
5. post-freeze external-normal evaluation
6. final native cohort evaluation

Failure analysis beyond the already identified representative
normal-side and attack-side mechanisms is optional future work unless
it reveals an experiment-invalidating bug.

## Fixed-Round Interpretation

D1, D2, D3, and D5 are fixed-budget endpoints.

They are NOT D_final.

Convergence-defined D_final remains a separate primary evaluation.

Fixed-round evaluation is a predeclared fairness/ablation analysis and
must not change training or model selection based on Test or A_final
results.
