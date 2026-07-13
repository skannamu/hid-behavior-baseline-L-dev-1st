# Baseline Design

## Goal

The goal of the baseline study is not to claim a new model architecture, but to validate the proposed detector-guided hard-negative hardening framework.

The main research questions are:

1. Does hard-negative hardening improve the detector compared with no hardening?
2. Does detector-guided hard-negative selection improve over random or unguided sample selection?
3. Are multiple decision signals necessary compared with reconstruction-only AE detection?
4. Does iterative co-evolution provide useful evidence beyond one-shot training?

---

## B1. No-hardening baseline

### Comparison

- D0: initial detector
- D3_guided: final detector hardened with V5, V6, and V7 hard negatives

### Purpose

To show that the initial detector is vulnerable to adaptive HID mimic samples and that iterative hardening improves robustness.

### Expected result

D0 is weak against V5/V6/V7, while D3 detects V4-V8 under OR95/OR90.

---

## B2. Signal ablation baseline

### Comparison

- Reconstruction-only
- Classifier-only
- Prototype/latent-distance-only
- Hybrid OR decision

### Purpose

To show that reconstruction error alone is insufficient for adaptive HID mimic detection and that the hybrid decision improves robustness.

### Expected result

Reconstruction-only is expected to be weaker on V7/V8 hardest samples, while hybrid OR remains stable.

---

## B3. Random hardening baseline

### Comparison

- D3_guided: trained with detector-bypassing hard negatives
- D3_random_matched: trained with randomly selected candidates of the same sample count

### Purpose

To test whether improvement comes from detector-guided hard-negative mining rather than simply adding more attack data.

### Fairness condition

The number of random samples must match the number of guided hard-negative samples used in training.

### Expected result

D3_guided should outperform D3_random_matched, especially on V7 and V8 hardest candidate sets.

---

## B4. Unguided augmentation baseline

### Comparison

- D3_guided
- D3_unguided_aug

### Purpose

To compare detector-guided generation with ordinary feature-space augmentation that does not use detector feedback.

### Expected result

Unguided augmentation may help against simple attacks but should be weaker than detector-guided hard negatives on adaptive candidates.

---

## B5. One-shot baseline

### Comparison

- D3_iterative
- D_oneshot_all

### Purpose

To evaluate whether training once with all discovered hard negatives yields similar final performance.

### Important caveat

One-shot training assumes that V5/V6/V7 are already known in advance. Therefore, it does not replace the proposed framework, whose contribution is discovering these hard negatives iteratively.

---

## Recommended minimum baseline set for the paper

1. D0 vs D3_guided
2. Reconstruction-only vs Hybrid OR
3. D3_guided vs D3_random_matched

These three baselines directly support the main claim of detector-guided hard-negative hardening.
