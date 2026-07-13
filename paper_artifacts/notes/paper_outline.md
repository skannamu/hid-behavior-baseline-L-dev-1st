# Paper Outline

## Tentative Title

Detector-Guided Hard-Negative Generation for Hardening HID-Based BadUSB Behavior Detectors

Korean title:
HID 기반 BadUSB 행위 탐지기 강건화를 위한 Detector-Guided Hard-Negative 생성 프레임워크

---

## 1. Abstract

### Key points
- BadUSB attacks abuse HID interfaces and appear as legitimate keyboard input.
- Behavior-based HID detection can detect such attacks, but adaptive human-mimic patterns can expose blind spots.
- This paper proposes a detector-guided hard-negative generation and hardening framework.
- The framework iteratively generates adaptive hard negatives, evaluates detector bypasses, and retrains the detector.
- In experiments, D0 was bypassed by V5, D1 by V6, and D2 by V7.
- After hardening with V7, D3 detected V4-V7 and no D3-bypassing V8 candidate was found among the generated candidates.
- Bypass yield decreased from 67.53% to 1.68% and finally to 0.00%.

---

## 2. Introduction

### Problem
- BadUSB/HID attacks are difficult to detect by device identity alone.
- HID devices appear as normal keyboards.
- Timing and behavior patterns can be used for detection.

### Gap
- A detector that performs well on fixed attack datasets may still have blind spots.
- Adaptive human-mimic sequences can be generated to exploit detector weaknesses.
- Existing evaluation often stops at static test sets.

### Main idea
- Treat detector evaluation and hardening as a co-evolution loop.
- Generate detector-guided hard negatives.
- Use discovered bypass samples to harden the detector.
- Repeat until bypass candidates are no longer found under the search budget.

### Contributions
1. Propose a detector-guided hard-negative generation framework for HID behavior detectors.
2. Implement ReCon-HID v2, a hybrid detector using reconstruction, classifier, and prototype/latent-distance signals.
3. Demonstrate iterative detector hardening through D0-D3 and V5-V8 co-evolution.
4. Show that bypass yield decreases across rounds, reaching 0.00% against D3 under the current search budget.

---

## 3. Background

### 3.1 BadUSB and HID-based injection
- BadUSB abuses HID functionality to inject keyboard-like input.
- The OS treats the device as a legitimate keyboard.

### 3.2 Behavior-based HID detection
- Instead of analyzing command semantics, use timing and behavioral features.
- Features include hold time, flight time, key rate, burst density, correction ratio, pause duration, and key category.

### 3.3 Hard negatives
- Hard negatives are attack-like samples that the detector incorrectly treats as normal.
- Training with hard negatives can improve detector robustness.

---

## 4. Method

### 4.1 HID behavior feature representation
- Input window: 50 timesteps × 15 features.
- Features summarize timing, overlap, correction, rate, entropy, and key category behavior.

### 4.2 ReCon-HID v2 detector
- Figure: `figures/v2/reconhid_v2_architecture.png`
- LSTM encoder.
- Reconstruction head.
- Classifier head.
- Prototype/latent-distance head.
- Fusion/oracle decision using OR90 and OR95.

### 4.3 Detector-guided candidate generation
- Generate candidate HID behavior sequences from attack seed data.
- Evaluate candidates with the current detector.
- Select bypass candidates as hard negatives.

### 4.4 Iterative hard-negative hardening
- Figure: `figures/v2/framework_loop_diagram_v3.png`
- Train D_r.
- Generate V_{r+1}.
- Select bypass hard negatives.
- Retrain D_{r+1}.
- Continue until no bypass candidate is found or stopping conditions are met.

---

## 5. Experimental Setup

### 5.1 Dataset
- Normal: `data/processed/window_typing_10000.csv`
- Initial attack seed: FeatureGuideHumanMimic V4
- Adaptive datasets:
  - V5: AdaptiveMimic_v5_constrained
  - V6: AdaptiveMimic_v6_D1_OR90_bypass
  - V7: AdaptiveMimic_v7_D2_OR90_bypass
  - V8: D3-guided candidate generation attempt

### 5.2 Detector rounds
- D0: initial ReCon-HID v2 detector
- D1: hardened with V5
- D2: hardened with V6
- D3: hardened with V7

### 5.3 Metrics
- OR95 detection rate
- OR90 detection rate
- OR95 bypass rate
- OR90 bypass rate
- Normal false positive rate
- Classifier detection rate
- Prototype detection rate
- Bypass yield

---

## 6. Results

### 6.1 D0-D3 benchmark against V4-V7
- Figure: `figures/v2/detector_attack_heatmap_or95_v2.png`
- Table: `tables/benchmark_summary_v4_v5_v6_v7_d0_d1_d2_d3.csv`

Main observation:
- D0 is bypassed by V5.
- D1 detects V5 but is bypassed by V6.
- D2 detects V6 but is bypassed by V7.
- D3 detects V4-V7.

### 6.2 Bypass yield reduction
- Figure: `figures/v2/bypass_yield_by_round_v2.png`
- Table: `tables/bypass_yield_by_round.csv`

Main observation:
- D1 → V6: 67.53%
- D2 → V7: 1.68%
- D3 → V8: 0.00%

### 6.3 V8 no-bypass result
- Table: `tables/v8_generation_summary.csv`
- Table: `tables/benchmark_v8_candidate_d0_d1_d2_d3.csv`
- Table: `tables/benchmark_v8_hardest_no_bypass_d0_d1_d2_d3.csv`

Main observation:
- No OR90/OR95 D3-bypassing V8 candidate was found under the current search budget.
- D3 detected both V8 candidate set and hardest no-bypass set.

### 6.4 OR90 bypass heatmap
- Figure: `figures/v2/detector_attack_heatmap_or90_bypass_v2.png`

Main observation:
- Earlier detectors are bypassed by later adaptive candidates.
- D3 shows 0% bypass rate across evaluated V4-V8 sets.

---

## 7. Discussion

### 7.1 Meaning of co-evolution
- Detector-guided generation reveals blind spots.
- Hard-negative retraining reduces exploitable blind spot density.

### 7.2 Why hybrid detection matters
- Reconstruction-only signals can be insufficient.
- Classifier/prototype/latent signals improve detection against adaptive hard negatives.

### 7.3 Framework generality
- The framework is not tied to one fixed detector checkpoint.
- Given new normal and attack seed datasets, the same loop can train customized detectors.
- However, the current implementation is a research prototype.

---

## 8. Limitations

- Current normal data comes from a controlled collection setting.
- Attack seeds were generated within the current HID behavior feature space.
- The framework generates behavior-level hard negatives, not executable BadUSB payloads.
- Cross-user, cross-device, and cross-attack generalization require further validation.
- Search budget limits the strength of the no-bypass claim.

---

## 9. Conclusion

- Proposed a detector-guided hard-negative hardening framework for HID-based BadUSB behavior detectors.
- Demonstrated D0-D3 / V5-V8 co-evolution.
- Showed bypass yield reduction from 67.53% to 1.68% to 0.00%.
- Results support the use of adaptive hard-negative generation for detector evaluation and hardening.
