# Figure Captions

## Figure 1. Detector-guided hard-negative hardening framework

The proposed framework takes normal HID behavior data and initial attack seed data as input, trains a detector D_r, and generates detector-guided candidate sequences against the current detector. The generated candidates are evaluated by an oracle decision rule. If bypass candidates are found, they are selected as hard negatives and used to retrain the detector into D_{r+1}. This loop continues until no bypass candidate is found under the configured search budget.

## Figure 2. ReCon-HID v2 hybrid detector architecture

The ReCon-HID v2 detector takes a 50×15 HID behavior feature sequence as input and encodes it using an LSTM encoder. The latent representation is used by three decision heads: a reconstruction head, a classifier head, and a prototype/latent-distance head. Their outputs are combined by a fusion/oracle decision rule using OR90 and OR95 thresholds.

## Figure 3. OR95 detection rate across detectors and attack rounds

OR95 detection rate of D0-D3 against V4-V8 attack and candidate sets. D0 is partially effective against V4 but is bypassed by the adaptive V5-V7 sequences. Each hardening round improves the detector against the previously bypassing attack family. The final detector D3 achieves 100% OR95 detection on V4-V7 and also detects all V8 candidate and hardest no-bypass samples.

## Figure 4. Bypass yield decreases across co-evolution rounds

Bypass yield of detector-guided candidate generation across co-evolution rounds. The bypass yield drops from 67.53% against D1 to 1.68% against D2 and finally to 0.00% against D3. This indicates that iterative hard-negative retraining progressively reduces the density of exploitable blind spots under the current feature-space mutation strategy and search budget.

## Figure 5. OR90 bypass rate across detectors and attack rounds

OR90 bypass rate of D0-D3 against V4-V8 attack and candidate sets. The heatmap shows a clear co-evolution pattern: earlier detectors are bypassed by later adaptive attack sets, while each subsequent detector hardening round eliminates the previously observed bypass family. D3 shows 0% OR90 bypass rate on all evaluated V4-V8 sets.

## Supplementary Figure S1. Detector performance across adaptive attack rounds

Grouped bar plot of OR95 detection rates for D0-D3 against V4-V7. This visualization provides an alternative view of the detector hardening progression shown in the OR95 detection heatmap.

## Supplementary Figure S2. Detector-guided co-evolution timeline

Concrete experimental timeline of the D0-D3 and V5-V8 co-evolution process. Starting from normal HID behavior data and the V4 seed, the framework generates V5, hardens D1, generates V6, hardens D2, generates V7, hardens D3, and finally attempts V8 generation, where no valid bypass candidate is found.
