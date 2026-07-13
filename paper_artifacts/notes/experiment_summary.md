# Experiment Summary

## Main Story
D0 was bypassed by V5.
D1 detected V5 but was bypassed by V6.
D2 detected V6 but was bypassed by V7.
D3 detected V4, V5, V6, and V7 test sets.
D3-guided V8 generation produced no valid OR90/OR95 bypass candidate within the current search budget.

## Main Detectors
- D0: recon_hid_v2.pt
- D1: recon_hid_v2_round1.pt
- D2: recon_hid_v2_round2.pt
- D3: recon_hid_v2_round3.pt

## Main Attacks
- V4: FeatureGuideHumanMimic
- V5: AdaptiveMimic_v5_constrained
- V6: AdaptiveMimic_v6_D1_OR90_bypass
- V7: AdaptiveMimic_v7_D2_OR90_bypass
- V8: D3-guided candidate generation attempt; no valid bypass found
