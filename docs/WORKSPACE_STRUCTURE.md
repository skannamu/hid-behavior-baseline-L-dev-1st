# Workspace Structure

## Active checkpoints

- `checkpoints/recon_hid_v2.pt`: D0, Round 0 ReCon-HID detector
- `checkpoints/recon_hid_v2_round1.pt`: D1, V5 hard-negative retrained detector
- `checkpoints/scaler.pkl`: shared feature scaler

## Active datasets

- `data/processed/window_typing_10000.csv`: normal HID behavior windows
- `data/attack/FeatureGuideHumanMimic`: V4 attack dataset
- `data/attack/AdaptiveMimic_v5_constrained_clean`: final V5 clean dataset
- `data/attack/AdaptiveMimic_v5_constrained_train`: V5 train split
- `data/attack/AdaptiveMimic_v5_constrained_test`: V5 test split

## Active reports

- `reports/round0_recon_hid_v2`: D0 evaluation results
- `reports/round1_adaptive_mimic_v5`: V5 generation results
- `reports/round1_detector_d1`: D1 evaluation results
- `reports/final_comparison`: model comparison figures

## Current research direction

The project is now organized around detector-guided adaptive HID attack generation.
D0 and D1 are used as benchmark/oracle detectors for evaluating attack datasets.
