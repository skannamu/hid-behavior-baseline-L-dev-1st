# Baseline Tables

This directory contains baseline and ablation tables generated from the existing paper artifacts.

## Files

- `baseline_no_hardening_vs_guided_or95.csv`
  - OR95 detection rate of D0-D3 against V4-V8.
  - Main table for showing hardening effectiveness.

- `baseline_no_hardening_vs_guided_or90_bypass.csv`
  - OR90 bypass rate of D0-D3 against V4-V8.
  - Main table for showing adaptive bypass reduction.

- `baseline_available_signal_ablation_all_detectors.csv`
  - Classifier-only, prototype-only, hybrid OR95, and hybrid OR90 detection rates from existing summary files.
  - This does not include reconstruction-only for V4-V7 unless those per-signal prediction files are reprocessed.

- `baseline_v8_hardest_d3_full_signal_ablation.csv`
  - Detailed D3-only signal ablation on V8 hardest no-bypass samples.
  - Includes recon-only, latent-distance, classifier, prototype, and hybrid decisions.

- `baseline_generation_yield_summary.csv`
  - Candidate generation / selected bypass yield by round.

## Interpretation

The current tables already support two baseline claims:

1. No-hardening baseline:
   D0 is vulnerable to adaptive candidates, while D3 detects V4-V8.

2. Signal ablation:
   On V8 hardest samples, reconstruction-only fails while classifier/prototype/hybrid detect the samples.

The next missing control experiment is:

- D3_guided vs D3_random_matched

This requires training a random-sample matched detector.
