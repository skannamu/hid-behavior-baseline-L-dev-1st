# Code Manifest

## Core implementation
- `paper_artifacts/code/core/`
  - Dataset loading
  - LSTM/Hybrid/ReCon-HID model definitions
  - Core model components used by D0-D3

## Benchmark code
- `paper_artifacts/code/benchmark/`
  - Attack dataset evaluation
  - Detector-vs-attack benchmark table generation

## ReCon-HID v2 detector
- `paper_artifacts/code/experiments/recon_hid_v2/`
  - D0 detector training/evaluation/oracle analysis code

## Co-evolution rounds
- `paper_artifacts/code/experiments/co_evolution_round1/`
  - V5 generation / D1 hardening
- `paper_artifacts/code/experiments/co_evolution_round2/`
  - D2 hardening with V6 hard negatives
- `paper_artifacts/code/experiments/co_evolution_round3/`
  - D3 hardening with V7 hard negatives

## Adaptive generation
- `paper_artifacts/code/experiments/attack_generation_v6/`
  - D1-guided V6 generation
- `paper_artifacts/code/experiments/attack_generation_v7/`
  - D2-guided V7 generation
- `paper_artifacts/code/experiments/attack_generation_v8/`
  - D3-guided V8 candidate generation and no-bypass analysis

## Root scripts
- `paper_artifacts/code/root_scripts/`
  - Root-level scripts that were used during the experiment before full reorganization

## Environment
- `paper_artifacts/code/requirements.txt`
  - Python package dependencies
