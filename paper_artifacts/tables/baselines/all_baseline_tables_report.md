# All Baseline Comparison Tables

This report collects all baseline and ablation tables for the detector-guided hard-negative hardening study.


## Table 1. No-hardening vs Guided Hardening: OR95 Detection

D0부터 D3까지 hardening round가 진행되면서 V4~V8 공격군 탐지율이 어떻게 변하는지 보여준다.

| detector | V4 | V5 | V6 | V7 | V8_candidate | V8_hardest |
| --- | --- | --- | --- | --- | --- | --- |
| d0 | 45.57 | 0.00 | 0.00 | 0.00 | 14.59 | 0.00 |
| d1 | 100.00 | 100.00 | 0.00 | 0.00 | 29.15 | 0.00 |
| d2 | 100.00 | 100.00 | 100.00 | 0.00 | 97.87 | 77.60 |
| d3 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |


## Table 2. No-hardening vs Guided Hardening: OR90 Bypass

공격자 관점의 우회율 표다. D0/D1/D2의 blind spot이 D3에서 제거되는지 확인한다.

| detector | V4 | V5 | V6 | V7 | V8_candidate | V8_hardest |
| --- | --- | --- | --- | --- | --- | --- |
| d0 | 26.56 | 100.00 | 87.68 | 100.00 | 79.39 | 100.00 |
| d1 | 0.00 | 0.00 | 100.00 | 100.00 | 67.18 | 100.00 |
| d2 | 0.00 | 0.00 | 0.00 | 100.00 | 2.12 | 22.40 |
| d3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


## Table 3. Available Signal Ablation Across Detectors

classifier-only, prototype-only, hybrid OR95, hybrid OR90 신호별 탐지율을 비교한다.

| detector | signal | V4 | V5 | V6 | V7 | V8_candidate | V8_hardest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d0 | classifier_only | 19.27 | 0.00 | 0.00 | 0.00 | 0.88 | 0.00 |
| d0 | prototype_only | 18.75 | 0.00 | 0.00 | 0.00 | 0.87 | 0.00 |
| d0 | hybrid_or95 | 45.57 | 0.00 | 0.00 | 0.00 | 14.59 | 0.00 |
| d0 | hybrid_or90 | 73.44 | 0.00 | 12.32 | 0.00 | 20.61 | 0.00 |
| d1 | classifier_only | 100.00 | 100.00 | 0.00 | 0.00 | 17.49 | 0.00 |
| d1 | prototype_only | 100.00 | 100.00 | 0.00 | 0.00 | 17.40 | 0.00 |
| d1 | hybrid_or95 | 100.00 | 100.00 | 0.00 | 0.00 | 29.15 | 0.00 |
| d1 | hybrid_or90 | 100.00 | 100.00 | 0.00 | 0.00 | 32.82 | 0.00 |
| d2 | classifier_only | 100.00 | 100.00 | 100.00 | 0.00 | 97.85 | 77.60 |
| d2 | prototype_only | 100.00 | 100.00 | 100.00 | 0.00 | 97.81 | 77.40 |
| d2 | hybrid_or95 | 100.00 | 100.00 | 100.00 | 0.00 | 97.87 | 77.60 |
| d2 | hybrid_or90 | 100.00 | 100.00 | 100.00 | 0.00 | 97.88 | 77.60 |
| d3 | classifier_only | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| d3 | prototype_only | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| d3 | hybrid_or95 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| d3 | hybrid_or90 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |


## Table 4. D3 Full Signal Ablation on V8 Hardest

V8 hardest에서 reconstruction-only AE가 실패하고 latent/classifier/prototype signal이 성공하는지 보여준다.

| attack_set | signal | samples | detected | detection_rate | bypass_rate |
| --- | --- | --- | --- | --- | --- |
| V8_hardest | recon_90 | 500 | 0 | 0.00 | 100.00 |
| V8_hardest | recon_95 | 500 | 0 | 0.00 | 100.00 |
| V8_hardest | latent_dist_90 | 500 | 500 | 100.00 | 0.00 |
| V8_hardest | latent_dist_95 | 500 | 500 | 100.00 | 0.00 |
| V8_hardest | classifier_only | 500 | 500 | 100.00 | 0.00 |
| V8_hardest | prototype_only | 500 | 500 | 100.00 | 0.00 |
| V8_hardest | hybrid_or90 | 500 | 500 | 100.00 | 0.00 |
| V8_hardest | hybrid_or95 | 500 | 500 | 100.00 | 0.00 |
| V8_hardest | hybrid_or95_with_latent | 500 | 500 | 100.00 | 0.00 |


## Table 5. Generation / Bypass Yield by Round

D1→V6, D2→V7, D3→V8로 갈수록 bypass yield가 감소하는지 보여준다.

| round | generated_candidates | selected_or_bypass | yield_rate | note |
| --- | --- | --- | --- | --- |
| D0 → V5 | 1672 | 1000 | 59.81 | selected constrained OR95 bypass |
| D1 → V6 | 3600 | 2431 | 67.53 | D1 OR90 bypass |
| D2 → V7 | 15000 | 252 | 1.68 | D2 OR90 bypass |
| D3 → V8 | 15000 | 0 | 0.00 | no D3 OR90/OR95 bypass found; top-500 hardest no-bypass separately saved |


## Table 6. Random-Matched Dataset Summary

D3_random_matched baseline이 guided baseline과 같은 수의 V6/V7 sample을 사용했는지 확인한다.

| candidate_path | guided_train_path | out_path | candidate_rows | guided_train_rows | random_rows | seed |
| --- | --- | --- | --- | --- | --- | --- |
| data/attack/AdaptiveMimic_v6_candidate/v6_candidates.csv | data/attack/AdaptiveMimic_v6_D1_OR90_bypass_train/v6_or90_train.csv | data/attack/RandomMatched_D3_train/v6_random_matched_train.csv | 3600 | 1944 | 1944 | 42 |
| data/attack/AdaptiveMimic_v7_candidate/v7_candidates.csv | data/attack/AdaptiveMimic_v7_D2_OR90_bypass_train/v7_or90_train.csv | data/attack/RandomMatched_D3_train/v7_random_matched_train.csv | 15000 | 201 | 201 | 43 |


## Table 7. D3 Guided vs Random-Matched: Full Summary

D3_guided와 D3_random_matched의 전체 상세 평가 결과다.

| detector | detector_description | attack | attack_short | prediction_csv | attack_samples | normal_samples | or95_detection_rate | or95_bypass_rate | or95_normal_fpr | or90_detection_rate | or90_bypass_rate | or90_normal_fpr | classifier_detection_rate | classifier_bypass_rate | classifier_normal_fpr | prototype_detection_rate | prototype_bypass_rate | prototype_normal_fpr | recon90_detection_rate | recon90_bypass_rate | recon90_normal_fpr | recon95_detection_rate | recon95_bypass_rate | recon95_normal_fpr | latent90_detection_rate | latent90_bypass_rate | latent90_normal_fpr | latent95_detection_rate | latent95_bypass_rate | latent95_normal_fpr | or95_with_latent_detection_rate | or95_with_latent_bypass_rate | or95_with_latent_normal_fpr | mean_reconstruction_error | mean_classifier_attack_probability | mean_prototype_attack_probability | mean_distance_to_normal_prototype | mean_distance_to_attack_prototype | mean_latent_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3_guided | D3 trained with detector-guided V6/V7 hard negatives | FeatureGuideHumanMimic_v4 | V4 | reports/benchmark/d3_random_matched_baseline/V4/oracle_D3_guided_FeatureGuideHumanMimic_v4/oracle_predictions.csv | 384 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 79.17 | 20.83 | 10.01 | 61.46 | 38.54 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.83 | 1.23 | 1.00 | 1.00 | 3.06 | 1.70 | 5.00 |
| D3_random_matched | D3 baseline trained with random-matched V6/V7 candidates | FeatureGuideHumanMimic_v4 | V4 | reports/benchmark/d3_random_matched_baseline/V4/oracle_D3_random_matched_FeatureGuideHumanMimic_v4/oracle_predictions.csv | 384 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 18.49 | 81.51 | 10.01 | 11.20 | 88.80 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.80 | 0.92 | 1.00 | 1.00 | 2.79 | 1.17 | 5.19 |
| D3_guided | D3 trained with detector-guided V6/V7 hard negatives | AdaptiveMimic_v5_constrained_test | V5_test | reports/benchmark/d3_random_matched_baseline/V5_test/oracle_D3_guided_AdaptiveMimic_v5_constrained_test/oracle_predictions.csv | 201 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 2.49 | 97.51 | 10.01 | 0.00 | 100.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.83 | 0.77 | 1.00 | 1.00 | 3.06 | 1.69 | 5.01 |
| D3_random_matched | D3 baseline trained with random-matched V6/V7 candidates | AdaptiveMimic_v5_constrained_test | V5_test | reports/benchmark/d3_random_matched_baseline/V5_test/oracle_D3_random_matched_AdaptiveMimic_v5_constrained_test/oracle_predictions.csv | 201 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 10.01 | 0.00 | 100.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.80 | 0.62 | 1.00 | 1.00 | 2.80 | 1.16 | 5.22 |
| D3_guided | D3 trained with detector-guided V6/V7 hard negatives | AdaptiveMimic_v6_D1_OR90_bypass_test | V6_test | reports/benchmark/d3_random_matched_baseline/V6_test/oracle_D3_guided_AdaptiveMimic_v6_D1_OR90_bypass_test/oracle_predictions.csv | 487 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 23.61 | 76.39 | 10.01 | 13.35 | 86.65 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.83 | 0.49 | 1.00 | 1.00 | 3.06 | 1.69 | 5.02 |
| D3_random_matched | D3 baseline trained with random-matched V6/V7 candidates | AdaptiveMimic_v6_D1_OR90_bypass_test | V6_test | reports/benchmark/d3_random_matched_baseline/V6_test/oracle_D3_random_matched_AdaptiveMimic_v6_D1_OR90_bypass_test/oracle_predictions.csv | 487 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 17.45 | 82.55 | 10.01 | 0.00 | 100.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.80 | 0.54 | 1.00 | 1.00 | 2.80 | 1.17 | 5.24 |
| D3_guided | D3 trained with detector-guided V6/V7 hard negatives | AdaptiveMimic_v7_D2_OR90_bypass_test | V7_test | reports/benchmark/d3_random_matched_baseline/V7_test/oracle_D3_guided_AdaptiveMimic_v7_D2_OR90_bypass_test/oracle_predictions.csv | 51 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 10.01 | 0.00 | 100.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.83 | 0.13 | 1.00 | 1.00 | 3.07 | 1.70 | 5.02 |
| D3_random_matched | D3 baseline trained with random-matched V6/V7 candidates | AdaptiveMimic_v7_D2_OR90_bypass_test | V7_test | reports/benchmark/d3_random_matched_baseline/V7_test/oracle_D3_random_matched_AdaptiveMimic_v7_D2_OR90_bypass_test/oracle_predictions.csv | 51 | 10013 | 76.47 | 23.53 | 5.00 | 76.47 | 23.53 | 10.01 | 76.47 | 23.53 | 0.00 | 74.51 | 25.49 | 0.00 | 0.00 | 100.00 | 10.01 | 0.00 | 100.00 | 5.00 | 88.24 | 11.76 | 10.01 | 84.31 | 15.69 | 5.00 | 84.31 | 15.69 | 8.80 | 0.44 | 0.76 | 0.74 | 2.31 | 1.71 | 4.60 |
| D3_guided | D3 trained with detector-guided V6/V7 hard negatives | AdaptiveMimic_v8_candidate | V8_candidate | reports/benchmark/d3_random_matched_baseline/V8_candidate/oracle_D3_guided_AdaptiveMimic_v8_candidate/oracle_predictions.csv | 15000 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 23.68 | 76.32 | 10.01 | 18.77 | 81.23 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.83 | 0.72 | 1.00 | 1.00 | 3.06 | 1.69 | 5.02 |
| D3_random_matched | D3 baseline trained with random-matched V6/V7 candidates | AdaptiveMimic_v8_candidate | V8_candidate | reports/benchmark/d3_random_matched_baseline/V8_candidate/oracle_D3_random_matched_AdaptiveMimic_v8_candidate/oracle_predictions.csv | 15000 | 10013 | 99.34 | 0.66 | 5.00 | 99.34 | 0.66 | 10.01 | 99.32 | 0.68 | 0.00 | 99.27 | 0.73 | 0.00 | 17.28 | 82.72 | 10.01 | 13.33 | 86.67 | 5.00 | 99.75 | 0.25 | 10.01 | 99.73 | 0.27 | 5.00 | 99.73 | 0.27 | 8.80 | 0.74 | 0.99 | 0.99 | 2.78 | 1.18 | 5.22 |
| D3_guided | D3 trained with detector-guided V6/V7 hard negatives | AdaptiveMimic_v8_D3_hardest_no_bypass | V8_hardest | reports/benchmark/d3_random_matched_baseline/V8_hardest/oracle_D3_guided_AdaptiveMimic_v8_D3_hardest_no_bypass/oracle_predictions.csv | 500 | 10013 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 10.01 | 0.00 | 100.00 | 5.00 | 100.00 | 0.00 | 10.01 | 100.00 | 0.00 | 5.00 | 100.00 | 0.00 | 8.83 | 0.08 | 1.00 | 1.00 | 3.06 | 1.69 | 5.03 |
| D3_random_matched | D3 baseline trained with random-matched V6/V7 candidates | AdaptiveMimic_v8_D3_hardest_no_bypass | V8_hardest | reports/benchmark/d3_random_matched_baseline/V8_hardest/oracle_D3_random_matched_AdaptiveMimic_v8_D3_hardest_no_bypass/oracle_predictions.csv | 500 | 10013 | 85.20 | 14.80 | 5.00 | 85.20 | 14.80 | 10.01 | 85.20 | 14.80 | 0.00 | 84.80 | 15.20 | 0.00 | 0.00 | 100.00 | 10.01 | 0.00 | 100.00 | 5.00 | 93.00 | 7.00 | 10.01 | 92.20 | 7.80 | 5.00 | 92.20 | 7.80 | 8.80 | 0.29 | 0.85 | 0.85 | 2.56 | 1.44 | 5.02 |


## Table 8. D3 Guided vs Random-Matched: OR95 Detection

D3_guided와 D3_random_matched의 OR95 탐지율 핵심 비교표다.

| detector | V4 | V5_test | V6_test | V7_test | V8_candidate | V8_hardest |
| --- | --- | --- | --- | --- | --- | --- |
| D3_guided | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| D3_random_matched | 100.00 | 100.00 | 100.00 | 76.47 | 99.34 | 85.20 |


## Table 9. D3 Guided vs Random-Matched: OR90 Bypass

D3_guided와 D3_random_matched의 OR90 우회율 비교표다.

| detector | V4 | V5_test | V6_test | V7_test | V8_candidate | V8_hardest |
| --- | --- | --- | --- | --- | --- | --- |
| D3_guided | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| D3_random_matched | 0.00 | 0.00 | 0.00 | 23.53 | 0.66 | 14.80 |


## Table 10. D3 Guided vs Random-Matched: Compact Final Table

논문 본문에 바로 넣기 좋은 compact 최종 비교표다.

| attack_set | D3_guided_OR95_detection | D3_random_matched_OR95_detection | OR95_detection_gap | D3_guided_OR90_bypass | D3_random_matched_OR90_bypass | OR90_bypass_gap |
| --- | --- | --- | --- | --- | --- | --- |
| V4 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| V5_test | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| V6_test | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| V7_test | 100.00 | 76.47 | 23.53 | 0.00 | 23.53 | 23.53 |
| V8_candidate | 100.00 | 99.34 | 0.66 | 0.00 | 0.66 | 0.66 |
| V8_hardest | 100.00 | 85.20 | 14.80 | 0.00 | 14.80 | 14.80 |
