# HID Behavior Feature Schema v2.0.0 — Stable

## 1. 목적과 고정 원칙

이 문서는 HID 기반 BadUSB 행동 탐지를 위한 Feature Schema v2의 안정 버전이다.

1. `raw.csv`는 삭제·수정하지 않는 source of truth다.
2. 한 timestep은 하나의 완성된 물리적 keystroke다.
3. state 순서는 key-up이 아니라 key-down 시각 순서다.
4. 정상 수집, offline replay, synthetic attack은 동일 extractor를 사용한다.
5. 모델 입력은 sequence `50 × 12`와 window context `12`로 구성한다.
6. category ID를 연속형 수치 feature로 사용하지 않는다.
7. 서로 다른 schema version/hash의 window는 함께 로드하지 않는다.

## 2. v2.0.0 최종 변경점

### Modifier snapshot

모든 modifier bit는 현재 key-down 직전에 이미 활성화되어 있던 modifier만 나타낸다.
현재 modifier 키는 자기 timestep을 활성화하지 않는다.

```text
Shift down  -> shift_at_press = 0
A down      -> shift_at_press = 1
A up
Shift up
```

### 시작 경계

수집 시작 Enter의 key-up이 Raw Input 등록 뒤 도착하는 현상을 방지한다.

- 준비 Enter 뒤 0.75초 settle
- 첫 key-down 전 1.5초 이내의 선행 unmatched key-up은 boundary info로 기록하고 모델 raw에서 제외
- 첫 key-down 이후에는 일반 unmatched key-up 정책을 그대로 적용

### 종료 경계

Ctrl+C, 창 종료 순간의 chord, 눌린 채 끝난 키가 마지막 window를 오염하지 않도록 한다.

- raw 이벤트는 모두 보존한다.
- 종료 시 남아 있는 incomplete key 중 가장 이른 key-down을 cutoff로 정한다.
- key-up이 cutoff보다 엄격히 앞선 completed keystroke만 state에 유지한다.
- cutoff와 겹치거나 그 뒤에 끝난 completed keystroke는 state/window에서 제외한다.
- trim 개수와 cutoff raw index/timestamp를 summary와 quality log에 기록한다.

예시:

```text
정상 입력 ...
Ctrl down       <- earliest incomplete boundary
C down
C up
프로그램 종료
```

위 경우 Ctrl은 incomplete이고 C는 completed지만 둘 다 모델 state/window에서 제외된다.
`raw.csv`에는 원본 그대로 남는다.

### Context 중복 제거

고정 window 크기 `N=50`에서 다음 관계가 성립한다.

```text
mean_burst_length_keys = N / (pause_count + 1)
pause_count = (N - 1) * pause_rate
```

따라서 `mean_burst_length_keys`는 `pause_rate`의 정확한 결정론적 변환이다.
이를 제거하고 독립적인 dwell variability 요약인 `hold_time_robust_cv`로 교체한다.

```text
hold_time_robust_cv
= 1.4826 × MAD(hold_time_s) / max(median(hold_time_s), 1e-6)
```

## 3. 데이터 계층

### Raw event

필수 필드:

```text
session_id, raw_event_index, timestamp_ns,
device_id_session_local, make_code, extended_flag,
vkey, message, action
```

행동 시간 계산은 `time.perf_counter_ns()` 기반 monotonic nanoseconds만 사용한다.
wall-clock은 메타데이터용이다.

### State

raw down/up을 physical key ID `(device, make_code, extended_flag)`로 pairing한
press-order canonical keystroke다.

### Window

- Sequence: `50 × 12`
- Context: `12`
- Metadata: 10 columns
- 총 window CSV columns: `10 + 600 + 12 = 622`

## 4. Sequence model features

| # | Feature | 의미 |
|---:|---|---|
| 1 | `hold_time_s` | 현재 release − 현재 press |
| 2 | `press_interval_s` | 현재 press − 이전 press |
| 3 | `signed_flight_time_s` | 현재 press − 이전 release; 음수 rollover 허용 |
| 4 | `overlap_fraction` | 현재 hold 중 다른 키와 겹친 union duration / hold |
| 5 | `concurrent_keys_at_press` | 현재 key-down 직전 이미 눌린 다른 physical key 수 |
| 6 | `release_inversion_flag` | 현재 release가 이전 press-order key release보다 빠르면 1 |
| 7 | `correction_key_flag` | Backspace/Delete면 1 |
| 8 | `repeat_flag` | 동일 physical key의 OS repeat down이 존재하면 1 |
| 9 | `shift_at_press` | 직전 Shift 활성 상태 |
| 10 | `ctrl_at_press` | 직전 Ctrl 활성 상태 |
| 11 | `alt_at_press` | 직전 Alt 활성 상태 |
| 12 | `meta_at_press` | 직전 Windows/Meta 활성 상태 |

## 5. Window context features

| # | Feature | 정의 |
|---:|---|---|
| 1 | `press_rate_hz` | `(N−1) / press span` |
| 2 | `active_press_interval_median_s` | pause 미만 press interval median |
| 3 | `active_press_interval_robust_cv` | active interval의 `1.4826 × MAD / median` |
| 4 | `pause_rate` | pause transition 수 / `(N−1)` |
| 5 | `pause_time_fraction` | pause interval 합 / 전체 interval 합 |
| 6 | `max_pause_interval_s` | 최대 pause interval, 없으면 0 |
| 7 | `hold_time_robust_cv` | hold time의 robust CV |
| 8 | `max_burst_length_keys` | pause 경계로 나눈 가장 긴 burst의 키 수 |
| 9 | `correction_rate` | correction key 수 / N |
| 10 | `command_shortcut_rate` | Ctrl/Alt/Meta 활성 비-modifier key 비율 |
| 11 | `overlap_key_rate` | overlap이 존재한 key 비율 |
| 12 | `release_inversion_rate` | window 내부 release inversion 비율 |

Shift만 활성화된 대문자·문장부호 입력은 command shortcut으로 계산하지 않는다.

## 6. Pairing 및 품질 정책

- duplicate down: 새 keystroke를 만들지 않고 repeat count 증가
- unmatched up: quality warning, state 제외
- negative hold: quality error, state 제외
- zero hold: state 유지, quality flag 추가
- incomplete at end: quality warning + trailing boundary truncate
- 모든 overlap fraction은 `[0,1]`
- 모든 binary sequence feature는 `{0,1}`
- 모든 window 수치는 finite여야 한다.

## 7. Window 생성

1. boundary-safe press-order state prefix를 생성한다.
2. 이전 keystroke가 필요한 transition 값이 없는 첫 state는 eligible sequence에서 제외한다.
3. 50개씩 stride 1로 생성한다.
4. metadata에 schema version과 YAML SHA-256을 기록한다.
5. exact 622-column header/order를 강제한다.

## 8. 분할 및 평가

정상 데이터는 최소 `participant_id + session_id` group으로 분리한다.
Synthetic attack은 `attack_trial_id + lineage_id`로 분리한다.

금지:

- stride-1 window random split
- calibration과 final test 재사용
- final test에서 threshold/fusion weight 최적화
- schema/hash 혼합

## 9. 안정 버전 식별

```text
Feature Schema: 2.0.0
Collector: v4.3 Final
Extractor: feature_core_v2_3
Dataset root: dataset_v2_final
```
