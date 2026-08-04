# HID Behavior Feature Schema v2 — Draft 0.1

## 1. 목표

이 문서는 HID 기반 BadUSB 행동 탐지를 위한 **Feature Schema v2**의 첫 번째 고정 초안이다.

핵심 원칙은 다음과 같다.

1. `raw.csv`를 source of truth로 보존한다.
2. 한 timestep은 하나의 완성된 물리적 keystroke이다.
3. timestep 순서는 key-up 순서가 아니라 **key-down 시각 순서**이다.
4. 개별 keystroke feature와 50-key window context feature를 분리한다.
5. 정상 데이터와 synthetic attack 데이터는 반드시 동일 extractor를 사용한다.
6. category ID를 연속형 숫자로 모델에 넣지 않는다.
7. threshold와 score policy는 final test에서 최적화하지 않는다.

---

## 2. 데이터 계층

### Raw event

키보드에서 받은 원시 key-down/key-up 이벤트다. 모든 후속 데이터를 다시 생성할 수 있는 원본이다.

### State

raw 이벤트를 press-order 기준 keystroke로 pairing한 표준 중간 표현이다. 정확한 keystroke 관측값과 품질 진단값만 저장한다.

### Window

50개의 press-ordered state를 하나의 모델 입력으로 만든다.

- Sequence tensor: `50 × 12`
- Window context vector: `12`
- Optional category token sequence: `50`개 토큰, 기본 비활성

---

## 3. Sequence model features

| # | 이름 | 정의 | 판정 |
|---:|---|---|---|
| 1 | `hold_time_s` | release − press | 유지 |
| 2 | `press_interval_s` | 현재 press − 이전 press | 기존 P2P 개명·유지 |
| 3 | `signed_flight_time_s` | 현재 press − 이전 release | signed로 재정의 |
| 4 | `overlap_fraction` | 현재 hold 구간에서 다른 키와 겹친 시간의 합집합 / hold | 공식 교체 |
| 5 | `concurrent_keys_at_press` | 현재 key-down 직전에 눌려 있던 다른 physical key 수 | 공식 교체 |
| 6 | `release_inversion_flag` | press-order상 현재 release가 이전 release보다 빠르면 1 | 추가 |
| 7 | `correction_key_flag` | Backspace/Delete면 1 | 추가 |
| 8 | `repeat_flag` | OS repeat가 한 번 이상 발생했으면 1 | 추가 |
| 9 | `shift_at_press` | key-down 당시 Shift 상태 | 추가 |
| 10 | `ctrl_at_press` | key-down 당시 Ctrl 상태 | 추가 |
| 11 | `alt_at_press` | key-down 당시 Alt 상태 | 추가 |
| 12 | `meta_at_press` | key-down 당시 Windows/Meta 상태 | 추가 |

### 중요한 부호 규칙

- `hold_time_s >= 0`
- `press_interval_s >= 0`
- `signed_flight_time_s`는 음수가 정상적으로 가능하다.
- `release_inversion_flag=1`은 오류가 아니라 release 순서 역전을 뜻한다.
- `overlap_fraction`은 반드시 `[0, 1]`이다.

---

## 4. Window context features

| # | 이름 | 정의 |
|---:|---|---|
| 1 | `press_rate_hz` | `(N-1) / window press span` |
| 2 | `active_press_interval_median_s` | pause가 아닌 press interval의 median |
| 3 | `active_press_interval_robust_cv` | `1.4826 × MAD / median` |
| 4 | `pause_rate` | pause interval 수 / 전체 transition 수 |
| 5 | `pause_time_fraction` | pause interval 합 / 전체 press span |
| 6 | `max_pause_interval_s` | 가장 긴 pause interval |
| 7 | `mean_burst_length_keys` | pause 경계로 나눈 burst의 평균 키 수 |
| 8 | `max_burst_length_keys` | 가장 긴 burst의 키 수 |
| 9 | `correction_rate` | correction key 수 / 전체 키 수 |
| 10 | `command_shortcut_rate` | Ctrl/Alt/Meta가 활성화된 비-modifier key 비율 |
| 11 | `overlap_key_rate` | overlap이 존재한 키 비율 |
| 12 | `release_inversion_rate` | release inversion transition 비율 |

`Shift`만 활성화된 대문자·문장부호 입력은 command shortcut으로 계산하지 않는다.

---

## 5. 기존 15개 판정

### 유지 또는 재정의 후 유지

- `hold_time`
- `flight_time` → `signed_flight_time_s`
- `press_to_press_time` → `press_interval_s`
- `overlap_ratio` → `overlap_fraction`
- `simultaneous_key_count` → `concurrent_keys_at_press`

### state 진단용으로만 유지

- `release_to_release_time`
- `modifier_count`
- `shortcut_flag`

### window context로 이동

- `correction_ratio`
- `keys_per_second`

### 기존 구현 폐기 및 교체

- `burst_density`
- `timing_variance`
- `pause_duration`

### 기본 모델에서 제외

- `timing_entropy`
- `current_key_category_id`

category는 문자열/token으로 보존하고 optional embedding ablation에서만 사용한다.

---

## 6. 중복 제거 근거

`release_interval[i]`는 다음으로 계산 가능하다.

```text
release_interval[i]
= press_interval[i] + hold_time[i] - hold_time[i-1]
```

따라서 sequence model에는 넣지 않는다.

또한 legacy `burst_density`는 실제 코드에서 `keys_per_second`와 동일했으므로 제거한다.

`pause_count`, `pause_rate`, `burst_count`는 window size가 고정일 때 강하게 종속되므로 baseline context에는 `pause_rate`만 넣고 burst는 길이 통계만 사용한다.

---

## 7. Pairing 및 edge-case 정책

- physical key ID: `(session-local device ID, make code, extended flag)`
- 같은 physical key의 중복 down:
  - 새 keystroke를 만들지 않는다.
  - `repeat_count`를 증가시킨다.
- unmatched key-up:
  - 품질 이벤트로 기록
  - 모델용 state에서는 제외
- 세션 종료 시 미완성 key:
  - 품질 이벤트로 기록
  - window에서 제외
- modifier 상태:
  - 반드시 현재 일반 키의 key-down 시점 snapshot 사용
- 시간 계산:
  - monotonic high-resolution integer nanoseconds 사용

---

## 8. Window 생성 규칙

1. 전체 세션에서 keystroke feature를 먼저 계산한다.
2. 이전 keystroke가 필요한 transition feature가 유효하지 않은 첫 keystroke는 model window 시작점에서 제외한다.
3. 50개씩 stride 1로 window를 생성한다.
4. row metadata에 schema version과 schema hash를 기록한다.
5. schema가 다른 파일은 합치지 않는다.

---

## 9. 분할 규칙

정상 데이터:

```text
participant_id + session_id
```

단위로 train/calibration/test를 분할한다.

Synthetic attack:

```text
attack_trial_id + lineage_id
```

단위로 분할한다.

금지:

- window row random split
- calibration set과 final test 동일 사용
- final test에서 threshold 또는 fusion weight 최적화

---

## 10. 다음 구현 순서

1. `src/features/schema.py`
2. `src/features/keystroke_builder.py`
3. `src/features/extractor.py`
4. `src/features/window_builder.py`
5. `src/features/validators.py`
6. Windows Collector v4에 raw writer + 공통 extractor 연결
7. Framework v9 dataset loader
8. event-level synthetic generator
9. participant/session group split
10. calibration/final-test 분리
