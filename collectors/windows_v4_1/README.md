# HID Behavior Collector v4

## 상태

- Feature Schema: `2.0.0-draft.1`
- Collector: `v4`
- Core: `feature_core_v2_1`
- Windows 전용
- 기존 v3 데이터와 출력 경로를 분리하기 위해 `dataset_v2/`를 사용

## 처리 구조

```text
Windows Raw Input
→ raw/raw.csv 즉시 저장
→ 세션 종료 또는 idle split
→ 공통 FeatureExtractorV2
→ press-order state/state.csv
→ WindowBuilder
→ 50×12 sequence + 12 context window/window.csv
```

수집 도중 feature를 즉석에서 계산하지 않는다. raw를 먼저 안전하게 보존하고,
세션 종료 때 동일한 공통 extractor로 state/window를 생성한다.

## 중요 개인정보 안내

`raw.csv`는 평문 문자열을 직접 저장하지 않지만 VKey와 scan code의 순서가
남기 때문에 입력 내용을 상당 부분 복원할 수 있다. 일반 익명 통계 데이터가
아니며 민감한 연구 데이터로 취급해야 한다.

- 암호화된 저장소 사용
- 접근 권한 최소화
- 메신저·공개 GitHub 업로드 금지
- 참가자의 명시적 raw-keycode 저장 동의 필요

## Windows에서 빌드

압축을 해제한 폴더에서:

```bat
build_exe.bat
```

빌드 성공 결과:

```text
dist\HID_Behavior_Collector_v4.exe
```

이 환경에서는 Windows EXE 자체를 미리 만들 수 없으므로, EXE는 반드시
Windows에서 위 스크립트로 생성해야 한다.

## 소스 실행

먼저 `build_exe.bat`을 한 번 실행해 가상환경과 의존성을 구성한 후:

```bat
run_source.bat
```

## 세션 출력

```text
dataset_v2/<participant_id>/<session_id>/
├── metadata.json
├── session_summary.json
├── quality_events.jsonl
├── raw/raw.csv
├── state/state.csv
└── window/window.csv
```

## 완료 후 필수 검증

1. `session_summary.json`
   - `raw_validation_ok: true`
   - `state_validation_ok: true`
   - `window_validation_ok: true`
2. 50개 이하의 paired keystroke면 window가 0개일 수 있음
3. 첫 keystroke는 transition predecessor가 없으므로 51개 이상의 paired
   keystroke부터 첫 50-step model window가 생성됨
4. `overlap_fraction`은 항상 0~1
5. `signed_flight_time_s`의 음수는 정상적인 overlap 표현

## 기존 v3

최종 연구 데이터 수집에는 사용하지 않는다. 다만 논문 재현성과 legacy 결과
설명을 위해 소스와 기존 결과는 archive에 읽기 전용으로 보존하는 편이 안전하다.
