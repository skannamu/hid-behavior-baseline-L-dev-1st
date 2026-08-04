# Collector v4.1 Ctrl+C 종료 수정

## 증상

```text
Python WNDPROC handler failed
Traceback ...
KeyboardInterrupt
```

pywin32의 WNDPROC 콜백 안에서 기본 SIGINT 처리기가 `KeyboardInterrupt`를
발생시켰고, pywin32가 그 예외를 출력한 뒤 메시지 루프를 계속 실행했다.
그 결과 `state.csv`, `window.csv`, `session_summary.json` 생성 단계로
진입하지 못했다.

## 수정

- `SIGINT`와 Windows `SIGBREAK`를 사용자 정의 signal handler로 받음
- callback 안에서 예외를 발생시키지 않음
- 숨겨진 Raw Input window에 `WM_CLOSE`를 게시
- 기존 정상 종료 경로에서 raw → state → window를 생성
- 중복 Ctrl+C는 한 번만 처리

## 빌드

```bat
build_exe.bat
```

결과:

```text
dist\HID_Behavior_Collector_v4_1.exe
```

## 정상 종료 예상 출력

```text
[Shutdown] Ctrl+C received. Finalizing the current session...
[Finalize] Building state.csv and window.csv...
[OK] Finalized: ...
Collector stopped.
```

Ctrl+C는 한 번만 누르고 finalize가 끝날 때까지 기다린다.
