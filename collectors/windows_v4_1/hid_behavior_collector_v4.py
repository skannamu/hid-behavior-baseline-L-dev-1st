from __future__ import annotations

import csv
import ctypes
import json
import os
import signal
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import win32gui

from src.features.extractor import FeatureExtractorV2
from src.features.models import KeystrokeState, QualityEvent, RawKeyEvent
from src.features.schema import (
    EXPECTED_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    PAUSE_THRESHOLD_SECONDS,
    STRIDE,
    WINDOW_CONTEXT_FEATURES,
    WINDOW_SIZE,
)
from src.features.validators import validate_raw_events, validate_states, validate_windows
from src.features.window_builder import WindowBuilder, write_windows_csv


COLLECTOR_VERSION = "hid_behavior_collector_v4_1_feature_schema_v2_draft1"
EXTRACTOR_VERSION = "feature_core_v2_1"
DATASET_DIR_NAME = "dataset_v2"

SESSION_SPLIT_SECONDS = 180.0
INPUT_SOURCE = "human"
LABEL = "normal"

WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002

RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1
RIDEV_INPUTSINK = 0x00000100

RI_KEY_BREAK = 0x0001
RI_KEY_E0 = 0x0002
RI_KEY_E1 = 0x0004
KEYBOARD_OVERRUN_MAKE_CODE = 0x00FF
UCHAR_MAX = 0x00FF
MAPVK_VK_TO_VSC_EX = 4

user32 = ctypes.windll.user32


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]


class RAWINPUTUNION(ctypes.Union):
    _fields_ = [("keyboard", RAWKEYBOARD)]


class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("data", RAWINPUTUNION),
    ]


SCENARIOS = {
    "1": {
        "name": "korean_typing",
        "duration_min": 7,
        "description": "한글 문장 따라 타이핑",
        "instruction": (
            "제시된 한글 문장을 보고 직접 타이핑하세요. "
            "오타는 평소처럼 수정하고 복사/붙여넣기는 사용하지 마세요."
        ),
        "recommended_usage": "train/calibration/test_by_group",
    },
    "2": {
        "name": "english_typing",
        "duration_min": 7,
        "description": "영어 문장 따라 타이핑",
        "instruction": (
            "제시된 영어 문장을 직접 타이핑하세요. "
            "대소문자와 문장부호를 자연스럽게 입력하고 복사/붙여넣기는 사용하지 마세요."
        ),
        "recommended_usage": "train/calibration/test_by_group",
    },
    "3": {
        "name": "free_writing",
        "duration_min": 6,
        "description": "자유 글쓰기",
        "instruction": (
            "아무 주제로 자유롭게 작성하세요. "
            "생각하며 멈추거나 문장을 수정하는 행동도 자연스럽게 진행하세요."
        ),
        "recommended_usage": "robustness/test_by_group",
    },
    "4": {
        "name": "coding_controlled",
        "duration_min": 8,
        "description": "통제된 코딩 입력",
        "instruction": (
            "제시된 코드 또는 문제를 직접 입력하세요. "
            "자동완성, 복사/붙여넣기, 코드 생성 도구는 사용하지 마세요."
        ),
        "recommended_usage": "train/calibration/test_by_group",
    },
    "5": {
        "name": "terminal_safe_commands",
        "duration_min": 6,
        "description": "안전한 터미널 명령 입력",
        "instruction": (
            "dir, cd, echo, python --version, git status 같은 안전한 명령만 직접 입력하세요. "
            "삭제·포맷·권한 변경 명령은 입력하지 마세요."
        ),
        "recommended_usage": "train/calibration/test_by_group",
    },
    "6": {
        "name": "mixed_real_use",
        "duration_min": 6,
        "description": "현실적 혼합 입력",
        "instruction": (
            "한글·영어·숫자·파일명·검색어·짧은 명령을 섞어 자연스럽게 입력하세요."
        ),
        "recommended_usage": "robustness/test_by_group",
    },
}

AGE_GROUP_OPTIONS = {
    "0": "undisclosed",
    "1": "10s",
    "2": "20s",
    "3": "30s",
    "4": "40s",
    "5": "50_plus",
}
OCCUPATION_GROUP_OPTIONS = {
    "0": "undisclosed",
    "1": "cs_student",
    "2": "non_cs_student",
    "3": "developer",
    "4": "researcher",
    "5": "office_worker",
    "6": "other",
}
FIELD_GROUP_OPTIONS = {
    "0": "undisclosed",
    "1": "computer_science",
    "2": "engineering",
    "3": "natural_science",
    "4": "humanities_social",
    "5": "arts_sports",
    "6": "other",
}
EXPERIENCE_OPTIONS = {
    "0": "undisclosed",
    "1": "none",
    "2": "beginner",
    "3": "intermediate",
    "4": "advanced",
    "5": "professional",
}
TYPING_STYLE_OPTIONS = {
    "0": "undisclosed",
    "1": "touch_typing",
    "2": "partial_touch_typing",
    "3": "two_finger_or_visual_typing",
    "4": "mixed",
}
KOREAN_IME_OPTIONS = {
    "0": "undisclosed",
    "1": "two_set_korean",
    "2": "three_set_korean",
    "3": "other",
}
KEYBOARD_TYPE_OPTIONS = {
    "0": "undisclosed",
    "1": "laptop_keyboard",
    "2": "external_membrane",
    "3": "external_mechanical",
    "4": "external_low_profile",
    "5": "other",
}
DOMINANT_HAND_OPTIONS = {
    "0": "undisclosed",
    "1": "right",
    "2": "left",
    "3": "both",
}


def get_project_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        return exe_dir.parent if exe_dir.name.lower() == "dist" else exe_dir
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
DATASET_ROOT = PROJECT_ROOT / DATASET_DIR_NAME


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def choose_from_options(title: str, options: dict[str, str], default_key: str = "0") -> str:
    print(f"\n--- {title} ---")
    for key in sorted(options, key=int):
        print(f"{key}. {options[key]}")
    while True:
        value = input(f"선택 번호 입력 default={default_key}: ").strip() or default_key
        if value in options:
            return options[value]
        print("[ERROR] 올바른 번호를 입력하세요.")


def ask_yes_no(title: str, *, default: bool) -> bool:
    default_text = "Y" if default else "N"
    value = input(f"{title} (Y/N, default={default_text}): ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true"}


def sanitize_participant_id(value: str) -> str:
    cleaned = "".join(ch for ch in value.strip() if ch.isalnum() or ch in {"-", "_"})
    return cleaned[:40] or "unknown"


def collect_participant_profile(participant_id: str) -> tuple[dict[str, Any], Path]:
    participant_dir = DATASET_ROOT / participant_id
    participant_dir.mkdir(parents=True, exist_ok=True)
    path = participant_dir / "participant_profile.json"

    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = None
        if old and ask_yes_no("기존 참가자 프로필을 재사용할까요?", default=True):
            old["last_used_at"] = datetime.now().isoformat()
            atomic_write_json(path, old)
            return old, path

    print("\n==============================================")
    print(" Participant Metadata / Explicit Consent")
    print(" 이름·학번·연락처·주소·학교명·회사명은 입력하지 마세요.")
    print(" 주의: raw VKey/scan-code 시퀀스는 평문을 저장하지 않더라도")
    print("       실제 입력 내용을 상당 부분 추정할 수 있는 민감정보입니다.")
    print("==============================================")

    if not ask_yes_no("연구 목적의 행동 데이터 수집에 동의합니까?", default=False):
        raise SystemExit("[STOP] 연구 사용 동의가 없습니다.")
    if not ask_yes_no(
        "재처리를 위해 raw 키코드 시퀀스를 저장하는 것에 명시적으로 동의합니까?",
        default=False,
    ):
        raise SystemExit("[STOP] raw 키코드 저장 동의가 없습니다.")

    profile = {
        "schema_version": "participant_profile_v2",
        "participant_id": participant_id,
        "created_at": datetime.now().isoformat(),
        "last_used_at": datetime.now().isoformat(),
        "privacy_level": "pseudonymous_but_raw_keycodes_are_content_reconstructable",
        "research_use_consent": True,
        "raw_keycode_storage_consent": True,
        "raw_keycode_privacy_warning_acknowledged": True,
        "age_group": choose_from_options("age_group", AGE_GROUP_OPTIONS),
        "occupation_group": choose_from_options("occupation_group", OCCUPATION_GROUP_OPTIONS),
        "field_group": choose_from_options("field_group", FIELD_GROUP_OPTIONS),
        "typing_style": choose_from_options("typing_style", TYPING_STYLE_OPTIONS),
        "korean_ime": choose_from_options("korean_ime", KOREAN_IME_OPTIONS),
        "keyboard_type": choose_from_options("keyboard_type", KEYBOARD_TYPE_OPTIONS),
        "dominant_hand": choose_from_options("dominant_hand", DOMINANT_HAND_OPTIONS),
        "programming_experience": choose_from_options(
            "programming_experience", EXPERIENCE_OPTIONS
        ),
        "terminal_experience": choose_from_options(
            "terminal_experience", EXPERIENCE_OPTIONS
        ),
        "daily_keyboard_usage": choose_from_options(
            "daily_keyboard_usage", EXPERIENCE_OPTIONS
        ),
    }
    atomic_write_json(path, profile)
    return profile, path


def choose_scenario() -> tuple[str, dict[str, Any]]:
    print("\n========== Scenario Menu ==========")
    for key in sorted(SCENARIOS, key=int):
        info = SCENARIOS[key]
        print(
            f"{key}. {info['name']:<24} | "
            f"{info['duration_min']:>2} min | {info['description']}"
        )
    print("===================================")

    while True:
        value = input("scenario 번호 또는 이름 입력: ").strip()
        if value in SCENARIOS:
            return value, SCENARIOS[value]
        for key, info in SCENARIOS.items():
            if value == info["name"]:
                return key, info
        print("[ERROR] 올바른 scenario를 입력하세요.")


def action_from_flags(flags: int) -> str:
    return "up" if flags & RI_KEY_BREAK else "down"


def extended_flag_from_raw(flags: int) -> int:
    return int(bool(flags & (RI_KEY_E0 | RI_KEY_E1)))


def normalize_make_code(make_code: int, vkey: int, flags: int) -> tuple[int, int]:
    extended = extended_flag_from_raw(flags)
    if make_code:
        return make_code, extended

    mapped = int(user32.MapVirtualKeyW(vkey, MAPVK_VK_TO_VSC_EX))
    if mapped:
        low = mapped & 0xFF
        high = (mapped >> 8) & 0xFF
        return low, int(extended or high in {0xE0, 0xE1})

    return make_code, extended


class DeviceAliasMap:
    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}

    def alias(self, raw_handle: Any) -> str:
        key = str(raw_handle)
        if key not in self._aliases:
            self._aliases[key] = f"kbd{len(self._aliases) + 1:03d}"
        return self._aliases[key]

    @property
    def device_count(self) -> int:
        return len(self._aliases)


class SessionRecorder:
    RAW_FIELDS = [
        "session_id",
        "raw_event_index",
        "timestamp_ns",
        "wall_time_ns",
        "device_id_session_local",
        "make_code",
        "extended_flag",
        "vkey",
        "message",
        "raw_flags",
        "action",
    ]

    def __init__(
        self,
        *,
        participant_id: str,
        participant_profile: dict[str, Any],
        participant_profile_file: Path,
        scenario_key: str,
        scenario_info: dict[str, Any],
        session_counter: int,
        split_reason: str,
    ) -> None:
        self.participant_id = participant_id
        self.participant_profile = participant_profile
        self.participant_profile_file = participant_profile_file
        self.scenario_key = scenario_key
        self.scenario_info = scenario_info
        self.session_counter = session_counter
        self.split_reason = split_reason

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.started_wall_time_ns = time.time_ns()
        self.started_monotonic_ns = time.perf_counter_ns()

        self.base_dir = DATASET_ROOT / participant_id / self.session_id
        self.raw_dir = self.base_dir / "raw"
        self.state_dir = self.base_dir / "state"
        self.window_dir = self.base_dir / "window"
        for directory in (self.raw_dir, self.state_dir, self.window_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.raw_path = self.raw_dir / "raw.csv"
        self.state_path = self.state_dir / "state.csv"
        self.window_path = self.window_dir / "window.csv"
        self.metadata_path = self.base_dir / "metadata.json"
        self.summary_path = self.base_dir / "session_summary.json"
        self.quality_path = self.base_dir / "quality_events.jsonl"

        self.raw_events: list[RawKeyEvent] = []
        self.ignored_raw_records: list[dict[str, Any]] = []
        self.device_aliases = DeviceAliasMap()
        self.released_key_count = 0
        self.closed = False

        self._raw_handle = self.raw_path.open("w", newline="", encoding="utf-8")
        self._raw_writer = csv.DictWriter(self._raw_handle, fieldnames=self.RAW_FIELDS)
        self._raw_writer.writeheader()

        self._write_metadata(status="collecting")

    def _write_metadata(self, *, status: str) -> None:
        payload = {
            "collector_version": COLLECTOR_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "schema_hash": EXPECTED_SCHEMA_SHA256,
            "status": status,
            "project_root": str(PROJECT_ROOT),
            "dataset_root": str(DATASET_ROOT),
            "participant_id": self.participant_id,
            "participant_profile_file": str(self.participant_profile_file),
            "participant_profile_snapshot": self.participant_profile,
            "session_id": self.session_id,
            "session_counter": self.session_counter,
            "split_reason": self.split_reason,
            "scenario_key": self.scenario_key,
            "scenario": self.scenario_info["name"],
            "scenario_description": self.scenario_info["description"],
            "scenario_instruction": self.scenario_info["instruction"],
            "scenario_recommended_usage": self.scenario_info["recommended_usage"],
            "planned_duration_min": self.scenario_info["duration_min"],
            "label": LABEL,
            "input_source": INPUT_SOURCE,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "pause_threshold_seconds": PAUSE_THRESHOLD_SECONDS,
            "session_split_seconds": SESSION_SPLIT_SECONDS,
            "behavior_clock": "time.perf_counter_ns_monotonic",
            "wall_clock": "time.time_ns_metadata_only",
            "raw_file": str(self.raw_path),
            "state_file": str(self.state_path),
            "window_file": str(self.window_path),
            "quality_events_file": str(self.quality_path),
            "created_at": datetime.now().isoformat(),
            "privacy_warning": (
                "Raw virtual-key and scan-code sequences may allow reconstruction "
                "of typed content. Store and transfer as sensitive research data."
            ),
        }
        atomic_write_json(self.metadata_path, payload)

    def append_keyboard_event(
        self,
        *,
        timestamp_ns: int,
        wall_time_ns: int,
        raw_device_handle: Any,
        make_code: int,
        vkey: int,
        message: int,
        raw_flags: int,
    ) -> RawKeyEvent | None:
        if make_code == KEYBOARD_OVERRUN_MAKE_CODE or vkey >= UCHAR_MAX:
            self.ignored_raw_records.append(
                {
                    "timestamp_ns": timestamp_ns,
                    "vkey": vkey,
                    "make_code": make_code,
                    "raw_flags": raw_flags,
                    "reason": "keyboard_overrun_or_invalid_vkey",
                }
            )
            return None

        normalized_make, extended = normalize_make_code(make_code, vkey, raw_flags)
        action = action_from_flags(raw_flags)
        alias = self.device_aliases.alias(raw_device_handle)

        event = RawKeyEvent(
            session_id=self.session_id,
            raw_event_index=len(self.raw_events),
            timestamp_ns=timestamp_ns,
            device_id_session_local=alias,
            make_code=normalized_make,
            extended_flag=extended,
            vkey=vkey,
            message=message,
            action=action,
        )
        self.raw_events.append(event)
        if action == "up":
            self.released_key_count += 1

        self._raw_writer.writerow(
            {
                "session_id": event.session_id,
                "raw_event_index": event.raw_event_index,
                "timestamp_ns": event.timestamp_ns,
                "wall_time_ns": wall_time_ns,
                "device_id_session_local": event.device_id_session_local,
                "make_code": event.make_code,
                "extended_flag": event.extended_flag,
                "vkey": event.vkey,
                "message": event.message,
                "raw_flags": raw_flags,
                "action": event.action,
            }
        )
        self._raw_handle.flush()
        return event

    def finalize(self, close_reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        self._raw_handle.flush()
        self._raw_handle.close()

        raw_report = validate_raw_events(self.raw_events)
        extractor = FeatureExtractorV2()
        build = extractor.extract_states(
            self.raw_events,
            participant_id=self.participant_id,
            scenario=self.scenario_info["name"],
            input_source=INPUT_SOURCE,
            label=LABEL,
        )
        state_report = validate_states(build.states)

        self._write_states(build.states)

        windows = ()
        window_report = None
        if raw_report.ok and state_report.ok:
            windows = WindowBuilder().build(build.states)
            window_report = validate_windows(windows)
            if window_report.ok:
                write_windows_csv(windows, self.window_path)

        all_quality: list[dict[str, Any]] = [
            event.to_dict() for event in build.quality_events
        ]
        all_quality.extend(
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "index": issue.index,
                "source": "raw_validator",
            }
            for issue in raw_report.issues
        )
        all_quality.extend(
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "index": issue.index,
                "source": "state_validator",
            }
            for issue in state_report.issues
        )
        if window_report is not None:
            all_quality.extend(
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "index": issue.index,
                    "source": "window_validator",
                }
                for issue in window_report.issues
            )
        all_quality.extend(
            {
                "code": "ignored_raw_record",
                "severity": "warning",
                "message": record["reason"],
                "details": record,
                "source": "collector",
            }
            for record in self.ignored_raw_records
        )
        self._write_quality_events(all_quality)

        ended_wall_ns = time.time_ns()
        summary = {
            "collector_version": COLLECTOR_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "schema_hash": EXPECTED_SCHEMA_SHA256,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "scenario": self.scenario_info["name"],
            "close_reason": close_reason,
            "started_wall_time_ns": self.started_wall_time_ns,
            "ended_wall_time_ns": ended_wall_ns,
            "duration_seconds": max(
                0.0, (ended_wall_ns - self.started_wall_time_ns) / 1e9
            ),
            "raw_event_count": len(self.raw_events),
            "ignored_raw_record_count": len(self.ignored_raw_records),
            "paired_keystroke_count": len(build.states),
            "window_count": len(windows),
            "device_count": self.device_aliases.device_count,
            "quality_event_count": len(all_quality),
            "raw_validation_ok": raw_report.ok,
            "state_validation_ok": state_report.ok,
            "window_validation_ok": window_report.ok if window_report else False,
            "raw_file": str(self.raw_path),
            "state_file": str(self.state_path),
            "window_file": str(self.window_path),
            "metadata_file": str(self.metadata_path),
            "quality_events_file": str(self.quality_path),
            "closed_at": datetime.now().isoformat(),
        }
        atomic_write_json(self.summary_path, summary)
        self._write_metadata(status="finalized")

    def _write_states(self, states: tuple[KeystrokeState, ...]) -> None:
        state_fields = [field.name for field in fields(KeystrokeState)]
        with self.state_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=state_fields)
            writer.writeheader()
            for state in states:
                row = state.to_dict()
                row["quality_flags"] = json.dumps(
                    row["quality_flags"], ensure_ascii=False
                )
                writer.writerow(row)

    def _write_quality_events(self, rows: list[dict[str, Any]]) -> None:
        with self.quality_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CollectorApp:
    def __init__(
        self,
        *,
        participant_id: str,
        participant_profile: dict[str, Any],
        participant_profile_file: Path,
        scenario_key: str,
        scenario_info: dict[str, Any],
        auto_stop: bool,
    ) -> None:
        self.participant_id = participant_id
        self.participant_profile = participant_profile
        self.participant_profile_file = participant_profile_file
        self.scenario_key = scenario_key
        self.scenario_info = scenario_info
        self.auto_stop = auto_stop

        self.session_counter = 0
        self.session: SessionRecorder | None = None
        self.last_release_timestamp_ns: int | None = None
        self.hwnd = None
        self.program_started_ns = time.perf_counter_ns()
        self._lock = threading.RLock()

    def start_session(self, reason: str) -> None:
        with self._lock:
            if self.session is not None:
                self.session.finalize(f"new_session_{reason}")
            self.session_counter += 1
            self.session = SessionRecorder(
                participant_id=self.participant_id,
                participant_profile=self.participant_profile,
                participant_profile_file=self.participant_profile_file,
                scenario_key=self.scenario_key,
                scenario_info=self.scenario_info,
                session_counter=self.session_counter,
                split_reason=reason,
            )
            self.last_release_timestamp_ns = None
            print("\n================ NEW SESSION ================")
            print(f"Reason      : {reason}")
            print(f"Session ID  : {self.session.session_id}")
            print(f"Output      : {self.session.base_dir}")
            print("=============================================\n")

    def handle_keyboard(
        self,
        *,
        timestamp_ns: int,
        wall_time_ns: int,
        raw_device_handle: Any,
        make_code: int,
        vkey: int,
        message: int,
        raw_flags: int,
    ) -> None:
        action = action_from_flags(raw_flags)

        with self._lock:
            if (
                action == "down"
                and self.last_release_timestamp_ns is not None
                and (timestamp_ns - self.last_release_timestamp_ns) / 1e9
                >= SESSION_SPLIT_SECONDS
            ):
                gap = (timestamp_ns - self.last_release_timestamp_ns) / 1e9
                self.start_session(f"idle_gap_{gap:.2f}s")

            if self.session is None:
                return

            event = self.session.append_keyboard_event(
                timestamp_ns=timestamp_ns,
                wall_time_ns=wall_time_ns,
                raw_device_handle=raw_device_handle,
                make_code=make_code,
                vkey=vkey,
                message=message,
                raw_flags=raw_flags,
            )
            if event is None:
                return

            if event.action == "up":
                self.last_release_timestamp_ns = event.timestamp_ns
                if self.session.released_key_count % 25 == 0:
                    elapsed = (time.perf_counter_ns() - self.program_started_ns) / 1e9
                    print(
                        f"session={self.session.session_id} "
                        f"raw={len(self.session.raw_events):06d} "
                        f"released={self.session.released_key_count:06d} "
                        f"elapsed={elapsed/60:.1f}m"
                    )

    def close(self, reason: str) -> None:
        with self._lock:
            if self.session is not None:
                print("\n[Finalize] Building state.csv and window.csv...")
                self.session.finalize(reason)
                print(f"[OK] Finalized: {self.session.base_dir}")
                self.session = None

    def timer_main(self) -> None:
        if not self.auto_stop:
            return
        time.sleep(self.scenario_info["duration_min"] * 60)
        if self.hwnd:
            win32gui.PostMessage(self.hwnd, WM_CLOSE, 0, 0)


APP: CollectorApp | None = None
SHUTDOWN_REQUESTED = False


def request_graceful_shutdown(signum=None, frame=None) -> None:
    """Convert console interrupts into a normal Windows WM_CLOSE request.

    The default SIGINT handler raises KeyboardInterrupt inside the Python
    WNDPROC callback. pywin32 catches that callback exception and keeps the
    message pump alive, so finalization never runs. Posting WM_CLOSE avoids
    raising inside WNDPROC and lets the normal close path build state/window.
    """
    global SHUTDOWN_REQUESTED

    if SHUTDOWN_REQUESTED:
        return
    SHUTDOWN_REQUESTED = True

    print("\n[Shutdown] Ctrl+C received. Finalizing the current session...")

    if APP is not None and APP.hwnd:
        win32gui.PostMessage(APP.hwnd, WM_CLOSE, 0, 0)


def install_console_signal_handlers() -> None:
    signal.signal(signal.SIGINT, request_graceful_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_graceful_shutdown)


def wnd_proc(hwnd, msg, wparam, lparam):
    global APP

    if msg == WM_INPUT:
        size = wintypes.UINT(0)
        user32.GetRawInputData(
            lparam,
            RID_INPUT,
            None,
            ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if size.value <= 0:
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        buffer = ctypes.create_string_buffer(size.value)
        result = user32.GetRawInputData(
            lparam,
            RID_INPUT,
            buffer,
            ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if result == 0xFFFFFFFF:
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
        if raw.header.dwType == RIM_TYPEKEYBOARD and APP is not None:
            kb = raw.data.keyboard
            APP.handle_keyboard(
                timestamp_ns=time.perf_counter_ns(),
                wall_time_ns=time.time_ns(),
                raw_device_handle=raw.header.hDevice,
                make_code=int(kb.MakeCode),
                vkey=int(kb.VKey),
                message=int(kb.Message),
                raw_flags=int(kb.Flags),
            )

    elif msg == WM_CLOSE:
        if APP is not None:
            APP.close("window_close")
        win32gui.DestroyWindow(hwnd)
        return 0

    elif msg == WM_DESTROY:
        if APP is not None:
            APP.close("window_destroy")
        win32gui.PostQuitMessage(0)
        return 0

    return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)


def main() -> None:
    global APP

    print("\n================================================")
    print(" HID Behavior Collector v4 / Feature Schema v2")
    print(" Raw first, press-order state, 50x12 + context12")
    print("================================================")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Schema       : {FEATURE_SCHEMA_VERSION}")
    print(f"Schema hash  : {EXPECTED_SCHEMA_SHA256}\n")

    participant_id = sanitize_participant_id(
        input("participant_id 예: p001: ").strip()
    )
    profile, profile_path = collect_participant_profile(participant_id)
    scenario_key, scenario_info = choose_scenario()
    auto_stop = ask_yes_no("계획 시간이 끝나면 자동 종료할까요?", default=True)

    print("\n============== Collection Plan ==============")
    print(f"Participant : {participant_id}")
    print(f"Scenario    : {scenario_info['name']}")
    print(f"Duration    : {scenario_info['duration_min']} min")
    print(f"Keyboard    : {profile.get('keyboard_type', 'undisclosed')}")
    print(f"Schema      : {FEATURE_SCHEMA_VERSION}")
    print("---------------------------------------------")
    print(scenario_info["instruction"])
    print("=============================================\n")

    input("준비되면 Enter를 누르세요...")

    APP = CollectorApp(
        participant_id=participant_id,
        participant_profile=profile,
        participant_profile_file=profile_path,
        scenario_key=scenario_key,
        scenario_info=scenario_info,
        auto_stop=auto_stop,
    )
    APP.start_session("initial_start")

    window_class = win32gui.WNDCLASS()
    window_class.lpfnWndProc = wnd_proc
    window_class.lpszClassName = (
        f"HIDBehaviorCollectorV4_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    class_atom = win32gui.RegisterClass(window_class)

    hwnd = win32gui.CreateWindow(
        class_atom,
        "HID Behavior Collector v4",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
    )
    APP.hwnd = hwnd
    install_console_signal_handlers()

    rid = RAWINPUTDEVICE()
    rid.usUsagePage = 0x01
    rid.usUsage = 0x06
    rid.dwFlags = RIDEV_INPUTSINK
    rid.hwndTarget = hwnd

    if not user32.RegisterRawInputDevices(
        ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE)
    ):
        raise ctypes.WinError()

    threading.Thread(target=APP.timer_main, daemon=True).start()

    print("\n수집 중입니다. 종료 시 raw를 다시 읽어 state/window를 생성합니다.")
    print("창을 닫거나 Ctrl+C를 누르면 안전하게 finalize합니다.\n")

    try:
        win32gui.PumpMessages()
    except KeyboardInterrupt:
        # Defensive fallback. The installed SIGINT handler should normally
        # convert Ctrl+C into WM_CLOSE before KeyboardInterrupt is raised.
        request_graceful_shutdown()
        APP.close("keyboard_interrupt_fallback")
        if hwnd:
            win32gui.DestroyWindow(hwnd)
    finally:
        if APP is not None:
            APP.close("finally")
        print("\nCollector stopped.")


if __name__ == "__main__":
    main()
