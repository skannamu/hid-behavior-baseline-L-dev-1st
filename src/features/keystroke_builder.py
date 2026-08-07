"""Convert raw key events into press-ordered canonical keystroke states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .models import BuildResult, KeystrokeState, QualityEvent, RawKeyEvent

BACKSPACE_VKEY = 8
DELETE_VKEY = 46
ENTER_VKEY = 13
SPACE_VKEY = 32

SHIFT_VKEYS = {16, 160, 161}
CTRL_VKEYS = {17, 162, 163}
ALT_VKEYS = {18, 164, 165}
META_VKEYS = {91, 92}
MODIFIER_VKEYS = SHIFT_VKEYS | CTRL_VKEYS | ALT_VKEYS | META_VKEYS

PUNCTUATION_VKEYS = set(range(186, 193)) | set(range(219, 223))


def key_category(vkey: int) -> str:
    if 65 <= vkey <= 90:
        return "LETTER"
    if 48 <= vkey <= 57 or 96 <= vkey <= 105:
        return "DIGIT"
    if vkey == SPACE_VKEY:
        return "SPACE"
    if vkey == ENTER_VKEY:
        return "ENTER"
    if vkey == BACKSPACE_VKEY:
        return "BACKSPACE"
    if vkey == DELETE_VKEY:
        return "DELETE"
    if vkey in MODIFIER_VKEYS:
        return "MODIFIER"
    if 112 <= vkey <= 135:
        return "FUNCTION"
    if vkey in {37, 38, 39, 40}:
        return "ARROW"
    if vkey in PUNCTUATION_VKEYS:
        return "PUNCTUATION"
    return "OTHER"


def modifier_kind(vkey: int) -> str | None:
    if vkey in SHIFT_VKEYS:
        return "shift"
    if vkey in CTRL_VKEYS:
        return "ctrl"
    if vkey in ALT_VKEYS:
        return "alt"
    if vkey in META_VKEYS:
        return "meta"
    return None


@dataclass
class _PendingKeystroke:
    down: RawKeyEvent
    concurrent_keys_at_press: int
    shift_at_press: int
    ctrl_at_press: int
    alt_at_press: int
    meta_at_press: int
    modifier_count_at_press: int
    command_shortcut_flag: int
    repeat_count: int = 0
    repeat_raw_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _CompletedKeystroke:
    pending: _PendingKeystroke
    up: RawKeyEvent


class KeystrokeBuilder:
    """Deterministic offline builder shared by collection and generation paths.

    Final boundary contract:
    if one or more physical keys remain down at the end of a session, the
    earliest such key-down defines the trailing contamination boundary. Any
    completed keystroke whose key-up is not strictly before that boundary is
    excluded from canonical states and therefore from all model windows.

    This removes shutdown chords such as Ctrl+C and also prevents partially
    observed final chords from changing overlap, modifier, pause, or shortcut
    features. Raw events remain untouched and are still the source of truth.
    """

    def build(
        self,
        events: Iterable[RawKeyEvent | Mapping[str, Any]],
        *,
        participant_id: str = "",
        scenario: str = "",
        input_source: str = "",
        label: str = "",
    ) -> BuildResult:
        normalized = [
            event if isinstance(event, RawKeyEvent) else RawKeyEvent.from_mapping(event)
            for event in events
        ]
        normalized.sort(key=lambda e: (e.timestamp_ns, e.raw_event_index))

        quality: list[QualityEvent] = []
        active: dict[tuple[str, int, int], _PendingKeystroke] = {}
        active_modifiers: dict[tuple[str, int, int], str] = {}
        completed: list[_CompletedKeystroke] = []

        seen_indices: set[int] = set()
        for event in normalized:
            if event.raw_event_index in seen_indices:
                quality.append(
                    QualityEvent(
                        code="duplicate_raw_event_index",
                        severity="error",
                        message=f"Duplicate raw_event_index={event.raw_event_index}",
                        raw_event_index=event.raw_event_index,
                        physical_key_id=event.physical_key_id,
                    )
                )
                continue
            seen_indices.add(event.raw_event_index)

            pid = event.physical_key_id

            if event.action == "down":
                if pid in active:
                    pending = active[pid]
                    pending.repeat_count += 1
                    pending.repeat_raw_indices.append(event.raw_event_index)
                    continue

                concurrent = len(active)
                kind = modifier_kind(event.vkey)

                # Modifier bits describe only modifiers that were already
                # active immediately before the current key-down. A modifier
                # never activates its own timestep.
                kinds_before_press = set(active_modifiers.values())
                shift = int("shift" in kinds_before_press)
                ctrl = int("ctrl" in kinds_before_press)
                alt = int("alt" in kinds_before_press)
                meta = int("meta" in kinds_before_press)
                modifier_count = shift + ctrl + alt + meta

                command_shortcut = int(
                    event.vkey not in MODIFIER_VKEYS and (ctrl or alt or meta)
                )

                active[pid] = _PendingKeystroke(
                    down=event,
                    concurrent_keys_at_press=concurrent,
                    shift_at_press=shift,
                    ctrl_at_press=ctrl,
                    alt_at_press=alt,
                    meta_at_press=meta,
                    modifier_count_at_press=modifier_count,
                    command_shortcut_flag=command_shortcut,
                )

                # Activate only after the current snapshot has been frozen.
                if kind is not None:
                    active_modifiers[pid] = kind
                continue

            pending = active.pop(pid, None)
            if pending is None:
                quality.append(
                    QualityEvent(
                        code="unmatched_key_up",
                        severity="warning",
                        message="Key-up had no active matching key-down",
                        raw_event_index=event.raw_event_index,
                        physical_key_id=pid,
                    )
                )
                continue

            if event.timestamp_ns < pending.down.timestamp_ns:
                quality.append(
                    QualityEvent(
                        code="negative_hold_interval",
                        severity="error",
                        message="Key-up timestamp precedes key-down timestamp",
                        raw_event_index=event.raw_event_index,
                        physical_key_id=pid,
                    )
                )
                active_modifiers.pop(pid, None)
                continue

            completed.append(_CompletedKeystroke(pending=pending, up=event))
            active_modifiers.pop(pid, None)

        incomplete = list(active.items())
        for pid, pending in incomplete:
            quality.append(
                QualityEvent(
                    code="incomplete_key_at_end",
                    severity="warning",
                    message="Key-down had no matching key-up before input ended",
                    raw_event_index=pending.down.raw_event_index,
                    physical_key_id=pid,
                )
            )

        cutoff_raw_index: int | None = None
        cutoff_timestamp_ns: int | None = None
        trimmed_completed = 0

        if incomplete:
            _, earliest_pending = min(
                incomplete,
                key=lambda pair: (
                    pair[1].down.timestamp_ns,
                    pair[1].down.raw_event_index,
                ),
            )
            cutoff_raw_index = earliest_pending.down.raw_event_index
            cutoff_timestamp_ns = earliest_pending.down.timestamp_ns
            cutoff_key = (cutoff_timestamp_ns, cutoff_raw_index)

            safe_completed = [
                item
                for item in completed
                if (item.up.timestamp_ns, item.up.raw_event_index) < cutoff_key
            ]
            trimmed_completed = len(completed) - len(safe_completed)
            completed = safe_completed

            if trimmed_completed:
                quality.append(
                    QualityEvent(
                        code="trailing_boundary_trim",
                        severity="info",
                        message=(
                            f"Excluded {trimmed_completed} completed keystroke(s) "
                            "at/after the earliest incomplete key-down boundary"
                        ),
                        raw_event_index=cutoff_raw_index,
                        physical_key_id=earliest_pending.down.physical_key_id,
                    )
                )

        completed.sort(
            key=lambda item: (
                item.pending.down.timestamp_ns,
                item.pending.down.raw_event_index,
            )
        )

        overlap_ns = self._compute_overlap_union_ns(completed)

        states: list[KeystrokeState] = []
        previous: _CompletedKeystroke | None = None

        for index, item in enumerate(completed):
            down = item.pending.down
            up = item.up
            hold_ns = up.timestamp_ns - down.timestamp_ns
            flags: list[str] = []

            if hold_ns == 0:
                flags.append("zero_hold_duration")

            if previous is None:
                press_interval_s = None
                signed_flight_s = None
                release_interval_s = None
                inversion = None
            else:
                press_interval_s = (
                    down.timestamp_ns - previous.pending.down.timestamp_ns
                ) / 1e9
                signed_flight_s = (
                    down.timestamp_ns - previous.up.timestamp_ns
                ) / 1e9
                release_interval_s = (
                    up.timestamp_ns - previous.up.timestamp_ns
                ) / 1e9
                inversion = int(up.timestamp_ns < previous.up.timestamp_ns)

            overlap_duration_s = overlap_ns[index] / 1e9
            overlap_fraction = overlap_ns[index] / hold_ns if hold_ns > 0 else 0.0

            states.append(
                KeystrokeState(
                    session_id=down.session_id,
                    keystroke_index=index,
                    source_raw_down_index=down.raw_event_index,
                    source_raw_up_index=up.raw_event_index,
                    device_id_session_local=down.device_id_session_local,
                    make_code=down.make_code,
                    extended_flag=down.extended_flag,
                    vkey=down.vkey,
                    key_category=key_category(down.vkey),
                    press_timestamp_ns=down.timestamp_ns,
                    release_timestamp_ns=up.timestamp_ns,
                    hold_time_s=hold_ns / 1e9,
                    press_interval_s=press_interval_s,
                    signed_flight_time_s=signed_flight_s,
                    release_interval_s=release_interval_s,
                    overlap_union_duration_s=overlap_duration_s,
                    overlap_fraction=overlap_fraction,
                    concurrent_keys_at_press=item.pending.concurrent_keys_at_press,
                    release_inversion_flag=inversion,
                    correction_key_flag=int(
                        down.vkey in {BACKSPACE_VKEY, DELETE_VKEY}
                    ),
                    repeat_count=item.pending.repeat_count,
                    repeat_flag=int(item.pending.repeat_count > 0),
                    shift_at_press=item.pending.shift_at_press,
                    ctrl_at_press=item.pending.ctrl_at_press,
                    alt_at_press=item.pending.alt_at_press,
                    meta_at_press=item.pending.meta_at_press,
                    modifier_count_at_press=item.pending.modifier_count_at_press,
                    command_shortcut_flag=item.pending.command_shortcut_flag,
                    quality_flags=tuple(flags),
                    participant_id=participant_id,
                    scenario=scenario,
                    input_source=input_source,
                    label=label,
                )
            )
            previous = item

        return BuildResult(
            states=tuple(states),
            quality_events=tuple(quality),
            boundary_cutoff_raw_event_index=cutoff_raw_index,
            boundary_cutoff_timestamp_ns=cutoff_timestamp_ns,
            incomplete_key_count=len(incomplete),
            trailing_completed_keystrokes_trimmed=trimmed_completed,
        )

    @staticmethod
    def _compute_overlap_union_ns(
        completed: list[_CompletedKeystroke],
    ) -> list[int]:
        """Accumulate time where global keyboard occupancy is at least two."""
        overlap = [0 for _ in completed]
        if not completed:
            return overlap

        starts: dict[int, list[int]] = {}
        ends: dict[int, list[int]] = {}

        for idx, item in enumerate(completed):
            press = item.pending.down.timestamp_ns
            release = item.up.timestamp_ns
            starts.setdefault(press, []).append(idx)
            ends.setdefault(release, []).append(idx)

        active: set[int] = set()
        previous_time: int | None = None

        for timestamp in sorted(set(starts) | set(ends)):
            if previous_time is not None:
                duration = timestamp - previous_time
                if duration > 0 and len(active) >= 2:
                    for idx in active:
                        overlap[idx] += duration

            for idx in starts.get(timestamp, ()):
                active.add(idx)
            for idx in ends.get(timestamp, ()):
                active.discard(idx)

            previous_time = timestamp

        return overlap
