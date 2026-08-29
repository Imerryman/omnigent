"""TUI→web forwarder for the qwen-native harness.

The ``omnigent qwen`` wrapper launches the real ``qwen`` TUI in a runner-owned
tmux pane with ``--json-file`` pointed at the bridge dir, and
:mod:`omnigent.qwen_native_bridge` appends web-UI messages to its ``--input-file``.
That covers the web→TUI direction, but the *embedded terminal* is then the only
surface that reflects the agent's work — the Omnigent conversation view stays
empty because nothing mirrors the transcript back into the session.

This module is that missing mirror — the qwen analog of
:mod:`omnigent.goose_native_forwarder`. Where goose has to scrape a SQLite store,
qwen emits a structured **stream-json event stream** (verified Anthropic-shaped
against ``qwen`` v0.18.1): we tail the ``--json-file`` NDJSON by byte offset and
POST each new ``user`` / ``assistant`` message as an ``external_conversation_item``
event (which also seeds the session title).

Event shapes consumed (others are ignored defensively):

- ``{"type":"user","message":{"role":"user","content":[{"type":"text","text":...}]}}``
- ``{"type":"assistant","message":{"role":"assistant","content":[{"type":"text"|
  "thinking"|"tool_use",...}]}}`` — only ``text`` blocks are mirrored.
- ``{"type":"control_request","request":{"subtype":"can_use_tool",...},
  "request_id":...}`` and the matching ``control_response`` — the permission
  control plane. NOT handled here: the tool-approval mirror
  (:mod:`omnigent.qwen_native_permissions`) tails the same stream and surfaces
  these as web elicitation cards. This forwarder ignores them (they carry no
  transcript prose to mirror).

Status (``running``/``idle``) is intentionally NOT posted here: the runner's
PTY-activity watcher owns those edges for qwen-native (see
:mod:`omnigent.runner.app`), exactly as for goose-/cursor-native.

This module also hosts the **compaction mirror** (:func:`supervise_qwen_compaction_mirror`).
qwen compaction (its *compression*) is invisible on the ``--json-file`` stream
(``session_start``'s ``supported_events`` omits it — verified live, ``qwen``
v0.18.2), but qwen writes a ``{"type":"system","subtype":"chat_compression",
"systemPayload":{"info":{originalTokenCount,newTokenCount,compressionStatus}}}``
record to its on-disk chat recording the instant compression finishes. The mirror
tails that recording and POSTs ``external_compaction_status`` (``completed`` on
success, ``failed`` otherwise) — the completion half of the web ``/compact`` →
qwen ``/compress`` flow whose ``in_progress`` edge the runner raises on injection.
It fires for both explicit ``/compress`` and auto-compaction.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Container, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

from omnigent.inner.native_attachments import ATTACHMENT_MARKER_STRIP_PATTERN
from omnigent.qwen_native_bridge import events_file_path, read_tmux_info

_logger = logging.getLogger(__name__)

#: Seconds between event-file polls. qwen flushes events per streaming step, so a
#: sub-second cadence keeps the mirrored chat tracking the terminal step by step.
_DEFAULT_POLL_INTERVAL_S = 0.4
_POST_TIMEOUT_S = 30.0
# Liveness probe budget; a hung tmux answers "assume alive" rather than blocking.
_TMUX_PROBE_TIMEOUT_S = 5.0
# Minimum gap between liveness probes. The probe spawns a `tmux has-session`
# subprocess, so it must not ride the sub-second poll cadence; this bounds the
# fallback wake latency after a process death to roughly this interval.
_TMUX_PROBE_INTERVAL_S = 5.0

# Supervisor backoff (mirrors goose_native_forwarder.supervise_goose_forwarder).
_SUPERVISOR_INITIAL_BACKOFF_S = 1.0
_SUPERVISOR_MAX_BACKOFF_S = 30.0
_SUPERVISOR_HEALTHY_UPTIME_S = 60.0

_STATE_FILE = "qwen_forwarder.json"

# LOCAL PATCH #6: the two terminal edges this forwarder posts. Both are accepted
# by the v0.11.0 Sessions API (``_EXTERNAL_SESSION_STATUS_VALUES`` is
# ``{"idle", "running", "waiting", "failed"}``) and both are treated as a
# sub-agent terminal that wakes the parent inbox (``status in {"idle",
# "failed"}`` in the server's ``external_session_status`` handler).
_STATUS_IDLE = "idle"
_STATUS_FAILED = "failed"

# Dedup window: the number of most-recently-posted event uuids persisted so a
# truncation/relaunch re-read (offset rewinds to 0) doesn't re-post them. The
# window must keep the *most recent* uuids, so ``seen`` is an insertion-ordered
# mapping (a ``dict`` used as an ordered set), not a ``set`` — ``list(set)`` is
# hash-ordered, which would make the ``[-_DEDUP_WINDOW:]`` cap keep an arbitrary
# subset and re-post recent history on a long-session relaunch.
_DEDUP_WINDOW = 512


def _new_seen(uuids: Iterable[str] | None = None) -> dict[str, None]:
    """Build the insertion-ordered dedup set (``dict`` used as an ordered set)."""
    return dict.fromkeys(uuids or [])


# The executor injects ``[Attached: <path>]`` (or the could-not-load marker
# from native_attachments) for web-UI attachments before submitting; strip them
# from the mirrored bubble (internal bridge details).
_ATTACHMENT_MARKER_RE = re.compile(ATTACHMENT_MARKER_STRIP_PATTERN)


@dataclass
class _ForwardState:
    """Durable forwarder cursor, persisted to ``bridge_dir/qwen_forwarder.json``.

    :param offset: Byte offset into the ``--json-file`` already consumed. The
        event file is append-only within a TUI lifetime; a relaunched terminal
        truncates it (see :func:`~omnigent.qwen_native_bridge.prepare_bridge_files`),
        which we detect as ``size < offset`` and reset to 0.
    :param seen_uuids: Recently posted item ids, for idempotent dedup across a
        truncation/restart. Bounded to the most recent entries. Entries are
        per-item ids (``"<event-uuid>#<index>"``, see :class:`_MirrorItem`) plus
        the bare envelope uuids of posted turn-end markers. State written by an
        older build holds bare event uuids; :func:`_item_already_seen` still
        honours those, so an upgrade does not re-post history.
    :param pending_window_id: ``message.id`` of the assistant message window
        currently open (a top-level ``message_start`` with no matching
        ``message_stop`` yet), else ``None``.
    :param pending_stop_reason: ``stop_reason`` of the most recent TOP-LEVEL
        ``assistant`` event inside that window — the classification the next
        ``message_stop`` consumes. ``"tool_use"`` means the message_stop closes a
        mid-loop tool step, NOT the user-facing turn.
    :param turn_open: ``True`` while a turn is in flight with no terminal status
        posted for it yet. Drives the process-exit fallback in
        :func:`forward_qwen_events_to_session` and makes the wake exactly-once.
    """

    offset: int = 0
    seen_uuids: list[str] | None = None
    # LOCAL PATCH #6: the tool-loop classification must survive a forwarder
    # restart. Without it `pending_stop_reason` restarts as None, so a
    # message_stop whose `stop_reason="tool_use"` assistant event was consumed
    # (and its offset persisted) BEFORE the restart is misread as a turn end and
    # wakes the parent mid tool-loop.
    pending_window_id: str | None = None
    pending_stop_reason: str | None = None
    turn_open: bool = False


def _read_state(bridge_dir: Path) -> _ForwardState:
    """Load the persisted forward cursor, or a cold default."""
    try:
        raw = (bridge_dir / _STATE_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return _ForwardState(offset=0, seen_uuids=[])
    offset = data.get("offset")
    seen = data.get("seen_uuids")
    # LOCAL PATCH #6: restore the pending tool-loop classification across a
    # restart. Absent (state written by an older build) => cold defaults, which
    # is the pre-patch behaviour for that one boundary and no worse.
    window_id = data.get("pending_window_id")
    stop_reason = data.get("pending_stop_reason")
    turn_open = data.get("turn_open")
    return _ForwardState(
        offset=offset if isinstance(offset, int) and offset >= 0 else 0,
        seen_uuids=[u for u in seen if isinstance(u, str)] if isinstance(seen, list) else [],
        pending_window_id=window_id if isinstance(window_id, str) and window_id else None,
        pending_stop_reason=stop_reason if isinstance(stop_reason, str) and stop_reason else None,
        turn_open=turn_open is True,
    )


def _write_state(bridge_dir: Path, state: _ForwardState) -> bool:
    """Atomically persist the forward cursor (tmp write + rename)."""
    try:
        bridge_dir.mkdir(parents=True, exist_ok=True)
        tmp = bridge_dir / (_STATE_FILE + ".tmp")
        # Cap the dedup window so the state file can't grow unbounded. The list
        # is insertion-ordered (see _new_seen), so this keeps the most recent
        # _DEDUP_WINDOW uuids — the ones a relaunch re-read is most likely to hit.
        seen = (state.seen_uuids or [])[-_DEDUP_WINDOW:]
        tmp.write_text(
            json.dumps(
                {
                    "offset": state.offset,
                    "seen_uuids": seen,
                    # LOCAL PATCH #6: persist the tool-loop classification.
                    "pending_window_id": state.pending_window_id,
                    "pending_stop_reason": state.pending_stop_reason,
                    "turn_open": state.turn_open,
                }
            ),
            encoding="utf-8",
        )
        os.replace(tmp, bridge_dir / _STATE_FILE)
        return True
    except OSError:
        _logger.warning("qwen forwarder could not persist state to %s", bridge_dir, exc_info=True)
        return False


def clear_qwen_bridge_state(bridge_dir: Path) -> None:
    """Remove the persisted forward cursor so a re-created terminal starts clean."""
    with contextlib.suppress(OSError):
        (bridge_dir / _STATE_FILE).unlink()


@dataclass
class _MirrorItem:
    """One conversation item ready to POST, plus the event uuid that produced it.

    :param uuid: Envelope uuid of the stream event this item came from. Several
        items can share it (a tool call plus prose in one event).
    :param index: Position of this item within its event's item list.
    :param item_type: AP conversation item type, e.g. ``"message"``.
    :param item_data: The AP item payload.
    :param response_id: Grouping id for the event's items.
    """

    uuid: str
    index: int
    item_type: str
    item_data: dict[str, object]
    response_id: str

    @property
    def item_uuid(self) -> str:
        """Deterministic per-ITEM dedup id, ``"<event-uuid>#<index>"``.

        LOCAL PATCH #6. Dedup used to be keyed on the bare event uuid, which is
        wrong once one event yields several items: marking the event seen after
        the FIRST successful POST meant a failure on a later item was replayed as
        "already seen" and that item was dropped for good, leaving the turn-end
        wake to fire over incomplete content. Keying on the item makes each POST
        independently replayable, and the id is derived (not random) so a
        truncation rewind re-derives exactly the same ids.
        """
        return f"{self.uuid}#{self.index}"


def _item_already_seen(item: _MirrorItem, seen: Container[str]) -> bool:
    """Whether *item* has already been posted.

    Checks the per-item id first, then falls back to the bare event uuid so
    state written by a pre-patch build (which recorded whole events) still
    suppresses history instead of re-posting it after an upgrade.
    """
    # LOCAL PATCH #6: per-item dedup, with backward-compatible event-uuid fallback.
    return item.item_uuid in seen or item.uuid in seen


def _text_from_content(content: object) -> str:
    """Join the ``text`` blocks of a stream-json message ``content`` array.

    ``thinking`` and ``tool_use`` blocks are skipped — only user-facing prose is
    mirrored into the chat bubble. Tolerant of a bare string or odd shapes so a
    schema tweak degrades to "best available text" rather than dropping the row.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _tool_result_output(block: dict[str, object]) -> str:
    """Return the UI-facing output string for a ``tool_result`` block.

    Mirrors :func:`omnigent.claude_native_bridge._tool_result_output` minus the
    Claude-transcript ``toolUseResult`` fallback qwen's stream does not carry:
    ``str`` content passes through, other non-null content is compact-JSON
    encoded, and missing content becomes ``""``.
    """
    # LOCAL PATCH #5: mirror tool_use/tool_result as function_call/function_call_output
    content = block.get("content")
    if isinstance(content, str):
        return content
    if content is not None:
        return json.dumps(content, separators=(",", ":"))
    return ""


def _event_to_items(event: dict[str, object], agent_name: str) -> list[_MirrorItem]:
    """Convert one qwen stream-json event to mirror items (empty list to skip it).

    A ``user`` event yields one ``function_call_output`` item per ``tool_result``
    block, then the prose ``message`` item (if any text). An ``assistant`` event
    yields one ``function_call`` item per ``tool_use`` block, then the prose
    ``message`` item. All items from one event share that event's uuid and
    ``response_id``, so the existing per-uuid dedupe skips a whole already-seen
    event exactly as the old single-item form did. The ``function_call`` /
    ``function_call_output`` item shapes match claude-/hermes-native. Ordering
    invariant: tool items precede prose within an event, and stream order puts
    the assistant ``tool_use`` event before the user ``tool_result`` event, so a
    ``function_call`` always posts before its ``function_call_output``.
    """
    # LOCAL PATCH #5: mirror tool_use/tool_result as function_call/function_call_output
    etype = event.get("type")
    if etype not in ("user", "assistant"):
        # control_request / control_response (the permission control plane) carry
        # no transcript prose; the tool-approval mirror
        # (omnigent.qwen_native_permissions) owns them off the same stream.
        return []
    uuid = event.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        return []
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    response_id = f"qwen:{uuid}"
    content = message.get("content")
    blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
    items: list[_MirrorItem] = []
    if etype == "user":
        for block in blocks:
            if block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            items.append(
                _MirrorItem(
                    uuid=uuid,
                    index=len(items),
                    item_type="function_call_output",
                    item_data={"call_id": call_id, "output": _tool_result_output(block)},
                    response_id=response_id,
                )
            )
    else:
        for block in blocks:
            if block.get("type") != "tool_use":
                continue
            tool_id = block.get("id")
            name = block.get("name")
            if not isinstance(tool_id, str) or not tool_id:
                continue
            if not isinstance(name, str) or not name:
                continue
            arguments = block.get("input")
            if not isinstance(arguments, dict):
                arguments = {}
            items.append(
                _MirrorItem(
                    uuid=uuid,
                    index=len(items),
                    item_type="function_call",
                    item_data={
                        "agent": agent_name,
                        "name": name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                        "call_id": tool_id,
                    },
                    response_id=response_id,
                )
            )
    text = _ATTACHMENT_MARKER_RE.sub("", _text_from_content(content)).strip()
    if not text:
        return items  # tool-only / thinking-only step with no prose
    if etype == "user":
        items.append(
            _MirrorItem(
                uuid=uuid,
                index=len(items),
                item_type="message",
                item_data={"role": "user", "content": [{"type": "input_text", "text": text}]},
                response_id=response_id,
            )
        )
        return items
    items.append(
        _MirrorItem(
            uuid=uuid,
            index=len(items),
            item_type="message",
            item_data={
                "role": "assistant",
                "agent": agent_name,
                "content": [{"type": "output_text", "text": text}],
            },
            response_id=response_id,
        )
    )
    return items


def _stream_inner(event: dict[str, object]) -> dict[str, object] | None:
    """Return the nested ``event`` of a TOP-LEVEL ``stream_event``, else ``None``.

    ``parent_tool_use_id is not None`` marks a nested sub-tool stream, whose
    message boundaries must never be mistaken for the session's own turn.
    """
    if event.get("type") != "stream_event":
        return None
    if event.get("parent_tool_use_id") is not None:
        return None
    inner = event.get("event")
    return inner if isinstance(inner, dict) else None


def _message_window_id(event: dict[str, object]) -> str | None:
    """Return the ``message.id`` a top-level ``message_start`` opens, else ``None``.

    Verified against real qwen output (v0.22.0, protocol 2): one turn is a
    ``message_start`` … ``message_stop`` window carrying several consolidated
    ``assistant`` events, and windows never overlap (max concurrent depth 1 over
    319k recorded events). That makes the opening ``message.id`` a stable,
    persistable name for "the turn currently being classified".
    """
    # LOCAL PATCH #6: stable turn identity, replacing "the most recent assistant".
    inner = _stream_inner(event)
    if inner is None or inner.get("type") != "message_start":
        return None
    message = inner.get("message")
    if not isinstance(message, dict):
        return None
    window_id = message.get("id")
    return window_id if isinstance(window_id, str) and window_id else None


def _turn_end_uuid(event: dict[str, object]) -> str | None:
    """Return the envelope uuid of a top-level ``message_stop`` event, else ``None``.

    A turn-end candidate is a ``stream_event`` whose nested ``event`` has
    ``type=="message_stop"`` and whose ``parent_tool_use_id`` is null — the
    ``parent_tool_use_id is None`` guard excludes any nested sub-tool turn-end.
    Whether the candidate is a real turn end depends on the classification of
    the message it closes; qwen's ``message_stop`` is bare (``{"type":
    "message_stop"}`` — no id, verified over 1016 recorded stops), so the
    classification is carried by the open window, not read off this event.
    """
    # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
    inner = _stream_inner(event)
    if inner is None or inner.get("type") != "message_stop":
        return None
    uuid = event.get("uuid")
    if isinstance(uuid, str) and uuid:
        return uuid
    return None


def _session_end_uuid(event: dict[str, object]) -> str | None:
    """Return the uuid of a top-level ``system``/``session_end`` record, else ``None``.

    qwen writes this when it closes the session cleanly (verified in real bridge
    output). It is a definitive terminal, but NOT a reliable one: only 2 of 23
    recorded sessions carry it — the rest are torn down with the pane — which is
    why the process-exit fallback in :func:`forward_qwen_events_to_session`
    exists alongside it.
    """
    # LOCAL PATCH #6: definitive terminal event #1 (clean close).
    if event.get("type") != "system" or event.get("subtype") != "session_end":
        return None
    if event.get("parent_tool_use_id") is not None:
        return None
    uuid = event.get("uuid")
    return uuid if isinstance(uuid, str) and uuid else None


def _result_terminal(event: dict[str, object]) -> tuple[str, str] | None:
    """Return ``(uuid, status)`` for a top-level ``result`` event, else ``None``.

    qwen advertises ``result`` in ``session_start.supported_events`` but never
    emits it in the interactive TUI mode this forwarder tails (0 occurrences in
    319k recorded events) — it belongs to headless ``-p`` runs. Handled anyway so
    a qwen build that starts emitting it produces a correct, deduplicated wake
    rather than falling through to the process-exit fallback.
    """
    # LOCAL PATCH #6: definitive terminal event #2 (declared, not yet observed).
    if event.get("type") != "result":
        return None
    if event.get("parent_tool_use_id") is not None:
        return None
    uuid = event.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        return None
    is_error = event.get("is_error") is True
    subtype = event.get("subtype")
    failed = is_error or (isinstance(subtype, str) and subtype not in ("", "success"))
    return uuid, (_STATUS_FAILED if failed else _STATUS_IDLE)


@dataclass
class _Wake:
    """One terminal ``external_session_status`` edge to POST, with its dedup id.

    :param uuid: Envelope uuid of the event that produced it; also the dedup key
        recorded in ``seen`` and the idempotency key sent to the server.
    :param status: ``"idle"`` or ``"failed"``.
    """

    uuid: str
    status: str


@dataclass
class _PollResult:
    """What one :func:`_read_new_events` pass produced.

    :param items: New mirror items to POST, in stream order.
    :param wakes: Terminal status edges to POST after the items.
    :param offset: New byte offset.
    :param pending_window_id: Open message window, threaded to the next poll.
    :param pending_stop_reason: Its classification, threaded to the next poll.
    :param turn_open: Whether work is in flight with no terminal edge posted yet.
    """

    items: list[_MirrorItem]
    wakes: list[_Wake]
    offset: int
    pending_window_id: str | None
    pending_stop_reason: str | None
    turn_open: bool


def _read_new_events(
    events_file: Path,
    offset: int,
    seen: Container[str],
    agent_name: str,
    state: _ForwardState | None = None,
) -> _PollResult:
    """Read NDJSON lines past *offset* into a :class:`_PollResult`.

    Detects a truncated/recreated event file (``size < offset``) and rewinds to 0.
    Only fully terminated lines (ending in ``\\n``) are consumed; a trailing
    partial line is left for the next poll by not advancing past it — so a wake
    is never decided on a half-written record.

    *state* threads the turn classification across poll batches AND across
    forwarder restarts (it is persisted); pass ``None`` for a cold read.
    """
    # LOCAL PATCH #6: classification is threaded through _ForwardState rather
    # than a bare `last_stop_reason` local, so it survives a restart.
    window_id = state.pending_window_id if state is not None else None
    stop_reason = state.pending_stop_reason if state is not None else None
    turn_open = state.turn_open if state is not None else False

    def _result(items: list[_MirrorItem], wakes: list[_Wake], off: int) -> _PollResult:
        return _PollResult(
            items=items,
            wakes=wakes,
            offset=off,
            pending_window_id=window_id,
            pending_stop_reason=stop_reason,
            turn_open=turn_open,
        )

    try:
        size = events_file.stat().st_size
    except OSError:
        return _result([], [], offset)
    if size < offset:
        offset = 0  # file truncated by a relaunched terminal
    if size == offset:
        return _result([], [], offset)
    try:
        with open(events_file, "rb") as fh:
            fh.seek(offset)
            data = fh.read(size - offset)
    except OSError:
        return _result([], [], offset)
    # Only consume up to the last newline; keep any trailing partial line.
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return _result([], [], offset)  # no complete line yet
    consumed = data[: last_nl + 1]
    new_offset = offset + len(consumed)
    items: list[_MirrorItem] = []
    wakes: list[_Wake] = []
    for raw in consumed.split(b"\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue  # tolerate a malformed line rather than stalling the tail
        if not isinstance(event, dict):
            continue
        top_level = event.get("parent_tool_use_id") is None
        etype = event.get("type")

        # --- turn classification (TOP-LEVEL ONLY) ---------------------------
        # LOCAL PATCH #6: a nested (sub-tool) assistant event must not overwrite
        # the top-level classification the next top-level message_stop consumes.
        opened = _message_window_id(event)
        if opened is not None:
            window_id = opened
            stop_reason = None
            turn_open = True
        elif etype == "assistant" and top_level:
            message = event.get("message")
            if isinstance(message, dict):
                raw_stop = message.get("stop_reason")
                stop_reason = raw_stop if isinstance(raw_stop, str) else None
            turn_open = True
        elif etype == "user" and top_level:
            # A prompt or a tool_result: work is in flight again.
            turn_open = True

        # --- terminal edges --------------------------------------------------
        te_uuid = _turn_end_uuid(event)
        if te_uuid is not None:
            # LOCAL PATCH #4: a message_stop closing a `stop_reason="tool_use"`
            # step is mid-loop (qwen auto-continues), NOT a user-facing turn end.
            if stop_reason != "tool_use":
                if te_uuid not in seen:
                    wakes.append(_Wake(uuid=te_uuid, status=_STATUS_IDLE))
                turn_open = False
            window_id = None
            stop_reason = None
        else:
            result_terminal = _result_terminal(event)
            se_uuid = _session_end_uuid(event)
            if result_terminal is not None:
                r_uuid, r_status = result_terminal
                if r_uuid not in seen:
                    wakes.append(_Wake(uuid=r_uuid, status=r_status))
                turn_open = False
                window_id = None
                stop_reason = None
            elif se_uuid is not None:
                # A clean qwen-side close. Only an edge when the turn never got
                # its own terminal, else the message_stop wake already fired.
                if turn_open and se_uuid not in seen:
                    wakes.append(_Wake(uuid=se_uuid, status=_STATUS_IDLE))
                turn_open = False
                window_id = None
                stop_reason = None

        # LOCAL PATCH #5: one event can yield several items (tool items then
        # prose); LOCAL PATCH #6 dedupes them per ITEM, not per event.
        for item in _event_to_items(event, agent_name):
            if not _item_already_seen(item, seen):
                items.append(item)
    return _result(items, wakes, new_offset)


async def _post_conversation_item(
    client: httpx.AsyncClient, *, session_id: str, item: _MirrorItem
) -> None:
    """POST one mirrored item as an ``external_conversation_item`` event."""
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "type": "external_conversation_item",
            "data": {
                "item_type": item.item_type,
                "item_data": item.item_data,
                "response_id": item.response_id,
            },
        },
    )
    resp.raise_for_status()


async def _post_external_session_status(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    status: str,
    idempotency_key: str | None = None,
) -> None:
    """POST one ``external_session_status`` event to the Sessions API.

    For a sub-agent conversation the server maps an ``idle`` edge to a terminal
    completion that wakes the parent orchestrator's inbox — the SAME contract
    claude-/codex-/cursor-/hermes-native use (see their
    ``_post_external_session_status`` / ``_post_status``). The runner's
    PTY-activity watcher edge is a UI signal only and never wakes a parent,
    which is why this explicit post off the stream's ``message_stop`` turn-end
    marker is required.

    *idempotency_key* names the stream event that justified the edge (the
    ``message_stop`` / ``result`` / ``session_end`` envelope uuid) so a replay
    after a crash between POST and cursor-persist is identifiable as the same
    edge rather than a second turn.

    Duplicate-wake safety today does NOT depend on the server honouring that
    key: the v0.11.0 Sessions endpoint takes ``data`` as a free-form dict and
    ignores unknown keys, but the RUNNER dedupes repeated terminal status for a
    child (``_drained_delivered_subagent_children`` / the ``entry.delivered``
    check in ``omnigent.runner.app``), so a replayed ``idle`` is acknowledged
    ``already_delivered`` and does not wake the parent twice. That dedup is
    runner-local in-memory state, so the key is sent as durable, forward-
    compatible evidence for a runner that was restarted in between.

    :raises httpx.HTTPError: If the Omnigent request fails or is rejected.
    """
    # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
    data: dict[str, object] = {"status": status}
    if idempotency_key is not None:
        # LOCAL PATCH #6: stable per-edge key (see the note above).
        data["idempotency_key"] = f"qwen:{idempotency_key}"
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_session_status", "data": data},
    )
    resp.raise_for_status()


def _tmux_pane_is_alive(bridge_dir: Path) -> bool:
    """Whether the qwen terminal's tmux pane still exists.

    Fail-SAFE: any uncertainty (no advertised ``tmux.json``, tmux missing, the
    command erroring) answers ``True`` — "assume alive". A false ``False`` would
    post a spurious terminal edge and wake the parent mid-turn, which is exactly
    the failure this module exists to prevent; a false ``True`` merely leaves the
    fallback to a later poll.
    """
    # LOCAL PATCH #6: liveness signal behind the process-exit wake fallback.
    info = read_tmux_info(bridge_dir)
    if info is None:
        return True
    try:
        proc = subprocess.run(
            ["tmux", "-S", info["socket_path"], "has-session", "-t", info["tmux_target"]],
            check=False,
            capture_output=True,
            timeout=_TMUX_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return proc.returncode == 0


async def forward_qwen_events_to_session(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    agent_name: str,
    events_file: Path | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    auth: httpx.Auth | None = None,
    terminal_is_alive: Callable[[], bool] | None = None,
) -> None:
    """Tail qwen's ``--json-file`` and mirror new messages into the AP session.

    Polls the event file past a persisted byte offset, posting each new
    user/assistant message as an ``external_conversation_item``. The offset +
    dedup set are persisted to ``bridge_dir`` so a supervisor restart resumes
    without re-posting.

    :param base_url: Omnigent server base URL.
    :param headers: Static HTTP headers (auth normally via ``auth``).
    :param session_id: Omnigent session/conversation id.
    :param bridge_dir: The qwen-native bridge dir (holds the persisted cursor).
    :param agent_name: Agent label stamped on mirrored assistant items.
    :param events_file: qwen ``--json-file`` path; defaults to the bridge dir's.
    :param poll_interval_s: Seconds between event-file polls.
    :param auth: Optional refresh-capable httpx Auth for remote deployments.
    :param terminal_is_alive: Liveness predicate for the qwen terminal; defaults
        to a tmux-pane probe over the bridge dir. When it answers ``False`` the
        stream is drained one last time and, if a turn was still open, a terminal
        edge is posted so the parent is not stranded (see LOCAL PATCH #6).
    :returns: Never normally returns; cancel the task to stop it.
    """
    target = events_file or events_file_path(bridge_dir)
    state = _read_state(bridge_dir)
    offset = state.offset
    seen = _new_seen(state.seen_uuids)
    # LOCAL PATCH #6: the process-exit fallback. `terminal_is_alive` is probed
    # BEFORE each read so the read that follows a death drains every byte qwen
    # flushed on its way out; the terminal edge is only posted after that drain.
    is_alive = terminal_is_alive or (lambda: _tmux_pane_is_alive(bridge_dir))
    exit_terminal_posted = False
    last_probe_at = 0.0
    timeout = httpx.Timeout(_POST_TIMEOUT_S)
    from omnigent.cli_auth import open_server_client

    async with open_server_client(base_url, headers=headers, auth=auth, timeout=timeout) as client:
        while True:
            try:
                # LOCAL PATCH #6: the probe costs a subprocess, so it only runs
                # when it could change anything — a turn is open (something is
                # waiting on a terminal edge) and the throttle has elapsed.
                # Anything else answers "alive": if no turn is open there is
                # nothing to strand, and a stream still producing bytes is
                # self-evidently alive.
                now = _supervisor_monotonic()
                if state.turn_open and now - last_probe_at >= _TMUX_PROBE_INTERVAL_S:
                    last_probe_at = now
                    alive = await asyncio.to_thread(is_alive)
                else:
                    alive = True
                # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
                poll = await asyncio.to_thread(
                    _read_new_events, target, offset, seen, agent_name, state
                )
                for item in poll.items:
                    await _post_conversation_item(client, session_id=session_id, item=item)
                    # LOCAL PATCH #6: mark the ITEM, not its event. A failure on a
                    # later item of the same event now re-posts only that item on
                    # the next poll instead of being suppressed as "seen".
                    seen[item.item_uuid] = None
                # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
                # Post the parent-waking terminal edge AFTER the batch's mirrored
                # items, so the reply content is in the store before the wake.
                for wake in poll.wakes:
                    await _post_external_session_status(
                        client,
                        session_id=session_id,
                        status=wake.status,
                        idempotency_key=wake.uuid,
                    )
                    seen[wake.uuid] = None
                if poll.offset != offset or poll.items or poll.wakes:
                    exit_terminal_posted = False  # fresh activity re-arms it
                offset = poll.offset
                state = _ForwardState(
                    offset=offset,
                    seen_uuids=list(seen),
                    pending_window_id=poll.pending_window_id,
                    pending_stop_reason=poll.pending_stop_reason,
                    turn_open=poll.turn_open,
                )
                _write_state(bridge_dir, state)
                # LOCAL PATCH #6: the qwen process is gone and the drain above
                # consumed its last bytes. If a turn was still in flight it will
                # never get its own message_stop, so the parent would wait
                # forever — post the terminal edge here instead. Deduplicated
                # against the normal path by `turn_open`, which the message_stop
                # / result / session_end branches clear.
                if not alive and not exit_terminal_posted:
                    if state.turn_open:
                        _logger.warning(
                            "qwen terminal exited with a turn still open; posting %r so the "
                            "parent is not left waiting; session=%s bridge_dir=%s",
                            _STATUS_FAILED,
                            session_id,
                            bridge_dir,
                        )
                        await _post_external_session_status(
                            client, session_id=session_id, status=_STATUS_FAILED
                        )
                        state = dataclasses.replace(state, turn_open=False)
                        _write_state(bridge_dir, state)
                    exit_terminal_posted = True
            except asyncio.CancelledError:
                raise
            except Exception:
                _logger.exception(
                    "qwen forwarder poll failed; session=%s bridge_dir=%s",
                    session_id,
                    bridge_dir,
                )
            await asyncio.sleep(poll_interval_s)


def _supervisor_monotonic() -> float:
    """Indirection so tests can stub the supervisor's clock."""
    return time.monotonic()


async def _supervisor_sleep(seconds: float) -> None:
    """Indirection so tests can stub the supervisor's backoff sleep."""
    await asyncio.sleep(seconds)


async def supervise_qwen_forwarder(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    agent_name: str,
    events_file: Path | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    auth: httpx.Auth | None = None,
    terminal_is_alive: Callable[[], bool] | None = None,
) -> None:
    """Run :func:`forward_qwen_events_to_session` under a restart supervisor.

    Mirrors :func:`omnigent.goose_native_forwarder.supervise_goose_forwarder`:
    bounded exponential backoff, :class:`asyncio.CancelledError` propagates for
    clean teardown, and the persisted offset means restarts resume exactly where
    they left off — including, since LOCAL PATCH #6, the open turn's tool-loop
    classification, so a restart mid tool-loop cannot mistake the next
    ``message_stop`` for a turn end.

    :returns: Never normally returns; cancel the task to stop it.
    """
    backoff_s = _SUPERVISOR_INITIAL_BACKOFF_S
    while True:
        run_started_at = _supervisor_monotonic()
        crash_exc: Exception | None = None
        try:
            await forward_qwen_events_to_session(
                base_url=base_url,
                headers=headers,
                session_id=session_id,
                bridge_dir=bridge_dir,
                agent_name=agent_name,
                events_file=events_file,
                poll_interval_s=poll_interval_s,
                auth=auth,
                terminal_is_alive=terminal_is_alive,
            )
            _logger.warning(
                "qwen forwarder returned unexpectedly; restarting; session=%s bridge_dir=%s",
                session_id,
                bridge_dir,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — supervisor restarts on any Exception
            crash_exc = exc
        if _supervisor_monotonic() - run_started_at >= _SUPERVISOR_HEALTHY_UPTIME_S:
            backoff_s = _SUPERVISOR_INITIAL_BACKOFF_S
        if crash_exc is not None:
            _logger.error(
                "qwen forwarder crashed; restarting in %.1fs; session=%s bridge_dir=%s",
                backoff_s,
                session_id,
                bridge_dir,
                exc_info=crash_exc,
            )
        await _supervisor_sleep(backoff_s)
        backoff_s = min(backoff_s * 2.0, _SUPERVISOR_MAX_BACKOFF_S)


# --- Compaction mirror (chat-recording tail → external_compaction_status) ------

#: qwen's CompressionStatus enum (verified, qwen v0.18.2): 1 = COMPRESSED (success),
#: 2/3 = COMPRESSION_FAILED_*. 1 → completed; anything else → failed.
_COMPRESSION_STATUS_OK = 1


def _compaction_status_from_record(record: dict[str, object]) -> str | None:
    """Map a chat-recording line to a compaction status, or ``None`` to skip it.

    Returns ``"completed"`` for a successful ``chat_compression`` record,
    ``"failed"`` for a failed one, and ``None`` for any other line.
    """
    if record.get("type") != "system" or record.get("subtype") != "chat_compression":
        return None
    payload = record.get("systemPayload")
    info = payload.get("info") if isinstance(payload, dict) else None
    status = info.get("compressionStatus") if isinstance(info, dict) else None
    return "completed" if status == _COMPRESSION_STATUS_OK else "failed"


def _read_new_compaction_statuses(recording: Path, offset: int) -> tuple[list[str], int]:
    """Read NDJSON lines past *offset*, returning new compaction statuses + offset.

    Same tail discipline as :func:`_read_new_events` (truncation rewind, only
    newline-terminated lines consumed), but scoped to ``chat_compression`` records.
    """
    try:
        size = recording.stat().st_size
    except OSError:
        return [], offset  # recording not created yet — retry next poll
    if size < offset:
        offset = 0  # truncated/recreated
    if size == offset:
        return [], offset
    try:
        with open(recording, "rb") as fh:
            fh.seek(offset)
            data = fh.read(size - offset)
    except OSError:
        return [], offset
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset
    consumed = data[: last_nl + 1]
    new_offset = offset + len(consumed)
    statuses: list[str] = []
    for raw in consumed.split(b"\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        status = _compaction_status_from_record(record)
        if status is not None:
            statuses.append(status)
    return statuses, new_offset


async def _post_external_compaction_status(
    client: httpx.AsyncClient, *, session_id: str, status: str
) -> None:
    """POST one ``external_compaction_status`` event; the server republishes the SSE."""
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_compaction_status", "data": {"status": status}},
    )
    resp.raise_for_status()


async def supervise_qwen_compaction_mirror(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    recording_path: Path,
    auth: httpx.Auth | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
) -> None:
    """Tail qwen's chat recording and mirror compaction completions to the session.

    Seeds the read offset at the recording's current end of file so only
    compactions that happen *after* launch are posted — a resumed session's
    recording already holds prior ``chat_compression`` records, and re-posting them
    would flash stale "Conversation compacted" dividers. Self-healing: any error is
    logged and the loop continues (a transient blip never abandons the mirror);
    cancellation propagates for clean teardown. Best-effort, like the approval
    mirror — the offset is in-memory, so a forwarder restart may miss a compaction
    that lands during the gap, which only drops one divider.

    :param recording_path: qwen's chat recording for this session (see
        :func:`omnigent.qwen_native_bridge.qwen_session_recording_path`).
    """
    try:
        offset = recording_path.stat().st_size
    except OSError:
        offset = 0  # not created yet; first poll reads from the start
    timeout = httpx.Timeout(_POST_TIMEOUT_S)
    from omnigent.cli_auth import open_server_client

    async with open_server_client(base_url, headers=headers, auth=auth, timeout=timeout) as client:
        while True:
            try:
                statuses, offset = await asyncio.to_thread(
                    _read_new_compaction_statuses, recording_path, offset
                )
                for status in statuses:
                    await _post_external_compaction_status(
                        client, session_id=session_id, status=status
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                _logger.exception(
                    "qwen compaction mirror poll failed; session=%s recording=%s",
                    session_id,
                    recording_path,
                )
            await asyncio.sleep(poll_interval_s)
