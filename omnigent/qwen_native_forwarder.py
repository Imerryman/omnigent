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
import json
import logging
import os
import re
import time
from collections.abc import Container, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

from omnigent.inner.native_attachments import ATTACHMENT_MARKER_STRIP_PATTERN
from omnigent.qwen_native_bridge import events_file_path

_logger = logging.getLogger(__name__)

#: Seconds between event-file polls. qwen flushes events per streaming step, so a
#: sub-second cadence keeps the mirrored chat tracking the terminal step by step.
_DEFAULT_POLL_INTERVAL_S = 0.4
_POST_TIMEOUT_S = 30.0

# Supervisor backoff (mirrors goose_native_forwarder.supervise_goose_forwarder).
_SUPERVISOR_INITIAL_BACKOFF_S = 1.0
_SUPERVISOR_MAX_BACKOFF_S = 30.0
_SUPERVISOR_HEALTHY_UPTIME_S = 60.0

_STATE_FILE = "qwen_forwarder.json"

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
    :param seen_uuids: Recently posted event uuids, for idempotent dedup across a
        truncation/restart. Bounded to the most recent entries.
    """

    offset: int = 0
    seen_uuids: list[str] | None = None


def _read_state(bridge_dir: Path) -> _ForwardState:
    """Load the persisted forward cursor, or a cold default."""
    try:
        raw = (bridge_dir / _STATE_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return _ForwardState(offset=0, seen_uuids=[])
    offset = data.get("offset")
    seen = data.get("seen_uuids")
    return _ForwardState(
        offset=offset if isinstance(offset, int) and offset >= 0 else 0,
        seen_uuids=[u for u in seen if isinstance(u, str)] if isinstance(seen, list) else [],
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
            json.dumps({"offset": state.offset, "seen_uuids": seen}),
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
    """One conversation item ready to POST, plus the event uuid that produced it."""

    uuid: str
    item_type: str
    item_data: dict[str, object]
    response_id: str


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
                item_type="message",
                item_data={"role": "user", "content": [{"type": "input_text", "text": text}]},
                response_id=response_id,
            )
        )
        return items
    items.append(
        _MirrorItem(
            uuid=uuid,
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


def _turn_end_uuid(event: dict[str, object]) -> str | None:
    """Return the envelope uuid of a top-level ``message_stop`` event, else ``None``.

    A turn-end candidate is a ``stream_event`` whose nested ``event`` has
    ``type=="message_stop"`` and whose ``parent_tool_use_id`` is null — the
    ``parent_tool_use_id is None`` guard excludes any nested sub-tool turn-end.
    """
    # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
    if event.get("type") != "stream_event":
        return None
    if event.get("parent_tool_use_id") is not None:
        return None
    inner = event.get("event")
    if not isinstance(inner, dict) or inner.get("type") != "message_stop":
        return None
    uuid = event.get("uuid")
    if isinstance(uuid, str) and uuid:
        return uuid
    return None


def _read_new_events(
    events_file: Path,
    offset: int,
    seen: Container[str],
    agent_name: str,
    last_stop_reason: str | None = None,
) -> tuple[list[_MirrorItem], list[str], int, str | None]:
    """Read NDJSON lines past *offset*, returning new mirror items, turn-end
    uuids, the new offset, and the threaded ``last_stop_reason``.

    Detects a truncated/recreated event file (``size < offset``) and rewinds to 0.
    Only fully terminated lines (ending in ``\\n``) are consumed; a trailing
    partial line is left for the next poll by not advancing past it.
    """
    # LOCAL PATCH #4: `last_stop_reason` (param + 4th return element) threads the
    # most recent consolidated assistant `message.stop_reason` across poll
    # batches, so a tool_use step whose message_stop lands in a LATER batch is
    # still recognised as intermediate (cross-poll carry).
    try:
        size = events_file.stat().st_size
    except OSError:
        return [], [], offset, last_stop_reason
    if size < offset:
        offset = 0  # file truncated by a relaunched terminal
    if size == offset:
        return [], [], offset, last_stop_reason
    try:
        with open(events_file, "rb") as fh:
            fh.seek(offset)
            data = fh.read(size - offset)
    except OSError:
        return [], [], offset, last_stop_reason
    # Only consume up to the last newline; keep any trailing partial line.
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], [], offset, last_stop_reason  # no complete line yet
    consumed = data[: last_nl + 1]
    new_offset = offset + len(consumed)
    items: list[_MirrorItem] = []
    # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
    turn_end_uuids: list[str] = []
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
        # LOCAL PATCH #4: record each consolidated assistant event's stop_reason.
        # A message_stop only closes a user-facing turn when the assistant message
        # it closes did NOT stop for tool_use (mid-loop, auto-continuing).
        if event.get("type") == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                stop_reason = message.get("stop_reason")
                last_stop_reason = stop_reason if isinstance(stop_reason, str) else None
        te_uuid = _turn_end_uuid(event)
        if te_uuid is not None and te_uuid not in seen:
            # LOCAL PATCH #4: skip a tool_use-step message_stop (not a turn-end).
            if last_stop_reason != "tool_use":
                turn_end_uuids.append(te_uuid)
        # LOCAL PATCH #5: one event can now yield several items (tool items then
        # prose); all share the event uuid, so dedupe-by-uuid skips whole events.
        for item in _event_to_items(event, agent_name):
            if item.uuid not in seen:
                items.append(item)
    # LOCAL PATCH #4: return last_stop_reason so the caller threads it across polls.
    return items, turn_end_uuids, new_offset, last_stop_reason


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
    client: httpx.AsyncClient, *, session_id: str, status: str
) -> None:
    """POST one ``external_session_status`` event to the Sessions API.

    For a sub-agent conversation the server maps an ``idle`` edge to a terminal
    completion that wakes the parent orchestrator's inbox — the SAME contract
    claude-/codex-/cursor-/hermes-native use (see their
    ``_post_external_session_status`` / ``_post_status``). The runner's
    PTY-activity watcher edge is a UI signal only and never wakes a parent,
    which is why this explicit post off the stream's ``message_stop`` turn-end
    marker is required.

    :raises httpx.HTTPError: If the Omnigent request fails or is rejected.
    """
    # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_session_status", "data": {"status": status}},
    )
    resp.raise_for_status()


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
    :returns: Never normally returns; cancel the task to stop it.
    """
    target = events_file or events_file_path(bridge_dir)
    persisted = _read_state(bridge_dir)
    offset = persisted.offset
    seen = _new_seen(persisted.seen_uuids)
    # LOCAL PATCH #4: cross-poll carry of the last consolidated assistant
    # stop_reason (a tool_use step and its message_stop can land in different
    # poll batches).
    last_stop_reason: str | None = None
    timeout = httpx.Timeout(_POST_TIMEOUT_S)
    from omnigent.cli_auth import open_server_client

    async with open_server_client(base_url, headers=headers, auth=auth, timeout=timeout) as client:
        while True:
            try:
                # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
                items, turn_end_uuids, new_offset, last_stop_reason = await asyncio.to_thread(
                    _read_new_events, target, offset, seen, agent_name, last_stop_reason
                )
                for item in items:
                    await _post_conversation_item(client, session_id=session_id, item=item)
                    seen[item.uuid] = None
                # LOCAL PATCH #3: qwen message_stop -> external_session_status wake
                # Post the parent-waking idle edge AFTER the batch's mirrored
                # items, so the reply content is in the store before the wake.
                for te_uuid in turn_end_uuids:
                    await _post_external_session_status(
                        client, session_id=session_id, status="idle"
                    )
                    seen[te_uuid] = None
                if new_offset != offset or items or turn_end_uuids:
                    offset = new_offset
                    _write_state(
                        bridge_dir,
                        _ForwardState(offset=offset, seen_uuids=list(seen)),
                    )
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
) -> None:
    """Run :func:`forward_qwen_events_to_session` under a restart supervisor.

    Mirrors :func:`omnigent.goose_native_forwarder.supervise_goose_forwarder`:
    bounded exponential backoff, :class:`asyncio.CancelledError` propagates for
    clean teardown, and the persisted offset means restarts resume exactly where
    they left off.

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
