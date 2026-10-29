"""Unit tests for the qwen-native bridge + forwarder (the file-based core).

These cover the logic that diverges from goose-native: appending JSONL commands
to qwen's ``--input-file`` and parsing its ``--json-file`` stream-json events.
The event shapes cover qwen v0.18.1 and v0.21.14 (see
``docs/QWEN_NATIVE_DESIGN.md``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from omnigent.harnesses.qwen_native import bridge as qnb
from omnigent.harnesses.qwen_native import forwarder as fwd
from omnigent.harnesses.qwen_native.bridge import (
    BRIDGE_DIR_ENV_VAR,
    bridge_dir_for_session_id,
    build_qwen_native_spawn_env,
    events_file_path,
    input_file_path,
    prepare_bridge_files,
    qwen_session_id_for_conversation,
    qwen_session_recording_exists,
    read_tmux_info,
    submit_confirmation,
    submit_user_message,
    wait_for_ready,
    write_tmux_target,
)
from omnigent.harnesses.qwen_native.forwarder import (
    _DEDUP_WINDOW,
    _compaction_status_from_record,
    _event_to_item,
    _ForwardState,
    _new_seen,
    _read_new_compaction_statuses,
    _read_new_forward_events,
    _read_state,
    _write_state,
    clear_qwen_bridge_state,
)
from omnigent.terminals import pane_progress

_AGENT = "qwen-native-ui"


def _user_ev(uuid: str, text: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _asst_ev(uuid: str, content: list[dict], stop_reason: str | None = None) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "message": {"role": "assistant", "content": content, "stop_reason": stop_reason},
    }


def _message_stop_ev(uuid: str) -> dict:
    return {"type": "stream_event", "uuid": uuid, "event": {"type": "message_stop"}}


def _ev_bytes(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def test_user_event_maps_to_input_text() -> None:
    item = _event_to_item(_user_ev("u1", "hi"), _AGENT)
    assert item is not None
    assert item.response_id == "qwen:u1"
    assert item.item_data == {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}


def test_assistant_text_maps_to_output_text_and_skips_thinking() -> None:
    item = _event_to_item(
        _asst_ev(
            "a1",
            [
                {"type": "thinking", "thinking": "secret reasoning"},
                {"type": "text", "text": "Hi there!"},
            ],
        ),
        _AGENT,
    )
    assert item is not None
    assert item.item_data["role"] == "assistant"
    assert item.item_data["agent"] == _AGENT
    assert item.item_data["content"] == [{"type": "output_text", "text": "Hi there!"}]


def test_thinking_only_assistant_skipped() -> None:
    item = _event_to_item(_asst_ev("a2", [{"type": "thinking", "thinking": "x"}]), _AGENT)
    assert item is None


def test_non_message_events_ignored() -> None:
    for etype in ("system", "stream_event", "result"):
        assert _event_to_item({"type": etype, "uuid": "x"}, _AGENT) is None
    # control_request is recognized but produces no mirror item.
    control = {
        "type": "control_request",
        "request": {"subtype": "can_use_tool", "tool_name": "shell"},
        "request_id": "r1",
    }
    assert _event_to_item(control, _AGENT) is None


def test_attachment_marker_stripped() -> None:
    item = _event_to_item(_user_ev("u2", "[Attached: /tmp/x.png]\n\nlook"), _AGENT)
    assert item is not None
    assert item.item_data["content"][0]["text"] == "look"


def test_could_not_load_marker_stripped() -> None:
    # The executor types the failed-attachment placeholder into the pane,
    # so it mirrors back as user text; strip it like the path markers.
    item = _event_to_item(
        _user_ev("u3", "[Attachment photo.png could not be loaded]\n\nlook"), _AGENT
    )
    assert item is not None
    assert item.item_data["content"][0]["text"] == "look"
    # Bracketed filenames arrive pre-sanitized ("shot [final].png" →
    # "shot _final_.png"), so the marker still strips cleanly.
    item = _event_to_item(
        _user_ev("u4", "[Attachment shot _final_.png could not be loaded]\n\nlook"), _AGENT
    )
    assert item is not None
    assert item.item_data["content"][0]["text"] == "look"


def test_read_new_events_incremental_and_partial_line(tmp_path: Path) -> None:
    f = tmp_path / "out.ndjson"
    f.write_bytes(_ev_bytes(_user_ev("u1", "q")))
    items, off, _, _ = _read_new_forward_events(f, 0, set(), _AGENT, "")
    assert [i.uuid for i in items] == ["u1"]
    assert off == f.stat().st_size

    # Append a complete assistant line plus a trailing *partial* line.
    with open(f, "ab") as fh:
        fh.write(_ev_bytes(_asst_ev("a1", [{"type": "text", "text": "a"}])))
        fh.flush()
        complete_size = f.stat().st_size
        fh.write(b'{"type":"assistant","uuid":"a2"')  # no newline yet
        fh.flush()
    items, off2, _, _ = _read_new_forward_events(f, off, {"u1"}, _AGENT, "")
    assert [i.uuid for i in items] == ["a1"]
    # Offset stops at the last newline — the partial line is not consumed.
    assert off2 == complete_size


def test_read_new_events_detects_truncation(tmp_path: Path) -> None:
    f = tmp_path / "out.ndjson"
    # A long first line so the stale offset exceeds the post-truncation size.
    f.write_bytes(_ev_bytes(_user_ev("u1", "first message, intentionally long " * 4)))
    _, off, _, _ = _read_new_forward_events(f, 0, set(), _AGENT, "")
    assert off > 0
    # A relaunched terminal truncates + writes a shorter line; size < offset
    # must rewind so the fresh content is not skipped.
    f.write_bytes(_ev_bytes(_user_ev("u2", "fresh")))
    assert f.stat().st_size < off
    items, _, _, _ = _read_new_forward_events(f, off, set(), _AGENT, "")
    assert [i.uuid for i in items] == ["u2"]


def test_malformed_line_tolerated(tmp_path: Path) -> None:
    f = tmp_path / "out.ndjson"
    f.write_bytes(b"not json\n" + _ev_bytes(_user_ev("u1", "ok")))
    items, _, _, _ = _read_new_forward_events(f, 0, set(), _AGENT, "")
    assert [i.uuid for i in items] == ["u1"]


def test_forward_events_preserve_item_then_terminal_output_order(tmp_path: Path) -> None:
    """The structured result wakes only after its assistant item is available."""
    events = tmp_path / "out.ndjson"
    events.write_bytes(
        _ev_bytes(_user_ev("u1", "question"))
        + _ev_bytes(_asst_ev("a1", [{"type": "text", "text": "final answer"}]))
        + _ev_bytes(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "uuid": "r1",
            }
        )
    )

    actions, offset, buffered, stop_reason = _read_new_forward_events(events, 0, set(), _AGENT, "")

    assert [type(action) for action in actions] == [
        fwd._MirrorItem,
        fwd._MirrorItem,
        fwd._TerminalStatus,
    ]
    terminal = actions[-1]
    assert isinstance(terminal, fwd._TerminalStatus)
    assert terminal.status == "idle"
    assert terminal.output == "final answer"
    assert offset == events.stat().st_size
    assert buffered == ""
    assert stop_reason is None


def test_forward_events_terminal_without_pty_idle_and_failure(tmp_path: Path) -> None:
    """Wire-protocol results terminate success and error turns without PTY edges."""
    events = tmp_path / "out.ndjson"
    events.write_bytes(
        _ev_bytes(_asst_ev("a1", [{"type": "text", "text": "partial answer"}]))
        + _ev_bytes(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "tool failed",
                "uuid": "r1",
            }
        )
    )

    actions, _, _, _ = _read_new_forward_events(events, 0, set(), _AGENT, "")

    terminal = actions[-1]
    assert isinstance(terminal, fwd._TerminalStatus)
    assert terminal.status == "failed"
    assert terminal.output == "tool failed"


def test_qwen_021_message_stop_completes_final_text_turn(tmp_path: Path) -> None:
    """Qwen 0.21.14 uses message_stop instead of a top-level result record."""
    events = tmp_path / "out.ndjson"
    events.write_bytes(
        _ev_bytes(_asst_ev("a1", [{"type": "text", "text": "final answer"}]))
        + _ev_bytes(_message_stop_ev("stop-1"))
    )

    actions, _, buffered, stop_reason = _read_new_forward_events(events, 0, set(), _AGENT, "")

    assert [type(action) for action in actions] == [
        fwd._MirrorItem,
        fwd._TerminalStatus,
    ]
    terminal = actions[-1]
    assert isinstance(terminal, fwd._TerminalStatus)
    assert terminal.status == "idle"
    assert terminal.output == "final answer"
    assert terminal.response_id == "qwen:stop-1"
    assert buffered == ""
    assert stop_reason is None


def test_qwen_021_message_stop_skips_tool_use_and_completes_final_turn(tmp_path: Path) -> None:
    """An intermediate tool-use message_stop must not terminalize the child."""
    events = tmp_path / "out.ndjson"
    events.write_bytes(
        _ev_bytes(
            _asst_ev(
                "a-tool",
                [{"type": "tool_use", "id": "call-1", "name": "read_file"}],
                stop_reason="tool_use",
            )
        )
        + _ev_bytes(_message_stop_ev("stop-tool"))
        + _ev_bytes(
            {
                "type": "user",
                "uuid": "tool-result",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call-1"}],
                },
            }
        )
        + _ev_bytes(_asst_ev("a-final", [{"type": "text", "text": "done"}]))
        + _ev_bytes(_message_stop_ev("stop-final"))
    )

    actions, _, _, _ = _read_new_forward_events(events, 0, set(), _AGENT, "")
    terminals = [action for action in actions if isinstance(action, fwd._TerminalStatus)]

    assert [terminal.uuid for terminal in terminals] == ["stop-final"]
    assert terminals[0].output == "done"


def test_qwen_021_message_stop_context_survives_poll_boundary(tmp_path: Path) -> None:
    """Assistant and message_stop can land in separate forwarder polls."""
    events = tmp_path / "out.ndjson"
    events.write_bytes(_ev_bytes(_asst_ev("a1", [{"type": "text", "text": "answer"}])))

    first, offset, buffered, stop_reason = _read_new_forward_events(events, 0, set(), _AGENT, "")
    assert [action.uuid for action in first] == ["a1"]
    assert buffered == "answer"
    assert stop_reason == ""

    with open(events, "ab") as fh:
        fh.write(_ev_bytes(_message_stop_ev("stop-1")))

    second, _, buffered, stop_reason = _read_new_forward_events(
        events,
        offset,
        {"a1"},
        _AGENT,
        buffered,
        stop_reason,
    )
    assert [action.uuid for action in second] == ["stop-1"]
    terminal = second[0]
    assert isinstance(terminal, fwd._TerminalStatus)
    assert terminal.output == "answer"
    assert buffered == ""
    assert stop_reason is None


def test_qwen_021_truncation_resets_persisted_assistant_context(tmp_path: Path) -> None:
    """A fresh event file must not inherit assistant context from a prior process."""
    state = _ForwardState(
        offset=10_000,
        seen_uuids=[],
        last_assistant_text="stale answer",
        last_assistant_stop_reason="tool_use",
    )
    assert _write_state(tmp_path, state) is True
    persisted = _read_state(tmp_path)

    events = tmp_path / "out.ndjson"
    events.write_bytes(
        _ev_bytes(_asst_ev("a-fresh", [{"type": "text", "text": "fresh answer"}]))
        + _ev_bytes(_message_stop_ev("stop-fresh"))
    )
    assert events.stat().st_size < persisted.offset

    actions, _, buffered, stop_reason = _read_new_forward_events(
        events,
        persisted.offset,
        set(persisted.seen_uuids or []),
        _AGENT,
        persisted.last_assistant_text,
        persisted.last_assistant_stop_reason,
    )

    terminals = [action for action in actions if isinstance(action, fwd._TerminalStatus)]
    assert len(terminals) == 1
    assert terminals[0].output == "fresh answer"
    assert buffered == ""
    assert stop_reason is None


def test_result_is_error_is_authoritative_for_custom_success_subtype() -> None:
    terminal = fwd._event_to_terminal(
        {
            "type": "result",
            "subtype": "success_with_warnings",
            "is_error": False,
            "result": "completed with warnings",
            "uuid": "r-warning",
        },
        "",
    )

    assert terminal is not None
    assert terminal.status == "idle"
    assert terminal.output == "completed with warnings"


def test_forward_events_dedup_and_genuinely_empty_result(tmp_path: Path) -> None:
    """Duplicate results stay suppressed and output-free success stays explicit."""
    events = tmp_path / "out.ndjson"
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "uuid": "r-empty",
    }
    events.write_bytes(_ev_bytes(result) + _ev_bytes(result))

    actions, _, _, _ = _read_new_forward_events(events, 0, {"r-empty"}, _AGENT, "")
    assert actions == []

    actions, _, _, _ = _read_new_forward_events(events, 0, set(), _AGENT, "")
    terminals = [action for action in actions if isinstance(action, fwd._TerminalStatus)]
    assert len(terminals) == 1
    assert all(terminal.output is None for terminal in terminals)


# --- Compaction mirror (chat-recording tail) -------------------------------


def _compression_record(status: int) -> dict:
    """A qwen chat-recording ``chat_compression`` system event."""
    return {
        "type": "system",
        "subtype": "chat_compression",
        "systemPayload": {
            "info": {
                "originalTokenCount": 19398,
                "newTokenCount": 17000,
                "compressionStatus": status,
            }
        },
    }


def test_compaction_status_from_record() -> None:
    assert _compaction_status_from_record(_compression_record(1)) == "completed"
    # COMPRESSION_FAILED_* statuses → failed.
    assert _compaction_status_from_record(_compression_record(2)) == "failed"
    assert _compaction_status_from_record(_compression_record(3)) == "failed"
    # Other recording lines are ignored.
    assert _compaction_status_from_record({"type": "system", "subtype": "ui_telemetry"}) is None
    assert _compaction_status_from_record(_user_ev("u1", "hi")) is None


def test_read_new_compaction_statuses_incremental_and_ignores_other_lines(tmp_path: Path) -> None:
    rec = tmp_path / "chat.jsonl"
    # A transcript line + a successful compression record.
    rec.write_bytes(_ev_bytes(_user_ev("u1", "hi")) + _ev_bytes(_compression_record(1)))
    statuses, off = _read_new_compaction_statuses(rec, 0)
    assert statuses == ["completed"]
    assert off == rec.stat().st_size
    # No new lines → nothing.
    assert _read_new_compaction_statuses(rec, off) == ([], off)


def test_read_new_compaction_statuses_detects_truncation(tmp_path: Path) -> None:
    rec = tmp_path / "chat.jsonl"
    rec.write_bytes(_ev_bytes(_user_ev("u1", "padding line, intentionally long " * 4)))
    off = rec.stat().st_size
    # A recreated (shorter) recording must rewind so a fresh compaction is seen.
    rec.write_bytes(_ev_bytes(_compression_record(1)))
    assert rec.stat().st_size < off
    statuses, _ = _read_new_compaction_statuses(rec, off)
    assert statuses == ["completed"]


def test_read_new_compaction_statuses_missing_file(tmp_path: Path) -> None:
    # Recording not created yet → no statuses, offset unchanged (retry next poll).
    assert _read_new_compaction_statuses(tmp_path / "absent.jsonl", 0) == ([], 0)


def test_wait_for_ready_times_out_without_boot_signal(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge"
    prepare_bridge_files(bridge)
    # No events written → never ready; returns False fast (tiny timeout).
    assert wait_for_ready(bridge, timeout_s=0.05, poll_interval_s=0.01) is False


def test_wait_for_ready_detects_system_event(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge"
    prepare_bridge_files(bridge)
    # qwen emits a compact system/session_start as its first event.
    events_file_path(bridge).write_bytes(
        _ev_bytes({"type": "system", "subtype": "session_start", "uuid": "s1"})
    )
    assert wait_for_ready(bridge, timeout_s=1.0, poll_interval_s=0.01) is True


def test_wait_for_ready_ignores_system_substring_in_non_system_event(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge"
    prepare_bridge_files(bridge)
    # An assistant event whose text payload contains the bytes '"type":"system"'
    # must NOT be read as the boot signal — readiness parses per-line and checks
    # event["type"], so a substring inside another event can't latch ready early.
    events_file_path(bridge).write_bytes(
        _ev_bytes(_asst_ev("a1", [{"type": "text", "text": 'note "type":"system" inside'}]))
    )
    assert wait_for_ready(bridge, timeout_s=0.05, poll_interval_s=0.01) is False


def test_qwen_session_id_is_deterministic_and_uuid() -> None:
    a = qwen_session_id_for_conversation("conv_abc123")
    b = qwen_session_id_for_conversation("conv_abc123")
    c = qwen_session_id_for_conversation("conv_other")
    assert a == b  # stable across calls → resume can recompute it
    assert a != c  # distinct per conversation
    # Valid UUID (qwen requires one for --session-id / --resume).
    import uuid as _uuid

    assert str(_uuid.UUID(a)) == a


def test_qwen_session_recording_exists_is_workspace_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnigent.harnesses.qwen_native.bridge import _qwen_project_slug

    monkeypatch.setenv("HOME", str(tmp_path))
    ws_a = tmp_path / "repo_a"
    ws_a.mkdir()
    ws_b = tmp_path / "repo_b"
    ws_b.mkdir()
    sid = qwen_session_id_for_conversation("conv_resume_me")
    # No recording yet → fresh launch (--session-id).
    assert qwen_session_recording_exists(sid, ws_a) is False
    # qwen records under the LAUNCH workspace's project slug.
    chats = tmp_path / ".qwen" / "projects" / _qwen_project_slug(ws_a) / "chats"
    chats.mkdir(parents=True)
    (chats / f"{sid}.jsonl").write_text("{}\n", encoding="utf-8")
    assert qwen_session_recording_exists(sid, ws_a) is True
    # Resuming the SAME conversation from a DIFFERENT workspace must NOT see it:
    # qwen --resume is per-project, so a cross-workspace True would pick --resume
    # and land on qwen's blocking "No saved session found" screen.
    assert qwen_session_recording_exists(sid, ws_b) is False
    # A different conversation's id under the same workspace is unaffected.
    assert qwen_session_recording_exists(qwen_session_id_for_conversation("conv_x"), ws_a) is False


def test_qwen_session_recording_path_is_workspace_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnigent.harnesses.qwen_native.bridge import (
        _qwen_project_slug,
        qwen_session_recording_path,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    ws = tmp_path / "repo"
    ws.mkdir()
    sid = qwen_session_id_for_conversation("conv_rec")
    expected = tmp_path / ".qwen" / "projects" / _qwen_project_slug(ws) / "chats" / f"{sid}.jsonl"
    assert qwen_session_recording_path(sid, ws) == expected


def test_bridge_submit_and_confirmation_append_jsonl(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge"
    prepare_bridge_files(bridge)
    in_file = input_file_path(bridge)
    assert in_file.exists() and in_file.read_text() == ""

    submit_user_message(bridge, content="hello world")
    submit_confirmation(bridge, request_id="r1", allowed=True)

    lines = [json.loads(line) for line in in_file.read_text().splitlines() if line.strip()]
    assert lines[0] == {"type": "submit", "text": "hello world"}
    assert lines[1] == {"type": "confirmation_response", "request_id": "r1", "allowed": True}


def test_forward_state_roundtrip_and_clear(tmp_path: Path) -> None:
    state = _ForwardState(
        offset=123,
        seen_uuids=["a", "b"],
        last_assistant_text="buffered answer",
        last_assistant_stop_reason="tool_use",
    )
    assert _write_state(tmp_path, state) is True
    loaded = _read_state(tmp_path)
    assert loaded.offset == 123
    assert loaded.seen_uuids == ["a", "b"]
    assert loaded.last_assistant_text == "buffered answer"
    assert loaded.last_assistant_stop_reason == "tool_use"
    # Clearing resets the cursor so a re-created terminal starts clean.
    clear_qwen_bridge_state(tmp_path)
    cleared = _read_state(tmp_path)
    assert cleared.offset == 0
    assert cleared.seen_uuids == []


def test_forward_state_caps_seen_uuids(tmp_path: Path) -> None:
    # The dedup window is bounded so the state file can't grow unbounded.
    state = _ForwardState(offset=1, seen_uuids=[str(i) for i in range(1000)])
    assert _write_state(tmp_path, state) is True
    loaded = _read_state(tmp_path)
    assert len(loaded.seen_uuids or []) == 512
    assert (loaded.seen_uuids or [])[-1] == "999"  # most-recent retained


def test_seen_dedup_window_keeps_most_recent_in_insertion_order(tmp_path: Path) -> None:
    """The persisted dedup window retains the *most recent* uuids, in order.

    Regression: ``seen`` used to be a ``set``, so the forwarder's
    ``list(seen)`` at persist time was hash-ordered and ``_write_state``'s
    ``[-_DEDUP_WINDOW:]`` cap kept an arbitrary subset — not the most recent.
    On a qwen relaunch past ``_DEDUP_WINDOW`` events (offset rewinds to 0 and
    the file is re-read from the top), recent uuids evicted from the window
    were re-posted as duplicate bubbles. Building ``seen`` through
    :func:`_new_seen` (an insertion-ordered ``dict``) keeps the real tail.

    This exercises the forwarder's own reload/persist path — ``_new_seen`` (the
    line-301 ``seen = _new_seen(persisted.seen_uuids)`` idiom) then
    ``list(seen)`` — so a revert to a ``set`` fails the ordering assertion here
    (unlike ``test_forward_state_caps_seen_uuids``, which feeds an
    already-ordered list straight into ``_write_state`` and so never sees the
    ordering bug).
    """
    uuids = [f"u{i:05d}" for i in range(_DEDUP_WINDOW * 2)]
    seen = _new_seen(uuids)

    assert _write_state(tmp_path, _ForwardState(offset=7, seen_uuids=list(seen))) is True

    kept = _read_state(tmp_path).seen_uuids or []
    # The cap must keep the most-recent _DEDUP_WINDOW uuids, in order — not an
    # arbitrary hash-ordered subset (which a set-backed ``seen`` produced).
    assert kept == uuids[-_DEDUP_WINDOW:]


def test_tmux_target_round_trip(tmp_path: Path) -> None:
    write_tmux_target(tmp_path, socket_path=Path("/tmp/qwen.sock"), tmux_target="sess:0.0")
    info = read_tmux_info(tmp_path)
    assert info == {"socket_path": "/tmp/qwen.sock", "tmux_target": "sess:0.0"}


def test_read_tmux_info_missing(tmp_path: Path) -> None:
    assert read_tmux_info(tmp_path) is None


def test_spawn_env_carries_bridge_dir() -> None:
    env = build_qwen_native_spawn_env("conv_spawn_env")
    assert env[BRIDGE_DIR_ENV_VAR] == str(bridge_dir_for_session_id("conv_spawn_env"))


def test_harness_registered_aliased_and_native() -> None:
    from omnigent.harness_aliases import canonicalize_harness, is_native_harness
    from omnigent.native.native_coding_agents import native_coding_agent_for_harness
    from omnigent.runtime.harnesses import _HARNESS_MODULES
    from omnigent.spec._omnigent_compat import OMNIGENT_HARNESSES

    # Registry entry resolves to the harness module.
    assert _HARNESS_MODULES["qwen-native"] == "omnigent.inner.qwen_native_harness"
    # Allowlisted + recognized as a native-terminal harness (both spellings).
    assert "qwen-native" in OMNIGENT_HARNESSES
    assert is_native_harness("qwen-native") is True
    assert is_native_harness("native-qwen") is True
    assert canonicalize_harness("native-qwen") == "qwen-native"
    # Native coding-agent metadata is wired for the picker / labels.
    agent = native_coding_agent_for_harness("qwen-native")
    assert agent is not None
    assert agent.terminal_name == "qwen"
    assert agent.display_name == "Qwen Code"
    assert agent.agent_name == "qwen-native-ui"


def test_harness_create_app_builds() -> None:
    from omnigent.inner.qwen_native_harness import create_app

    app = create_app()
    assert app is not None


# --- bridge tmux helpers (interrupt / hard-stop) -----------------------------


class _FakeProc:
    """Minimal ``subprocess.run`` result stand-in."""

    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_inject_interrupt_sends_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tmux_target(tmp_path, socket_path=Path("/tmp/q.sock"), tmux_target="sess:0.0")
    captured: list[list[str]] = []
    monkeypatch.setattr(
        qnb.subprocess, "run", lambda cmd, **_k: captured.append(cmd) or _FakeProc(0)
    )
    qnb.inject_interrupt(tmp_path, timeout_s=1.0)
    # No ``-l`` flag: tmux interprets ``Escape`` as a key name, not literal text.
    assert captured[0] == ["tmux", "-S", "/tmp/q.sock", "send-keys", "-t", "sess:0.0", "Escape"]


def test_kill_session_kills_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tmux_target(tmp_path, socket_path=Path("/tmp/q.sock"), tmux_target="sess:0.0")
    captured: list[list[str]] = []
    monkeypatch.setattr(
        qnb.subprocess, "run", lambda cmd, **_k: captured.append(cmd) or _FakeProc(0)
    )
    qnb.kill_session(tmp_path, timeout_s=1.0)
    assert captured[0] == ["tmux", "-S", "/tmp/q.sock", "kill-session", "-t", "sess:0.0"]


def test_run_tmux_raises_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qnb.subprocess, "run", lambda *_a, **_k: _FakeProc(1, stderr="boom"))
    with pytest.raises(RuntimeError, match="boom"):
        qnb._run_tmux("/tmp/q.sock", "send-keys")


def test_inject_interrupt_raises_when_target_unadvertised(tmp_path: Path) -> None:
    # No tmux.json written → the wait times out fast and raises.
    with pytest.raises(RuntimeError):
        qnb.inject_interrupt(tmp_path, timeout_s=0.05)


# --- forwarder: post + poll loop + supervisor --------------------------------


class _RecordingClient:
    """Async httpx-client stub that records POSTs and returns HTTP 200."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    async def post(self, url: str, *, json: dict) -> httpx.Response:
        self.posts.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))


async def test_post_conversation_item_shape() -> None:
    client = _RecordingClient()
    item = fwd._MirrorItem(
        uuid="u1",
        item_type="message",
        item_data={"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        response_id="qwen:u1",
    )
    await fwd._post_conversation_item(client, session_id="conv_1", item=item)  # type: ignore[arg-type]
    url, body = client.posts[0]
    assert url == "/v1/sessions/conv_1/events"
    assert body["type"] == "external_conversation_item"
    assert body["data"] == {
        "item_type": "message",
        "item_data": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        "response_id": "qwen:u1",
    }


async def test_post_external_session_status_shape() -> None:
    client = _RecordingClient()
    terminal = fwd._TerminalStatus(
        uuid="r1", status="idle", output="answer", response_id="qwen:r1"
    )

    await fwd._post_external_session_status(  # type: ignore[arg-type]
        client, session_id="conv_1", terminal=terminal
    )

    _, body = client.posts[0]
    assert body == {
        "type": "external_session_status",
        "data": {"status": "idle", "response_id": "qwen:r1", "output": "answer"},
    }


async def test_partial_batch_failure_retries_only_terminal_with_buffered_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "out.ndjson"
    events.write_bytes(
        _ev_bytes(_asst_ev("a1", [{"type": "text", "text": "durable answer"}]))
        + _ev_bytes(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "uuid": "r1",
            }
        )
    )
    state = _ForwardState(offset=0, seen_uuids=[], last_assistant_text="")
    item_posts: list[str] = []
    terminal_posts: list[str | None] = []

    async def _post_item(_client: object, *, session_id: str, item: object) -> None:
        item_posts.append(item.uuid)  # type: ignore[attr-defined]

    async def _post_terminal(_client: object, *, session_id: str, terminal: object) -> None:
        if not terminal_posts:
            terminal_posts.append(None)
            raise httpx.ConnectError("terminal unavailable")
        terminal_posts.append(terminal.output)  # type: ignore[attr-defined]

    monkeypatch.setattr(fwd, "_post_conversation_item", _post_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _post_terminal)
    actions, _, _, _ = _read_new_forward_events(events, 0, set(), _AGENT, "")

    with pytest.raises(httpx.ConnectError, match="terminal unavailable"):
        await fwd._deliver_forward_actions(  # type: ignore[arg-type]
            object(), session_id="conv", bridge_dir=tmp_path, state=state, actions=actions
        )

    persisted = _read_state(tmp_path)
    assert persisted.offset == 0
    assert persisted.seen_uuids == ["a1"]
    assert persisted.last_assistant_text == "durable answer"
    retry_actions, retry_offset, _, _ = _read_new_forward_events(
        events,
        persisted.offset,
        set(persisted.seen_uuids or []),
        _AGENT,
        persisted.last_assistant_text,
        persisted.last_assistant_stop_reason,
    )
    assert [action.uuid for action in retry_actions] == ["r1"]

    await fwd._deliver_forward_actions(  # type: ignore[arg-type]
        object(),
        session_id="conv",
        bridge_dir=tmp_path,
        state=persisted,
        actions=retry_actions,
    )
    persisted.offset = retry_offset
    assert _write_state(tmp_path, persisted) is True

    assert item_posts == ["a1"]
    assert terminal_posts == [None, "durable answer"]
    final = _read_state(tmp_path)
    assert final.seen_uuids == ["a1", "r1"]
    assert final.last_assistant_text == ""


async def test_forward_loop_posts_new_events_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(_ev_bytes(_user_ev("u1", "hello")))

    posted: list[str] = []

    async def _fake_post(_client: object, *, session_id: str, item: object) -> None:
        posted.append(item.response_id)  # type: ignore[attr-defined]

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_post)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
        )
    )
    for _ in range(200):
        if posted:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    # Awaiting the cancelled task must propagate CancelledError (clean teardown).
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert posted == ["qwen:u1"]
    # Offset + dedup set are persisted so a restart resumes without re-posting.
    state = _read_state(bridge)
    assert state.offset > 0
    assert "u1" in (state.seen_uuids or [])


async def test_compaction_mirror_seeds_at_eof_and_posts_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = tmp_path / "chat.jsonl"
    # A pre-existing compression record (resume case): must NOT be re-posted.
    rec.write_bytes(_ev_bytes(_compression_record(1)))

    posted: list[str] = []

    async def _fake_post(_client: object, *, session_id: str, status: str) -> None:
        posted.append(status)

    monkeypatch.setattr(fwd, "_post_external_compaction_status", _fake_post)

    task = asyncio.create_task(
        fwd.supervise_qwen_compaction_mirror(
            base_url="http://test",
            headers={},
            session_id="conv",
            recording_path=rec,
            poll_interval_s=0.01,
        )
    )
    # Let it seed at EOF; the pre-existing record stays unposted.
    await asyncio.sleep(0.05)
    assert posted == []
    # A new compression now lands and is mirrored.
    with open(rec, "ab") as fh:
        fh.write(_ev_bytes(_compression_record(1)))
    for _ in range(200):
        if posted:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task
    assert posted == ["completed"]


async def test_supervise_restarts_then_propagates_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    async def _fake_forward(**_kw: object) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")  # first run crashes → supervisor restarts
        raise asyncio.CancelledError()  # second run cancelled → propagates out

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(fwd, "forward_qwen_events_to_session", _fake_forward)
    monkeypatch.setattr(fwd, "_supervisor_sleep", _fake_sleep)
    # Never "healthy" (uptime 0) so the backoff is not reset between runs.
    monkeypatch.setattr(fwd, "_supervisor_monotonic", lambda: 0.0)

    with pytest.raises(asyncio.CancelledError):
        await fwd.supervise_qwen_forwarder(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=tmp_path,
            agent_name=_AGENT,
        )

    assert calls["n"] == 2
    assert sleeps == [1.0]  # initial backoff before the one restart


# ---------------------------------------------------------------------------
# Pane-reaper liveness wiring (issue #1349).
#
# The forwarder's stream cursor is the reaper's only token-level evidence that a
# qwen pane is alive. Two cursors are involved and they must NOT be conflated:
# ``state.offset`` is the DELIVERY cursor and only advances once the mirror POST
# succeeds, while the liveness cursor tracks how far the FILE has been read. If
# liveness were stamped off the delivery cursor, a permanently-failing POST would
# re-stamp the same bytes on every poll and a dead pane would never be reapable.
# ---------------------------------------------------------------------------


async def _run_forward_loop(
    bridge: Path,
    *,
    until: Callable[[], bool],
    max_waits: int = 400,
) -> None:
    """Drive the real forward loop until *until* holds (bounded), then stop it."""
    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.001,
        )
    )
    try:
        for _ in range(max_waits):
            if until():
                break
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_forward_loop_records_stream_progress_for_the_pane_reaper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real loop populates the real ledger the pane reaper reads.

    Asserted against ``pane_progress`` itself rather than a stubbed
    ``note_stream_progress``, so a refactor that severs the wiring fails here
    instead of passing silently.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(_ev_bytes(_user_ev("u1", "hello")))

    posted: list[str] = []

    async def _fake_post(_client: object, *, session_id: str, item: object) -> None:
        posted.append(item.response_id)  # type: ignore[attr-defined]

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_post)
    pane_progress.reset_stream_progress()
    assert pane_progress.stream_progress_age_s("conv") is None

    await _run_forward_loop(bridge, until=lambda: bool(posted))

    assert posted == ["qwen:u1"]
    assert pane_progress.stream_progress_age_s("conv") is not None
    pane_progress.reset_stream_progress()


async def test_failed_delivery_stamps_progress_once_then_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: a retry loop over unchanged bytes must let a pane age out.

    ``state.offset`` never advances while delivery fails, so the loop re-reads the
    same bytes forever. Stamping liveness off that condition renewed the pane's
    lease on every poll and made a dead pane immortal — worse than the
    reaped-while-healthy bug the stream signal was added to fix.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(_ev_bytes(_user_ev("u1", "hello")))

    stamps: list[str] = []
    attempts: list[int] = []
    monkeypatch.setattr(fwd, "note_stream_progress", lambda sid: stamps.append(sid))

    async def _delivery_fails(*_args: object, **_kwargs: object) -> None:
        attempts.append(1)
        raise RuntimeError("mirror endpoint down")

    monkeypatch.setattr(fwd, "_deliver_forward_actions", _delivery_fails)

    # Run until the loop has retried the SAME bytes several times. Driving off
    # the retry count rather than a wall-clock settle keeps this deterministic:
    # each failed poll logs a traceback, so the retry cadence is not the poll
    # interval.
    await _run_forward_loop(bridge, until=lambda: len(attempts) >= 4, max_waits=2000)

    # The loop really did keep retrying the same bytes...
    assert len(attempts) >= 4
    # ...and stamped liveness exactly ONCE for them, so the pane ages out.
    assert stamps == ["conv"]
    # The delivery cursor is still parked, which is what made the bug possible.
    assert _read_state(bridge).offset == 0


async def test_growing_file_keeps_stamping_while_delivery_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live generation with a broken mirror must still be spared.

    The distinction is once per NEW BYTE, not once per delivery attempt: qwen
    still producing tokens is alive even when the mirror endpoint is down.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events = events_file_path(bridge)
    events.write_bytes(_ev_bytes(_user_ev("u1", "hello")))

    stamps: list[str] = []
    monkeypatch.setattr(fwd, "note_stream_progress", lambda sid: stamps.append(sid))

    async def _delivery_fails(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("mirror endpoint down")

    monkeypatch.setattr(fwd, "_deliver_forward_actions", _delivery_fails)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.001,
        )
    )
    try:
        for _ in range(400):
            if stamps:
                break
            await asyncio.sleep(0.005)
        assert stamps == ["conv"]
        # qwen keeps generating even though every POST keeps failing.
        with events.open("ab") as handle:
            handle.write(_ev_bytes(_asst_ev("a1", [{"type": "text", "text": "still working"}])))
        for _ in range(400):
            if len(stamps) > 1:
                break
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stamps == ["conv", "conv"]


# --- Stress cases for the liveness cursor -----------------------------------
#
# The two doors through which the immortal-pane bug could be reopened by a future
# refactor. Both were judged acceptable by inspection in review; these pin them
# so the property is enforced by tests rather than by reasoning. The shape that
# must NEVER return is "stamps scale with POLLS"; what is correct is "stamps scale
# with genuine file mutation, once per restart".


def _spawn_forward_loop(bridge: Path) -> asyncio.Task[None]:
    """Start the real forward loop against *bridge*, polling as fast as possible."""
    return asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.001,
        )
    )


async def _stop_forward_loop(task: asyncio.Task[None]) -> None:
    """Cancel the loop and confirm the clean-teardown contract."""
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _wait_until(predicate: Callable[[], bool], *, max_waits: int = 600) -> None:
    """Yield until *predicate* holds, bounded so a regression fails fast."""
    for _ in range(max_waits):
        if predicate():
            return
        await asyncio.sleep(0.005)


def _count_polls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count real reads of the event file, delegating to the real reader.

    A counter, not a stub: the loop still gets genuine read results, so "how many
    polls happened" can be asserted against "how many stamps happened".
    """
    polls = {"n": 0}
    real_read = fwd._read_new_forward_events

    def _counting_read(*args: object, **kwargs: object) -> object:
        polls["n"] += 1
        return real_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fwd, "_read_new_forward_events", _counting_read)
    return polls


def _spy_on_stamps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every liveness stamp while still writing the REAL ledger."""
    stamps: list[str] = []

    def _spy(session_id: str) -> None:
        stamps.append(session_id)
        pane_progress.note_stream_progress(session_id)

    monkeypatch.setattr(fwd, "note_stream_progress", _spy)
    return stamps


async def test_truncate_rewrite_flap_stamps_per_mutation_not_per_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncate/rewrite flap stamps once per file MUTATION, never once per poll.

    ``_read_new_forward_events`` rewinds to 0 when ``size < offset`` (a bridge
    relaunch truncating the event file), and the liveness cursor compares with
    ``!=``, so a shrinking offset stamps. That is intentional — a relaunch IS
    activity — but only if it is bounded to genuine file mutation rather than to
    poll count; the latter is the immortal-pane bug wearing a different hat.

    The unit is one file MUTATION, not one "relaunch cycle". A relaunch is two
    mutations, because ``bridge.prepare_bridge_files`` truncates the events file
    to EMPTY and qwen then appends to it — so the truncate stamps and the first
    append stamps. Two stamps per relaunch, a fixed constant; the property that
    matters is that no number of polls in between adds a third.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events = events_file_path(bridge)

    def _write_events(count: int, tag: str) -> None:
        events.write_bytes(
            b"".join(_ev_bytes(_user_ev(f"{tag}{i}", "hello")) for i in range(count))
        )

    def _truncate_to_empty() -> None:
        # Exactly what bridge.prepare_bridge_files does on a relaunch.
        events.write_text("", encoding="utf-8")

    _write_events(3, "a")

    posted: list[str] = []

    async def _fake_post(_client: object, *, session_id: str, item: object) -> None:
        posted.append(item.response_id)  # type: ignore[attr-defined]

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_post)
    polls = _count_polls(monkeypatch)
    stamps = _spy_on_stamps(monkeypatch)
    pane_progress.reset_stream_progress()

    async def _settle(expected_stamps: int) -> None:
        """Drive several more polls over unchanged content; nothing may stamp."""
        target = polls["n"] + 5
        await _wait_until(lambda: polls["n"] >= target)
        assert polls["n"] >= target
        assert len(stamps) == expected_stamps, "unchanged content stamped again"

    task = _spawn_forward_loop(bridge)
    try:
        # The initial read of an unseen file is itself one mutation's worth.
        await _wait_until(lambda: len(stamps) >= 1)
        assert len(stamps) == 1
        await _settle(1)

        expected = 1
        for cycle, (count, tag) in enumerate(((2, "b"), (4, "c")), start=1):
            # Mutation 1 of the relaunch: truncate to empty.
            _truncate_to_empty()
            expected += 1
            await _wait_until(lambda want=expected: len(stamps) >= want)
            assert len(stamps) == expected, f"cycle {cycle} truncate stamped twice"
            await _settle(expected)

            # Mutation 2 of the relaunch: qwen writes into the fresh file.
            _write_events(count, tag)
            expected += 1
            await _wait_until(lambda want=expected: len(stamps) >= want)
            assert len(stamps) == expected, f"cycle {cycle} rewrite stamped twice"
            await _settle(expected)
    finally:
        await _stop_forward_loop(task)

    # One initial read + two relaunches x two mutations == five stamps, over many
    # more polls than that. This ratio is the whole point of the test.
    assert len(stamps) == 5
    assert stamps == ["conv"] * 5
    assert polls["n"] > len(stamps) * 4
    # Delivery worked throughout, so every stamp tracked genuine file mutation,
    # and the real ledger the pane reaper reads was populated.
    assert posted
    assert pane_progress.stream_progress_age_s("conv") is not None
    pane_progress.reset_stream_progress()


async def test_restart_with_undelivered_bytes_stamps_once_per_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart stamps ONCE for undelivered bytes, then nothing more.

    The liveness cursor is in-memory and reseeds from the durable ``state.offset``
    on every restart, so bytes that were read but never delivered are stamped
    again by the new run. Review judged that acceptable because restarts back off
    from 1s. The load-bearing half is the second assertion: after that one stamp,
    repeated polls with no new file content must add NOTHING — otherwise a restart
    loop becomes the old per-poll renewal by another route.

    A restart is modelled as a fresh invocation of the loop, which is exactly what
    both restart paths produce: a new in-memory cursor over the same durable state.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(_ev_bytes(_user_ev("u1", "hello")))

    async def _delivery_fails(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("mirror endpoint down")

    monkeypatch.setattr(fwd, "_deliver_forward_actions", _delivery_fails)
    polls = _count_polls(monkeypatch)
    stamps = _spy_on_stamps(monkeypatch)
    pane_progress.reset_stream_progress()

    # Run 1: reads the bytes, stamps once, never delivers them.
    first = _spawn_forward_loop(bridge)
    try:
        await _wait_until(lambda: len(stamps) >= 1)
        assert len(stamps) == 1
    finally:
        await _stop_forward_loop(first)
    # The delivery cursor never advanced, which is what makes the restart re-read
    # the same bytes.
    assert _read_state(bridge).offset == 0

    # Run 2 == the restart. One stamp for the re-read bytes...
    polls_before_restart = polls["n"]
    second = _spawn_forward_loop(bridge)
    try:
        await _wait_until(lambda: len(stamps) >= 2)
        assert len(stamps) == 2
        # ...and then nothing, however long it keeps polling unchanged content.
        settled = polls["n"] + 10
        await _wait_until(lambda: polls["n"] >= settled)
        assert polls["n"] >= settled
        assert len(stamps) == 2, "the restart re-stamped per poll"
    finally:
        await _stop_forward_loop(second)

    assert stamps == ["conv", "conv"]
    # Bounded per restart, not per poll: the restart polled many times for its
    # single stamp.
    assert polls["n"] - polls_before_restart > 10
    assert _read_state(bridge).offset == 0
    pane_progress.reset_stream_progress()
