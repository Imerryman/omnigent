"""Unit tests for the qwen-native bridge + forwarder (the file-based core).

These cover the logic that diverges from goose-native: appending JSONL commands
to qwen's ``--input-file`` and parsing its ``--json-file`` stream-json events.
The event shapes are pinned to ``qwen`` v0.18.1 (see ``docs/QWEN_NATIVE_DESIGN.md``).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from omnigent import qwen_native_bridge as qnb
from omnigent import qwen_native_forwarder as fwd
from omnigent.qwen_native_bridge import (
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
from omnigent.qwen_native_forwarder import (
    _DEDUP_WINDOW,
    _STATUS_FAILED,
    _STATUS_IDLE,
    _compaction_status_from_record,
    _event_to_items,
    _ForwardState,
    _item_already_seen,
    _new_seen,
    _read_new_compaction_statuses,
    _read_state,
    _write_state,
    clear_qwen_bridge_state,
)
from omnigent.qwen_native_forwarder import (
    _read_new_events as _read_new_events_raw,
)

_AGENT = "qwen-native-ui"


def _event_to_item(event: dict, agent_name: str) -> fwd._MirrorItem | None:
    """Single-item view of :func:`_event_to_items` for the prose-only cases below.

    ``_event_to_items`` returns a list because one event can now also yield
    ``function_call`` / ``function_call_output`` items; every event these tests
    feed it is prose-only, so it collapses to at most one item.
    """
    items = _event_to_items(event, agent_name)
    assert len(items) <= 1, items
    return items[0] if items else None


def _poll(
    f: Path,
    offset: int = 0,
    seen: object = frozenset(),
    agent_name: str = _AGENT,
    state: fwd._ForwardState | None = None,
) -> fwd._PollResult:
    """Read *f* past *offset* and return the WHOLE :class:`_PollResult`.

    Nothing is discarded: the wake edges and the threaded turn classification
    are the parent-wake contract these tests exist to pin, so they are asserted
    on directly rather than dropped by a convenience shim.
    """
    return _read_new_events_raw(f, offset, seen, agent_name, state)


def _read_new_events(f: Path, offset: int, seen, agent_name: str) -> tuple[list, int]:
    """``(items, new_offset)`` narrowing for the mirroring-only tests below."""
    r = _poll(f, offset, seen, agent_name)
    return r.items, r.offset


def _user_ev(uuid: str, text: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _asst_ev(uuid: str, content: list[dict]) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "message": {"role": "assistant", "content": content},
    }


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
    items, off = _read_new_events(f, 0, set(), _AGENT)
    assert [i.uuid for i in items] == ["u1"]
    assert off == f.stat().st_size

    # Append a complete assistant line plus a trailing *partial* line.
    with open(f, "ab") as fh:
        fh.write(_ev_bytes(_asst_ev("a1", [{"type": "text", "text": "a"}])))
        fh.flush()
        complete_size = f.stat().st_size
        fh.write(b'{"type":"assistant","uuid":"a2"')  # no newline yet
        fh.flush()
    items, off2 = _read_new_events(f, off, {"u1"}, _AGENT)
    assert [i.uuid for i in items] == ["a1"]
    # Offset stops at the last newline — the partial line is not consumed.
    assert off2 == complete_size


def test_read_new_events_detects_truncation(tmp_path: Path) -> None:
    f = tmp_path / "out.ndjson"
    # A long first line so the stale offset exceeds the post-truncation size.
    f.write_bytes(_ev_bytes(_user_ev("u1", "first message, intentionally long " * 4)))
    _, off = _read_new_events(f, 0, set(), _AGENT)
    assert off > 0
    # A relaunched terminal truncates + writes a shorter line; size < offset
    # must rewind so the fresh content is not skipped.
    f.write_bytes(_ev_bytes(_user_ev("u2", "fresh")))
    assert f.stat().st_size < off
    items, _ = _read_new_events(f, off, set(), _AGENT)
    assert [i.uuid for i in items] == ["u2"]


def test_malformed_line_tolerated(tmp_path: Path) -> None:
    f = tmp_path / "out.ndjson"
    f.write_bytes(b"not json\n" + _ev_bytes(_user_ev("u1", "ok")))
    items, _ = _read_new_events(f, 0, set(), _AGENT)
    assert [i.uuid for i in items] == ["u1"]


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
    from omnigent.qwen_native_bridge import _qwen_project_slug

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
    from omnigent.qwen_native_bridge import _qwen_project_slug, qwen_session_recording_path

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
    state = _ForwardState(offset=123, seen_uuids=["a", "b"])
    assert _write_state(tmp_path, state) is True
    loaded = _read_state(tmp_path)
    assert loaded.offset == 123
    assert loaded.seen_uuids == ["a", "b"]
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
    from omnigent.native_coding_agents import native_coding_agent_for_harness
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
        index=0,
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
    # The recorded id is the per-ITEM id ("<event-uuid>#<index>"), not the bare
    # event uuid — see _MirrorItem.item_uuid.
    state = _read_state(bridge)
    assert state.offset > 0
    assert "u1#0" in (state.seen_uuids or [])


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


# --------------------------------------------------------------------------
# Parent-wake path (P11-P13). These replace the old shims that discarded the
# turn-end uuids and the stop-reason classification.
#
# The fixtures under tests/fixtures/qwen_native/ are REAL qwen bridge output
# (qwen v0.22.0, stream-json protocol 2) distilled down: envelopes and the
# fields the forwarder reads are verbatim, content_block_delta noise is dropped
# and long prose is truncated. Facts they pin, measured over 319k recorded
# events across 23 sessions:
#   * a turn is a message_start...message_stop window, and windows never nest
#     (max concurrent depth 1), so the opening message.id names the turn;
#   * message_stop is bare -- {"type": "message_stop"}, no id -- so the
#     classification has to come from the open window, not the stop event;
#   * stop_reason is only ever None (turn end) or "tool_use" (mid-loop);
#   * qwen never emits a `result` event in interactive TUI mode (0 occurrences)
#     even though session_start advertises it, and only 2 of 23 sessions carry
#     system/session_end -- hence the process-exit fallback.
# --------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures" / "qwen_native"


def _fixture_events(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (_FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_bytes(b"".join(_ev_bytes(e) for e in events))


def _stream_ev(uuid: str, inner: dict, parent_tool_use_id: str | None = None) -> dict:
    return {
        "type": "stream_event",
        "uuid": uuid,
        "parent_tool_use_id": parent_tool_use_id,
        "event": inner,
    }


def _msg_start(uuid: str, msg_id: str, parent_tool_use_id: str | None = None) -> dict:
    return _stream_ev(
        uuid,
        {"type": "message_start", "message": {"id": msg_id, "role": "assistant", "content": []}},
        parent_tool_use_id,
    )


def _msg_stop(uuid: str, parent_tool_use_id: str | None = None) -> dict:
    return _stream_ev(uuid, {"type": "message_stop"}, parent_tool_use_id)


def _asst_stop(uuid: str, stop_reason: str | None, blocks: list[dict] | None = None) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parent_tool_use_id": None,
        "message": {
            "role": "assistant",
            "stop_reason": stop_reason,
            "content": blocks if blocks is not None else [{"type": "text", "text": "ok"}],
        },
    }


# --- fixture-driven: the real recorded sessions ----------------------------


def test_real_tool_loop_session_wakes_exactly_once(tmp_path: Path) -> None:
    """A real 7-message_stop tool loop must wake the parent exactly ONCE.

    Regression pin for the premature-wake bug: six of those stops close
    ``stop_reason="tool_use"`` steps that qwen auto-continues from.
    """
    events = _fixture_events("tool_loop_session.ndjson")
    raw_stops = [e for e in events if (e.get("event") or {}).get("type") == "message_stop"]
    assert len(raw_stops) == 7, "fixture must exercise the multi-stop tool loop"

    f = tmp_path / "events.ndjson"
    _write_events(f, events)
    result = _poll(f)

    assert [w.status for w in result.wakes] == [_STATUS_IDLE]
    assert result.turn_open is False
    # The single wake is the LAST message_stop, not one of the tool-loop ones.
    assert result.wakes[0].uuid == raw_stops[-1]["uuid"]


def test_real_tool_loop_wake_is_last_and_follows_all_items(tmp_path: Path) -> None:
    """Ordering: every mirrored item of the turn precedes the wake edge."""
    events = _fixture_events("tool_loop_session.ndjson")
    f = tmp_path / "events.ndjson"
    _write_events(f, events)
    result = _poll(f)

    assert result.items, "the fixture mirrors real tool calls and prose"
    assert {i.item_type for i in result.items} >= {"function_call", "function_call_output"}
    # _PollResult keeps them in separate lists and the loop drains items first,
    # so the wake can only be posted after every item of the batch.
    assert len(result.wakes) == 1


def test_real_api_error_session_still_wakes(tmp_path: Path) -> None:
    """A real API-failure turn (error prose, no tool calls) still wakes the parent."""
    events = _fixture_events("api_error_session.ndjson")
    f = tmp_path / "events.ndjson"
    _write_events(f, events)
    result = _poll(f)

    assert [w.status for w in result.wakes] == [_STATUS_IDLE]
    prose = [i for i in result.items if i.item_type == "message"]
    assert any("API Error" in str(i.item_data) for i in prose)


def test_real_session_end_after_message_stop_does_not_double_wake(tmp_path: Path) -> None:
    """``system/session_end`` after a clean turn end must NOT add a second wake."""
    events = _fixture_events("session_end_session.ndjson")
    assert events[-1]["subtype"] == "session_end", "fixture must end with session_end"
    f = tmp_path / "events.ndjson"
    _write_events(f, events)
    result = _poll(f)

    assert [w.status for w in result.wakes] == [_STATUS_IDLE]
    assert result.turn_open is False


def test_session_end_wakes_when_turn_never_stopped(tmp_path: Path) -> None:
    """``session_end`` while a turn is still open IS the terminal edge."""
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _msg_start("s1", "m1"),
            _asst_stop("m1", None),
            {
                "type": "system",
                "subtype": "session_end",
                "uuid": "se1",
                "parent_tool_use_id": None,
            },
        ],
    )
    result = _poll(f)
    assert [(w.uuid, w.status) for w in result.wakes] == [("se1", _STATUS_IDLE)]
    assert result.turn_open is False


# --- BLOCKING #1: tool-loop state must survive a forwarder restart ----------


def test_tool_use_stop_does_not_wake_within_one_poll(tmp_path: Path) -> None:
    f = tmp_path / "events.ndjson"
    _write_events(f, [_msg_start("s1", "m1"), _asst_stop("m1", "tool_use"), _msg_stop("t1")])
    result = _poll(f)
    assert result.wakes == []
    assert result.turn_open is True, "the tool loop is still in flight"


def test_tool_use_stop_does_not_wake_across_a_split_poll(tmp_path: Path) -> None:
    """The classification survives a poll boundary (the cross-poll carry)."""
    f = tmp_path / "events.ndjson"
    _write_events(f, [_msg_start("s1", "m1"), _asst_stop("m1", "tool_use")])
    first = _poll(f, 0, set(), _AGENT, _ForwardState())
    assert first.wakes == []
    assert first.pending_stop_reason == "tool_use"

    state = _ForwardState(
        offset=first.offset,
        seen_uuids=[],
        pending_window_id=first.pending_window_id,
        pending_stop_reason=first.pending_stop_reason,
        turn_open=first.turn_open,
    )
    with open(f, "ab") as fh:
        fh.write(_ev_bytes(_msg_stop("t1")))
    second = _poll(f, first.offset, set(), _AGENT, state)
    assert second.wakes == [], "a tool_use stop is never a turn end"


def test_tool_use_classification_survives_a_forwarder_RESTART(tmp_path: Path) -> None:
    """BLOCKING #1. The bug: ``last_stop_reason`` was a local that started None on
    every forwarder start, so a ``tool_use`` step whose offset was persisted before
    the restart lost its classification and the following ``message_stop`` was
    misread as a turn end -> premature parent wake mid tool-loop.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    f = events_file_path(bridge)

    # Poll 1: the tool_use step is consumed and its offset persisted.
    _write_events(f, [_msg_start("s1", "m1"), _asst_stop("m1", "tool_use")])
    first = _poll(f, 0, set(), _AGENT, _ForwardState())
    _write_state(
        bridge,
        _ForwardState(
            offset=first.offset,
            seen_uuids=[],
            pending_window_id=first.pending_window_id,
            pending_stop_reason=first.pending_stop_reason,
            turn_open=first.turn_open,
        ),
    )

    # ---- forwarder restarts here: all in-memory state is gone ----
    restored = _read_state(bridge)
    assert restored.pending_stop_reason == "tool_use", "classification must be durable"
    assert restored.turn_open is True

    # The message_stop now arrives, after the restart.
    with open(f, "ab") as fh:
        fh.write(_ev_bytes(_msg_stop("t1")))
    after = _poll(f, restored.offset, set(), _AGENT, restored)
    assert after.wakes == [], "restart must not resurrect the premature wake"


def test_nested_assistant_event_cannot_overwrite_top_level_classification(
    tmp_path: Path,
) -> None:
    """BLOCKING #1 (second half). ``last_stop_reason`` was updated from EVERY
    assistant event while the turn-end check filters to top level, so a nested
    sub-tool assistant event could overwrite the top-level classification.
    """
    f = tmp_path / "events.ndjson"
    nested = {
        "type": "assistant",
        "uuid": "n1",
        "parent_tool_use_id": "call_42",  # nested: belongs to a sub-tool stream
        "message": {"role": "assistant", "stop_reason": None, "content": []},
    }
    _write_events(
        f,
        [_msg_start("s1", "m1"), _asst_stop("m1", "tool_use"), nested, _msg_stop("t1")],
    )
    result = _poll(f)
    assert result.wakes == [], "the nested event must not clear the tool_use classification"


def test_nested_message_stop_is_never_a_turn_end(tmp_path: Path) -> None:
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _msg_start("s1", "m1"),
            _asst_stop("m1", None),
            _msg_stop("t_nested", parent_tool_use_id="call_42"),
        ],
    )
    assert _poll(f).wakes == []


def test_assistant_and_stop_are_correlated_by_the_open_window(tmp_path: Path) -> None:
    """Ordering pin: a new ``message_start`` resets the classification, so the
    previous window's ``tool_use`` cannot leak into the next window's stop.
    """
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _msg_start("s1", "m1"),
            _asst_stop("m1", "tool_use"),
            _msg_stop("t1"),  # mid-loop: no wake
            _msg_start("s2", "m2"),
            _asst_stop("m2", None),
            _msg_stop("t2"),  # real turn end: wake
        ],
    )
    result = _poll(f)
    assert [w.uuid for w in result.wakes] == ["t2"]


# --- BLOCKING #2: a partial multi-item event must not lose items ------------


def test_multi_item_event_gets_distinct_deterministic_item_ids() -> None:
    """BLOCKING #2. All items of one event used to share the event uuid."""
    event = _asst_ev(
        "a1",
        [
            {"type": "tool_use", "id": "call_1", "name": "read", "input": {"p": "/x"}},
            {"type": "text", "text": "reading now"},
        ],
    )
    items = _event_to_items(event, _AGENT)
    assert len(items) == 2
    assert [i.item_uuid for i in items] == ["a1#0", "a1#1"]
    assert len({i.item_uuid for i in items}) == 2
    # Deterministic: re-deriving after a truncation rewind yields the same ids.
    assert [i.item_uuid for i in _event_to_items(event, _AGENT)] == ["a1#0", "a1#1"]


def test_partial_event_replays_only_the_failed_item(tmp_path: Path) -> None:
    """BLOCKING #2. Item 1 POSTs, item 2 fails; the retry must re-offer item 2
    and ONLY item 2. Before the fix the event uuid was marked seen after item 1,
    so the whole event was suppressed and item 2 was dropped for good.
    """
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _asst_ev(
                "a1",
                [
                    {"type": "tool_use", "id": "call_1", "name": "read", "input": {}},
                    {"type": "text", "text": "prose that must not be lost"},
                ],
            )
        ],
    )
    first = _poll(f)
    assert [i.item_uuid for i in first.items] == ["a1#0", "a1#1"]

    # Item 1 succeeded, item 2's POST raised: only item 1 is recorded.
    seen = _new_seen(["a1#0"])
    # The retry re-reads from the SAME offset (the cursor was not advanced).
    retry = _poll(f, 0, seen, _AGENT, _ForwardState())
    assert [i.item_uuid for i in retry.items] == ["a1#1"]
    assert "prose that must not be lost" in str(retry.items[0].item_data)


def test_partial_event_of_several_tool_calls_replays_only_the_gap(tmp_path: Path) -> None:
    """Same guarantee for an event carrying several tool calls."""
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _asst_ev(
                "a1",
                [
                    {"type": "tool_use", "id": "c1", "name": "read", "input": {}},
                    {"type": "tool_use", "id": "c2", "name": "grep", "input": {}},
                    {"type": "tool_use", "id": "c3", "name": "edit", "input": {}},
                ],
            )
        ],
    )
    seen = _new_seen(["a1#0", "a1#2"])  # the middle POST failed
    retry = _poll(f, 0, seen, _AGENT, _ForwardState())
    assert [i.item_uuid for i in retry.items] == ["a1#1"]
    assert retry.items[0].item_data["call_id"] == "c2"


def test_legacy_bare_event_uuid_in_state_still_suppresses_the_event() -> None:
    """Upgrade safety: state written by the pre-patch build holds bare event
    uuids, and must keep suppressing that event rather than re-posting history.
    """
    items = _event_to_items(_asst_ev("a1", [{"type": "text", "text": "hi"}]), _AGENT)
    assert _item_already_seen(items[0], {"a1"}) is True
    assert _item_already_seen(items[0], {"a1#0"}) is True
    assert _item_already_seen(items[0], {"a1#1"}) is False


# --- BLOCKING #3: a terminal edge on failure / EOF --------------------------


def test_eof_without_message_stop_leaves_the_turn_open(tmp_path: Path) -> None:
    """BLOCKING #3, precondition: a stream that dies mid-turn never produces a
    turn-end marker, so nothing on the message_stop path can wake the parent.
    """
    events = _fixture_events("tool_loop_session.ndjson")
    # Truncate the real session just before its final message_stop -- the shape
    # of the one recorded session that really did end mid-stream.
    cut = len(events) - 1
    assert (events[cut].get("event") or {}).get("type") == "message_stop"
    f = tmp_path / "events.ndjson"
    _write_events(f, events[:cut])

    result = _poll(f)
    assert result.wakes == [], "no message_stop means no wake on the normal path"
    assert result.turn_open is True, "so the fallback must have something to fire on"


async def test_process_exit_posts_terminal_status_after_draining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING #3. The qwen process is gone with a turn still open: the
    forwarder must drain the stream, mirror what is there, and THEN post a
    terminal edge so the parent is not stranded.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events = _fixture_events("tool_loop_session.ndjson")
    _write_events(events_file_path(bridge), events[: len(events) - 1])  # no message_stop

    posted_items: list[str] = []
    statuses: list[str] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        posted_items.append(item.item_uuid)  # type: ignore[attr-defined]

    async def _fake_status(_c: object, *, session_id: str, status: str, **_kw: object) -> None:
        statuses.append(status)

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _fake_status)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: False,  # the pane is gone
        )
    )
    for _ in range(300):
        if statuses:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert statuses == [_STATUS_FAILED], "exactly one terminal edge"
    assert posted_items, "the stream was drained BEFORE the terminal edge"
    # Durably closed, so a restart does not fire it again.
    assert _read_state(bridge).turn_open is False


async def test_process_exit_does_not_double_wake_after_a_clean_turn_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING #3, dedup half. A complete turn already posted its ``idle``; the
    process-exit fallback must stay silent.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    _write_events(events_file_path(bridge), _fixture_events("tool_loop_session.ndjson"))

    statuses: list[str] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        return None

    async def _fake_status(_c: object, *, session_id: str, status: str, **_kw: object) -> None:
        statuses.append(status)

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _fake_status)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: False,
        )
    )
    for _ in range(300):
        if statuses:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)  # let several more polls run
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert statuses == [_STATUS_IDLE], "the message_stop wake only, never a second edge"


async def test_wake_is_posted_after_items_and_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering + exactly-once, asserted on one interleaved call log."""
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    _write_events(events_file_path(bridge), _fixture_events("tool_loop_session.ndjson"))

    calls: list[str] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        calls.append(f"item:{item.item_uuid}")  # type: ignore[attr-defined]

    async def _fake_status(_c: object, *, session_id: str, status: str, **_kw: object) -> None:
        calls.append(f"status:{status}")

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _fake_status)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: True,  # still running
        )
    )
    for _ in range(300):
        if any(c.startswith("status:") for c in calls):
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert [c for c in calls if c.startswith("status:")] == ["status:idle"]
    # Every mirrored item precedes the single wake.
    assert calls.index("status:idle") == len(calls) - 1
    assert all(c.startswith("item:") for c in calls[:-1])


def test_result_event_is_honoured_when_qwen_emits_one(tmp_path: Path) -> None:
    """qwen advertises ``result`` in ``supported_events`` but never emits it in
    the interactive mode we tail; handled anyway so a future build is correct.
    """
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _msg_start("s1", "m1"),
            _asst_stop("m1", "tool_use"),
            {"type": "result", "uuid": "r1", "parent_tool_use_id": None, "is_error": True},
        ],
    )
    result = _poll(f)
    assert [(w.uuid, w.status) for w in result.wakes] == [("r1", _STATUS_FAILED)]
    assert result.turn_open is False


def test_result_event_success_maps_to_idle(tmp_path: Path) -> None:
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _msg_start("s1", "m1"),
            {
                "type": "result",
                "uuid": "r1",
                "parent_tool_use_id": None,
                "subtype": "success",
                "is_error": False,
            },
        ],
    )
    assert [w.status for w in _poll(f).wakes] == [_STATUS_IDLE]


# --- #5: idempotency key on the terminal edge ------------------------------


async def test_status_post_carries_a_stable_idempotency_key() -> None:
    client = _RecordingClient()
    await fwd._post_external_session_status(
        client,  # type: ignore[arg-type]
        session_id="conv_1",
        status="idle",
        idempotency_key="turn-end-uuid",
    )
    _url, body = client.posts[0]
    assert body["data"] == {"status": "idle", "idempotency_key": "qwen:turn-end-uuid"}


async def test_status_post_without_a_key_is_unchanged() -> None:
    client = _RecordingClient()
    await fwd._post_external_session_status(
        client,  # type: ignore[arg-type]
        session_id="conv_1",
        status="failed",
    )
    _url, body = client.posts[0]
    assert body["data"] == {"status": "failed"}


# --- liveness probe fail-safety --------------------------------------------


def test_tmux_probe_is_inconclusive_when_target_unadvertised(tmp_path: Path) -> None:
    """No tmux.json is not evidence of death -- it is no verdict at all."""
    assert fwd._tmux_pane_is_alive(tmp_path) is None


def test_tmux_probe_is_inconclusive_when_tmux_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_tmux_target(tmp_path, socket_path=tmp_path / "sock", tmux_target="s:0.0")

    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("tmux missing")

    monkeypatch.setattr(fwd.subprocess, "run", _boom)
    assert fwd._tmux_pane_is_alive(tmp_path) is None


def test_tmux_probe_is_inconclusive_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_tmux_target(tmp_path, socket_path=tmp_path / "sock", tmux_target="s:0.0")

    def _slow(*_a: object, **_k: object) -> object:
        raise subprocess.TimeoutExpired(cmd="tmux", timeout=5.0)

    monkeypatch.setattr(fwd.subprocess, "run", _slow)
    assert fwd._tmux_pane_is_alive(tmp_path) is None


def _fake_tmux(returncode: int, stdout: bytes):
    class _Proc:
        pass

    _Proc.returncode = returncode
    _Proc.stdout = stdout
    return lambda *_a, **_k: _Proc()


def test_tmux_probe_reports_dead_when_the_server_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default spec (no remain-on-exit): the process exit reaps the server."""
    write_tmux_target(tmp_path, socket_path=tmp_path / "sock", tmux_target="s:0.0")
    monkeypatch.setattr(fwd.subprocess, "run", _fake_tmux(1, b""))
    assert fwd._tmux_pane_is_alive(tmp_path) is False


def test_tmux_probe_reports_dead_on_pane_dead_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """remain-on-exit: the session SURVIVES, so only ``#{pane_dead}`` is truth.

    Verified live on this host: with ``keep_alive_after_exit=True`` a
    ``has-session`` probe still returns rc=0 after the inner process exits,
    while ``list-panes -F '#{pane_dead}'`` returns ``1``. A has-session probe
    would report a dead pane as alive forever and the fallback would never fire.
    """
    write_tmux_target(tmp_path, socket_path=tmp_path / "sock", tmux_target="s:0.0")
    monkeypatch.setattr(fwd.subprocess, "run", _fake_tmux(0, b"1\n"))
    assert fwd._tmux_pane_is_alive(tmp_path) is False


def test_tmux_probe_reports_alive_on_live_pane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_tmux_target(tmp_path, socket_path=tmp_path / "sock", tmux_target="s:0.0")
    monkeypatch.setattr(fwd.subprocess, "run", _fake_tmux(0, b"0\n"))
    assert fwd._tmux_pane_is_alive(tmp_path) is True


def test_tmux_probe_reports_dead_when_no_panes_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_tmux_target(tmp_path, socket_path=tmp_path / "sock", tmux_target="s:0.0")
    monkeypatch.setattr(fwd.subprocess, "run", _fake_tmux(0, b"\n"))
    assert fwd._tmux_pane_is_alive(tmp_path) is False


# --- state round-trip for the new durable fields ---------------------------


def test_forward_state_roundtrips_the_turn_classification(tmp_path: Path) -> None:
    state = _ForwardState(
        offset=42,
        seen_uuids=["a#0"],
        pending_window_id="m1",
        pending_stop_reason="tool_use",
        turn_open=True,
    )
    assert _write_state(tmp_path, state) is True
    assert _read_state(tmp_path) == state


def test_forward_state_from_a_pre_patch_file_reads_cold_defaults(tmp_path: Path) -> None:
    """A state file written by the old build has no classification fields."""
    (tmp_path / "qwen_forwarder.json").write_text(
        json.dumps({"offset": 9, "seen_uuids": ["u1"]}), encoding="utf-8"
    )
    state = _read_state(tmp_path)
    assert state.offset == 9
    assert state.pending_window_id is None
    assert state.pending_stop_reason is None
    assert state.turn_open is False


# --------------------------------------------------------------------------
# Second review round: final-drain gap, state-write churn, and the remaining
# non-blocking items.
# --------------------------------------------------------------------------


# --- BLOCKING A: complete-but-unterminated final record --------------------


def test_unterminated_final_record_is_left_alone_on_a_normal_poll(tmp_path: Path) -> None:
    """Normal polls keep the old discipline: never classify on a partial tail."""
    events = _fixture_events("tool_loop_session.ndjson")
    f = tmp_path / "events.ndjson"
    body = b"".join(_ev_bytes(e) for e in events[:-1])
    f.write_bytes(body + json.dumps(events[-1]).encode())  # no trailing newline

    result = _poll(f)  # final_drain defaults to False
    assert result.wakes == [], "the unterminated stop must not be consumed yet"
    assert result.offset == len(body)


def test_final_drain_consumes_a_complete_unterminated_record(tmp_path: Path) -> None:
    """BLOCKING A. Once the pane is dead nothing will ever append the newline,
    so a COMPLETE final record must still be read -- otherwise a real turn-end
    ``message_stop`` is discarded and the fallback posts ``failed`` instead of
    ``idle``.
    """
    f = _FIXTURES / "tool_loop_unterminated_stop.ndjson"
    assert not f.read_bytes().endswith(b"\n"), "fixture must lack the terminator"

    without = _read_new_events_raw(f, 0, set(), _AGENT, _ForwardState(), False)
    assert without.wakes == [], "precondition: the newline-gated read misses it"
    assert without.turn_open is True

    drained = _read_new_events_raw(f, 0, set(), _AGENT, _ForwardState(), True)
    assert [w.status for w in drained.wakes] == [_STATUS_IDLE]
    assert drained.turn_open is False
    assert drained.offset == f.stat().st_size, "the whole file is consumed"


def test_final_drain_leaves_a_genuinely_truncated_record(tmp_path: Path) -> None:
    """A record cut mid-write is NOT complete: leave it, stay turn_open."""
    events = _fixture_events("tool_loop_session.ndjson")
    f = tmp_path / "events.ndjson"
    body = b"".join(_ev_bytes(e) for e in events[:-1])
    truncated = json.dumps(events[-1]).encode()[:40]  # sliced mid-object
    f.write_bytes(body + truncated)

    drained = _read_new_events_raw(f, 0, set(), _AGENT, _ForwardState(), True)
    assert drained.wakes == []
    assert drained.turn_open is True
    assert drained.offset == len(body), "the malformed tail is retained"


def test_is_complete_json_object() -> None:
    assert fwd._is_complete_json_object(b'{"a": 1}') is True
    assert fwd._is_complete_json_object(b'{"a": 1') is False
    assert fwd._is_complete_json_object(b"[1, 2]") is False  # not an object
    assert fwd._is_complete_json_object(b"\xff\xfe") is False


async def test_process_exit_with_unterminated_stop_posts_idle_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING A, end to end: exactly one ``idle``, zero ``failed``."""
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(
        (_FIXTURES / "tool_loop_unterminated_stop.ndjson").read_bytes()
    )

    statuses: list[str] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        return None

    async def _fake_status(_c: object, *, session_id: str, status: str, **_kw: object) -> None:
        statuses.append(status)

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _fake_status)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: False,
        )
    )
    for _ in range(300):
        if statuses:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.15)  # let more polls run
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert statuses == [_STATUS_IDLE]
    assert _STATUS_FAILED not in statuses


# --- BLOCKING B: no state write when nothing changed -----------------------


async def test_idle_polls_perform_zero_state_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING B. The loop runs every poll_interval_s for the life of the
    session; an unconditional _write_state was ~216k temp-file write+renames a
    day per idle session.
    """
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(_ev_bytes(_user_ev("u1", "hello")))

    writes: list[int] = []
    real_write = fwd._write_state

    def _counting_write(bd: Path, st: object) -> bool:
        writes.append(st.offset)  # type: ignore[attr-defined]
        return real_write(bd, st)  # type: ignore[arg-type]

    monkeypatch.setattr(fwd, "_write_state", _counting_write)

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        return None

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: True,
        )
    )
    for _ in range(300):
        if writes:
            break
        await asyncio.sleep(0.01)
    writes_after_first_change = len(writes)
    await asyncio.sleep(0.3)  # ~30 further polls with nothing new
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert writes_after_first_change == 1, "the real change is persisted once"
    assert len(writes) == 1, f"idle polls must not write state; got {len(writes)}"


async def test_state_is_written_again_when_new_events_arrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suppression must not stop a genuine change from being persisted."""
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    f = events_file_path(bridge)
    f.write_bytes(_ev_bytes(_user_ev("u1", "one")))

    writes: list[int] = []
    real_write = fwd._write_state

    def _counting_write(bd: Path, st: object) -> bool:
        writes.append(st.offset)  # type: ignore[attr-defined]
        return real_write(bd, st)  # type: ignore[arg-type]

    monkeypatch.setattr(fwd, "_write_state", _counting_write)

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        return None

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: True,
        )
    )
    for _ in range(200):
        if writes:
            break
        await asyncio.sleep(0.01)
    with open(f, "ab") as fh:
        fh.write(_ev_bytes(_user_ev("u2", "two")))
    for _ in range(200):
        if len(writes) >= 2:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert len(writes) == 2, "one write per real change"
    assert writes[1] > writes[0]


# --- C: inconclusive liveness must not delay the wake forever --------------


async def test_indefinitely_inconclusive_probe_eventually_fires_the_terminal_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable tmux must not strand the parent. After the grace window
    with a silent stream, the pane is treated as dead.
    """
    monkeypatch.setattr(fwd, "_LIVENESS_UNKNOWN_GRACE_S", 0.05)
    monkeypatch.setattr(fwd, "_TMUX_PROBE_INTERVAL_S", 0.0)
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(
        b"".join(_ev_bytes(e) for e in [_msg_start("s1", "m1"), _asst_stop("m1", "tool_use")])
    )

    statuses: list[str] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        return None

    async def _fake_status(_c: object, *, session_id: str, status: str, **_kw: object) -> None:
        statuses.append(status)

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _fake_status)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: None,  # never a verdict
        )
    )
    for _ in range(400):
        if statuses:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert statuses == [_STATUS_FAILED], "the grace expired, so the edge fired"


async def test_inconclusive_probe_holds_alive_inside_the_grace_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside the grace window an inconclusive probe must NOT wake the parent."""
    monkeypatch.setattr(fwd, "_LIVENESS_UNKNOWN_GRACE_S", 3600.0)
    monkeypatch.setattr(fwd, "_TMUX_PROBE_INTERVAL_S", 0.0)
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(
        b"".join(_ev_bytes(e) for e in [_msg_start("s1", "m1"), _asst_stop("m1", "tool_use")])
    )

    statuses: list[str] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        return None

    async def _fake_status(_c: object, *, session_id: str, status: str, **_kw: object) -> None:
        statuses.append(status)

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _fake_status)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: None,
        )
    )
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert statuses == [], "no spurious wake while the verdict is merely unknown"


# --- D: `result` gated on turn_open ----------------------------------------


def test_message_stop_then_result_yields_one_wake(tmp_path: Path) -> None:
    """D. A trailing ``result`` after a turn-end ``message_stop`` must not
    double-wake -- the same gate ``session_end`` already had.
    """
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _msg_start("s1", "m1"),
            _asst_stop("m1", None),
            _msg_stop("t1"),
            {"type": "result", "uuid": "r1", "parent_tool_use_id": None, "subtype": "success"},
        ],
    )
    result = _poll(f)
    assert [w.uuid for w in result.wakes] == ["t1"], "one wake, from the message_stop"


# --- E: the live seen mapping is bounded -----------------------------------


def test_live_seen_mapping_evicts_beyond_the_dedup_window() -> None:
    """E. _write_state capped only what it SERIALISED; the live mapping grew
    for the lifetime of the process and was copied with list(seen) every poll.
    """
    seen = _new_seen()
    for i in range(_DEDUP_WINDOW * 3):
        fwd._remember_seen(seen, f"item-{i}")
        assert len(seen) <= _DEDUP_WINDOW
    assert len(seen) == _DEDUP_WINDOW
    # Oldest evicted, newest retained.
    assert "item-0" not in seen
    assert f"item-{_DEDUP_WINDOW * 3 - 1}" in seen


def test_new_seen_caps_an_oversized_seed() -> None:
    seen = _new_seen([f"u{i}" for i in range(_DEDUP_WINDOW * 2)])
    assert len(seen) == _DEDUP_WINDOW
    assert f"u{_DEDUP_WINDOW * 2 - 1}" in seen
    assert "u0" not in seen


async def test_forward_loop_keeps_seen_bounded_over_many_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound holds through the real loop, not just the helper."""
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    n = _DEDUP_WINDOW + 200
    events_file_path(bridge).write_bytes(
        b"".join(_ev_bytes(_user_ev(f"u{i}", f"m{i}")) for i in range(n))
    )

    posted: list[str] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        posted.append(item.item_uuid)  # type: ignore[attr-defined]

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: True,
        )
    )
    for _ in range(600):
        if len(posted) >= n:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert len(posted) == n
    assert len(_read_state(bridge).seen_uuids or []) <= _DEDUP_WINDOW


# --- F: the process-exit edge carries a deterministic key ------------------


async def test_process_exit_terminal_carries_a_deterministic_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F. The normal terminals already send one; so must the fallback."""
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    events_file_path(bridge).write_bytes(
        b"".join(_ev_bytes(e) for e in [_msg_start("s1", "m1"), _asst_stop("m1", "tool_use")])
    )

    keys: list[object] = []

    async def _fake_item(_c: object, *, session_id: str, item: object) -> None:
        return None

    async def _fake_status(
        _c: object, *, session_id: str, status: str, idempotency_key: object = None
    ) -> None:
        keys.append(idempotency_key)

    monkeypatch.setattr(fwd, "_post_conversation_item", _fake_item)
    monkeypatch.setattr(fwd, "_post_external_session_status", _fake_status)

    task = asyncio.create_task(
        fwd.forward_qwen_events_to_session(
            base_url="http://test",
            headers={},
            session_id="conv",
            bridge_dir=bridge,
            agent_name=_AGENT,
            poll_interval_s=0.01,
            terminal_is_alive=lambda: False,
        )
    )
    for _ in range(300):
        if keys:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert keys == ["process-exit:m1"], "keyed on the turn window it terminates"


# --- G: the legacy bare-uuid fallback is narrowed to solo items ------------


def test_legacy_bare_uuid_closes_a_single_item_event() -> None:
    """A pre-patch build only marked the event uuid AFTER a successful POST, so
    for a one-item event that uuid does prove delivery.
    """
    items = _event_to_items(_asst_ev("a1", [{"type": "text", "text": "hi"}]), _AGENT)
    assert len(items) == 1
    assert _item_already_seen(items[0], {"a1"}, solo=True) is True


def test_legacy_bare_uuid_does_not_close_a_MULTI_item_event() -> None:
    """G. The pre-patch build marked the event uuid after the FIRST item, so a
    bare uuid does NOT prove the later items were ever delivered. Honouring it
    would permanently suppress an item that never reached the server.
    """
    items = _event_to_items(
        _asst_ev(
            "a1",
            [
                {"type": "tool_use", "id": "c1", "name": "read", "input": {}},
                {"type": "text", "text": "prose"},
            ],
        ),
        _AGENT,
    )
    assert len(items) == 2
    for item in items:
        assert _item_already_seen(item, {"a1"}, solo=False) is False
    # The per-item ids still suppress correctly.
    assert _item_already_seen(items[0], {"a1#0"}, solo=False) is True
    assert _item_already_seen(items[1], {"a1#0"}, solo=False) is False


def test_legacy_state_replays_a_multi_item_event_rather_than_dropping_it(
    tmp_path: Path,
) -> None:
    """End to end through the reader: legacy state must not silently swallow
    the undelivered items of a multi-item event.
    """
    f = tmp_path / "events.ndjson"
    _write_events(
        f,
        [
            _asst_ev(
                "a1",
                [
                    {"type": "tool_use", "id": "c1", "name": "read", "input": {}},
                    {"type": "text", "text": "prose that was never delivered"},
                ],
            )
        ],
    )
    legacy_seen = _new_seen(["a1"])  # written by the pre-patch build
    result = _poll(f, 0, legacy_seen, _AGENT, _ForwardState())
    assert [i.item_uuid for i in result.items] == ["a1#0", "a1#1"]
    assert any("never delivered" in str(i.item_data) for i in result.items)


def test_legacy_state_still_suppresses_a_solo_event(tmp_path: Path) -> None:
    f = tmp_path / "events.ndjson"
    _write_events(f, [_user_ev("u1", "already mirrored")])
    result = _poll(f, 0, _new_seen(["u1"]), _AGENT, _ForwardState())
    assert result.items == [], "a one-item event is genuinely closed by its uuid"
