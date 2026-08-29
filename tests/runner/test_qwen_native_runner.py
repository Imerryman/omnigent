"""Unit tests for qwen-native runner-side helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent import qwen_native_bridge as qnb
from omnigent.qwen_native_settings import QWEN_SUBAGENT_DISABLED_TOOLS
from omnigent.runner.app import _build_qwen_fork_recording, _persist_qwen_external_session_id
from omnigent.runner.native.orchestration import (
    _pi_native_launch_config,
    _PiNativeLaunchConfig,
    _trim_qwen_subagent_tool_surface,
)


class _RecordingClient:
    """Async httpx-client stub recording PATCHes; returns a chosen status."""

    def __init__(self, status: int = 200) -> None:
        self.patches: list[tuple[str, dict]] = []
        self._status = status

    async def patch(self, url: str, *, json: dict, timeout: float | None = None) -> httpx.Response:
        self.patches.append((url, json))
        return httpx.Response(self._status, request=httpx.Request("PATCH", url))


async def test_persist_external_session_id_patches_session() -> None:
    client = _RecordingClient()
    await _persist_qwen_external_session_id(client, "conv_abc", "qsid-1")  # type: ignore[arg-type]
    assert client.patches == [("/v1/sessions/conv_abc", {"external_session_id": "qsid-1"})]


async def test_persist_external_session_id_noop_without_client() -> None:
    # No server client (e.g. embedded/test runner) → silent no-op, no raise.
    await _persist_qwen_external_session_id(None, "conv_abc", "qsid-1")


async def test_persist_external_session_id_swallows_errors() -> None:
    # Best-effort: a rejected PATCH or transport error must not raise (only
    # resume/fork carry-over degrades, never the live turn).
    rejected = _RecordingClient(status=500)
    await _persist_qwen_external_session_id(rejected, "conv_abc", "qsid-1")  # type: ignore[arg-type]

    class _Boom:
        async def patch(self, *_a: object, **_k: object) -> httpx.Response:
            raise httpx.ConnectError("down")

    await _persist_qwen_external_session_id(_Boom(), "conv_abc", "qsid-1")  # type: ignore[arg-type]


class _ItemsClient:
    """Async httpx-client stub serving one page of session items from GET /items."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    async def get(self, url: str, *, params: dict | None = None, **_k: object) -> httpx.Response:
        body = {"data": self._items, "has_more": False}
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json=body,
        )


async def test_build_qwen_fork_recording_writes_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate ~/.qwen at a temp HOME so the synthesized recording lands there.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        },
    ]
    client = _ItemsClient(items)

    qsid = await _build_qwen_fork_recording(
        client,  # type: ignore[arg-type]
        session_id="conv_fork",
        workspace=str(workspace),
    )

    # Returns the clone's deterministic id, and a resumable recording now exists.
    assert qsid == qnb.qwen_session_id_for_conversation("conv_fork")
    assert qnb.qwen_session_recording_exists(qsid, workspace)
    recording = qnb.qwen_session_recording_path(qsid, workspace)
    types = [json.loads(line)["type"] for line in recording.read_text().splitlines()]
    assert types == ["user", "assistant"]


async def test_build_qwen_fork_recording_returns_none_when_no_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing carryable → None so the caller launches fresh (no recording written).
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    qsid = await _build_qwen_fork_recording(
        _ItemsClient([]),  # type: ignore[arg-type]
        session_id="conv_empty",
        workspace=str(workspace),
    )
    assert qsid is None
    assert not qnb.qwen_session_recording_exists(
        qnb.qwen_session_id_for_conversation("conv_empty"), workspace
    )


async def test_build_qwen_fork_recording_does_not_clobber_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B2: a relaunch (e.g. after a failed external_session_id persist) re-enters
    # the fork path. If qwen has already built and since appended live, full-
    # fidelity turns, the rebuild must NOT overwrite them — it resumes as-is.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    qsid = qnb.qwen_session_id_for_conversation("conv_fork")
    # Simulate qwen's live recording already on disk (a richer transcript than a
    # text-only rebuild would produce).
    recording = qnb.qwen_session_recording_path(qsid, workspace)
    recording.parent.mkdir(parents=True, exist_ok=True)
    sentinel = '{"type":"assistant","message":{"role":"model","parts":[{"text":"LIVE"}]}}\n'
    recording.write_text(sentinel)

    returned = await _build_qwen_fork_recording(
        _ItemsClient(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "rebuilt"}],
                }
            ]
        ),  # type: ignore[arg-type]
        session_id="conv_fork",
        workspace=str(workspace),
    )

    # Returns the id to resume, and the live recording is untouched.
    assert returned == qsid
    assert recording.read_text() == sentinel


# ---------------------------------------------------------------------------
# Sub-agent tool-surface trim (``_trim_qwen_subagent_tool_surface``)
#
# A qwen sub-agent boots with qwen-code's whole 63-tool registry declared
# upfront; the runner materializes a workspace ``.qwen/settings.json`` at launch
# to cut that to the implementer's coding tools. Gated to sub-agent sessions:
# ``omnigent qwen`` (the interactive TUI, where the human is the orchestrator)
# must keep the full surface.
# ---------------------------------------------------------------------------


def _qwen_launch_config(workspace: Path, *, parent_session_id: str | None) -> Any:
    """Build the snapshot-derived launch config the qwen launch path reads."""
    return _PiNativeLaunchConfig(
        workspace=workspace,
        server_url="http://localhost:6767",
        terminal_launch_args=None,
        external_session_id=None,
        parent_session_id=parent_session_id,
    )


def _trim_settings(workspace: Path) -> dict:
    return json.loads((workspace / ".qwen" / "settings.json").read_text(encoding="utf-8"))


def test_subagent_launch_writes_the_trimmed_workspace_settings(tmp_path: Path) -> None:
    workspace = tmp_path / "worktree"
    workspace.mkdir()

    _trim_qwen_subagent_tool_surface(
        "conv_child",
        _qwen_launch_config(workspace, parent_session_id="conv_parent"),
        str(workspace),
    )

    settings = _trim_settings(workspace)
    assert settings["tools"]["toolSearch"]["threshold"] == 0
    assert settings["tools"]["computerUse"]["enabled"] is False
    assert settings["memory"]["enableManagedAutoMemory"] is False
    assert set(QWEN_SUBAGENT_DISABLED_TOOLS) <= set(settings["tools"]["disabled"])


def test_interactive_launch_leaves_the_workspace_untouched(tmp_path: Path) -> None:
    """No parent session → ``omnigent qwen``'s TUI, which keeps every tool."""
    workspace = tmp_path / "repo"
    workspace.mkdir()

    _trim_qwen_subagent_tool_surface(
        "conv_top_level",
        _qwen_launch_config(workspace, parent_session_id=None),
        str(workspace),
    )

    assert list(workspace.iterdir()) == []


def test_subagent_launch_merges_an_existing_workspace_settings_file(tmp_path: Path) -> None:
    """A worktree carrying the user's own ``.qwen/settings.json`` keeps it."""
    workspace = tmp_path / "worktree"
    (workspace / ".qwen").mkdir(parents=True)
    (workspace / ".qwen" / "settings.json").write_text(
        json.dumps({"ui": {"theme": "Default"}, "tools": {"approvalMode": "yolo"}}),
        encoding="utf-8",
    )

    _trim_qwen_subagent_tool_surface(
        "conv_child",
        _qwen_launch_config(workspace, parent_session_id="conv_parent"),
        str(workspace),
    )

    settings = _trim_settings(workspace)
    assert settings["ui"] == {"theme": "Default"}
    assert settings["tools"]["approvalMode"] == "yolo"
    assert settings["tools"]["toolSearch"]["threshold"] == 0


def test_subagent_launch_survives_an_unwritable_workspace(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Losing the trim must not fail the launch — qwen still starts, untrimmed."""
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    (workspace / ".qwen").write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="omnigent.runner.native.orchestration"):
        _trim_qwen_subagent_tool_surface(
            "conv_child",
            _qwen_launch_config(workspace, parent_session_id="conv_parent"),
            str(workspace),
        )

    assert "full tool registry" in caplog.text


async def test_launch_config_reads_parent_session_id_from_the_snapshot() -> None:
    """The sub-agent discriminator comes off ``GET /v1/sessions/{id}``."""

    class _Snapshot:
        def __init__(self, payload: dict) -> None:
            self.status_code = 200
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    class _Client:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        async def get(self, url: str, timeout: float | None = None) -> _Snapshot:
            del url, timeout
            return _Snapshot(self._payload)

    child = await _pi_native_launch_config(
        session_id="conv_child",
        server_client=_Client({"workspace": "/ws", "parent_session_id": "conv_parent"}),  # type: ignore[arg-type]
    )
    assert child.parent_session_id == "conv_parent"

    # Top-level sessions report it absent (or null); both read as "not a child".
    top = await _pi_native_launch_config(
        session_id="conv_top",
        server_client=_Client({"workspace": "/ws"}),  # type: ignore[arg-type]
    )
    assert top.parent_session_id is None
