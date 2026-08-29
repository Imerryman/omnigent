"""Unit tests for qwen-native runner-side helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent import qwen_native_bridge as qnb
from omnigent.qwen_native_settings import (
    QWEN_SUBAGENT_DISABLED_TOOLS,
    QWEN_SYSTEM_SETTINGS_ENV_VAR,
)
from omnigent.runner.app import _build_qwen_fork_recording, _persist_qwen_external_session_id
from omnigent.runner.native.orchestration import (
    _pi_native_launch_config,
    _PiNativeLaunchConfig,
    _qwen_subagent_trim_env,
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
# Sub-agent tool-surface trim (``_qwen_subagent_trim_env``)
#
# A qwen sub-agent otherwise boots with qwen's full built-in registry declared
# upfront. The runner points it at an ephemeral per-session trim through
# ``QWEN_CODE_SYSTEM_SETTINGS_PATH`` — qwen's highest-precedence settings scope —
# and writes NOTHING into the launch cwd. Gated to sub-agent sessions:
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


def test_subagent_launch_points_qwen_at_an_ephemeral_trim(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bridge_dir = tmp_path / "bridge"

    env = _qwen_subagent_trim_env(
        "conv_child",
        _qwen_launch_config(workspace, parent_session_id="conv_parent"),
        bridge_dir,
    )

    settings_path = Path(env[QWEN_SYSTEM_SETTINGS_ENV_VAR])
    assert settings_path.parent == bridge_dir
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["tools"]["toolSearch"]["threshold"] == 0
    assert settings["tools"]["computerUse"]["enabled"] is False
    assert settings["memory"]["enableManagedAutoMemory"] is False
    assert set(QWEN_SUBAGENT_DISABLED_TOOLS) <= set(settings["tools"]["disabled"])


def test_subagent_launch_writes_nothing_into_the_workspace(tmp_path: Path) -> None:
    """The launch cwd is often the user's real checkout — never touch it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _qwen_subagent_trim_env(
        "conv_child",
        _qwen_launch_config(workspace, parent_session_id="conv_parent"),
        tmp_path / "bridge",
    )

    assert list(workspace.iterdir()) == []


def test_interactive_launch_gets_no_trim_env(tmp_path: Path) -> None:
    """No parent session -> ``omnigent qwen``'s TUI, which keeps every tool."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    env = _qwen_subagent_trim_env(
        "conv_top_level",
        _qwen_launch_config(workspace, parent_session_id=None),
        tmp_path / "bridge",
    )

    assert env == {}


def test_a_child_launch_does_not_trim_a_later_top_level_launch(tmp_path: Path) -> None:
    """The leak this design exists to prevent.

    Sub-agent sessions frequently carry no explicit workspace and fall back to
    ``OMNIGENT_RUNNER_WORKSPACE`` — the user's real checkout. A workspace
    ``.qwen/settings.json`` is persistent project config, so a child writing one
    there would trim every later interactive ``omnigent qwen`` in that same cwd.
    Here the child and the top-level session share a cwd, and the top-level
    launch must still come out untrimmed.
    """
    shared_cwd = tmp_path / "the-users-checkout"
    shared_cwd.mkdir()

    child_env = _qwen_subagent_trim_env(
        "conv_child",
        _qwen_launch_config(shared_cwd, parent_session_id="conv_parent"),
        tmp_path / "bridge_child",
    )
    assert child_env  # the child IS trimmed

    top_level_env = _qwen_subagent_trim_env(
        "conv_top_level",
        _qwen_launch_config(shared_cwd, parent_session_id=None),
        tmp_path / "bridge_top",
    )

    # Nothing was left in the shared cwd for the TUI to pick up, and its own
    # process carries no settings override.
    assert list(shared_cwd.iterdir()) == []
    assert top_level_env == {}
    assert QWEN_SYSTEM_SETTINGS_ENV_VAR not in top_level_env


def test_subagent_launch_survives_an_unwritable_bridge_dir(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Losing the trim must not fail the launch — qwen still starts, untrimmed."""
    occupied = tmp_path / "bridge"
    occupied.write_text("", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with caplog.at_level(logging.WARNING, logger="omnigent.runner.native.orchestration"):
        env = _qwen_subagent_trim_env(
            "conv_child",
            _qwen_launch_config(workspace, parent_session_id="conv_parent"),
            occupied,
        )

    assert env == {}
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
