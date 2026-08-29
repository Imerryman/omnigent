"""Tests for the qwen sub-agent workspace tool-surface trim.

Covers :mod:`omnigent.qwen_native_settings` — the four settings knobs that take a
qwen implementer sub-agent from qwen-code's whole 63-tool registry (~35.5k
prefill tokens) down to its coding tools (~11.1k), and the merge semantics that
keep a workspace's pre-existing ``.qwen/settings.json`` intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnigent.qwen_native_settings import (
    QWEN_SUBAGENT_CORE_TOOLS,
    QWEN_SUBAGENT_DISABLED_TOOLS,
    QWEN_WORKSPACE_SETTINGS_PATH,
    subagent_settings_overlay,
    write_subagent_workspace_settings,
)


def _written(workspace: Path) -> dict:
    return json.loads((workspace / QWEN_WORKSPACE_SETTINGS_PATH).read_text(encoding="utf-8"))


def test_writes_the_four_trim_keys(tmp_path: Path) -> None:
    """The written file pins every knob the token measurement depends on."""
    path = write_subagent_workspace_settings(tmp_path)

    assert path == tmp_path / ".qwen" / "settings.json"
    settings = _written(tmp_path)
    # threshold 0 = never preload deferred tools (10% of a 1M window otherwise
    # fits the whole registry, so qwen declares all of it upfront).
    assert settings["tools"]["toolSearch"]["threshold"] == 0
    assert settings["tools"]["computerUse"]["enabled"] is False
    assert settings["memory"]["enableManagedAutoMemory"] is False
    assert settings["tools"]["disabled"] == sorted(QWEN_SUBAGENT_DISABLED_TOOLS)


def test_creates_the_settings_dir_when_absent(tmp_path: Path) -> None:
    """A fresh worktree has no ``.qwen/``; the writer makes one."""
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    assert not (workspace / ".qwen").exists()

    write_subagent_workspace_settings(workspace)

    assert (workspace / ".qwen" / "settings.json").is_file()


def test_accepts_a_string_workspace(tmp_path: Path) -> None:
    """The runner hands over a realpath'd ``str``, not a ``Path``."""
    write_subagent_workspace_settings(str(tmp_path))

    assert _written(tmp_path)["tools"]["toolSearch"]["threshold"] == 0


def test_disables_orchestration_tools_only(tmp_path: Path) -> None:
    """The implementer keeps its coding tools; the parent keeps orchestration."""
    write_subagent_workspace_settings(tmp_path)
    disabled = set(_written(tmp_path)["tools"]["disabled"])

    # Spawning, worktree state, scheduling and goal/artifact state belong to the
    # Omnigent parent session that dispatched this sub-agent.
    assert {
        "agent",
        "create_sub_session",
        "workflow",
        "send_message",
        "monitor",
        "loop_wakeup",
        "cron_create",
        "cron_list",
        "cron_delete",
        "enter_worktree",
        "exit_worktree",
        "record_artifact",
        "get_goal",
        "update_goal",
    } <= disabled
    # ... and nothing an implementer actually calls is taken away.
    assert disabled.isdisjoint(QWEN_SUBAGENT_CORE_TOOLS)


def test_merges_into_an_existing_settings_file(tmp_path: Path) -> None:
    """Unrelated user settings survive; only the trim's own keys are pinned."""
    settings_path = tmp_path / ".qwen" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "model": {"name": "qwen3-coder"},
                "mcpServers": {"searxng": {"command": "npx"}},
                "tools": {"approvalMode": "yolo", "toolSearch": {"enabled": True}},
                "memory": {"importFormat": "flat"},
            }
        ),
        encoding="utf-8",
    )

    write_subagent_workspace_settings(tmp_path)

    settings = _written(tmp_path)
    assert settings["model"] == {"name": "qwen3-coder"}
    assert settings["mcpServers"] == {"searxng": {"command": "npx"}}
    # Sibling keys inside the branches we touch are kept, not clobbered.
    assert settings["tools"]["approvalMode"] == "yolo"
    assert settings["tools"]["toolSearch"]["enabled"] is True
    assert settings["memory"]["importFormat"] == "flat"
    # The trim still wins where it overlaps.
    assert settings["tools"]["toolSearch"]["threshold"] == 0
    assert settings["memory"]["enableManagedAutoMemory"] is False


def test_unions_a_pre_existing_disabled_list(tmp_path: Path) -> None:
    """A workspace that already hid a tool keeps hiding it."""
    settings_path = tmp_path / ".qwen" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"tools": {"disabled": ["enter_plan_mode", "agent"]}}), encoding="utf-8"
    )

    write_subagent_workspace_settings(tmp_path)

    disabled = _written(tmp_path)["tools"]["disabled"]
    assert "enter_plan_mode" in disabled
    assert set(QWEN_SUBAGENT_DISABLED_TOOLS) <= set(disabled)
    # Union, not concatenation: the overlap appears once.
    assert disabled.count("agent") == 1
    assert disabled == sorted(disabled)


def test_overwrites_an_unparseable_existing_file(tmp_path: Path) -> None:
    """qwen would reject the broken file anyway; preserving it helps nobody."""
    settings_path = tmp_path / ".qwen" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{ not json", encoding="utf-8")

    write_subagent_workspace_settings(tmp_path)

    assert _written(tmp_path)["tools"]["computerUse"]["enabled"] is False


def test_is_idempotent(tmp_path: Path) -> None:
    """Relaunching in the same worktree rewrites identical bytes."""
    first = write_subagent_workspace_settings(tmp_path).read_text(encoding="utf-8")
    second = write_subagent_workspace_settings(tmp_path).read_text(encoding="utf-8")

    assert first == second


def test_leaves_no_temp_files_behind(tmp_path: Path) -> None:
    """The atomic write replaces in place rather than littering ``.qwen/``."""
    write_subagent_workspace_settings(tmp_path)

    assert [p.name for p in (tmp_path / ".qwen").iterdir()] == ["settings.json"]


def test_overlay_is_a_fresh_object() -> None:
    """Callers mutating the overlay must not corrupt the module constants."""
    overlay = subagent_settings_overlay()
    overlay["tools"]["disabled"].append("read_file")

    assert "read_file" not in subagent_settings_overlay()["tools"]["disabled"]
    assert "read_file" not in QWEN_SUBAGENT_DISABLED_TOOLS


def test_raises_when_the_settings_dir_cannot_be_made(tmp_path: Path) -> None:
    """The caller decides how to degrade; the writer reports the failure."""
    # ``.qwen`` occupied by a regular file — mkdir cannot make a dir there.
    # Chosen over a chmod'd workspace because root ignores the mode bits.
    (tmp_path / ".qwen").write_text("", encoding="utf-8")

    with pytest.raises(OSError):
        write_subagent_workspace_settings(tmp_path)
