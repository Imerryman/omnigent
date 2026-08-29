"""Tests for the qwen sub-agent tool-surface trim.

Covers :mod:`omnigent.qwen_native_settings` — the four settings knobs that take a
qwen implementer sub-agent from qwen-code's whole built-in registry (67 declared
tools, ~47.7k prefill tokens) down to its coding tools (~12.0k), delivered as an
ephemeral per-session file via ``QWEN_CODE_SYSTEM_SETTINGS_PATH`` rather than a
workspace ``.qwen/settings.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnigent.qwen_native_settings import (
    QWEN_SUBAGENT_CORE_TOOLS,
    QWEN_SUBAGENT_DISABLED_SKILL_LEVELS,
    QWEN_SUBAGENT_DISABLED_TOOLS,
    QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND,
    QWEN_SYSTEM_SETTINGS_ENV_VAR,
    subagent_launch_overrides,
    subagent_settings_overlay,
    system_settings_path,
    write_subagent_system_settings,
)


def _written(settings_dir: Path) -> dict:
    return json.loads(system_settings_path(settings_dir).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# What the trim declares
# ---------------------------------------------------------------------------


def test_writes_the_four_trim_keys(tmp_path: Path) -> None:
    """The written file pins every knob the token measurement depends on."""
    path = write_subagent_system_settings(tmp_path)

    assert path == tmp_path / "qwen_system_settings.json"
    settings = _written(tmp_path)
    # threshold 0 = never preload deferred tools (10% of a 1M window otherwise
    # fits the whole registry, so qwen declares all of it upfront).
    assert settings["tools"]["toolSearch"]["threshold"] == 0
    assert settings["tools"]["computerUse"]["enabled"] is False
    assert settings["memory"]["enableManagedAutoMemory"] is False
    assert settings["tools"]["disabled"] == sorted(QWEN_SUBAGENT_DISABLED_TOOLS)
    # No skill level is scanned, so the bundled catalogue never reaches the
    # prompt (-1,034 tokens measured on qwen v0.22.0).
    assert settings["skills"]["disabledLevels"] == list(QWEN_SUBAGENT_DISABLED_SKILL_LEVELS)
    assert set(settings["skills"]["disabledLevels"]) == {
        "project",
        "user",
        "extension",
        "bundled",
    }
    # No MCP server connects, dropping the deferred MCP tool summary (-291).
    assert settings["mcp"]["excluded"] == ["*"]


def test_disables_orchestration_tools_only(tmp_path: Path) -> None:
    """The implementer keeps its coding tools; the parent keeps orchestration."""
    write_subagent_system_settings(tmp_path)
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


def test_overlay_is_a_fresh_object() -> None:
    """Callers mutating the overlay must not corrupt the module constants."""
    overlay = subagent_settings_overlay()
    overlay["tools"]["disabled"].append("read_file")

    assert "read_file" not in subagent_settings_overlay()["tools"]["disabled"]
    assert "read_file" not in QWEN_SUBAGENT_DISABLED_TOOLS


# ---------------------------------------------------------------------------
# How it is delivered: ephemeral file + env var, never the workspace
# ---------------------------------------------------------------------------


def test_launch_overrides_point_qwen_at_the_written_file(tmp_path: Path) -> None:
    overrides = subagent_launch_overrides(tmp_path)

    assert overrides.env == {QWEN_SYSTEM_SETTINGS_ENV_VAR: str(system_settings_path(tmp_path))}
    assert _written(tmp_path)["tools"]["toolSearch"]["threshold"] == 0


def test_launch_overrides_append_the_implementer_prompt(tmp_path: Path) -> None:
    """qwen exposes no settings key for this — only the CLI flag."""
    overrides = subagent_launch_overrides(tmp_path)

    assert overrides.args == ["--append-system-prompt", QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND]
    # The two base-prompt prescriptions this exists to correct.
    assert "durable pre-authorization" in QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND
    assert "rather than stopping to clarify" in QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND
    # It must not try to replace qwen's base prompt.
    assert "--system-prompt" not in overrides.args


def test_tool_search_is_kept(tmp_path: Path) -> None:
    """Disabling it looks free and is not.

    Measured on qwen v0.22.0 with skills + MCP already excluded: removing
    ``tool_search`` moved the prefill from 10,674 to 11,340 tokens (+666,
    reproduced three times). With no discovery path left qwen stops deferring
    and pays for the reminder upfront, so keeping it is the cheaper arm.
    """
    write_subagent_system_settings(tmp_path)

    assert "tool_search" not in _written(tmp_path)["tools"]["disabled"]
    assert "tool_search" in QWEN_SUBAGENT_CORE_TOOLS


def test_creates_the_settings_dir_when_absent(tmp_path: Path) -> None:
    """The bridge dir may not exist yet on a first launch."""
    settings_dir = tmp_path / "bridge" / "nested"

    subagent_launch_overrides(settings_dir)

    assert system_settings_path(settings_dir).is_file()


def test_accepts_a_string_settings_dir(tmp_path: Path) -> None:
    """The runner may hand over a ``str`` rather than a ``Path``."""
    subagent_launch_overrides(str(tmp_path))

    assert _written(tmp_path)["tools"]["computerUse"]["enabled"] is False


def test_writes_nothing_outside_the_settings_dir(tmp_path: Path) -> None:
    """The launch cwd must never gain a ``.qwen/`` — that is project config."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings_dir = tmp_path / "bridge"

    subagent_launch_overrides(settings_dir)

    assert list(workspace.iterdir()) == []
    assert [p.name for p in settings_dir.iterdir()] == ["qwen_system_settings.json"]


def test_raises_when_the_settings_dir_cannot_be_made(tmp_path: Path) -> None:
    """The caller decides how to degrade; the writer reports the failure."""
    # The dir path is occupied by a regular file — mkdir cannot make a dir there.
    # Chosen over a chmod'd parent because root ignores the mode bits.
    occupied = tmp_path / "bridge"
    occupied.write_text("", encoding="utf-8")

    with pytest.raises(OSError):
        subagent_launch_overrides(occupied)


# ---------------------------------------------------------------------------
# Living alongside qwen's own writes to the same file
# ---------------------------------------------------------------------------


def test_leaves_a_file_qwen_stamped_untouched(tmp_path: Path) -> None:
    """The real lifecycle: write -> qwen boots and adds ``$version`` -> relaunch.

    qwen persists ``"$version": 4`` into a versionless settings file when it
    loads one, so the file after a boot is not the file we wrote. A relaunch must
    not fight that: the trim is already satisfied, so the file is left exactly as
    qwen left it (same bytes, same mtime).
    """
    path = write_subagent_system_settings(tmp_path)
    booted = json.loads(path.read_text(encoding="utf-8"))
    booted["$version"] = 4  # what qwen writes back on boot
    path.write_text(json.dumps(booted, indent=2), encoding="utf-8")
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    write_subagent_system_settings(tmp_path)

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_mtime
    assert json.loads(path.read_text(encoding="utf-8"))["$version"] == 4


def test_rewrites_when_the_trim_is_no_longer_satisfied(tmp_path: Path) -> None:
    """A file that drifted (or was never ours) is brought back to the trim."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"$version": 4, "tools": {"toolSearch": {"threshold": 10}}}),
        encoding="utf-8",
    )

    write_subagent_system_settings(tmp_path)

    settings = _written(tmp_path)
    assert settings["tools"]["toolSearch"]["threshold"] == 0
    # Whatever qwen added is carried forward, not stripped.
    assert settings["$version"] == 4


def test_a_true_boolean_does_not_count_as_zero(tmp_path: Path) -> None:
    """``False == 0`` in Python; the satisfied-check must not conflate them."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tools": {"toolSearch": {"threshold": False}}}), encoding="utf-8")

    write_subagent_system_settings(tmp_path)

    assert _written(tmp_path)["tools"]["toolSearch"]["threshold"] == 0
    assert _written(tmp_path)["tools"]["toolSearch"]["threshold"] is not False


# ---------------------------------------------------------------------------
# Reading an existing file: lenient, never destructive, never fatal
# ---------------------------------------------------------------------------


def test_preserves_a_jsonc_file_with_comments(tmp_path: Path) -> None:
    """qwen accepts JSONC; ``json.loads`` does not. Merge it, do not discard it."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """{
  // the operator pinned a model here
  "model": { "name": "qwen3-coder" },
  /* and a server */
  "mcpServers": { "searxng": { "command": "npx" } },
  "tools": { "disabled": ["enter_plan_mode"], },
}""",
        encoding="utf-8",
    )

    write_subagent_system_settings(tmp_path)

    settings = _written(tmp_path)
    assert settings["model"] == {"name": "qwen3-coder"}
    assert settings["mcpServers"] == {"searxng": {"command": "npx"}}
    # Union, not replace: what was already hidden stays hidden.
    assert "enter_plan_mode" in settings["tools"]["disabled"]
    assert set(QWEN_SUBAGENT_DISABLED_TOOLS) <= set(settings["tools"]["disabled"])
    assert settings["tools"]["toolSearch"]["threshold"] == 0


def test_unions_the_new_list_keys_too(tmp_path: Path) -> None:
    """``skills.disabledLevels`` and ``mcp.excluded`` merge as unions, like qwen's."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "skills": {"disabledLevels": ["some-future-level"]},
                "mcp": {"excluded": ["a-named-server"]},
            }
        ),
        encoding="utf-8",
    )

    write_subagent_system_settings(tmp_path)

    settings = _written(tmp_path)
    assert "some-future-level" in settings["skills"]["disabledLevels"]
    assert set(QWEN_SUBAGENT_DISABLED_SKILL_LEVELS) <= set(settings["skills"]["disabledLevels"])
    assert "a-named-server" in settings["mcp"]["excluded"]
    assert "*" in settings["mcp"]["excluded"]


def test_a_double_slash_inside_a_string_is_not_a_comment(tmp_path: Path) -> None:
    """The JSONC strip must skip string literals, or URLs lose their tail."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"env": {"OPENAI_BASE_URL": "http://192.168.2.7:18010/v1"}}),
        encoding="utf-8",
    )

    write_subagent_system_settings(tmp_path)

    assert _written(tmp_path)["env"]["OPENAI_BASE_URL"] == "http://192.168.2.7:18010/v1"


def test_invalid_utf8_does_not_abort_the_launch(tmp_path: Path) -> None:
    """``read_text`` raises UnicodeDecodeError, which is not an OSError."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"tools": "\xff\xfe not utf-8"}')

    # Must not raise; the trim is re-established over an empty base.
    write_subagent_system_settings(tmp_path)

    assert _written(tmp_path)["tools"]["toolSearch"]["threshold"] == 0


def test_unparseable_content_does_not_abort_the_launch(tmp_path: Path) -> None:
    """Garbage in our own session-private file must not fail the session."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json at all", encoding="utf-8")

    write_subagent_system_settings(tmp_path)

    assert _written(tmp_path)["tools"]["computerUse"]["enabled"] is False


def test_a_json_array_reads_as_empty_rather_than_crashing(tmp_path: Path) -> None:
    """Valid JSON that is not an object still has to merge cleanly."""
    path = system_settings_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")

    write_subagent_system_settings(tmp_path)

    assert _written(tmp_path)["memory"]["enableManagedAutoMemory"] is False


def test_leaves_no_temp_files_behind(tmp_path: Path) -> None:
    """The atomic write replaces in place rather than littering the dir."""
    subagent_launch_overrides(tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == ["qwen_system_settings.json"]
