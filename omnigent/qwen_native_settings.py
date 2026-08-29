"""Workspace ``.qwen/settings.json`` trim for headless qwen SUB-AGENT launches.

A qwen sub-agent (a ``qwen-native`` child session dispatched by an orchestrator
via ``sys_session_send``, launched with ``cwd`` = its git worktree) boots with
qwen-code's whole 63-tool built-in registry declared upfront. Measured on qwen
v0.18.2, turn 1 of such a sub-agent prefills ~35.5k tokens before it has done any
work; 25.8k of that (72%) is tool schemas, and ~80% of *those* are tools an
implementer cannot use — the 30 ``computer_use__*`` desktop-automation tools plus
qwen's own orchestration surface (``agent``, ``cron_*``, ``create_sub_session``,
``enter_worktree``, …). The Omnigent side is the orchestrator; a sub-agent that
spawns its own fan-out or re-enters a worktree is a bug, not a capability.

The whole surface is revealed because qwen's deferred-tool preload budget is
``tools.toolSearch.threshold`` percent of the context window — 10% of a 1M window
is 100k tokens, which the ~16k estimate for every deferred tool fits under, so
qwen declares them all instead of leaving them behind ``tool_search``.

qwen-code's settings cascade is user (``~/.qwen``) → workspace
(``<cwd>/.qwen/settings.json``), and only the workspace file is scoped to one
launch cwd. So the trim is materialized there, at sub-agent launch, by
:func:`write_subagent_workspace_settings`:

- ``tools.toolSearch.threshold: 0`` — never preload deferred tools; the model
  reaches the rest through ``tool_search`` if it genuinely needs them.
- ``tools.computerUse.enabled: false`` — drop the ``computer_use__*`` family
  outright (a headless worktree has no desktop to drive).
- ``memory.enableManagedAutoMemory: false`` — drop the ~3.3k-token ``# auto
  memory`` block, 40% of the system prompt, and the extra LLM pre-pass that
  builds it.
- ``tools.disabled`` — :data:`QWEN_SUBAGENT_DISABLED_TOOLS`, so orchestration
  tools are never *registered* (unlike ``permissions.deny``, which only blocks
  the call), and ``tool_search`` cannot surface them either.

Measured together: 8 declared tools and a ~11.1k-token floor, −69%.

The interactive ``omnigent qwen`` TUI keeps the full surface — a human at the
terminal is the orchestrator there — so this is never applied on that path; see
:func:`omnigent.runner.native.orchestration._auto_create_qwen_terminal`.

Unlike the per-session MCP config (which deliberately lives in the bridge dir so
Omnigent drops no file in the user's repo — see
:func:`omnigent.qwen_native_bridge.write_mcp_config`), this file has to be in the
workspace: qwen offers no CLI flag or env var for a settings path, and the
workspace scope is the only one that can differ per launch. It is written only
for sub-agent sessions, whose cwd is an orchestrator-owned worktree rather than
the user's own checkout, and it merges into an existing file rather than
replacing it.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from omnigent.json_types import JsonObject as _JsonObject

#: Workspace-scope settings file qwen-code reads, relative to the launch cwd.
QWEN_WORKSPACE_SETTINGS_PATH = Path(".qwen") / "settings.json"

#: Core coding tools a qwen implementer sub-agent actually calls. Documentary:
#: nothing is written from this list — the trim is subtractive (a
#: ``tools.disabled`` denylist plus the deferred-preload and computer-use
#: switches), so a tool qwen adds later stays available rather than silently
#: disappearing behind a stale allowlist. Kept here so the intent of
#: :data:`QWEN_SUBAGENT_DISABLED_TOOLS` is readable as "everything an
#: implementer needs survives".
QWEN_SUBAGENT_CORE_TOOLS: tuple[str, ...] = (
    "run_shell_command",
    "read_file",
    "edit",
    "write_file",
    "glob",
    "grep_search",
    "todo_write",
    "tool_search",
)

#: qwen-code tool names hidden from a sub-agent's registry entirely.
#:
#: Names are qwen-code's own (``ToolNames`` in its bundle, v0.18.2). Every one is
#: an *orchestration* capability that belongs to the Omnigent parent session, not
#: to the implementer it dispatched; qwen draws the same line for its in-process
#: subagents in ``EXCLUDED_TOOLS_FOR_SUBAGENTS``, which this mirrors (minus
#: ``todo_write``, which an implementer does use, plus the goal/loop/monitor/
#: sub-session tools qwen gates elsewhere). Listing a name qwen has not
#: registered in this session — several are behind their own feature switches —
#: is inert, so the list is deliberately a superset and does not have to track
#: which gates happen to be on.
QWEN_SUBAGENT_DISABLED_TOOLS: tuple[str, ...] = (
    # Fan-out: a sub-agent that spawns its own workers produces work the
    # orchestrator never collects.
    "agent",
    "create_sub_session",
    "workflow",
    # Teams / task board: the parent owns the plan and its task rows.
    "team_create",
    "team_delete",
    "team_plan_approval",
    "task_create",
    "task_update",
    "task_list",
    "task_stop",
    # Inter-agent messaging and roster: the parent addresses the sub-agent, not
    # the other way round.
    "list_agents",
    "send_message",
    # Scheduling / self-pacing: an implementer runs to completion in one turn.
    "cron_create",
    "cron_list",
    "cron_delete",
    "loop_wakeup",
    "monitor",
    # Worktree state belongs to the parent — a sub-agent must never enter or
    # leave the worktree it was launched into.
    "enter_worktree",
    "exit_worktree",
    # Session-scoped artifacts and goal state are the parent session's.
    "artifact",
    "record_artifact",
    "get_goal",
    "update_goal",
)


def subagent_settings_overlay() -> _JsonObject:
    """
    Build the qwen-code settings overlay applied to a sub-agent's workspace.

    :returns: The four-knob trim described in the module docstring, shaped as
        qwen-code's settings schema.
    """
    return {
        "tools": {
            # Percent of the context window budgeted for preloading deferred
            # tools; 0 keeps every one of them behind ``tool_search``.
            "toolSearch": {"threshold": 0},
            "computerUse": {"enabled": False},
            # Sorted, not in declaration order: the merge path unions with an
            # existing list and sorts, so writing sorted here keeps a first
            # launch and every relaunch producing byte-identical output.
            "disabled": sorted(QWEN_SUBAGENT_DISABLED_TOOLS),
        },
        "memory": {"enableManagedAutoMemory": False},
    }


def write_subagent_workspace_settings(workspace: Path | str) -> Path:
    """
    Materialize the sub-agent tool-surface trim into *workspace*.

    Merges :func:`subagent_settings_overlay` into any existing
    ``<workspace>/.qwen/settings.json`` rather than replacing it: unrelated keys
    (auth, model, MCP servers a user put there) survive untouched, and
    ``tools.disabled`` is unioned with whatever the file already denies. The
    trim's own four keys win — the point of writing the file is to pin them.

    An unreadable or non-object existing file is treated as absent and
    overwritten; qwen would reject it at startup anyway, so preserving it would
    only propagate the breakage.

    :param workspace: Launch cwd for the qwen sub-agent process. ``.qwen/`` is
        created when absent.
    :returns: Path to the written settings file.
    :raises OSError: If the settings file cannot be written.
    """
    path = Path(workspace) / QWEN_WORKSPACE_SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge_settings(_read_existing_settings(path), subagent_settings_overlay())
    # Atomic replace so a qwen process reading the file mid-write (a concurrent
    # launch in the same worktree) never sees a truncated document.
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _read_existing_settings(path: Path) -> _JsonObject:
    """Return the JSON object at *path*, or ``{}`` when absent/unusable."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _merge_settings(base: _JsonObject, overlay: _JsonObject) -> _JsonObject:
    """
    Deep-merge *overlay* onto *base*, unioning the ``tools.disabled`` list.

    :param base: Settings already on disk.
    :param overlay: The trim to apply; its scalars win on conflict.
    :returns: A new merged object; neither argument is mutated.
    """
    merged: _JsonObject = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _merge_settings(existing, value)
        elif key == "disabled" and isinstance(value, list) and isinstance(existing, list):
            # Union, not replace: a workspace that already hid tools keeps
            # hiding them. Sorted so repeated launches rewrite the same bytes.
            merged[key] = sorted({str(name) for name in [*existing, *value]})
        else:
            merged[key] = value
    return merged
