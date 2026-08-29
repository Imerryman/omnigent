"""Tool-surface trim for headless qwen SUB-AGENT launches.

A qwen sub-agent (a ``qwen-native`` child session dispatched by an Omnigent
orchestrator and launched headless) boots with qwen-code's whole built-in tool
registry declared upfront. Measured on qwen v0.22.0: 67 declared tools and a
47,697-token turn-1 prefill before it has done any work. ~80% of that surface is
unusable by an implementer — the 35 ``computer_use__*`` desktop-automation tools
plus qwen's own orchestration tools, which belong to the Omnigent parent session
(a sub-agent that spawns its own fan-out or re-enters a worktree is a bug, not a
capability).

The whole surface is revealed because qwen's deferred-tool preload budget is
``tools.toolSearch.threshold`` percent of the context window — 10% of a 1M window
is 100k tokens, which the entire registry fits under, so nothing stays behind
``tool_search``.

Four knobs cut it (see :func:`subagent_settings_overlay`):

- ``tools.toolSearch.threshold: 0`` — never preload deferred tools; the model
  reaches the rest through ``tool_search`` if it genuinely needs them.
- ``tools.computerUse.enabled: false`` — drop the ``computer_use__*`` family
  outright (a headless launch has no desktop to drive).
- ``memory.enableManagedAutoMemory: false`` — drop the ~3.3k-token ``# auto
  memory`` block, 40% of the system prompt, and the extra LLM pre-pass that
  builds it.
- ``tools.disabled`` — :data:`QWEN_SUBAGENT_DISABLED_TOOLS`, so orchestration
  tools are never *registered* (unlike ``permissions.deny``, which only blocks
  the call), and ``tool_search`` cannot surface them either.

Delivery: an EPHEMERAL per-session file, never the workspace
---------------------------------------------------------

qwen resolves settings from four scopes, merged by its ``mergeSettings`` in the
order system-defaults → user (``~/.qwen``) → workspace (``<cwd>/.qwen``) →
**system**, so the system scope has the highest precedence. Its path is
per-process overridable via the ``QWEN_CODE_SYSTEM_SETTINGS_PATH`` environment
variable (qwen's ``getSystemSettingsPath``; documented in qwen's
``docs/configuration/settings.md``).

So the trim is written to a file in the session's own bridge dir and handed to
the sub-agent's process through that env var. Nothing is written into the launch
cwd. This matters because a workspace ``.qwen/settings.json`` would be
**persistent project config, not session state**: a sub-agent that carries no
explicit workspace falls back to ``OMNIGENT_RUNNER_WORKSPACE`` — the user's real
checkout — so a workspace-file trim would silently apply to every later
interactive ``omnigent qwen`` (and every bare ``qwen``) run in that directory.
The env-var route is session-scoped by construction: the variable is set on the
sub-agent's terminal process only, so a top-level launch in the very same cwd is
untouched.

It is also stronger. Because the system scope wins the merge, a workspace file
that a repo already carries cannot re-enable what the trim disables — verified by
launching against a workspace ``.qwen/settings.json`` asking for
``toolSearch.threshold: 90``, ``computerUse.enabled: true`` and
``enableManagedAutoMemory: true``: the trim held (17 declared tools, 12,020
prefill tokens, no ``computer_use__*``, no orchestration tools).

The same reasoning already put qwen's per-session MCP config in the bridge dir
rather than the workspace; see :func:`omnigent.qwen_native_bridge.write_mcp_config`.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path

from omnigent.json_types import JsonObject as _JsonObject

#: qwen-code env var naming the highest-precedence ("system") settings file.
#: Set on the sub-agent's terminal process only.
QWEN_SYSTEM_SETTINGS_ENV_VAR = "QWEN_CODE_SYSTEM_SETTINGS_PATH"

#: Filename for the trim inside the caller-supplied settings dir (the session's
#: bridge dir in production). Distinct from qwen's own ``settings.json`` names so
#: a stray copy is recognisable.
SYSTEM_SETTINGS_FILE = "qwen_system_settings.json"

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
#: Names are qwen-code's own (``ToolNames`` in its bundle). Every one is an
#: *orchestration* capability that belongs to the Omnigent parent session, not to
#: the implementer it dispatched; qwen draws the same line for its in-process
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
    Build the qwen-code settings the sub-agent's system scope declares.

    :returns: The four-knob trim described in the module docstring, shaped as
        qwen-code's settings schema.
    """
    return {
        "tools": {
            # Percent of the context window budgeted for preloading deferred
            # tools; 0 keeps every one of them behind ``tool_search``.
            "toolSearch": {"threshold": 0},
            "computerUse": {"enabled": False},
            "disabled": sorted(QWEN_SUBAGENT_DISABLED_TOOLS),
        },
        "memory": {"enableManagedAutoMemory": False},
    }


def system_settings_path(settings_dir: Path | str) -> Path:
    """Return the trim file's path inside *settings_dir*."""
    return Path(settings_dir) / SYSTEM_SETTINGS_FILE


def subagent_launch_env(settings_dir: Path | str) -> dict[str, str]:
    """
    Materialize the trim and return the env that points qwen at it.

    :param settings_dir: Session-private directory to write the trim into (the
        qwen bridge dir in production) — never the launch cwd.
    :returns: ``{QWEN_CODE_SYSTEM_SETTINGS_PATH: <path>}`` to merge into the
        sub-agent terminal's environment.
    :raises OSError: If the trim cannot be written; the caller decides how to
        degrade (the launch should proceed untrimmed rather than fail).
    """
    return {QWEN_SYSTEM_SETTINGS_ENV_VAR: str(write_subagent_system_settings(settings_dir))}


def write_subagent_system_settings(settings_dir: Path | str) -> Path:
    """
    Write the trim into *settings_dir*, preserving whatever qwen put there.

    The file is ours and session-private, but qwen *writes back* to it: loading a
    versionless settings file stamps ``"$version": 4`` and persists it, so the
    file on disk after a boot is not the file we wrote. Rewriting it wholesale
    each launch would strip that stamp and make qwen re-add it — churn for no
    reason — so this merges the trim onto the current contents and, when those
    already satisfy the trim, leaves the file completely untouched.

    Existing content is parsed leniently (qwen accepts JSONC) and a file that
    cannot be parsed at all is left in place rather than clobbered: the trim is
    re-applied over ``{}`` only in memory, and the write below replaces the file
    with a valid document. Nothing outside *settings_dir* is ever touched.

    :param settings_dir: Session-private directory; created when absent.
    :returns: Path to the trim file.
    :raises OSError: If the directory or file cannot be written.
    """
    path = system_settings_path(settings_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_settings(path)
    overlay = subagent_settings_overlay()
    if _satisfies(existing, overlay):
        # Already trimmed (typically: we wrote it, qwen booted and added
        # ``$version``). Touching it would only fight qwen's own migration.
        return path
    merged = _merge_settings(existing, overlay)
    # Atomic replace so a qwen process reading the file mid-write never sees a
    # truncated document.
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


#: ``//`` and ``/* */`` comments outside string literals, plus trailing commas —
#: the JSONC affordances qwen's own settings loader accepts and ``json.loads``
#: rejects. The leading alternation consumes whole string literals so a ``//``
#: inside one (a URL, a Windows path) is never mistaken for a comment.
_JSONC_NOISE = re.compile(
    r'"(?:\\.|[^"\\])*"'  # string literal — matched, then re-emitted verbatim
    r"|//[^\n]*"  # line comment
    r"|/\*.*?\*/",  # block comment
    re.DOTALL,
)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _strip_jsonc(raw: str) -> str:
    """Return *raw* with JSONC comments and trailing commas removed."""
    without_comments = _JSONC_NOISE.sub(lambda m: m.group(0) if m.group(0)[0] == '"' else "", raw)
    return _TRAILING_COMMA.sub(r"\1", without_comments)


def _read_settings(path: Path) -> _JsonObject:
    """
    Return the settings object at *path*, or ``{}`` when it cannot be read.

    Never raises. A missing file, an unreadable one, invalid UTF-8, or content
    that is not a JSON object all read as ``{}`` — the caller then writes a
    valid document. Comments and trailing commas are tolerated so a legitimately
    JSONC file is merged rather than discarded.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    for candidate in (raw, _strip_jsonc(raw)):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _satisfies(existing: _JsonObject, overlay: _JsonObject) -> bool:
    """
    Report whether *existing* already declares everything in *overlay*.

    Extra keys are ignored, so qwen's own additions (``$version``) do not make a
    trimmed file look stale. ``disabled`` compares as a superset because the
    merge unions rather than replaces.
    """
    for key, value in overlay.items():
        current = existing.get(key)
        if isinstance(value, dict):
            if not isinstance(current, dict) or not _satisfies(current, value):
                return False
        elif key == "disabled" and isinstance(value, list):
            if not isinstance(current, list) or not set(value) <= set(current):
                return False
        elif current != value or type(current) is not type(value):
            return False
    return True


def _merge_settings(base: _JsonObject, overlay: _JsonObject) -> _JsonObject:
    """
    Deep-merge *overlay* onto *base*, unioning the ``tools.disabled`` list.

    :param base: Settings already on disk (including anything qwen added).
    :param overlay: The trim to apply; its scalars win on conflict.
    :returns: A new merged object; neither argument is mutated.
    """
    merged: _JsonObject = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _merge_settings(existing, value)
        elif key == "disabled" and isinstance(value, list) and isinstance(existing, list):
            # Union, not replace: anything already hidden stays hidden.
            merged[key] = sorted({str(name) for name in [*existing, *value]})
        else:
            merged[key] = value
    return merged
