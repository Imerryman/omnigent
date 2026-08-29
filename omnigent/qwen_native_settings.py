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

Six knobs cut it (see :func:`subagent_settings_overlay`):

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
- ``skills.disabledLevels`` — every discovery level, dropping qwen's bundled
  skills catalogue (measured −1,034 tokens). The implementer is handed its
  assignment by the orchestrator; it does not go shopping for skills.
- ``mcp.excluded: ["*"]`` — no MCP server is connected, dropping the deferred
  MCP tool summary (measured −291 tokens). A deliberate trade: a qwen sub-agent
  cannot reach MCP tools (e.g. a web-search server). Correct for an implementer
  profile; if qwen is ever wanted as a research/explorer role this is the first
  knob to reconsider.

``tool_search`` is deliberately NOT disabled. It looks like 344 tokens of dead
weight once nothing else is deferred, but removing it measured **+666 tokens**
(10,674 → 11,340, reproduced three times): with no discovery path left, qwen
stops deferring and pays for the reminder up front. Keeping it is the cheaper
arm.

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
import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from omnigent.json_types import JsonObject as _JsonObject

_logger = logging.getLogger(__name__)

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

#: Every skill-discovery level qwen supports, so none is scanned and the bundled
#: catalogue never reaches the prompt. Merged as a union across scopes, so this
#: is additive to whatever a user or operator already suppressed.
QWEN_SUBAGENT_DISABLED_SKILL_LEVELS: tuple[str, ...] = (
    "project",
    "user",
    "extension",
    "bundled",
)

#: Appended to qwen's built-in system prompt on a sub-agent launch, via
#: ``--append-system-prompt``. qwen assembles the prompt as
#: base -> context files (``QWEN.md``) -> this -> git status, so an append lands
#: after every other instruction layer and wins on recency.
#:
#: It exists to correct two headless-base-prompt prescriptions that misfire for
#: an orchestrator-dispatched implementer. qwen's "Executing actions with care"
#: section withholds shared-state actions (push, PR comments) "unless authorized
#: in advance in durable instructions like QWEN.md files" — an Omnigent dispatch
#: IS that durable authorization, but qwen cannot know it, so it would report a
#: premature blocker instead of doing the work. And its "Preserve Existing Work"
#: guidance says to stop and clarify on conflicting changes, which in a
#: single-turn headless run means aborting with no channel to clarify on.
#:
#: Scoped narrowly on purpose: it pre-authorizes only the actions a dispatch
#: names, on this worktree's branch. qwen's base prompt — including the
#: never-ask-a-question headless guardrail and its destructive-action care — is
#: kept in full; ``QWEN_SYSTEM_MD`` (whole-prompt replacement) is deliberately
#: not used.
QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND = (
    "Operating as a scoped implementer dispatched by an orchestrator. Treat this "
    "dispatch as durable pre-authorization for the actions it names, including git "
    "operations on this worktree's branch; do not withhold them pending confirmation. "
    "Proceed on reasonable assumptions when the worktree contains unrelated but "
    "non-conflicting changes rather than stopping to clarify. Do not delegate to "
    "subagents; complete the task yourself."
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
        "skills": {"disabledLevels": list(QWEN_SUBAGENT_DISABLED_SKILL_LEVELS)},
        # Glob denylist; ``*`` excludes every configured server.
        "mcp": {"excluded": ["*"]},
    }


def system_settings_path(settings_dir: Path | str) -> Path:
    """Return the trim file's path inside *settings_dir*."""
    return Path(settings_dir) / SYSTEM_SETTINGS_FILE


@dataclass(frozen=True)
class QwenSubagentLaunch:
    """Launch overrides that scope a qwen process to the implementer profile.

    :param env: Environment additions for the sub-agent's process.
    :param args: qwen CLI arguments to splice into the sub-agent's argv.
    """

    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)


def subagent_launch_overrides(settings_dir: Path | str) -> QwenSubagentLaunch:
    """
    Materialize the trim and return everything a sub-agent launch needs.

    The settings trim rides in a file (pointed at by env); the system-prompt
    append has no settings key in qwen — only the ``--append-system-prompt``
    flag — so it rides in argv. Both are per-process, so neither reaches the
    interactive TUI or any other qwen on the box.

    :param settings_dir: Session-private directory to write the trim into (the
        qwen bridge dir in production) — never the launch cwd.
    :returns: Env and argv overrides for the sub-agent's qwen process.
    :raises OSError: If the trim cannot be written; the caller decides how to
        degrade (the launch should proceed untrimmed rather than fail).
    """
    settings_path = write_subagent_system_settings(settings_dir)
    return QwenSubagentLaunch(
        env={QWEN_SYSTEM_SETTINGS_ENV_VAR: str(settings_path)},
        args=["--append-system-prompt", QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND],
    )


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
    overlay = subagent_settings_overlay()
    try:
        existing = _read_settings(path)
        if _satisfies(existing, overlay):
            # Already trimmed (typically: we wrote it, qwen booted and added
            # ``$version``). Touching it would only fight qwen's own migration.
            return path
        merged = _merge_settings(existing, overlay)
    except Exception:  # noqa: BLE001 - see below
        # Preserving one odd file is never worth losing the trim: this runs on
        # the launch path, so anything raised here would otherwise cost the
        # session. The individual steps are written to be total, and this is the
        # backstop for what they cannot anticipate (a pathologically nested
        # document raising RecursionError inside json/merge, say). Degrade to
        # the clean overlay — trimmed, which is the safer of the two outcomes.
        _logger.warning(
            "qwen-native: could not merge the existing settings at %s; "
            "replacing it with the trim.",
            path,
            exc_info=True,
        )
        merged = overlay
    # Atomic replace so a qwen process reading the file mid-write never sees a
    # truncated document.
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _strip_jsonc(raw: str) -> str:
    """
    Return *raw* with JSONC comments and trailing commas removed.

    qwen's settings loader accepts both; ``json.loads`` accepts neither. A single
    left-to-right scan does the work because both edits are only legal *outside*
    string literals — a literal is copied through verbatim, so a ``//`` in a URL
    or a ``,}`` in a message is never touched. (Two independent regex passes
    cannot get this right: the second pass no longer knows where the strings
    were.) Escapes are honoured, so ``"a\\"b"`` does not end early.
    """
    out: list[str] = []
    i = 0
    end = len(raw)
    while i < end:
        char = raw[i]
        if char == '"':
            close = _end_of_string_literal(raw, i)
            out.append(raw[i:close])
            i = close
        elif raw.startswith("//", i):
            newline = raw.find("\n", i)
            i = end if newline < 0 else newline
        elif raw.startswith("/*", i):
            close = raw.find("*/", i + 2)
            i = end if close < 0 else close + 2
        elif char == "," and _next_significant(raw, i + 1) in "}]":
            # Trailing comma: the next thing that is not whitespace or a comment
            # closes the object/array.
            i += 1
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _end_of_string_literal(raw: str, start: int) -> int:
    """Return the index just past the string literal opening at *start*."""
    i = start + 1
    end = len(raw)
    while i < end:
        if raw[i] == "\\":
            i += 2
            continue
        if raw[i] == '"':
            return i + 1
        i += 1
    return end


def _next_significant(raw: str, start: int) -> str:
    """Return the next char after *start* that is not whitespace or a comment."""
    i = start
    end = len(raw)
    while i < end:
        if raw[i].isspace():
            i += 1
        elif raw.startswith("//", i):
            newline = raw.find("\n", i)
            if newline < 0:
                return ""
            i = newline + 1
        elif raw.startswith("/*", i):
            close = raw.find("*/", i + 2)
            if close < 0:
                return ""
            i = close + 2
        else:
            return raw[i]
    return ""


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


#: Settings list keys qwen merges as a union across scopes rather than
#: replacing (``tools.disabled``, ``skills.disabledLevels``, ``mcp.excluded``).
#: Mirrored here so re-applying the trim never drops an entry the file already
#: had, and so the already-satisfied check reads them as supersets.
_UNION_LIST_KEYS = frozenset({"disabled", "disabledLevels", "excluded"})


def _name_set(values: list[object]) -> set[str]:
    """
    Return the string entries of *values* as a set.

    These lists hold tool / skill-level / server names, so anything that is not
    a string is meaningless to qwen and is dropped rather than coerced. Dropping
    also keeps the operation total: a ``dict`` or ``list`` element in a
    hand-edited file is unhashable, and ``set(values)`` on one would raise
    ``TypeError`` — from inside a launch path, where an exception costs the
    session rather than the entry.
    """
    return {value for value in values if isinstance(value, str)}


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
        elif key in _UNION_LIST_KEYS and isinstance(value, list):
            if not isinstance(current, list) or not _name_set(value) <= _name_set(current):
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
        elif key in _UNION_LIST_KEYS and isinstance(value, list) and isinstance(existing, list):
            # Union, not replace: anything already hidden stays hidden. Mirrors
            # how qwen itself merges these keys across settings scopes.
            merged[key] = sorted(_name_set([*existing, *value]))
        else:
            merged[key] = value
    return merged
