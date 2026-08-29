"""Conformance probe: the qwen-code behaviours our sub-agent safety rests on.

The qwen sub-agent trim (:mod:`omnigent.qwen_native_settings`) is safe to run
headless only because of two things qwen-code does today:

1. its headless base prompt tells the model **"Never ask the user a question"** —
   a single-turn run has no channel to answer on, so a question is a hung turn;
2. neither ``agent`` (fan-out the orchestrator would never collect) nor
   ``ask_user_question`` is in the sub-agent's tool set — the first because the
   trim disables it, the second because qwen only registers it when the
   interaction mode supports user interaction.

Both are qwen-code implementation details that a version bump could change
silently, and no unit test over our own code would notice. So this probe runs
the **installed qwen binary** headlessly, with the exact settings file the
runner materializes, and asserts both properties against reality.

Behaviour with and without a live qwen
--------------------------------------

- **no qwen binary**: every test skips. That is the ONLY skip. The main suite
  must not depend on a vendor CLI being installed.
- **qwen on PATH** (or ``OMNIGENT_QWEN_PATH``): the probe runs and **fails
  loudly** if the guardrail text is gone, if either tool reappears, or if qwen
  does not produce both startup artifacts. A missing artifact is not an excuse
  to skip — a qwen that stopped rendering its prompt or stopped announcing its
  tools IS the regression this file exists to catch, and skipping there would
  let a real one through as a green run.
- **no model server**: still runs. The probe deliberately points the OpenAI-
  compatible base URL at a closed port — qwen writes the system prompt
  (``QWEN_WRITE_SYSTEM_MD``) and emits its ``init`` event with the resolved tool
  list *before* any completion request, so no LLM backend, credentials, or
  network access is needed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator

import pytest

from omnigent.qwen_native_settings import (
    QWEN_SYSTEM_SETTINGS_ENV_VAR,
    subagent_launch_overrides,
)

#: Guardrail sentence qwen's headless interaction mode injects. Matched
#: case-insensitively on the load-bearing clause only, so wording around it can
#: drift without a false alarm.
_NEVER_ASK = "never ask the user a question"

#: Tools that must never reach a qwen sub-agent.
_FORBIDDEN_TOOLS = ("agent", "ask_user_question")

_PROBE_TIMEOUT_S = 120


def _qwen_binary() -> str | None:
    """Resolve the qwen CLI the runner would launch, or ``None``."""
    configured = os.environ.get("OMNIGENT_QWEN_PATH", "").strip()
    return shutil.which(configured or "qwen")


# ---------------------------------------------------------------------------
# Stream parsing. Pure, so these run with or without a qwen binary — and they
# have to: a parser that quietly returns ``None`` would now turn every probe
# into a hard failure, so its own robustness cannot rest on the probe passing.
# ---------------------------------------------------------------------------

_INIT_EVENT = '[{"type":"system","subtype":"init","tools":["read_file","edit"]}]'


def test_tools_are_found_after_a_bracketed_log_line() -> None:
    """A ``[INFO]``-style prefix must not capture the parse."""
    stdout = f"[INFO] starting up\n[warn] not json either\n{_INIT_EVENT}\n"

    assert _tools_from_stream(stdout) == ["read_file", "edit"]


def test_tools_are_found_in_line_delimited_output() -> None:
    """stream-json emits one event per line rather than a single array."""
    stdout = (
        "Warning: something\n"
        '{"type":"system","subtype":"other"}\n'
        '{"type":"system","subtype":"init","tools":["glob"]}\n'
    )

    assert _tools_from_stream(stdout) == ["glob"]


def test_tools_are_found_after_a_decodable_but_wrong_payload() -> None:
    """The first parseable JSON need not be the one carrying the init event."""
    stdout = f'[1, 2, 3]\n{{"unrelated": true}}\n{_INIT_EVENT}'

    assert _tools_from_stream(stdout) == ["read_file", "edit"]


def test_no_init_event_reads_as_none() -> None:
    """``None`` is the signal the fixture turns into a loud failure."""
    assert _tools_from_stream("[INFO] nothing here\n[1,2,3]\n") is None
    assert _tools_from_stream("") is None


@pytest.fixture(scope="module")
def qwen_probe(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the installed qwen once under the real trim; skip if unavailable."""
    binary = _qwen_binary()
    if binary is None:
        pytest.skip("qwen CLI not installed; conformance probe needs the real binary")

    root = tmp_path_factory.mktemp("qwen-conformance")
    workspace = root / "workspace"
    workspace.mkdir()
    overrides = subagent_launch_overrides(root / "bridge")
    system_md = root / "base_prompt.md"

    env = {
        **os.environ,
        **overrides.env,
        "QWEN_WRITE_SYSTEM_MD": str(system_md),
        # Closed port: the probe reads startup artifacts only, and must not
        # reach (or depend on) whatever backend this machine is configured for.
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "OPENAI_API_KEY": "conformance-probe-no-auth",
        "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
    }
    try:
        completed = subprocess.run(
            [binary, *overrides.args, "-p", "probe", "-o", "json"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.fail(f"qwen is installed at {binary} but could not be run: {exc}", pytrace=False)

    tools = _tools_from_stream(completed.stdout)
    prompt = system_md.read_text(encoding="utf-8") if system_md.is_file() else ""
    missing = [
        name
        for name, present in (
            ("system-prompt dump", bool(prompt)),
            ("init event", tools is not None),
        )
        if not present
    ]
    if missing:
        # Deliberately a failure, not a skip: both artifacts are produced during
        # startup, before any model call, so their absence means qwen changed —
        # exactly what this file watches for. Skipping here would retire the
        # probe silently on the very upgrade it exists to catch.
        pytest.fail(
            f"qwen at {binary} produced no {' and no '.join(missing)} "
            f"(exit {completed.returncode}). The conformance probe cannot verify "
            "the sub-agent safety premises, so treat this as a qwen regression "
            f"until proven otherwise.\nstderr tail: {completed.stderr[-800:]!r}",
            pytrace=False,
        )
    return {"prompt": prompt, "tools": tools, "settings": overrides.env}


def _tools_from_stream(stdout: str) -> list[str] | None:
    """
    Pull the declared tool list out of qwen's ``system``/``init`` event.

    qwen may print warnings or log lines around its JSON, and one of those can
    itself start with ``[`` (an ``[INFO]``-style prefix), so locking onto the
    first bracket and giving up when it fails to parse would silently report
    "no tools" — which now fails the suite rather than skipping it. Instead every
    plausible JSON start is tried with ``raw_decode`` until one yields an init
    event, and each line is also tried alone for stream-json output.

    :returns: The declared tool names, or ``None`` if no init event was found.
    """
    for payload in _candidate_json_payloads(stdout):
        tools = _tools_from_payload(payload)
        if tools is not None:
            return tools
    return None


def _candidate_json_payloads(stdout: str) -> Iterator[object]:
    """Yield every JSON value decodable from *stdout*, junk lines tolerated."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(stdout, index)
            except ValueError:
                continue
            yield value
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped[:1] in ("[", "{"):
            try:
                yield json.loads(stripped)
            except ValueError:
                continue


def _tools_from_payload(payload: object) -> list[str] | None:
    """Return the tool names from an init event inside *payload*, if present."""
    events = payload if isinstance(payload, list) else [payload]
    for event in events:
        if not isinstance(event, dict) or event.get("subtype") != "init":
            continue
        tools = event.get("tools")
        if isinstance(tools, list):
            return [name for name in tools if isinstance(name, str)]
    return None


def test_headless_prompt_still_forbids_asking_questions(qwen_probe: dict) -> None:
    """A headless sub-agent that asks a question hangs its turn with no answerer.

    If this fails after a qwen upgrade, the never-ask guardrail moved or was
    removed: either re-anchor the assertion on the new wording, or add the
    instruction to ``QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND`` ourselves.
    """
    assert _NEVER_ASK in qwen_probe["prompt"].lower(), (
        "qwen's headless base prompt no longer tells the model never to ask a "
        "question. The qwen sub-agent trim relies on that guardrail."
    )


def test_the_trimmed_tool_set_excludes_agent_and_ask_user_question(qwen_probe: dict) -> None:
    """``agent`` would fan out uncollectably; ``ask_user_question`` would hang.

    If this fails after a qwen upgrade, add the offending name to
    ``QWEN_SUBAGENT_DISABLED_TOOLS``.
    """
    tools = qwen_probe["tools"]
    present = sorted(set(_FORBIDDEN_TOOLS) & set(tools))
    assert not present, (
        f"qwen registered {present} for a trimmed sub-agent. "
        "Add the name(s) to QWEN_SUBAGENT_DISABLED_TOOLS."
    )
    # Sanity: the probe really did run under our trim, so an empty tool list
    # cannot pass this vacuously.
    assert "run_shell_command" in tools
    assert QWEN_SYSTEM_SETTINGS_ENV_VAR in qwen_probe["settings"]


def test_the_trim_survives_qwen_startup(qwen_probe: dict) -> None:
    """The knobs must still be honored at system scope after a real boot."""
    tools = qwen_probe["tools"]
    assert not [t for t in tools if t.startswith("computer_use__")]
    assert not [t for t in tools if t.startswith("mcp__")]
    assert not [t for t in tools if t in {"monitor", "create_sub_session", "enter_worktree"}]
