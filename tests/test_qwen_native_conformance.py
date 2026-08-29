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

- **qwen on PATH** (or ``OMNIGENT_QWEN_PATH``): the probe runs and **fails
  loudly** if the guardrail text is gone or if either tool reappears.
- **no qwen binary**: every test skips. The main suite must not depend on a
  vendor CLI being installed.
- **no model server**: still runs. The probe deliberately points the OpenAI-
  compatible base URL at a closed port — qwen writes the system prompt
  (``QWEN_WRITE_SYSTEM_MD``) and emits its ``init`` event with the resolved tool
  list *before* any completion request, so no LLM backend, credentials, or
  network access is needed. A run that nevertheless produces neither artifact is
  treated as an unusable environment and skips rather than fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

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
        pytest.skip(f"qwen could not be run in this environment: {exc}")

    tools = _tools_from_stream(completed.stdout)
    prompt = system_md.read_text(encoding="utf-8") if system_md.is_file() else ""
    if not prompt and tools is None:
        pytest.skip(
            "qwen produced neither a system-prompt dump nor an init event "
            f"(exit {completed.returncode}); environment cannot run the probe"
        )
    return {"prompt": prompt, "tools": tools, "settings": overrides.env}


def _tools_from_stream(stdout: str) -> list[str] | None:
    """Pull the declared tool list out of qwen's ``system``/``init`` event."""
    start = stdout.find("[")
    if start < 0:
        return None
    try:
        events = json.loads(stdout[start:])
    except ValueError:
        return None
    for event in events if isinstance(events, list) else []:
        if isinstance(event, dict) and event.get("subtype") == "init":
            tools = event.get("tools")
            if isinstance(tools, list):
                return [t for t in tools if isinstance(t, str)]
    return None


def test_headless_prompt_still_forbids_asking_questions(qwen_probe: dict) -> None:
    """A headless sub-agent that asks a question hangs its turn with no answerer.

    If this fails after a qwen upgrade, the never-ask guardrail moved or was
    removed: either re-anchor the assertion on the new wording, or add the
    instruction to ``QWEN_SUBAGENT_SYSTEM_PROMPT_APPEND`` ourselves.
    """
    prompt = qwen_probe["prompt"]
    if not prompt:
        pytest.skip("qwen did not write a system-prompt dump in this environment")

    assert _NEVER_ASK in prompt.lower(), (
        "qwen's headless base prompt no longer tells the model never to ask a "
        "question. The qwen sub-agent trim relies on that guardrail."
    )


def test_the_trimmed_tool_set_excludes_agent_and_ask_user_question(qwen_probe: dict) -> None:
    """``agent`` would fan out uncollectably; ``ask_user_question`` would hang.

    If this fails after a qwen upgrade, add the offending name to
    ``QWEN_SUBAGENT_DISABLED_TOOLS``.
    """
    tools = qwen_probe["tools"]
    if tools is None:
        pytest.skip("qwen did not emit an init event in this environment")

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
    if tools is None:
        pytest.skip("qwen did not emit an init event in this environment")

    assert not [t for t in tools if t.startswith("computer_use__")]
    assert not [t for t in tools if t.startswith("mcp__")]
    assert not [t for t in tools if t in {"monitor", "create_sub_session", "enter_worktree"}]
