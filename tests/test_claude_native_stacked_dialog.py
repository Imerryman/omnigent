"""
Dialogs Claude Code stacks ABOVE a still-rendered composer.

Claude Code 2.1.280 opens "Teach auto mode about your environment?" as a turn
ends, once auto mode has recorded enough denials. The composer stays drawn
under it, so every "is the input box mounted?" check passes, yet the dialog
owns the keyboard: a pasted message is swallowed and the composer stays empty.
Before this fix delivery waited out the full paste-commit timeout and failed
("pasted draft could not be confirmed") on every message after the first of a
session; stock upstream instead pressed Enter blind, answering the dialog
"1. Yes" and launching the auto-mode setup wizard while reporting success.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnigent.harnesses.claude_native import bridge as claude_native_bridge
from omnigent.harnesses.claude_native.bridge import (
    ClaudeTerminalDialog,
    _occupying_surface,
    _stacked_dialog_headline,
    inject_user_message,
    write_tmux_target,
)

_RULE = "─" * 80

# Captured verbatim (blank rows dropped) from a live Claude Code 2.1.280 pane.
_TEACH_AUTO_MODE_PANE = f"""\
 ▐▛███▛█   Claude Code v2.1.280
▝▜██████▀  Sonnet 5 · Claude API
  ▝▝ ▝▝    ~/dev/ab-bridge/work
❯ reply with just the word ok
● ok
{_RULE}
  Teach auto mode about your environment?
  Auto mode works better when it knows your environment. Takes about a minute.
  ❯ 1. Yes
    2. Not now
    3. Don't show again
  Enter to confirm · Esc to cancel
{_RULE}
❯
{_RULE}
  ⏵⏵ auto mode on (shift+tab to cycle) · ← for agents
"""


def _composer_pane(draft: str = "") -> str:
    """
    Render the same session with the dialog gone and *draft* in the box.

    :param draft: Text sitting in the composer; empty means idle.
    :returns: The pane text.
    """
    return f"""\
❯ reply with just the word ok
● ok
{_RULE}
❯ {draft}
{_RULE}
  ⏵⏵ auto mode on (shift+tab to cycle) · ← for agents
"""


def test_the_auto_mode_setup_prompt_is_read_as_a_stacked_dialog() -> None:
    """The real 2.1.280 pane names its headline and occupies the input box."""
    assert _stacked_dialog_headline(_TEACH_AUTO_MODE_PANE) == (
        "Teach auto mode about your environment?"
    )
    surface = _occupying_surface(_TEACH_AUTO_MODE_PANE)
    assert surface is not None and surface.startswith("a dialog above the input box"), surface


@pytest.mark.parametrize(
    "pane",
    [
        _composer_pane(),
        _composer_pane("fix the bug"),
        # The dialog's text echoed in the transcript, but the framed region
        # above the composer ends in ordinary output, not the footer.
        f"""\
{_RULE}
  Teach auto mode about your environment?
  ❯ 1. Yes
  Enter to confirm · Esc to cancel
● I quoted that dialog above; nothing to answer.
{_RULE}
❯
{_RULE}
""",
        # Footer flush on the composer rule but no numbered selector row.
        f"""\
{_RULE}
  Some note
  Enter to confirm · Esc to cancel
{_RULE}
❯
{_RULE}
""",
        # A framed region far taller than any dialog is transcript.
        _RULE
        + "\n"
        + "\n".join(f"line {i}" for i in range(20))
        + "\n  ❯ 1. Yes\n  Enter to confirm · Esc to cancel\n"
        + f"{_RULE}\n❯ \n{_RULE}\n",
        "",
    ],
    ids=[
        "idle-composer",
        "draft-in-composer",
        "echoed-in-transcript",
        "no-selector",
        "too-tall",
        "torn-capture",
    ],
)
def test_panes_without_a_stacked_dialog_draw_no_escape(pane: str) -> None:
    """Only the narrow structural shape reads as a dialog — never plain chat."""
    assert _stacked_dialog_headline(pane) is None
    assert _occupying_surface(pane) is None


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Point the bridge at a fake tmux target with fast polling.

    :param tmp_path: pytest temp dir.
    :param monkeypatch: pytest monkeypatch.
    :returns: The bridge directory.
    """
    monkeypatch.setattr(claude_native_bridge, "_TRUSTED_PARENT", tmp_path)
    monkeypatch.setattr(claude_native_bridge, "_BRIDGE_ROOT", tmp_path)
    monkeypatch.setattr(claude_native_bridge, "_CLAUDE_READY_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(claude_native_bridge, "_OCCUPIED_INPUT_DISMISS_RETRY_INTERVAL_S", 0.0)
    bridge_dir = tmp_path / "bridge"
    write_tmux_target(
        bridge_dir, socket_path=Path("/tmp/example/tmux.sock"), tmux_target="claude:0.0"
    )
    return bridge_dir


def test_a_stacked_dialog_is_escaped_before_the_message_is_pasted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The prompt already up when a message arrives is dismissed, then delivery runs.

    Escape is the dialog's documented non-committal exit ("Not now"); Enter
    would answer "1. Yes" and launch the setup wizard, so no Enter may reach
    the pane before the dialog is gone.
    """
    bridge_dir = _setup(tmp_path, monkeypatch)
    tui = {"pane": _TEACH_AUTO_MODE_PANE}
    keys: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if "capture-pane" in cmd:
            return SimpleNamespace(returncode=0, stdout=tui["pane"], stderr="")
        if cmd[-1] == "Escape":
            tui["pane"] = _composer_pane()
        if "paste-buffer" in cmd:
            if tui["pane"] is not _TEACH_AUTO_MODE_PANE:
                tui["pane"] = _composer_pane("what is 17 plus 25")
            keys.append("paste")
        elif cmd[-1] in ("Escape", "Enter", "C-a", "C-k"):
            if cmd[-1] == "Enter":
                assert tui["pane"] is not _TEACH_AUTO_MODE_PANE, "Enter answered the dialog"
                tui["pane"] = _composer_pane()
            keys.append(cmd[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)
    inject_user_message(bridge_dir, content="what is 17 plus 25")

    assert keys == ["Escape", "C-a", "C-k", "paste", "Enter"], keys


def test_a_dialog_that_opens_mid_paste_is_dismissed_and_the_paste_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The dialog opens as a turn ends — after the pre-paste check — and eats the paste.

    Delivery must notice the dialog instead of waiting out the whole
    paste-commit timeout, Escape it, and paste once more.
    """
    bridge_dir = _setup(tmp_path, monkeypatch)
    tui = {"pane": _composer_pane(), "pastes": 0}
    keys: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if "capture-pane" in cmd:
            return SimpleNamespace(returncode=0, stdout=tui["pane"], stderr="")
        if "paste-buffer" in cmd:
            tui["pastes"] += 1
            keys.append("paste")
            # First paste: the dialog pops and swallows it (composer stays empty).
            tui["pane"] = (
                _TEACH_AUTO_MODE_PANE if tui["pastes"] == 1 else _composer_pane("hello there")
            )
        elif cmd[-1] == "Escape":
            keys.append("Escape")
            tui["pane"] = _composer_pane()
        elif cmd[-1] == "Enter":
            keys.append("Enter")
            tui["pane"] = _composer_pane()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)
    started = time.monotonic()
    inject_user_message(bridge_dir, content="hello there")
    elapsed = time.monotonic() - started

    assert keys == ["paste", "Escape", "paste", "Enter"], keys
    assert elapsed < claude_native_bridge._PASTE_COMMIT_TIMEOUT_S / 2, elapsed


def test_a_dialog_that_will_not_close_fails_fast_by_name_without_enter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stubborn dialog is reported by headline quickly; no Enter ever answers it."""
    bridge_dir = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(claude_native_bridge, "_OCCUPIED_INPUT_DISMISS_TIMEOUT_S", 0.2)
    keys: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if "capture-pane" in cmd:
            return SimpleNamespace(returncode=0, stdout=_TEACH_AUTO_MODE_PANE, stderr="")
        keys.append("paste" if "paste-buffer" in cmd else cmd[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)
    started = time.monotonic()
    with pytest.raises(ClaudeTerminalDialog, match="Teach auto mode about your environment"):
        inject_user_message(bridge_dir, content="hello")
    elapsed = time.monotonic() - started

    assert "Enter" not in keys, keys
    assert elapsed < claude_native_bridge._PASTE_COMMIT_TIMEOUT_S / 2, elapsed
