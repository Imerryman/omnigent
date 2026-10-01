"""Idle reaper for native-harness terminal panes (issue #1349).

Native CLI sessions (``claude-native`` / ``codex-native`` / ``cursor-native`` /
...) run their vendor CLI plus a full MCP fleet inside a persistent tmux pane
held for the whole conversation lifetime. Unlike the SDK harness proxies — which
``HarnessProcessManager._idle_reaper_loop`` reaps after an idle window — these
panes have no idle reaper, so on a shared/multi-conversation runner memory grows
without bound as idle conversations accumulate, independent of how many are
actually active (#1349).

This reaps a single native pane only when it is genuinely unused. "Busy" is a
disjunction of signals wired by ``create_runner_app``, evaluated in this order
(the first that fires spares the pane; cheap in-process checks run before any
that shell out to tmux or read ``/proc``):

  1. an in-flight runner turn (``_active_turns`` / ``has_active_turn``), OR
  2. the session's last recorded status is ``running`` (not consulted for codex,
     whose local status goes stale) — fed both by the in-runner pane watcher and,
     since #8058, by forwarder ``external_session_status`` edges. Native turns
     clear ``_active_turns`` right after the prompt is pasted, so this is the
     load-bearing signal for a long autonomous turn, OR
  3. a fresh approval-wait marker (a pane parked on a permission prompt), OR
  4. the harness stream made progress recently
     (:mod:`omnigent.terminals.pane_progress`; stamped once per new byte by the
     qwen forwarder), OR
  5. a human recently drove an attached viewer (#8320) — a keypress on a regular
     tmux client (``client_activity``), or any event on the web attach bridge.
     An attached viewer that is merely *present* does NOT count, else a tab left
     open overnight pins the pane (and its MCP fleet) until the host OOMs, OR
  6. the pane's tmux window emitted output recently, OR
  7. codex only: the AUTHORITATIVE server status is ``running``/``waiting`` or
     cannot be confirmed. A confirmed-idle codex verdict is final and skips the
     next signal, OR
  8. the pane's own process subtree has descendants averaging >= 5% of a core
     across two scans (:mod:`omnigent.terminals.pane_cpu`).

Signal 8 exists because every other one is *output-shaped*: they all ask "did
something appear recently?". A worker blocked inside one long silent child
process (``mypy .``, a migration, a full test run) produces nothing to see and
was being reaped while perfectly healthy. It measures the pane's process tree,
never tmux clients, so an attached-but-idle viewer cannot satisfy it either.

Recency windows: signals 4 and 6 use ``resolve_pane_output_busy_window_s``
(``OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S``, default 120s); signal 5 always uses the
fixed :data:`PANE_OUTPUT_BUSY_WINDOW_S`, so widening the knob cannot extend how
long an idle viewer is honoured. Signals 4 and 8 fail safe to busy (logged) if
their evidence-gathering raises.

A pane idle on every signal for longer than the idle window (20 min by default)
is reaped, with a **second busy re-check immediately before teardown** to close
the select→reap race. The tmux probes are blocking ``subprocess`` calls and the
CPU probe reads ``/proc``, so they run off the event loop via
``asyncio.to_thread``.

Teardown is **pane-scoped** (``reap`` closes only the one native terminal, not
the conversation's other terminals), leaving the session's primary OSEnv +
server-side transcript intact — the next message re-creates the pane and the
vendor CLI resumes via its own ``--resume``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NamedTuple

_logger = logging.getLogger(__name__)

# A pane whose window emitted output, or whose viewer took human input, this
# recently counts as busy. tmux's own clocks are evidence independent of the
# harness status pipeline, whose silent stall must not get a live, producing
# terminal reaped. Two reaper scan intervals, so activity between scans re-arms
# the idle clock. For the output-shaped signals this is the DEFAULT, which
# ``resolve_pane_output_busy_window_s`` lets the env override; the viewer-input
# signal always uses this fixed value.
PANE_OUTPUT_BUSY_WINDOW_S = 120.0
_OUTPUT_BUSY_WINDOW_ENV = "OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S"

# Native CLI panes are keyed (conversation_id, <harness short name>, "main") in
# the terminal registry. These short names match the ``terminal_name`` the
# per-harness ``_auto_create_<harness>_terminal`` paths launch with. This is the
# cheap name pre-filter; the wiring additionally confirms the registry resource
# ROLE is a native harness (so a user terminal that merely shares the name is not
# reaped — see ``create_runner_app``).
#
# "kimi" is deliberately absent: kimi records no resumable chat id, so a reaped
# pane cannot be re-created with its context — the next turn would silently
# start a fresh TUI. Keep kimi panes alive until the session is torn down.
NATIVE_PANE_TERMINAL_NAMES: frozenset[str] = frozenset(
    {
        "claude",
        "codex",
        "cursor",
        "goose",
        "hermes",
        "kiro",
        "qwen",
        "pi",
        "antigravity",
        "opencode",
    }
)

# Default idle window before an unused native pane is reaped. Shortened from
# the 1-hour HarnessProcessManager default to 20 min for the swapless dev box:
# a 60-min window let bursty fan-out pile ~1GB/tree resident to OOM. The
# OMNIGENT_NATIVE_PANE_IDLE_TIMEOUT_S env var still overrides, but env
# propagation across the zygote spawn paths proved unreliable, so the safe
# window lives in code.
_DEFAULT_IDLE_TIMEOUT_S = 20 * 60
_DEFAULT_REAPER_INTERVAL_S = 60.0
_IDLE_TIMEOUT_ENV = "OMNIGENT_NATIVE_PANE_IDLE_TIMEOUT_S"


class PaneRef(NamedTuple):
    """A live native CLI pane the reaper may reclaim.

    :param conversation_id: AP-allocated conversation id, e.g. ``"conv_abc123"``.
    :param terminal_id: Resource id of the native terminal, e.g.
        ``terminal_resource_id("claude", "main")`` — used for pane-scoped close.
    :param terminal_name: Harness short-name, e.g. ``"claude"``.
    :param socket_path: tmux socket for the viewer-input, output and pane-pid probes.
    """

    conversation_id: str
    terminal_id: str
    terminal_name: str
    socket_path: Path


def resolve_native_pane_idle_timeout_s() -> float:
    """Resolve the native-pane idle window in seconds.

    Honors :envvar:`OMNIGENT_NATIVE_PANE_IDLE_TIMEOUT_S` (``0`` disables pane
    reaping); otherwise the 20-minute default (:data:`_DEFAULT_IDLE_TIMEOUT_S`).
    An unparseable or negative value logs a warning and falls back to the
    default rather than failing the runner at boot — an env typo shouldn't take
    the runner down or (worse) make the reaper act on a bogus window.
    """
    raw = os.environ.get(_IDLE_TIMEOUT_ENV)
    if not raw:
        return float(_DEFAULT_IDLE_TIMEOUT_S)
    try:
        value = float(raw)
    except ValueError:
        _logger.warning(
            "%s=%r is not a number; using default %ss",
            _IDLE_TIMEOUT_ENV,
            raw,
            _DEFAULT_IDLE_TIMEOUT_S,
        )
        return float(_DEFAULT_IDLE_TIMEOUT_S)
    if value < 0:
        _logger.warning(
            "%s=%r is negative; using default %ss",
            _IDLE_TIMEOUT_ENV,
            raw,
            _DEFAULT_IDLE_TIMEOUT_S,
        )
        return float(_DEFAULT_IDLE_TIMEOUT_S)
    return value


def resolve_pane_output_busy_window_s() -> float:
    """Resolve the pane "recent progress" window in seconds.

    Honors :envvar:`OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S`; otherwise
    :data:`PANE_OUTPUT_BUSY_WINDOW_S`. Same shape and same failure policy as
    :func:`resolve_native_pane_idle_timeout_s` — an unparseable or negative value
    logs and falls back to the default rather than letting an env typo widen or
    (worse) collapse the window the busy predicate trusts.

    This window governs the OUTPUT-shaped "did something happen recently?"
    signals of the busy predicate: the tmux output clock and the harness
    stream-progress ledger (:mod:`omnigent.terminals.pane_progress`). It does
    NOT govern the viewer-input signal, which stays on the fixed
    :data:`PANE_OUTPUT_BUSY_WINDOW_S` so a widened knob cannot let an idle
    attached viewer pin a pane for longer. Unlike the idle timeout, ``0`` is
    not special-cased here — it simply means "only progress in the last zero
    seconds counts", which disables these two signals without touching the rest.
    """
    raw = os.environ.get(_OUTPUT_BUSY_WINDOW_ENV)
    if not raw:
        return PANE_OUTPUT_BUSY_WINDOW_S
    try:
        value = float(raw)
    except ValueError:
        _logger.warning(
            "%s=%r is not a number; using default %ss",
            _OUTPUT_BUSY_WINDOW_ENV,
            raw,
            PANE_OUTPUT_BUSY_WINDOW_S,
        )
        return PANE_OUTPUT_BUSY_WINDOW_S
    if value < 0:
        _logger.warning(
            "%s=%r is negative; using default %ss",
            _OUTPUT_BUSY_WINDOW_ENV,
            raw,
            PANE_OUTPUT_BUSY_WINDOW_S,
        )
        return PANE_OUTPUT_BUSY_WINDOW_S
    return value


class NativePaneReaper:
    """Background task that reaps idle, unattended native terminal panes.

    :param list_native_panes: Returns the currently-live native panes (already
        role-confirmed by the caller) as :class:`PaneRef` values.
    :param is_busy: ``async`` predicate — ``True`` if any liveness signal in the
        module docstring's chain fires (in-flight turn, ``running`` status,
        approval wait, stream progress, recent viewer *input*, tmux output,
        codex server status, descendant CPU). An attached viewer without recent
        input is not busy. Async so the blocking tmux and ``/proc`` probes run
        off the event loop.
    :param reap: ``async`` pane-scoped teardown — closes only this one native
        terminal, leaving the session resumable.
    :param idle_timeout_s: Idle window before reaping. ``None`` resolves the env
        knob; ``<= 0`` disables reaping.
    :param reaper_interval_s: Seconds between scans.
    """

    def __init__(
        self,
        *,
        list_native_panes: Callable[[], list[PaneRef]],
        is_busy: Callable[[PaneRef], Awaitable[bool]],
        reap: Callable[[PaneRef], Awaitable[None]],
        idle_timeout_s: float | None = None,
        reaper_interval_s: float = _DEFAULT_REAPER_INTERVAL_S,
    ) -> None:
        self._list_native_panes = list_native_panes
        self._is_busy = is_busy
        self._reap = reap
        self._idle_timeout_s = (
            idle_timeout_s if idle_timeout_s is not None else resolve_native_pane_idle_timeout_s()
        )
        self._reaper_interval_s = reaper_interval_s
        # conversation_id -> monotonic time it was last observed busy.
        self._last_busy_at: dict[str, float] = {}
        self._task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        """Spawn the reaper loop (idempotent)."""
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._reap_loop(), name="native-pane-idle-reaper")
        _logger.info(
            "native pane reaper started (idle_timeout=%ss, interval=%ss%s)",
            self._idle_timeout_s,
            self._reaper_interval_s,
            "; DISABLED" if self._idle_timeout_s <= 0 else "",
        )

    async def shutdown(self) -> None:
        """Cancel the reaper loop."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._started = False

    def _classify(self, now: float, panes: list[PaneRef], busy_convs: set[str]) -> list[PaneRef]:
        """Pure idle-clock decision: which panes are reapable right now.

        Given the set of conversation ids observed busy this scan, maintain the
        per-conversation idle clock and return the panes idle for at least
        ``idle_timeout_s``. A busy pane re-arms its clock; a newly-observed idle
        pane gets one full window of grace before it is eligible. No I/O, so it is
        unit-testable with an injected ``now`` and ``busy_convs``.
        """
        live: set[str] = set()
        reapable: list[PaneRef] = []
        for pane in panes:
            conv = pane.conversation_id
            live.add(conv)
            if conv in busy_convs:
                self._last_busy_at[conv] = now
                continue
            last = self._last_busy_at.get(conv)
            if last is None:
                self._last_busy_at[conv] = now
                continue
            if now - last >= self._idle_timeout_s:
                reapable.append(pane)
        # Forget conversations whose pane is gone so the clock map can't grow.
        for gone in self._last_busy_at.keys() - live:
            del self._last_busy_at[gone]
        return reapable

    async def _reap_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._reaper_interval_s)
            except asyncio.CancelledError:
                return
            # ``<= 0`` disables reaping entirely (mirrors the SDK reaper guard).
            if self._idle_timeout_s <= 0:
                continue
            try:
                await self._scan_once()
            except Exception:  # never let a scan error kill the loop
                _logger.exception("native pane reaper: scan failed")

    async def _scan_once(self) -> None:
        panes = self._list_native_panes()
        now = time.monotonic()
        busy_convs = {p.conversation_id for p in panes if await self._is_busy(p)}
        for pane in self._classify(now, panes, busy_convs):
            # Re-check immediately before teardown: selection happened above with
            # possibly-stale signals, and a turn / client / autonomous run may
            # have started since (the select→reap race). Re-arm and skip if so.
            if await self._is_busy(pane):
                self._last_busy_at[pane.conversation_id] = time.monotonic()
                continue
            _logger.info(
                "reaping idle native pane for conversation %s (%s; idle > %.0fs)",
                pane.conversation_id,
                pane.terminal_name,
                self._idle_timeout_s,
            )
            # Drop the clock entry up front: a reap failure then re-arms the grace
            # window next scan instead of permanently skipping the conversation.
            self._last_busy_at.pop(pane.conversation_id, None)
            try:
                await self._reap(pane)
            except Exception:
                _logger.exception(
                    "native pane reaper: reap failed for conversation %s", pane.conversation_id
                )
