"""Tests for the native-pane idle reaper (issue #1349)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from omnigent.entities.session_resources import terminal_resource_id
from omnigent.harnesses.claude_native import bridge as claude_native_bridge
from omnigent.inner.terminal import TerminalInstance
from omnigent.native import native_cost_popup
from omnigent.runner.app import create_runner_app
from omnigent.runner.resource_registry import SessionResourceRegistry
from omnigent.terminals import pane_cpu, pane_progress
from omnigent.terminals.pane_reaper import (
    _DEFAULT_IDLE_TIMEOUT_S,
    _IDLE_TIMEOUT_ENV,
    PANE_OUTPUT_BUSY_WINDOW_S,
    NativePaneReaper,
    PaneRef,
    resolve_native_pane_idle_timeout_s,
    resolve_pane_output_busy_window_s,
)
from omnigent.terminals.registry import TerminalRegistry
from tests.runner.helpers import NullServerClient


def _pane(conv: str, name: str = "claude") -> PaneRef:
    return PaneRef(conv, f"terminal:{name}:main", name, Path(f"/tmp/omni-test/{conv}.sock"))


class _Fakes:
    """Mutable test doubles so a test can flip busy/panes between scans."""

    def __init__(self) -> None:
        self.panes: list[PaneRef] = []
        self.busy: set[str] = set()
        self.reaped: list[str] = []
        # Optional per-call override: (pane, call_index) -> bool | None (None is
        # "unknown"). Lets a test make is_busy answer differently on the
        # classify pass vs the re-check pass.
        self.busy_override: Callable[[PaneRef, int], bool | None] | None = None
        self.busy_calls = 0

    async def is_busy(self, pane: PaneRef) -> bool | None:
        self.busy_calls += 1
        if self.busy_override is not None:
            return self.busy_override(pane, self.busy_calls)
        return pane.conversation_id in self.busy

    async def reap(self, pane: PaneRef) -> None:
        self.reaped.append(pane.conversation_id)
        self.panes = [p for p in self.panes if p.conversation_id != pane.conversation_id]


def _make(fakes: _Fakes, *, timeout: float = 100.0, interval: float = 0.01) -> NativePaneReaper:
    return NativePaneReaper(
        list_native_panes=lambda: list(fakes.panes),
        is_busy=fakes.is_busy,
        reap=fakes.reap,
        idle_timeout_s=timeout,
        reaper_interval_s=interval,
    )


# ── Pure idle-clock decision (_classify) ────────────────────────────────────


def test_classify_reaps_only_after_full_window() -> None:
    f = _Fakes()
    p = _pane("conv_a")
    f.panes = [p]
    r = _make(f, timeout=100.0)
    assert r._classify(1000.0, [p], busy_convs=set()) == []  # first obs: grace
    assert r._classify(1099.0, [p], busy_convs=set()) == []  # 99s < 100s
    assert r._classify(1100.0, [p], busy_convs=set()) == [p]  # window elapsed


def test_classify_busy_rearms_clock() -> None:
    f = _Fakes()
    p = _pane("conv_a")
    r = _make(f, timeout=10.0)
    r._classify(0.0, [p], busy_convs={"conv_a"})
    assert r._classify(1000.0, [p], busy_convs={"conv_a"}) == []  # busy re-arms
    r._classify(1000.0, [p], busy_convs=set())  # now idle, grace
    assert r._classify(1010.0, [p], busy_convs=set()) == [p]


def test_classify_first_observation_grace() -> None:
    f = _Fakes()
    p = _pane("conv_a")
    r = _make(f, timeout=0.001)
    assert r._classify(5.0, [p], busy_convs=set()) == []  # clock seeded this pass


def test_classify_forgets_gone_panes() -> None:
    f = _Fakes()
    p = _pane("conv_a")
    r = _make(f, timeout=10.0)
    r._classify(0.0, [p], busy_convs=set())
    assert "conv_a" in r._last_busy_at
    r._classify(1.0, [], busy_convs=set())  # pane gone
    assert "conv_a" not in r._last_busy_at


def test_classify_unknown_leaves_clock_untouched() -> None:
    """An unknown scan neither re-arms the idle clock nor makes the pane reapable."""
    f = _Fakes()
    p = _pane("conv_a")
    r = _make(f, timeout=10.0)
    r._classify(0.0, [p], busy_convs={"conv_a"})
    # Long past the window, but liveness could not be confirmed: no reap, no re-arm.
    assert r._classify(1000.0, [p], busy_convs=set(), unknown_convs={"conv_a"}) == []
    assert r._last_busy_at["conv_a"] == 0.0
    # The next confirmed-idle scan judges against the last CONFIRMED busy time.
    assert r._classify(1001.0, [p], busy_convs=set()) == [p]


def test_classify_unknown_first_observation_seeds_nothing() -> None:
    f = _Fakes()
    p = _pane("conv_a")
    r = _make(f, timeout=10.0)
    assert r._classify(0.0, [p], busy_convs=set(), unknown_convs={"conv_a"}) == []
    assert "conv_a" not in r._last_busy_at
    r._classify(5.0, [p], busy_convs=set())  # first confirmed idle: grace starts here
    assert r._classify(14.0, [p], busy_convs=set()) == []
    assert r._classify(15.0, [p], busy_convs=set()) == [p]


# ── Scan behaviour (_scan_once): reap, skip-busy, TOCTOU re-check ────────────


async def test_scan_reaps_idle_unbusy_pane() -> None:
    f = _Fakes()
    p = _pane("conv_a")
    f.panes = [p]
    r = _make(f, timeout=10.0)
    r._last_busy_at["conv_a"] = time.monotonic() - 1000  # already idle past window
    await r._scan_once()
    assert f.reaped == ["conv_a"]


async def test_scan_skips_busy_pane() -> None:
    f = _Fakes()
    p = _pane("conv_a")
    f.panes = [p]
    f.busy = {"conv_a"}
    r = _make(f, timeout=10.0)
    r._last_busy_at["conv_a"] = time.monotonic() - 1000
    await r._scan_once()
    assert f.reaped == []  # busy → not reaped, clock re-armed


async def test_scan_recheck_spares_pane_that_became_busy() -> None:
    """TOCTOU guard: a pane idle at selection but busy at the pre-reap re-check
    must NOT be reaped (a turn/client/autonomous run started in between)."""
    f = _Fakes()
    p = _pane("conv_a")
    f.panes = [p]
    # is_busy: False on the classify-phase call (call 1), True on the re-check
    # call (call 2) — simulating a turn starting between selection and teardown.
    f.busy_override = lambda pane, n: n >= 2
    r = _make(f, timeout=10.0)
    r._last_busy_at["conv_a"] = time.monotonic() - 1000
    await r._scan_once()
    assert f.reaped == []  # spared by the re-check
    assert f.busy_calls == 2  # classify + re-check


async def test_scan_unknown_neither_reaps_nor_rearms() -> None:
    """A failed liveness probe on the selection pass spares the pane, clock intact."""
    f = _Fakes()
    p = _pane("conv_a")
    f.panes = [p]
    f.busy_override = lambda pane, n: None
    r = _make(f, timeout=10.0)
    stale = time.monotonic() - 1000
    r._last_busy_at["conv_a"] = stale
    await r._scan_once()
    assert f.reaped == []
    assert r._last_busy_at["conv_a"] == stale


async def test_scan_recheck_unknown_skips_without_rearming() -> None:
    """Idle at selection but unconfirmable at the pre-reap re-check: no teardown."""
    f = _Fakes()
    p = _pane("conv_a")
    f.panes = [p]
    f.busy_override = lambda pane, n: False if n == 1 else None
    r = _make(f, timeout=10.0)
    stale = time.monotonic() - 1000
    r._last_busy_at["conv_a"] = stale
    await r._scan_once()
    assert f.reaped == []
    assert f.busy_calls == 2
    assert r._last_busy_at["conv_a"] == stale


# ── Env resolver ────────────────────────────────────────────────────────────


def test_resolve_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_IDLE_TIMEOUT_ENV, raising=False)
    assert resolve_native_pane_idle_timeout_s() == float(_DEFAULT_IDLE_TIMEOUT_S)


def test_resolve_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_IDLE_TIMEOUT_ENV, "120")
    assert resolve_native_pane_idle_timeout_s() == 120.0


def test_resolve_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_IDLE_TIMEOUT_ENV, "0")
    assert resolve_native_pane_idle_timeout_s() == 0.0


@pytest.mark.parametrize("bad", ["abc", "-5", ""])
def test_resolve_invalid_falls_back(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv(_IDLE_TIMEOUT_ENV, bad)
    assert resolve_native_pane_idle_timeout_s() == float(_DEFAULT_IDLE_TIMEOUT_S)


# ── Loop smoke (start/shutdown + disable) ───────────────────────────────────


async def test_loop_reaps_idle_pane() -> None:
    f = _Fakes()
    f.panes = [_pane("conv_a")]
    r = _make(f, timeout=0.0001, interval=0.01)
    await r.start()
    try:
        for _ in range(100):
            if f.reaped:
                break
            await asyncio.sleep(0.01)
    finally:
        await r.shutdown()
    assert f.reaped == ["conv_a"]


async def test_loop_disabled_when_timeout_non_positive() -> None:
    f = _Fakes()
    f.panes = [_pane("conv_a")]
    r = _make(f, timeout=0.0, interval=0.01)  # 0 disables
    await r.start()
    try:
        await asyncio.sleep(0.1)
    finally:
        await r.shutdown()
    assert f.reaped == []


def test_kimi_is_exempt_from_pane_reaping() -> None:
    # kimi records no resumable chat id, so a reaped pane cannot be re-created
    # with its context; the name filter must never offer kimi panes to the reaper.
    from omnigent.terminals.pane_reaper import NATIVE_PANE_TERMINAL_NAMES

    assert "kimi" not in NATIVE_PANE_TERMINAL_NAMES
    assert "claude" in NATIVE_PANE_TERMINAL_NAMES


async def test_runner_busy_check_spares_a_pane_parked_on_an_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The runner's busy check treats a fresh approval-wait marker as busy.

    A pane parked on a permission prompt has no active turn, no ``running``
    status, no attached client and no output, so every other signal reads idle
    and the reaper would kill the prompt under a still-answerable card.
    """
    # Bound by name when the app is built, so stub before building: no tmux here.
    monkeypatch.setattr(native_cost_popup, "_tmux_last_client_input_at", lambda *_args: None)
    monkeypatch.setattr(native_cost_popup, "_tmux_window_activity_at", lambda *_args: None)
    monkeypatch.setattr(claude_native_bridge, "_APPROVAL_WAIT_ROOT", tmp_path / "approval-waits")
    registry = TerminalRegistry()
    app = create_runner_app(
        terminal_registry=registry,
        resource_registry=SessionResourceRegistry(terminal_registry=registry),
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )
    reaper = app.state.native_pane_reaper
    assert reaper is not None
    pane = PaneRef(
        "conv_parked", terminal_resource_id("claude", "main"), "claude", tmp_path / "tmux.sock"
    )

    assert not await reaper._is_busy(pane)

    marker = claude_native_bridge.approval_wait_marker_path("conv_parked")
    marker.parent.mkdir(parents=True)
    claude_native_bridge.touch_approval_wait_marker(marker)
    assert await reaper._is_busy(pane)
    # Another session's parked prompt does not spare this pane.
    other = PaneRef("conv_other", pane.terminal_id, "claude", pane.socket_path)
    assert not await reaper._is_busy(other)


async def test_runner_busy_check_counts_a_viewer_only_on_recent_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An attached viewer spares the pane only while a human recently drove it.

    A CLI tab's keypress (tmux ``client_activity``) or a web-bridge event (the
    terminal's interaction stamp) inside the busy window reads busy; an idle
    attached viewer, or stale input, does not — a tab left open overnight must
    not keep the native stack resident.
    """
    tmux_input_at: dict[str, float | None] = {"value": None}
    monkeypatch.setattr(
        native_cost_popup, "_tmux_last_client_input_at", lambda *_args: tmux_input_at["value"]
    )
    monkeypatch.setattr(native_cost_popup, "_tmux_window_activity_at", lambda *_args: None)
    monkeypatch.setattr(claude_native_bridge, "_APPROVAL_WAIT_ROOT", tmp_path / "approval-waits")
    registry = TerminalRegistry()
    instance = TerminalInstance(
        name="claude",
        session_key="main",
        socket_path=tmp_path / "tmux.sock",
        private_dir=tmp_path,
        running=True,
    )
    registry._by_conversation["conv_viewed"] = {("claude", "main"): instance}
    app = create_runner_app(
        terminal_registry=registry,
        resource_registry=SessionResourceRegistry(terminal_registry=registry),
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )
    reaper = app.state.native_pane_reaper
    assert reaper is not None
    pane = PaneRef(
        "conv_viewed", terminal_resource_id("claude", "main"), "claude", tmp_path / "tmux.sock"
    )

    # Attached but idle on both signals: not busy.
    assert not await reaper._is_busy(pane)
    # A CLI keypress inside the window spares the pane; a stale one does not.
    tmux_input_at["value"] = time.time() - 1.0
    assert await reaper._is_busy(pane)
    tmux_input_at["value"] = time.time() - PANE_OUTPUT_BUSY_WINDOW_S - 1.0
    assert not await reaper._is_busy(pane)
    # A web-bridge event on the pane's terminal spares it; a stale one does not.
    instance.note_client_interaction()
    assert await reaper._is_busy(pane)
    instance._last_client_interaction_at = time.monotonic() - PANE_OUTPUT_BUSY_WINDOW_S - 1.0
    assert not await reaper._is_busy(pane)


# ---------------------------------------------------------------------------
# Process-level + stream-level liveness (issue #1349).
#
# Every pre-existing busy signal is output-shaped: an in-flight turn, a
# ``running`` status, recent viewer input, a byte in the tmux window. A worker
# blocked inside ONE long silent child (``mypy .``, a migration, a full pytest
# run) trips none of them and was being reaped while perfectly healthy. These
# cover the two signals that close that gap, and — mandatorily — that a pane
# with no evidence at all is still reaped.
# ---------------------------------------------------------------------------


class _FixedClock:
    """Injected monotonic clock so the CPU probe yields a rate without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _silent_pane_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pane_pid: int | None = 4242,
    server_client: NullServerClient | None = None,
    window_activity_at: Callable[..., float | None] = lambda *_args: None,
) -> object:
    """A runner app whose panes are silent on every pre-existing busy signal.

    No recent viewer input, no tmux output, no approval marker, no active turn,
    no ``running`` status — so whatever the new signals say is the whole answer.
    """
    monkeypatch.setattr(native_cost_popup, "_tmux_last_client_input_at", lambda *_args: None)
    monkeypatch.setattr(native_cost_popup, "_tmux_window_activity_at", window_activity_at)
    monkeypatch.setattr(native_cost_popup, "_tmux_pane_pid", lambda *_args: pane_pid)
    monkeypatch.setattr(claude_native_bridge, "_APPROVAL_WAIT_ROOT", tmp_path / "approval-waits")
    pane_progress.reset_stream_progress()
    registry = TerminalRegistry()
    return create_runner_app(
        terminal_registry=registry,
        resource_registry=SessionResourceRegistry(terminal_registry=registry),
        server_client=server_client or NullServerClient(),  # type: ignore[arg-type]
    )


async def test_pane_with_cpu_active_descendant_is_not_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance 1: a CPU-burning descendant spares a pane with zero output.

    This is the whole point of the change: ``mypy .`` under the vendor CLI emits
    nothing for minutes, so before this the pane read idle on every signal.
    """
    clock = _FixedClock()
    # 60s apart, +6000 ticks == 60 CPU-seconds == a descendant pinning one core.
    ticks = iter([{11: 5_000, 12: 100}, {11: 11_000, 12: 100}])
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: next(ticks))
    app = _silent_pane_app(monkeypatch, tmp_path)
    app.state.native_pane_cpu_probe = pane_cpu.PaneDescendantCpuProbe(clock=clock)
    reaper = app.state.native_pane_reaper
    pane = _pane("conv_compiling")

    # First scan only records the CPU baseline, so the pane is not yet spared by
    # this signal (it still has a full idle window of grace from _classify).
    assert not await reaper._is_busy(pane)
    clock.now += 60.0
    assert await reaper._is_busy(pane)


async def test_pane_with_quiet_descendant_is_still_reapable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle TUI's housekeeping CPU must not pin the pane forever."""
    clock = _FixedClock()
    # 60s apart, +60 ticks == 0.6 CPU-seconds == 1% of a core: below threshold.
    ticks = iter([{11: 5_000, 12: 100}, {11: 5_060, 12: 100}])
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: next(ticks))
    app = _silent_pane_app(monkeypatch, tmp_path)
    app.state.native_pane_cpu_probe = pane_cpu.PaneDescendantCpuProbe(clock=clock)
    reaper = app.state.native_pane_reaper
    pane = _pane("conv_idle_tui")

    assert not await reaper._is_busy(pane)
    clock.now += 60.0
    assert not await reaper._is_busy(pane)


async def test_streaming_qwen_pane_is_not_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance 2: recent stream progress spares a qwen pane with quiet tmux.

    qwen is the least-protected harness — it has no server-status fallback like
    codex — so its forwarder's sub-second stream cursor is the only token-level
    evidence available.
    """
    app = _silent_pane_app(monkeypatch, tmp_path, pane_pid=None)
    reaper = app.state.native_pane_reaper
    pane = _pane("conv_streaming", "qwen")

    assert not await reaper._is_busy(pane)

    pane_progress.note_stream_progress("conv_streaming")
    assert await reaper._is_busy(pane)
    # Another conversation's stream does not spare this pane.
    assert not await reaper._is_busy(_pane("conv_other_stream", "qwen"))


async def test_stale_stream_progress_does_not_spare_a_pane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress older than the output window is not evidence of a live stream."""
    app = _silent_pane_app(monkeypatch, tmp_path, pane_pid=None)
    reaper = app.state.native_pane_reaper
    pane = _pane("conv_stale_stream", "qwen")

    now = time.monotonic()
    pane_progress.note_stream_progress(
        "conv_stale_stream", now=now - (PANE_OUTPUT_BUSY_WINDOW_S + 5.0)
    )
    assert not await reaper._is_busy(pane)


async def test_dead_pane_is_still_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance 3 (mandatory): the reaper did not become a no-op.

    No recent viewer input, no turn, no output, no CPU-burning descendant, no
    stream progress — the pane is genuinely dead and must be torn down. Driven through
    a real reaper loop (including its pre-teardown re-check), not just the
    predicate, so a signal that wrongly latched "busy" would fail here.
    """
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: {11: 5_000})
    app = _silent_pane_app(monkeypatch, tmp_path)
    pane = _pane("conv_dead")
    reaped: list[str] = []

    reaper = NativePaneReaper(
        list_native_panes=lambda: [pane],
        is_busy=app.state.native_pane_reaper._is_busy,
        reap=lambda p: _record_reaped(reaped, p),
        idle_timeout_s=0.0001,
        reaper_interval_s=0.01,
    )
    await reaper._scan_once()  # arms the idle clock
    await asyncio.sleep(0.01)
    await reaper._scan_once()  # window elapsed -> reap
    await reaper.shutdown()

    assert reaped == ["conv_dead"]


async def _record_reaped(sink: list[str], pane: PaneRef) -> None:
    sink.append(pane.conversation_id)


@pytest.mark.parametrize("failing", ["cpu", "stream"])
async def test_new_signal_exception_means_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failing: str,
) -> None:
    """Acceptance 4: a broken new signal fails safe to busy, logged, never raised.

    The established convention for this predicate is that any doubt spares the
    pane; a bug in evidence-gathering must never become a teardown.
    """

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("probe exploded")

    # The stream reader is bound by name when the app is built, so patch first.
    if failing == "stream":
        monkeypatch.setattr(pane_progress, "stream_progress_age_s", _boom)
    app = _silent_pane_app(monkeypatch, tmp_path)
    if failing == "cpu":
        monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", _boom)
    reaper = app.state.native_pane_reaper

    with caplog.at_level(logging.ERROR):
        assert await reaper._is_busy(_pane("conv_broken"))
    assert any("treating pane as busy" in record.message for record in caplog.records)


def test_output_busy_window_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The output/progress window is env-overridable, default UNCHANGED at 120s."""
    monkeypatch.delenv("OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S", raising=False)
    assert PANE_OUTPUT_BUSY_WINDOW_S == 120.0
    assert resolve_pane_output_busy_window_s() == 120.0
    monkeypatch.setenv("OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S", "45")
    assert resolve_pane_output_busy_window_s() == 45.0


@pytest.mark.parametrize("bad", ["", "abc", "-1"])
def test_output_busy_window_invalid_falls_back(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """An env typo must not widen or collapse the window the predicate trusts."""
    monkeypatch.setenv("OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S", bad)
    assert resolve_pane_output_busy_window_s() == PANE_OUTPUT_BUSY_WINDOW_S


def test_stream_progress_ledger_is_bounded() -> None:
    """The module-level ledger evicts oldest-first so it cannot grow unbounded."""
    pane_progress.reset_stream_progress()
    for index in range(pane_progress._MAX_TRACKED + 10):
        pane_progress.note_stream_progress(f"conv_{index}", now=float(index))
    assert len(pane_progress._last_progress_at) == pane_progress._MAX_TRACKED
    assert "conv_0" not in pane_progress._last_progress_at
    assert f"conv_{pane_progress._MAX_TRACKED + 9}" in pane_progress._last_progress_at
    pane_progress.reset_stream_progress()


def test_stream_progress_unknown_conversation_is_none() -> None:
    """No recorded progress is "no evidence", never "idle"."""
    pane_progress.reset_stream_progress()
    assert pane_progress.stream_progress_age_s("conv_never") is None
    pane_progress.note_stream_progress("conv_never", now=100.0)
    assert pane_progress.stream_progress_age_s("conv_never", now=130.0) == 30.0
    pane_progress.clear_stream_progress("conv_never")
    assert pane_progress.stream_progress_age_s("conv_never") is None


async def test_authoritative_codex_status_wins_over_descendant_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A codex session the server calls idle IS idle, whatever its subtree burns.

    The codex app-server is the one component that actually knows whether a
    session has a turn in flight; descendant CPU is a proxy for that question.
    Letting the proxy override a confirmed-idle verdict would make a codex pane
    with a runaway child immortal. ``NullServerClient`` answers 200 with no
    status, i.e. authoritatively not running.
    """
    clock = _FixedClock()
    # A descendant pinning a full core: enough to spare any non-codex pane.
    ticks = iter([{11: 5_000}, {11: 11_000}, {11: 5_000}, {11: 11_000}])
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: next(ticks))
    app = _silent_pane_app(monkeypatch, tmp_path)
    app.state.native_pane_cpu_probe = pane_cpu.PaneDescendantCpuProbe(clock=clock)
    reaper = app.state.native_pane_reaper

    codex_pane = _pane("conv_codex", "codex")
    claude_pane = _pane("conv_claude", "claude")
    assert not await reaper._is_busy(codex_pane)  # baseline sample
    assert not await reaper._is_busy(claude_pane)  # baseline sample
    clock.now += 60.0
    # Same CPU evidence, opposite verdicts: codex defers to its server.
    assert not await reaper._is_busy(codex_pane)
    assert await reaper._is_busy(claude_pane)


async def test_sparing_on_cpu_alone_logs_the_culprit_pid_and_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pane spared SOLELY by descendant CPU says so, with the pid and the rate.

    Accepted tradeoff of acceptance item 1: a dead agent with a busy-looping
    descendant (a wedged MCP server, a polling sidecar) above the threshold stays
    alive while it burns CPU. Not redesigned — made observable, so the case is
    diagnosable from the log rather than mysterious.
    """
    clock = _FixedClock()
    ticks = iter([{11: 0, 12: 0}, {11: 600, 12: 5_400}])
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: next(ticks))
    app = _silent_pane_app(monkeypatch, tmp_path)
    app.state.native_pane_cpu_probe = pane_cpu.PaneDescendantCpuProbe(clock=clock)
    reaper = app.state.native_pane_reaper
    pane = _pane("conv_runaway")

    await reaper._is_busy(pane)
    clock.now += 60.0
    with caplog.at_level(logging.INFO):
        assert await reaper._is_busy(pane)

    logged = [r.getMessage() for r in caplog.records if "descendant CPU alone" in r.getMessage()]
    assert len(logged) == 1
    assert "conv_runaway" in logged[0]
    assert "busiest pid 12" in logged[0]
    assert "90.0%" in logged[0]  # pid 12's share of the 100%-of-a-core total


async def test_session_cleanup_clears_both_liveness_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanly-ended session leaves no liveness residue behind.

    Both ledgers were previously cleared only on the reaper's own teardown path,
    so a normal session end left stale entries alive until eviction.
    """
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: {11: 5_000})
    app = _silent_pane_app(monkeypatch, tmp_path)
    probe = app.state.native_pane_cpu_probe

    pane_progress.note_stream_progress("conv_ending")
    probe.is_cpu_active("conv_ending", 4242)
    assert pane_progress.stream_progress_age_s("conv_ending") is not None
    assert "conv_ending" in probe._samples

    client = TestClient(app)
    client.delete("/v1/sessions/conv_ending")

    assert pane_progress.stream_progress_age_s("conv_ending") is None
    assert "conv_ending" not in probe._samples


# ---------------------------------------------------------------------------
# Composition with #8320 (an attached viewer is busy only on recent input).
#
# The liveness signals above are layered onto upstream's viewer model, not in
# place of it: a viewer that is merely attached must never spare a pane, and
# neither new signal may read the tmux clients. Otherwise a tab left open
# overnight pins every idle native stack until the host runs out of memory.
# ---------------------------------------------------------------------------


def _idle_viewer_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    conv_id: str,
) -> tuple[object, TerminalInstance]:
    """A runner app whose pane has a CLI and a web viewer attached, both idle.

    The CLI client attached (its last keypress) eight hours ago, the web
    bridge's last event is equally stale, and ``_list_tmux_clients`` reports
    the attached client — so any code that treated "a client is attached" as
    busy would spare this pane.
    """
    monkeypatch.setattr(native_cost_popup, "_list_tmux_clients", lambda *_args: ["/dev/pts/7"])
    monkeypatch.setattr(
        native_cost_popup, "_tmux_last_client_input_at", lambda *_args: time.time() - 8 * 3600.0
    )
    monkeypatch.setattr(native_cost_popup, "_tmux_window_activity_at", lambda *_args: None)
    monkeypatch.setattr(native_cost_popup, "_tmux_pane_pid", lambda *_args: 4242)
    monkeypatch.setattr(claude_native_bridge, "_APPROVAL_WAIT_ROOT", tmp_path / "approval-waits")
    pane_progress.reset_stream_progress()
    registry = TerminalRegistry()
    instance = TerminalInstance(
        name="claude",
        session_key="main",
        socket_path=tmp_path / "tmux.sock",
        private_dir=tmp_path,
        running=True,
    )
    instance._last_client_interaction_at = time.monotonic() - 8 * 3600.0
    registry._by_conversation[conv_id] = {("claude", "main"): instance}
    app = create_runner_app(
        terminal_registry=registry,
        resource_registry=SessionResourceRegistry(terminal_registry=registry),
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )
    return app, instance


async def test_attached_idle_viewer_without_cpu_or_stream_is_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for #8320: an attached-but-idle viewer does not pin the pane.

    A CLI client and a web viewer are attached, neither has taken input in
    hours, the pane's subtree is idle (flat CPU ticks) and its stream never
    progressed. The process- and stream-level signals must not resurrect the
    "client attached means busy" rule, so the pane is reaped — driven through a
    real reaper loop, including its pre-teardown re-check.
    """
    clock = _FixedClock()
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: {11: 5_000, 12: 100})
    app, _instance = _idle_viewer_app(monkeypatch, tmp_path, "conv_viewed_idle")
    app.state.native_pane_cpu_probe = pane_cpu.PaneDescendantCpuProbe(clock=clock)
    pane = PaneRef(
        "conv_viewed_idle",
        terminal_resource_id("claude", "main"),
        "claude",
        tmp_path / "tmux.sock",
    )

    busy = app.state.native_pane_reaper._is_busy
    assert not await busy(pane)  # CPU baseline
    clock.now += 60.0
    assert not await busy(pane)  # flat ticks: no CPU evidence either

    reaped: list[str] = []
    reaper = NativePaneReaper(
        list_native_panes=lambda: [pane],
        is_busy=busy,
        reap=lambda p: _record_reaped(reaped, p),
        idle_timeout_s=0.0001,
        reaper_interval_s=0.01,
    )
    await reaper._scan_once()  # arms the idle clock
    await asyncio.sleep(0.01)
    clock.now += 60.0
    await reaper._scan_once()  # window elapsed -> reap
    await reaper.shutdown()

    assert reaped == ["conv_viewed_idle"]


async def test_idle_viewer_pane_is_spared_only_by_real_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an idle viewer attached, only stream, CPU or fresh input spare the pane.

    Proves the new signals compose with #8320 rather than replace it: each one
    independently flips the verdict to busy, and removing it flips it back.
    """
    clock = _FixedClock()
    ticks = iter([{11: 5_000}, {11: 11_000}, {11: 11_000}])
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: next(ticks))
    app, instance = _idle_viewer_app(monkeypatch, tmp_path, "conv_viewed_live")
    app.state.native_pane_cpu_probe = pane_cpu.PaneDescendantCpuProbe(clock=clock)
    busy = app.state.native_pane_reaper._is_busy
    pane = PaneRef(
        "conv_viewed_live",
        terminal_resource_id("claude", "main"),
        "claude",
        tmp_path / "tmux.sock",
    )

    assert not await busy(pane)  # idle viewer + CPU baseline: not busy
    # A descendant pinning a core spares it...
    clock.now += 60.0
    assert await busy(pane)
    # ...and once that work stops, the idle viewer alone does not.
    clock.now += 60.0
    assert not await busy(pane)
    # Fresh stream bytes spare it.
    pane_progress.note_stream_progress("conv_viewed_live")
    assert await busy(pane)
    pane_progress.clear_stream_progress("conv_viewed_live")
    # A fresh web-bridge event (#8320's own signal) still spares it.
    instance.note_client_interaction()
    assert await busy(pane)


async def test_output_busy_window_env_reaches_the_busy_predicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S`` changes the predicate's verdict.

    host/connect.py allowlists the knob into the runner env, so it must not be
    inert: a 10-minute window spares a pane whose stream or tmux output last
    moved 5 minutes ago, which the 120s default would not. The viewer-input
    signal deliberately stays on the fixed default, so input 5 minutes old does
    not spare the pane even under the widened window.
    """
    five_min_ago_mono = time.monotonic() - 300.0
    five_min_ago_wall = time.time() - 300.0
    window_activity: dict[str, float | None] = {"at": None}
    input_at: dict[str, float | None] = {"at": None}

    def _build(window: str | None) -> object:
        if window is None:
            monkeypatch.delenv("OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S", raising=False)
        else:
            monkeypatch.setenv("OMNIGENT_PANE_OUTPUT_BUSY_WINDOW_S", window)
        # The predicate binds the probes by name at app build, so stub first.
        monkeypatch.setattr(
            native_cost_popup, "_tmux_window_activity_at", lambda *_args: window_activity["at"]
        )
        monkeypatch.setattr(
            native_cost_popup, "_tmux_last_client_input_at", lambda *_args: input_at["at"]
        )
        monkeypatch.setattr(native_cost_popup, "_tmux_pane_pid", lambda *_args: None)
        monkeypatch.setattr(
            claude_native_bridge, "_APPROVAL_WAIT_ROOT", tmp_path / "approval-waits"
        )
        registry = TerminalRegistry()
        return create_runner_app(
            terminal_registry=registry,
            resource_registry=SessionResourceRegistry(terminal_registry=registry),
            server_client=NullServerClient(),  # type: ignore[arg-type]
        )

    pane = _pane("conv_window")

    for window, expect_spared in ((None, False), ("600", True)):
        busy = _build(window).state.native_pane_reaper._is_busy
        pane_progress.reset_stream_progress()
        pane_progress.note_stream_progress("conv_window", now=five_min_ago_mono)
        assert await busy(pane) is expect_spared, f"stream, window={window}"
        pane_progress.reset_stream_progress()
        window_activity["at"] = five_min_ago_wall
        assert await busy(pane) is expect_spared, f"tmux output, window={window}"
        window_activity["at"] = None
        input_at["at"] = five_min_ago_wall
        assert not await busy(pane), f"viewer input, window={window}"
        input_at["at"] = None


async def test_reaped_claude_pane_leaves_no_forwarder_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reaping a non-codex pane cancels its transcript forwarder.

    The codex teardown helper is a no-op for every other harness, and the claude
    forwarder restarts forever, so without an explicit cancel each reaped claude
    pane left a task polling its dead bridge dir for the runner's lifetime.
    """
    from omnigent.runner.native import orchestration

    app = _silent_pane_app(monkeypatch, tmp_path)
    reaper = app.state.native_pane_reaper
    parked = asyncio.Event()

    async def _forwarder() -> None:
        parked.set()
        await asyncio.Event().wait()

    task: asyncio.Task[object] = asyncio.create_task(_forwarder())
    orchestration._register_auto_forwarder_task("conv_reaped_claude", task)
    try:
        await parked.wait()
        await reaper._reap(_pane("conv_reaped_claude", "claude"))

        assert task.cancelled()
        assert "conv_reaped_claude" not in orchestration._AUTO_FORWARDER_TASKS
        # Idempotent: a second reap with nothing registered is a no-op.
        await reaper._reap(_pane("conv_reaped_claude", "claude"))
    finally:
        task.cancel()
        orchestration._AUTO_FORWARDER_TASKS.pop("conv_reaped_claude", None)


class _SessionStatusClient(NullServerClient):
    """Server client whose session GET answers with a scripted outcome.

    ``outcome`` is a status string (a 200 carrying that status), an int (that
    HTTP status with an empty body), or an exception to raise.
    """

    def __init__(self) -> None:
        self.outcome: str | int | Exception = "idle"

    async def get(self, url: str, **kwargs: object) -> object:  # type: ignore[override]
        del url, kwargs
        outcome = self.outcome
        if isinstance(outcome, Exception):
            raise outcome
        status_code = outcome if isinstance(outcome, int) else 200
        body = {} if isinstance(outcome, int) else {"status": outcome}
        return SimpleNamespace(status_code=status_code, json=lambda: body)


@pytest.mark.parametrize(
    ("outcome", "verdict"),
    [
        ("idle", False),
        ("running", True),
        ("waiting", True),
        (503, None),
        (httpx.ReadError("connection reset"), None),
        (httpx.ConnectTimeout("timed out"), None),
    ],
)
async def test_codex_status_check_is_tri_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str | int | Exception,
    verdict: bool | None,
) -> None:
    """A failed codex status check is unknown, never busy and never idle."""
    client = _SessionStatusClient()
    client.outcome = outcome
    app = _silent_pane_app(monkeypatch, tmp_path, server_client=client)

    assert await app.state.native_pane_reaper._is_busy(_pane("conv_codex", "codex")) is verdict


async def test_status_check_failures_do_not_reset_the_idle_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic GET failures neither re-arm the idle clock nor trigger a reap.

    Treating a failed check as busy re-armed the clock, so a codex session idle
    for well past the window survived ~1.6h on bursts of ReadError and
    ConnectTimeout. The first scan that confirms idle must reap it.
    """
    client = _SessionStatusClient()
    app = _silent_pane_app(monkeypatch, tmp_path, server_client=client)
    pane = _pane("conv_flaky", "codex")
    reaped: list[str] = []
    reaper = NativePaneReaper(
        list_native_panes=lambda: [pane],
        is_busy=app.state.native_pane_reaper._is_busy,
        reap=lambda p: _record_reaped(reaped, p),
        idle_timeout_s=600.0,
        reaper_interval_s=0.01,
    )
    stale = time.monotonic() - 3600.0
    reaper._last_busy_at["conv_flaky"] = stale

    for failure in (httpx.ReadError("reset"), httpx.ConnectTimeout("slow"), 502):
        client.outcome = failure
        await reaper._scan_once()
        assert reaped == []
        assert reaper._last_busy_at["conv_flaky"] == stale

    client.outcome = "idle"
    await reaper._scan_once()
    assert reaped == ["conv_flaky"]


def _spinning_title(*_args: object) -> float:
    """tmux's window_activity for a pane whose title spinner writes ~20x a second."""
    return time.time()


async def test_idle_codex_pane_with_spinning_title_is_reapable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server's idle verdict beats tmux's activity clock for codex.

    An idle codex TUI can keep a stuck braille spinner in its terminal title.
    tmux counts each title write as window activity, so the pane looked busy
    forever on a byte-identical screen while the server reported it idle.
    """
    client = _SessionStatusClient()
    client.outcome = "idle"
    app = _silent_pane_app(
        monkeypatch, tmp_path, server_client=client, window_activity_at=_spinning_title
    )
    busy = app.state.native_pane_reaper._is_busy
    pane = _pane("conv_spinner", "codex")

    assert await busy(pane) is False
    reaped: list[str] = []
    reaper = NativePaneReaper(
        list_native_panes=lambda: [pane],
        is_busy=busy,
        reap=lambda p: _record_reaped(reaped, p),
        idle_timeout_s=0.0001,
        reaper_interval_s=0.01,
    )
    await reaper._scan_once()  # arms the idle clock
    await asyncio.sleep(0.01)
    await reaper._scan_once()  # window elapsed -> reap
    assert reaped == ["conv_spinner"]


@pytest.mark.parametrize(
    ("status", "verdict"),
    [("running", True), ("waiting", True), (httpx.ReadError("reset"), None)],
)
async def test_codex_pane_server_reports_working_stays_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str | Exception,
    verdict: bool | None,
) -> None:
    """With the same fresh output, codex follows its server, not the clock."""
    client = _SessionStatusClient()
    client.outcome = status
    app = _silent_pane_app(
        monkeypatch, tmp_path, server_client=client, window_activity_at=_spinning_title
    )

    assert await app.state.native_pane_reaper._is_busy(_pane("conv_working", "codex")) is verdict


async def test_output_clock_still_spares_non_codex_panes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only codex defers to its server; other harnesses keep the output signal."""
    client = _SessionStatusClient()
    client.outcome = "idle"
    app = _silent_pane_app(
        monkeypatch, tmp_path, server_client=client, window_activity_at=_spinning_title
    )

    assert await app.state.native_pane_reaper._is_busy(_pane("conv_claude", "claude")) is True
