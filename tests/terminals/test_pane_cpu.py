"""Tests for the pane descendant-CPU liveness probe (issue #1349)."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest

from omnigent.terminals import pane_cpu
from omnigent.terminals.pane_cpu import PaneDescendantCpuProbe


class _Clock:
    """Injected monotonic clock so a CPU *rate* is testable without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _probe_with(
    monkeypatch: pytest.MonkeyPatch,
    samples: list[dict[int, int]],
    clock: _Clock,
    **kwargs: float,
) -> PaneDescendantCpuProbe:
    """A probe whose per-pid subtree sampler yields *samples* in order."""
    queue = list(samples)
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: queue.pop(0))
    return PaneDescendantCpuProbe(clock=clock, **kwargs)


def test_first_sample_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold cache records a baseline and reports no evidence, not busy.

    Reporting busy on a cold cache would pin every pane forever on a runner that
    restarts often; abstaining is safe because a pane still has to read idle for
    the whole idle window (twenty scans at the defaults) before it is reapable.
    """
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{11: 5_000, 12: 100}], clock)
    assert probe.is_cpu_active("conv_a", 4242).active is False


def test_busy_descendant_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A descendant pinning a core across the scan interval reads busy."""
    clock = _Clock()
    # 60s later, +6000 ticks == 60 CPU-seconds == 100% of one core.
    probe = _probe_with(monkeypatch, [{11: 5_000, 12: 100}, {11: 11_000, 12: 100}], clock)
    assert probe.is_cpu_active("conv_a", 4242).active is False
    clock.now += 60.0
    assert probe.is_cpu_active("conv_a", 4242).active is True


def test_idle_descendant_below_threshold_is_not_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle TUI's housekeeping (well under 5% of a core) is not work."""
    clock = _Clock()
    # 60s later, +60 ticks == 0.6 CPU-seconds == 1% of one core.
    probe = _probe_with(monkeypatch, [{11: 5_000, 12: 100}, {11: 5_060, 12: 100}], clock)
    assert probe.is_cpu_active("conv_a", 4242).active is False
    clock.now += 60.0
    assert probe.is_cpu_active("conv_a", 4242).active is False


def test_threshold_is_the_documented_five_percent() -> None:
    """The committed threshold is 5% of one core (see the module rationale)."""
    assert pane_cpu._DEFAULT_MIN_CPU_FRACTION == 0.05


def test_threshold_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same 1%-of-a-core sample reads busy under a 0.5% threshold."""
    clock = _Clock()
    probe = _probe_with(
        monkeypatch, [{11: 5_000, 12: 100}, {11: 5_060, 12: 100}], clock, min_cpu_fraction=0.005
    )
    probe.is_cpu_active("conv_a", 4242)
    clock.now += 60.0
    assert probe.is_cpu_active("conv_a", 4242).active is True


def test_no_descendants_is_not_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing under the pane shell (or no ``/proc``) is no evidence of work."""
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{}, {}], clock)
    assert probe.is_cpu_active("conv_a", 4242).active is False
    clock.now += 60.0
    assert probe.is_cpu_active("conv_a", 4242).active is False


def test_missing_pane_pid_is_not_busy_and_drops_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolvable pane pid leaves nothing to measure; the baseline is dropped."""
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{11: 5_000, 12: 100}], clock)
    probe.is_cpu_active("conv_a", 4242)
    assert "conv_a" in probe._samples
    assert probe.is_cpu_active("conv_a", None).active is False
    assert "conv_a" not in probe._samples


def test_tick_total_dropping_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-heavy descendant exiting shrinks the total; that is not activity."""
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{11: 9_000}, {11: 1_000}], clock)
    probe.is_cpu_active("conv_a", 4242)
    clock.now += 60.0
    assert probe.is_cpu_active("conv_a", 4242).active is False


def test_zero_elapsed_is_not_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two samples at the same instant yield no rate, so no evidence."""
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{11: 5_000}, {11: 99_000}], clock)
    probe.is_cpu_active("conv_a", 4242)
    assert probe.is_cpu_active("conv_a", 4242).active is False


def test_forget_drops_the_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{11: 5_000, 12: 100}], clock)
    probe.is_cpu_active("conv_a", 4242)
    probe.forget("conv_a")
    assert "conv_a" not in probe._samples


def test_baselines_are_per_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    """One pane's CPU burn never spares another pane."""
    clock = _Clock()
    counts = {"conv_hot": [5_000, 11_000], "conv_cold": [5_000, 5_000]}

    def _sample(pid: int) -> dict[int, int]:
        key = "conv_hot" if pid == 1 else "conv_cold"
        return {pid * 100: counts[key].pop(0)}

    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", _sample)
    probe = PaneDescendantCpuProbe(clock=clock)
    probe.is_cpu_active("conv_hot", 1)
    probe.is_cpu_active("conv_cold", 2)
    clock.now += 60.0
    assert probe.is_cpu_active("conv_hot", 1).active is True
    assert probe.is_cpu_active("conv_cold", 2).active is False


def test_subtree_walk_is_node_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pathological process table cannot make the descendant walk unbounded."""
    monkeypatch.setattr(pane_cpu, "_child_pids", lambda pid: [pid * 2, pid * 2 + 1])
    assert len(pane_cpu._descendant_pids(1)) <= pane_cpu._MAX_SUBTREE_NODES


def test_cycle_in_process_table_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parent reported as its own descendant does not hang the walk."""
    monkeypatch.setattr(pane_cpu, "_child_pids", lambda pid: [1, 2])
    assert pane_cpu._descendant_pids(1) == [2]


def test_cpu_ticks_handles_comm_with_spaces_and_parens(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``/proc/<pid>/stat``'s comm field is parsed from the LAST ``)``."""
    proc = tmp_path / "proc"
    (proc / "7").mkdir(parents=True)
    # comm deliberately contains a space and a close paren.
    fields = " ".join(str(n) for n in range(3, 53))
    (proc / "7" / "stat").write_text(f"7 (weird ) name) S {fields}\n")
    monkeypatch.setattr(pane_cpu, "_PROC_FS", str(proc))
    # After the last ')': state(0) then fields 3..; utime/stime/cutime/cstime sit
    # at offsets 11..14, i.e. the values 13, 14, 15, 16.
    assert pane_cpu._cpu_ticks(7) == 13 + 14 + 15 + 16


def test_cpu_ticks_missing_process_is_zero(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A process that exits between the walk and the read contributes nothing."""
    monkeypatch.setattr(pane_cpu, "_PROC_FS", str(tmp_path))
    assert pane_cpu._cpu_ticks(999_999) == 0


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="Linux /proc required")
def test_real_descendants_and_cpu_are_observed() -> None:
    """End-to-end against the real process table: a busy grandchild is seen.

    Guards the ``/proc`` parsing itself — the unit tests above stub the sampler,
    so nothing else would catch a wrong ``stat`` field offset or a children-file
    path typo.
    """
    burner = f"{sys.executable} -c 'import time; t=time.time()\nwhile time.time()-t<3: pass'"
    # Own process group so the whole subtree dies with it: killing the shell
    # alone would strand the busy grandchild for the rest of the run.
    parent = subprocess.Popen(["/bin/sh", "-c", burner], start_new_session=True)
    try:
        deadline = time.monotonic() + 5.0
        descendants: list[int] = []
        while time.monotonic() < deadline:
            descendants = pane_cpu._descendant_pids(parent.pid)
            if descendants:
                break
            time.sleep(0.05)
        assert descendants, "the busy grandchild was not found under the shell"
        probe = PaneDescendantCpuProbe()
        assert probe.is_cpu_active("conv_real", parent.pid).active is False  # baseline
        time.sleep(0.5)
        assert probe.is_cpu_active("conv_real", parent.pid).active is True
    finally:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(parent.pid), signal.SIGKILL)
        parent.wait(timeout=5)


def test_exiting_hot_descendant_does_not_mask_a_working_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-pid clamping: one descendant leaving must not cancel another's work.

    With a single aggregate total, pid 11 (9000 ticks) exiting while pid 12 does
    60 CPU-seconds of real work nets out NEGATIVE and reads idle.
    """
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{11: 9_000, 12: 100}, {12: 6_100}], clock)
    probe.is_cpu_active("conv_a", 4242)
    clock.now += 60.0
    activity = probe.is_cpu_active("conv_a", 4242)
    assert activity.active is True
    assert activity.busiest_pid == 12


def test_activity_reports_the_busiest_descendant(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verdict names the culprit pid and its rate, for the reaper's log line."""
    clock = _Clock()
    probe = _probe_with(
        monkeypatch,
        [{11: 0, 12: 0}, {11: 600, 12: 5_400}],
        clock,
    )
    probe.is_cpu_active("conv_a", 4242)
    clock.now += 60.0
    activity = probe.is_cpu_active("conv_a", 4242)
    assert activity.active is True
    assert activity.busiest_pid == 12
    assert activity.descendants == 2
    # 6000 ticks / 100 per s / 60s == 100% of a core; pid 12 supplied 90% of it.
    assert activity.cpu_fraction == pytest.approx(1.0)
    assert activity.busiest_fraction == pytest.approx(0.9)


def test_no_evidence_verdict_carries_no_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """An abstaining verdict reports no pid and no rate to log."""
    clock = _Clock()
    probe = _probe_with(monkeypatch, [{11: 5_000}], clock)
    activity = probe.is_cpu_active("conv_a", 4242)
    assert activity == pane_cpu.NO_CPU_ACTIVITY
    assert activity.busiest_pid is None
    assert activity.cpu_fraction == 0.0


def test_sample_map_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseline map evicts oldest-first so it cannot grow without bound.

    ``forget`` covers the reap and session-cleanup paths; this is the backstop for
    a conversation that vanishes without either.
    """
    clock = _Clock()
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: {11: 1})
    probe = PaneDescendantCpuProbe(clock=clock)
    for index in range(pane_cpu._MAX_TRACKED + 10):
        clock.now += 1.0
        probe.is_cpu_active(f"conv_{index}", 4242)
    assert len(probe._samples) == pane_cpu._MAX_TRACKED
    assert "conv_0" not in probe._samples
    assert f"conv_{pane_cpu._MAX_TRACKED + 9}" in probe._samples


def test_rescanning_a_conversation_refreshes_its_eviction_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conversation scanned every cycle is never the one evicted."""
    clock = _Clock()
    monkeypatch.setattr(pane_cpu, "_subtree_cpu_ticks", lambda _pid: {11: 1})
    probe = PaneDescendantCpuProbe(clock=clock)
    probe.is_cpu_active("conv_keep", 4242)
    for index in range(pane_cpu._MAX_TRACKED + 10):
        clock.now += 1.0
        probe.is_cpu_active(f"conv_{index}", 4242)
        probe.is_cpu_active("conv_keep", 4242)
    assert "conv_keep" in probe._samples
