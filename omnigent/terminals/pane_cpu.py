"""Child-process CPU liveness for native harness panes (issue #1349).

The pane reaper's busy predicate had no process-level check at all: its evidence
was an in-flight runner turn, a ``running`` PTY status, an attached tmux client,
and the tmux window's output clock. A worker blocked inside **one long silent
child process** — ``mypy .``, ``alembic upgrade head``, a full ``pytest`` run —
emits no bytes and reports no turn, so every one of those signals reads idle and
a perfectly healthy pane gets torn down.

This module supplies the missing evidence: *does the pane's process tree contain
a descendant that is actually burning CPU?* A pane whose subtree consumed real
CPU since the previous scan is working, however quiet its terminal is.

Method
------
Each sample walks the descendants of the tmux pane's process (the pane process
itself is the harmless always-present login shell and is excluded) and sums
``utime + stime + cutime + cstime`` from ``/proc/<pid>/stat`` — the last two
fields carry the CPU of *reaped* children, which is what catches a build or test
run that spawns many short-lived workers none of which survive between two
60-second scans. Summing a parent's ``cutime`` while also counting a live child
double-counts, but only ever *inflates* the measured busy-ness, i.e. it biases
toward not reaping, which is the established fail-safe direction.

The sum is cached per conversation with its timestamp, so the next scan turns two
samples into a rate without the probe ever sleeping: the predicate must not block.
A conversation with no cached baseline (its very first scan) records one and
**abstains** — it reports "no evidence", not "busy". Abstaining is safe: a pane
must read idle on every signal for the full idle window (twenty scans at the
defaults) before it is reapable, and ``NativePaneReaper._classify`` already grants
a newly-observed pane one full window of grace, so a cold cache can never be the
thing that decides a reap. Reporting "busy" instead would make a runner that
restarts frequently pin every pane forever.

Platforms without ``/proc`` (macOS) find no descendants and the probe abstains
permanently, leaving the pre-existing signals exactly as they were. It never
turns the reaper into a no-op there.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

_logger = logging.getLogger(__name__)

_PROC_FS = "/proc"

# Fraction of ONE core a pane's descendants must average, across the interval
# between two scans, to count as actively working.
#
# Why 5%: the workloads this exists to protect (a compiler, a type checker, a
# test suite, a migration) pin a core at or near 100%, and even a heavily
# I/O-bound one — a slow ``alembic upgrade`` waiting on the database — stays well
# clear of this line once averaged over a full scan interval. The noise floor it
# has to clear is an idle vendor TUI: a select/poll loop with a redraw timer and
# a spinner, which costs a fraction of a percent. At the 60s default interval 5%
# means 3 CPU-seconds of real work between scans — orders of magnitude above the
# idle floor, and far below anything doing genuine work. The gap between those
# two populations is wide enough that the exact number is not load-bearing;
# picking the low end of it keeps the bias toward sparing the pane.
_DEFAULT_MIN_CPU_FRACTION = 0.05

# Bound on the pane->descendant walk so a pathological or cyclic process table
# cannot make it run unbounded. A real pane subtree is shallow (shell -> vendor
# CLI -> the child it is blocked on), far under this. Mirrors the bound in
# ``harnesses/antigravity_native/rpc.py``'s pane-subtree walk.
_MAX_SUBTREE_DEPTH = 32
_MAX_SUBTREE_NODES = 512

# Cap on conversations holding a cached CPU baseline, matching
# ``pane_progress._MAX_TRACKED``. Baselines are normally dropped by ``forget`` on
# reap and on session cleanup; this is the backstop for a conversation that
# vanishes without either (a crashed runner path, a pane closed out from under
# us). Evicted oldest-sample-first, so eviction only ever discards a
# conversation nobody has scanned in a long time -- and the cost of evicting a
# live one is a single extra cold-cache scan, which abstains.
_MAX_TRACKED = 512

# Clock ticks per second for /proc/<pid>/stat's utime/stime fields. Resolved once
# at import; POSIX guarantees it is constant for the life of the process.
try:
    _CLOCK_TICKS_PER_S = float(os.sysconf("SC_CLK_TCK"))
except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
    _CLOCK_TICKS_PER_S = 100.0
if _CLOCK_TICKS_PER_S <= 0:  # pragma: no cover - defensive
    _CLOCK_TICKS_PER_S = 100.0


def _child_pids(pid: int) -> list[int]:
    """Direct child pids of *pid*, read from ``/proc/<pid>/task/*/children``.

    Linux exposes each thread's children directly, so this costs a couple of
    small reads rather than a scan of the whole process table. Unreadable or
    vanished entries yield ``[]`` — a process that exits mid-walk simply
    contributes nothing.

    :param pid: Parent process id.
    :returns: Direct child pids, or ``[]`` when none are resolvable.
    """
    children: list[int] = []
    task_dir = Path(_PROC_FS, str(pid), "task")
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return []
    for tid in tids:
        try:
            raw = (task_dir / tid / "children").read_text(encoding="ascii")
        except OSError:
            continue  # thread exited, or CONFIG_PROC_CHILDREN is off
        children.extend(int(tok) for tok in raw.split() if tok.isdigit())
    return children


def _descendant_pids(pane_pid: int) -> list[int]:
    """Breadth-first descendants of *pane_pid*, excluding *pane_pid* itself.

    The pane process is the login shell tmux exec'd into the pane; it is always
    present and always idle, so it is never evidence of work. Everything beneath
    it — the vendor CLI and whatever that CLI is currently blocked on — is.

    Depth- and node-bounded, and visited-tracked, so a recycled pid or a cyclic
    process table cannot hang the walk.

    :param pane_pid: The tmux pane's process id.
    :returns: Descendant pids (possibly empty).
    """
    seen: set[int] = {pane_pid}
    out: list[int] = []
    frontier = [pane_pid]
    for _depth in range(_MAX_SUBTREE_DEPTH):
        if not frontier or len(out) >= _MAX_SUBTREE_NODES:
            break
        next_frontier: list[int] = []
        for parent in frontier:
            for child in _child_pids(parent):
                if child in seen:
                    continue
                seen.add(child)
                out.append(child)
                next_frontier.append(child)
                if len(out) >= _MAX_SUBTREE_NODES:
                    break
            if len(out) >= _MAX_SUBTREE_NODES:
                break
        frontier = next_frontier
    return out


def _cpu_ticks(pid: int) -> int:
    """Total CPU ticks charged to *pid*, including its reaped children.

    Reads ``/proc/<pid>/stat`` and sums ``utime + stime + cutime + cstime``. The
    ``comm`` field (field 2) is parenthesized and may itself contain spaces and
    parens, so fields are indexed from the LAST ``)`` rather than by a naive
    whitespace split — the same parsing care ``rpc._child_pids_from_proc`` takes.
    After that close paren the fields are ``state utime stime cutime cstime`` at
    offsets 0, 11, 12, 13, 14.

    :param pid: Process id to read.
    :returns: Ticks, or ``0`` when the process vanished or ``/proc`` is absent.
    """
    try:
        stat = Path(_PROC_FS, str(pid), "stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return 0  # exited between the walk and this read, or no /proc
    close_paren = stat.rfind(")")
    if close_paren == -1:
        return 0
    fields = stat[close_paren + 1 :].split()
    if len(fields) < 15:
        return 0
    total = 0
    for index in (11, 12, 13, 14):
        try:
            total += int(fields[index])
        except ValueError:
            return 0
    return total


def _subtree_cpu_ticks(pane_pid: int) -> dict[int, int]:
    """CPU ticks PER descendant of *pane_pid*.

    Per-pid rather than one aggregate for two reasons: a descendant exiting can
    no longer mask a sibling's work behind a shrinking grand total (each pid's
    delta is clamped independently), and the busiest pid is available to name in
    the log line the reaper emits when this signal alone spares a pane.

    :param pane_pid: The tmux pane's process id.
    :returns: ``{pid: ticks}``, empty when there are no descendants.
    """
    return {pid: _cpu_ticks(pid) for pid in _descendant_pids(pane_pid)}


class CpuActivity(NamedTuple):
    """One descendant-CPU verdict, with the evidence behind it.

    :param active: ``True`` only on positive evidence of CPU work. ``False``
        means "this signal has no evidence", never "the pane is dead".
    :param cpu_fraction: Fraction of one core the whole subtree averaged over the
        interval, or ``0.0`` when no rate could be computed.
    :param busiest_pid: The descendant that contributed the most CPU, or ``None``.
    :param busiest_fraction: That descendant's own fraction of one core.
    :param descendants: How many descendants were sampled.
    """

    active: bool
    cpu_fraction: float = 0.0
    busiest_pid: int | None = None
    busiest_fraction: float = 0.0
    descendants: int = 0


#: The "no evidence" verdict, returned whenever no rate can be derived.
NO_CPU_ACTIVITY = CpuActivity(active=False)


class PaneDescendantCpuProbe:
    """Per-conversation CPU-rate probe over a pane's descendant processes.

    Holds one cached ``(timestamp, ticks)`` baseline per conversation so a rate
    can be derived from two successive reaper scans without the probe ever
    sleeping. :meth:`is_cpu_active` does synchronous ``/proc`` reads and is meant
    to be called via :func:`asyncio.to_thread` from the async busy predicate.

    :param min_cpu_fraction: Fraction of one core the descendants must average to
        count as active. Defaults to :data:`_DEFAULT_MIN_CPU_FRACTION`.
    :param clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        *,
        min_cpu_fraction: float = _DEFAULT_MIN_CPU_FRACTION,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._min_cpu_fraction = min_cpu_fraction
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        # conversation_id -> (monotonic sample time, {descendant pid: ticks}).
        # Insertion order tracks recency so the bound below evicts oldest-first.
        self._samples: dict[str, tuple[float, dict[int, int]]] = {}

    def is_cpu_active(self, conversation_id: str, pane_pid: int | None) -> CpuActivity:
        """Whether *conversation_id*'s pane has a descendant burning CPU.

        :param conversation_id: AP-allocated conversation id, e.g. ``"conv_abc"``.
        :param pane_pid: The tmux pane's process id, or ``None`` when it could not
            be resolved (the pane is gone, or ``tmux`` failed) — then there is
            nothing to measure and the probe abstains.
        :returns: A :class:`CpuActivity`. ``active`` is ``True`` only on positive
            evidence of CPU work; ``False`` means "this signal has no evidence",
            never "the pane is dead" — the caller's other signals still apply.
        """
        if pane_pid is None:
            self._samples.pop(conversation_id, None)
            return NO_CPU_ACTIVITY
        now = self._clock()
        ticks = _subtree_cpu_ticks(pane_pid)
        previous = self._samples.pop(conversation_id, None)
        self._samples[conversation_id] = (now, ticks)
        while len(self._samples) > _MAX_TRACKED:
            del self._samples[next(iter(self._samples))]
        if not ticks:
            # Nothing under the shell at all: no /proc, or the vendor CLI is gone.
            return NO_CPU_ACTIVITY
        if previous is None:
            # First sample: baseline recorded, abstain (see the module docstring).
            return NO_CPU_ACTIVITY
        last_at, last_ticks = previous
        elapsed = now - last_at
        if elapsed <= 0:
            return NO_CPU_ACTIVITY
        total_delta = 0
        busiest_pid: int | None = None
        busiest_delta = 0
        for pid, value in ticks.items():
            # Clamped per pid: a descendant whose total shrinks (it exited and a
            # recycled pid took its place) is not evidence of work, and must not
            # cancel out a sibling that really is working. A pid with no previous
            # sample started within the interval, so all of its ticks accrued
            # inside it and counting them in full is correct.
            delta = max(0, value - last_ticks.get(pid, 0))
            total_delta += delta
            if delta > busiest_delta:
                busiest_delta, busiest_pid = delta, pid
        scale = _CLOCK_TICKS_PER_S * elapsed
        activity = CpuActivity(
            active=(total_delta / scale) >= self._min_cpu_fraction,
            cpu_fraction=total_delta / scale,
            busiest_pid=busiest_pid,
            busiest_fraction=busiest_delta / scale,
            descendants=len(ticks),
        )
        if activity.active:
            _logger.debug(
                "pane cpu probe: conversation %s busy (%.1f%% of a core over %.1fs, "
                "%d descendants, busiest pid %s at %.1f%%)",
                conversation_id,
                activity.cpu_fraction * 100.0,
                elapsed,
                activity.descendants,
                activity.busiest_pid,
                activity.busiest_fraction * 100.0,
            )
        return activity

    def forget(self, conversation_id: str) -> None:
        """Drop *conversation_id*'s cached baseline (its pane or session is gone)."""
        self._samples.pop(conversation_id, None)
