"""Process-wide stream-progress ledger for native harness panes (issue #1349).

The native-pane reaper's liveness evidence is entirely *output-shaped*: an
in-flight runner turn, a ``running`` PTY status, an attached tmux client, or a
byte emitted into the tmux window recently. During a long autonomous turn the
first two go quiet, so the only thing standing between a healthy worker and
teardown is "did the pane emit a byte lately?".

The qwen-native forwarder (:mod:`omnigent.harnesses.qwen_native.forwarder`)
already tails qwen's ``--json-file`` stream every 0.4s and mirrors each new
message into the session. That is a genuine token-level progress signal the
reaper was ignoring. This module is the seam between the two: the forwarder
stamps a timestamp here whenever it consumes new stream bytes, and the reaper's
busy predicate treats a recent stamp as "working, do not reap".

Why not ``HarnessProcessManager.note_activity``: that method refreshes
``entry.last_used_at`` for a **registered harness subprocess** and is an explicit
no-op when the conversation has none (``process_manager.py`` ``note_activity``).
A native qwen pane runs its TUI inside a runner-owned tmux pane, not as a
manager-spawned proxy subprocess, so for a native-only conversation the call
does nothing at all. It also has no public reader — the pane predicate would have
to reach into the manager's private ``_entries`` to observe it. So this is a
small dedicated ledger rather than a reuse of that seam.

Clock: ``time.monotonic()``, deliberately. ``process_manager.get_client`` documents
why (``process_manager.py``): the forwarder and the reaper can run on different
event loops whose ``loop.time()`` bases disagree across a host sleep, which once
produced bogus multi-hour diffs and reaped actively-streaming sessions.
``time.monotonic()`` is the one process-wide source both sides agree on.
"""

from __future__ import annotations

import time

# Cap on tracked conversations so a long-lived runner churning through sessions
# cannot grow this map without bound. Entries are evicted oldest-stamp-first;
# the cap is far above any plausible concurrent native-pane count, so eviction
# only ever discards conversations that stopped progressing long ago.
_MAX_TRACKED = 512

# conversation_id -> monotonic timestamp of the last observed stream progress.
_last_progress_at: dict[str, float] = {}


def note_stream_progress(conversation_id: str, *, now: float | None = None) -> None:
    """Record that *conversation_id*'s harness stream just made progress.

    Called from the qwen forwarder's tail loop each time it consumes new bytes
    from the stream-json file. Cheap enough (one dict write) to call at the
    forwarder's 0.4s cadence.

    :param conversation_id: AP-allocated conversation id, e.g. ``"conv_abc123"``.
    :param now: Monotonic timestamp override (tests); defaults to
        :func:`time.monotonic`.
    """
    stamp = time.monotonic() if now is None else now
    # Re-insert so dict insertion order tracks recency, making the eviction
    # below drop the least-recently-progressing conversation.
    _last_progress_at.pop(conversation_id, None)
    _last_progress_at[conversation_id] = stamp
    while len(_last_progress_at) > _MAX_TRACKED:
        oldest = next(iter(_last_progress_at))
        del _last_progress_at[oldest]


def stream_progress_age_s(conversation_id: str, *, now: float | None = None) -> float | None:
    """Seconds since *conversation_id*'s stream last made progress.

    :param conversation_id: AP-allocated conversation id, e.g. ``"conv_abc123"``.
    :param now: Monotonic timestamp override (tests).
    :returns: Age in seconds, or ``None`` when no progress was ever recorded for
        this conversation (no evidence either way — the caller must fall through
        to its other signals rather than read ``None`` as "idle").
    """
    last = _last_progress_at.get(conversation_id)
    if last is None:
        return None
    stamp = time.monotonic() if now is None else now
    # A clock that appears to run backwards (never expected from monotonic, but
    # cheap to neutralize) reads as "just progressed", i.e. do not reap.
    return max(0.0, stamp - last)


def clear_stream_progress(conversation_id: str) -> None:
    """Forget *conversation_id*'s progress stamp (pane torn down / session ended)."""
    _last_progress_at.pop(conversation_id, None)


def reset_stream_progress() -> None:
    """Drop every recorded stamp. Test-only hygiene for this module-level state."""
    _last_progress_at.clear()
