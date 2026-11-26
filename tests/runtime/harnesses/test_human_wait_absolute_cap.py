"""A pending human wait holds the absolute turn ceiling open, within a budget.

The absolute ceiling (``HARNESS_TURN_ABSOLUTE_TIMEOUT_S``) is anchored at turn
start and only real progress extends it. A turn parked on a human (an
AskUserQuestion elicitation, a human-gated policy verdict) emits nothing but
heartbeats, so a question asked late in a turn used to be killed at the
ceiling while the human was still answering — "turn exceeded the 10800s
harness absolute watchdog" — and the human's answer was then dropped.

The rules covered here (in-process, driving ``_guarded_run_turn`` directly):

1. a question asked just before the ceiling and answered well after it
   completes;
2. progress after the answer keeps extending the ceiling as before;
3. with the idle watchdog disabled the extension is still bounded — an
   abandoned question terminates and names the human-wait limit, an answered
   one completes;
4. ``HARNESS_HUMAN_WAIT_TIMEOUT_S <= 0`` restores the old behavior;
5. the begin/end bracket stamps and clears the wait start on every exit
   path, for both elicitations and human-gated policy evaluations — and a
   policy-only wait (no elicitation to stamp it) holds the cap on its own;
6. the hold applies as the wait opens, so a cap that falls before the first
   heartbeat of the wait does not fire.

How to run::

    pytest tests/runtime/harnesses/test_human_wait_absolute_cap.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omnigent.runtime.harnesses import _scaffold
from omnigent.runtime.harnesses._scaffold import HarnessApp, TurnContext
from omnigent.server.schemas import (
    ElicitationRequestParams,
    ElicitationResult,
    HeartbeatEvent,
    OutputTextDeltaEvent,
)

_ELICITATION_ID = "elicit_late_1"


def _make_ctx(response_id: str) -> TurnContext:
    return TurnContext(
        response_id=response_id,
        event_queue=asyncio.Queue(),
        cancelled=asyncio.Event(),
    )


def _set_timeouts(
    monkeypatch: pytest.MonkeyPatch, *, idle: float, absolute: float, human_wait: float
) -> None:
    monkeypatch.setattr(_scaffold, "_TURN_IDLE_TIMEOUT_S", idle)
    monkeypatch.setattr(_scaffold, "_TURN_ABSOLUTE_TIMEOUT_S", absolute)
    monkeypatch.setattr(_scaffold, "_HUMAN_WAIT_TIMEOUT_S", human_wait)


class _LateQuestionApp(HarnessApp):
    """Works, asks a question late, then optionally works some more.

    ``_guarded_run_turn`` does not run the scaffold's heartbeat loop, so
    ``run_turn`` emits its own fast heartbeats (the production loop's 15s
    cadence scaled down) for the whole turn.
    """

    def __init__(
        self,
        *,
        ask_after: float,
        work_tick: float,
        post_answer_ticks: int = 0,
        heartbeat: float = 0.05,
    ) -> None:
        super().__init__()
        self._ask_after = ask_after
        self._work_tick = work_tick
        self._post_answer_ticks = post_answer_ticks
        self._heartbeat = heartbeat
        self.answer: str | None = None

    async def _beat(self, ctx: TurnContext) -> None:
        while True:
            await asyncio.sleep(self._heartbeat)
            ctx.emit(HeartbeatEvent(type="response.heartbeat"))

    def _tick(self, ctx: TurnContext, label: str) -> None:
        ctx.emit(OutputTextDeltaEvent(type="response.output_text.delta", delta=label))

    async def run_turn(self, request: Any, ctx: TurnContext) -> None:
        del request
        beat = asyncio.create_task(self._beat(ctx))
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._ask_after
            while (remaining := deadline - loop.time()) > 0:
                self._tick(ctx, "work ")
                await asyncio.sleep(min(self._work_tick, remaining))
            result = await ctx.elicit(
                _ELICITATION_ID,
                ElicitationRequestParams(mode="form", message="which option?"),
            )
            self.answer = result.action
            for _ in range(self._post_answer_ticks):
                await asyncio.sleep(self._work_tick)
                self._tick(ctx, "after ")
            self._tick(ctx, f"action:{result.action}")
        finally:
            beat.cancel()


async def _answer_at(ctx: TurnContext, at: float) -> None:
    """Answer the pending question at loop time *at*."""
    while _ELICITATION_ID not in ctx._pending_elicitations:
        await asyncio.sleep(0.01)
    await asyncio.sleep(max(0.0, at - asyncio.get_running_loop().time()))
    assert ctx._complete_elicitation(_ELICITATION_ID, ElicitationResult(action="accept")), (
        "the question was no longer pending when the human answered — the turn "
        "was torn down while waiting on the human"
    )


async def _run_answered(app: _LateQuestionApp, ctx: TurnContext, answer_after: float) -> float:
    loop = asyncio.get_running_loop()
    started = loop.time()
    answerer = asyncio.create_task(_answer_at(ctx, started + answer_after))
    try:
        async with asyncio.timeout(30):
            await app._guarded_run_turn(None, ctx)  # type: ignore[arg-type]
    finally:
        answerer.cancel()
    return loop.time() - started


async def test_question_asked_near_the_cap_survives_until_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute cap 3s, idle 2s, question at ~2.5s, answer at ~6s -> completes.

    Before the fix the last progress (the question itself, ~2.5s) left the
    ceiling at ~4.5s, so the turn died ~1.5s before the human answered.
    """
    _set_timeouts(monkeypatch, idle=2.0, absolute=3.0, human_wait=86400.0)
    app = _LateQuestionApp(ask_after=2.5, work_tick=1.0)
    ctx = _make_ctx("resp_late_question")

    elapsed = await _run_answered(app, ctx, answer_after=6.0)

    assert app.answer == "accept"
    assert elapsed >= 6.0, f"turn ended before the answer arrived ({elapsed:.1f}s)"
    assert ctx._pending_human_waits == 0
    assert ctx._human_wait_started_at is None, "the wait start is cleared once answered"


async def test_progress_after_the_answer_still_extends_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-answer work longer than one idle window still completes.

    The human-wait hold leaves the ceiling ~one idle window past the answer;
    the ~2s of steady work after it must keep pushing the ceiling via the
    ordinary progress reset, exactly as before the human wait.
    """
    _set_timeouts(monkeypatch, idle=0.5, absolute=1.0, human_wait=86400.0)
    app = _LateQuestionApp(ask_after=0.8, work_tick=0.2, post_answer_ticks=10)
    ctx = _make_ctx("resp_post_answer_work")

    elapsed = await _run_answered(app, ctx, answer_after=1.6)

    assert app.answer == "accept"
    # Answer at ~1.6s plus ~2s of post-answer work: well past the 1s cap and
    # past answer + one idle window.
    assert elapsed >= 3.4, f"post-answer work was cut short ({elapsed:.1f}s)"
    assert ctx._human_wait_started_at is None


async def test_abandoned_question_with_idle_watchdog_disabled_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle watchdog off: an abandoned question dies after budget + one window.

    With no idle window the hold extends the ceiling by the absolute window
    instead, so the turn outlives the 0.5s cap while the 1s budget lasts and
    then terminates at ~1.6s naming the human-wait limit — never hangs.
    """
    _set_timeouts(monkeypatch, idle=0.0, absolute=0.5, human_wait=1.0)
    app = _LateQuestionApp(ask_after=0.1, work_tick=0.05)
    ctx = _make_ctx("resp_idle_off_abandoned")

    loop = asyncio.get_running_loop()
    started = loop.time()
    async with asyncio.timeout(10):
        with pytest.raises(RuntimeError) as excinfo:
            await app._guarded_run_turn(None, ctx)  # type: ignore[arg-type]
    elapsed = loop.time() - started

    # Question at ~0.1s + 1s budget + 0.5s window.
    assert 1.1 <= elapsed < 3.0, f"expected death at ~1.6s, got {elapsed:.2f}s"
    message = str(excinfo.value)
    assert "human-wait limit was reached" in message, message
    assert "absolute watchdog" in message, message
    assert ctx._pending_human_waits == 0, "the cancelled wait must release its bracket"
    assert ctx._human_wait_started_at is None


async def test_answered_question_with_idle_watchdog_disabled_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle watchdog off: a question answered after the strict cap completes."""
    _set_timeouts(monkeypatch, idle=0.0, absolute=0.5, human_wait=10.0)
    app = _LateQuestionApp(ask_after=0.3, work_tick=0.1)
    ctx = _make_ctx("resp_idle_off_answered")

    elapsed = await _run_answered(app, ctx, answer_after=1.2)

    assert app.answer == "accept"
    assert elapsed >= 1.2


async def test_disabled_human_wait_budget_keeps_the_old_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_HUMAN_WAIT_TIMEOUT_S <= 0``: human waits count against the ceiling."""
    _set_timeouts(monkeypatch, idle=0.3, absolute=0.6, human_wait=0.0)
    app = _LateQuestionApp(ask_after=0.0, work_tick=0.1)
    ctx = _make_ctx("resp_budget_off")

    loop = asyncio.get_running_loop()
    started = loop.time()
    async with asyncio.timeout(10):
        with pytest.raises(RuntimeError) as excinfo:
            await app._guarded_run_turn(None, ctx)  # type: ignore[arg-type]
    elapsed = loop.time() - started

    assert elapsed < 1.5, f"expected death at the 0.6s cap, got {elapsed:.2f}s"
    message = str(excinfo.value)
    assert "absolute watchdog" in message, message
    assert "human answer" in message, (
        f"the error should say the turn died waiting on a human; got {message!r}"
    )


async def test_cap_without_a_human_wait_keeps_the_duration_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cap that fires after an answered wait reports the plain duration cap."""
    _set_timeouts(monkeypatch, idle=0.0, absolute=0.3, human_wait=10.0)

    class _AskThenHangApp(_LateQuestionApp):
        async def run_turn(self, request: Any, ctx: TurnContext) -> None:
            await super().run_turn(request, ctx)
            await asyncio.Event().wait()  # outlive the cap after the answer

    app = _AskThenHangApp(ask_after=0.0, work_tick=0.05)
    ctx = _make_ctx("resp_cap_after_answer")
    answerer = asyncio.create_task(_answer_at(ctx, asyncio.get_running_loop().time() + 0.1))
    try:
        async with asyncio.timeout(10):
            with pytest.raises(RuntimeError) as excinfo:
                await app._guarded_run_turn(None, ctx)  # type: ignore[arg-type]
    finally:
        answerer.cancel()

    message = str(excinfo.value)
    assert "total turn duration cap" in message, message
    assert "human" not in message, message


async def test_elicit_bracket_stamps_and_clears_on_cancel() -> None:
    ctx = _make_ctx("resp_bracket_cancel")
    loop = asyncio.get_running_loop()
    before = loop.time()
    task = asyncio.create_task(
        ctx.elicit(_ELICITATION_ID, ElicitationRequestParams(mode="form", message="q?"))
    )
    while _ELICITATION_ID not in ctx._pending_elicitations:
        await asyncio.sleep(0)
    assert ctx._pending_human_waits == 1
    assert ctx._human_wait_started_at is not None
    assert ctx._human_wait_started_at >= before

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ctx._pending_human_waits == 0
    assert ctx._human_wait_started_at is None
    assert ctx._last_human_wait is not None
    started, ended = ctx._last_human_wait
    assert ctx._human_wait_pending_since((started + ended) / 2) == started


async def test_overlapping_waits_keep_the_first_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The budget runs from the first of overlapping waits (elicit + policy)."""
    monkeypatch.setattr(_scaffold, "_UNGATED_POLICY_EVAL_TIMEOUT_S", 0.05)
    ctx = _make_ctx("resp_bracket_overlap")
    elicit = asyncio.create_task(
        ctx.elicit(_ELICITATION_ID, ElicitationRequestParams(mode="form", message="q?"))
    )
    while _ELICITATION_ID not in ctx._pending_elicitations:
        await asyncio.sleep(0)
    first_start = ctx._human_wait_started_at
    # custom-lint: disable-next=sleep-then-assert -- separates the two wait starts in time
    await asyncio.sleep(0.02)

    policy = asyncio.create_task(ctx.evaluate_policy("poleval_1", "PHASE_TOOL_CALL", {}))
    while "poleval_1" not in ctx._pending_policy_evaluations:
        await asyncio.sleep(0)
    assert ctx._pending_human_waits == 2
    assert ctx._human_wait_started_at == first_start

    ctx._complete_policy_evaluation(
        "poleval_1", _scaffold.PolicyVerdictPayload(action="POLICY_ACTION_ALLOW")
    )
    await policy
    assert ctx._pending_human_waits == 1
    assert ctx._human_wait_started_at == first_start, "one wait still pending"

    ctx._complete_elicitation(_ELICITATION_ID, ElicitationResult(action="accept"))
    await elicit
    assert ctx._pending_human_waits == 0
    assert ctx._human_wait_started_at is None


class _PolicyGateApp(HarnessApp):
    """Works briefly, then parks on a human-gated policy verdict (no elicitation).

    Records whether the wait start was stamped while the gate was open.
    """

    def __init__(self, *, ask_after: float, heartbeat: float = 0.05) -> None:
        super().__init__()
        self._ask_after = ask_after
        self._heartbeat = heartbeat
        self.verdict: str | None = None

    async def run_turn(self, request: Any, ctx: TurnContext) -> None:
        del request

        async def _beat() -> None:
            while True:
                await asyncio.sleep(self._heartbeat)
                ctx.emit(HeartbeatEvent(type="response.heartbeat"))

        beat = asyncio.create_task(_beat())
        try:
            ctx.emit(OutputTextDeltaEvent(type="response.output_text.delta", delta="work "))
            await asyncio.sleep(self._ask_after)
            verdict = await ctx.evaluate_policy("poleval_gate", "PHASE_TOOL_CALL", {})
            self.verdict = verdict.action
            ctx.emit(OutputTextDeltaEvent(type="response.output_text.delta", delta="done"))
        finally:
            beat.cancel()


async def test_policy_only_wait_holds_the_cap_until_approved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human-gated policy wait alone stamps the wait and outlives the cap.

    No elicitation in the turn, so nothing else can stamp the wait start: the
    policy path's own bracket must. Without it, heartbeats only hold the idle
    window and the turn dies at the 0.5s cap before the 1.2s approval.
    """
    _set_timeouts(monkeypatch, idle=0.3, absolute=0.5, human_wait=86400.0)
    app = _PolicyGateApp(ask_after=0.1)
    ctx = _make_ctx("resp_policy_only")
    loop = asyncio.get_running_loop()
    started = loop.time()
    stamped_while_open: list[float | None] = []

    async def _approve() -> None:
        while "poleval_gate" not in ctx._pending_policy_evaluations:
            await asyncio.sleep(0.01)
        stamped_while_open.append(ctx._human_wait_started_at)
        await asyncio.sleep(max(0.0, started + 1.2 - loop.time()))
        assert ctx._complete_policy_evaluation(
            "poleval_gate", _scaffold.PolicyVerdictPayload(action="POLICY_ACTION_ALLOW")
        ), "the policy gate was no longer pending when the human approved"

    approver = asyncio.create_task(_approve())
    try:
        async with asyncio.timeout(10):
            await app._guarded_run_turn(None, ctx)  # type: ignore[arg-type]
    finally:
        approver.cancel()
    elapsed = loop.time() - started

    assert stamped_while_open and stamped_while_open[0] is not None, (
        "a human-gated policy wait must stamp _human_wait_started_at when it opens"
    )
    assert app.verdict == "POLICY_ACTION_ALLOW"
    assert elapsed >= 1.2, f"turn ended before the approval arrived ({elapsed:.2f}s)"
    assert ctx._pending_human_waits == 0
    assert ctx._human_wait_started_at is None, "the stamp is cleared once the verdict lands"


async def test_wait_opened_just_before_the_cap_is_held_before_any_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hold applies as the wait opens, not only at the next heartbeat.

    Idle watchdog off (so the request event's progress reset can't help),
    1s cap, question at ~0.8s, heartbeats every 5s (none before the answer),
    answer at ~1.4s. Only the hold applied at wait-open keeps the turn past
    the 1s cap.
    """
    _set_timeouts(monkeypatch, idle=0.0, absolute=1.0, human_wait=86400.0)
    app = _LateQuestionApp(ask_after=0.8, work_tick=0.2, heartbeat=5.0)
    ctx = _make_ctx("resp_hold_at_open")

    elapsed = await _run_answered(app, ctx, answer_after=1.4)

    assert app.answer == "accept"
    assert elapsed >= 1.4, f"turn ended before the answer arrived ({elapsed:.2f}s)"
