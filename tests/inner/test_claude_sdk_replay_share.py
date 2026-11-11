"""LOCAL: OMNIGENT_CLAUDE_SDK_REPLAY_WINDOW_SHARE sizes a fresh client's replay."""

from unittest.mock import patch

import pytest

from omnigent.inner import claude_sdk_executor as mod
from omnigent.inner.claude_sdk_executor import ClaudeSDKExecutor
from omnigent.inner.executor import ExecutorConfig

_ENV = "OMNIGENT_CLAUDE_SDK_REPLAY_WINDOW_SHARE"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 0.5),
        ("", 0.5),
        ("0.25", 0.25),
        ("3", 1.0),
        ("0", None),
        ("-1", None),
        ("half", 0.5),
        ("nan", 0.5),
        ("NaN", 0.5),
        ("inf", 0.5),
        ("-inf", 0.5),
    ],
)
def test_window_share_env(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float | None
) -> None:
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    assert mod._replayed_history_window_share() == expected


def _history() -> list[dict[str, object]]:
    # Four ~100-token prior messages (300 bytes at 3 bytes per token).
    messages: list[dict[str, object]] = []
    for i in range(2):
        messages.append({"role": "user", "content": f"q{i} " + "q" * 291})
        messages.append({"role": "assistant", "content": f"a{i} " + "a" * 286})
    messages.append({"role": "user", "content": "What next?"})
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("share", "omitted"),
    [("0.5", 2), ("0.25", 3), ("1", None), ("0", None)],
)
async def test_window_share_sizes_the_fresh_replay(
    monkeypatch: pytest.MonkeyPatch, share: str, omitted: int | None
) -> None:
    monkeypatch.setenv(_ENV, share)
    executor = ClaudeSDKExecutor(model="claude-x")
    with patch.object(mod, "_resolve_replay_context_window", return_value=420):
        prompt = await executor._fresh_session_prompt(_history(), ExecutorConfig())

    assert isinstance(prompt, str)
    assert prompt.endswith("user: What next?")
    if omitted is None:
        assert "omitted" not in prompt
        assert "q0 " in prompt
    else:
        assert f"context window: {omitted}.]" in prompt
        assert "a1 " in prompt


def test_non_finite_share_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(_ENV, "nan")
    with caplog.at_level("WARNING", logger=mod.logger.name):
        assert mod._replayed_history_window_share() == 0.5
    assert _ENV in caplog.text


@pytest.mark.asyncio
async def test_nan_share_keeps_the_default_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "nan")
    executor = ClaudeSDKExecutor(model="claude-x")
    with patch.object(mod, "_resolve_replay_context_window", return_value=420):
        prompt = await executor._fresh_session_prompt(_history(), ExecutorConfig())

    assert isinstance(prompt, str)
    assert "context window: 2.]" in prompt
