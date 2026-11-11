"""LOCAL: a budgeted fresh-client replay keeps tool calls whole and prices text honestly."""

import pytest

from omnigent.inner import claude_sdk_executor as mod
from omnigent.inner.claude_sdk_executor import ClaudeSDKExecutor

_MARKER = "[Earlier messages omitted to fit the model's context window:"


def _call(call_id: str | None, tool: str = "probe") -> dict[str, object]:
    message: dict[str, object] = {"role": "tool_call", "content": {"tool": tool, "args": {}}}
    if call_id is not None:
        message["metadata"] = {"call_id": call_id}
    return message


def _result(call_id: str | None, output: str) -> dict[str, object]:
    message: dict[str, object] = {"role": "tool_result", "content": output}
    if call_id is not None:
        message["metadata"] = {"call_id": call_id}
    return message


def _replayed_lines(prompt: object) -> list[str]:
    """The replayed history lines between the omission marker and the trailer."""
    assert isinstance(prompt, str)
    history = prompt.split("\n\nRespond to the latest user message", 1)[0]
    lines = history.splitlines()[1:]  # drop "Conversation so far:"
    return [line for line in lines if not line.startswith(_MARKER)]


def _assert_no_orphans(prompt: object, pairs: dict[str, str]) -> None:
    """Every replayed result has its call, and the replay never opens on a result."""
    lines = _replayed_lines(prompt)
    if lines:
        assert not lines[0].startswith("tool result:"), lines[0]
    text = "\n".join(lines)
    for output, tool in pairs.items():
        if f"tool result: {output}" in text:
            assert f'"tool": "{tool}"' in text, (output, text)


def test_codex_repro_never_replays_a_result_without_its_call() -> None:
    messages = [
        {"role": "user", "content": "Run the probe. " + "q" * 600},
        _call("c1", tool="probe_" + "p" * 300),
        _result("c1", "done"),
        {"role": "assistant", "content": "It finished."},
        {"role": "user", "content": "What next?"},
    ]
    # Fits the result and the reply, but not the call.
    prompt = ClaudeSDKExecutor._build_prompt(
        messages, resume_session=False, history_token_budget=40
    )

    assert isinstance(prompt, str)
    assert "tool result: done" not in prompt
    assert f"{_MARKER} 3.]" in prompt
    assert "assistant: It finished." in prompt
    assert prompt.endswith("user: What next?")


def test_parallel_calls_are_kept_or_dropped_together() -> None:
    messages = [
        {"role": "user", "content": "Check both."},
        _call("a", tool="alpha"),
        _call("b", tool="beta"),
        _result("a", "alpha-out"),
        _result("b", "beta-out"),
        {"role": "assistant", "content": "Both done."},
        {"role": "user", "content": "And now?"},
    ]
    pairs = {"alpha-out": "alpha", "beta-out": "beta"}
    full = ClaudeSDKExecutor._build_prompt(messages, resume_session=False)
    assert isinstance(full, str)
    for budget in range(len(full)):
        prompt = ClaudeSDKExecutor._build_prompt(
            messages, resume_session=False, history_token_budget=budget
        )
        _assert_no_orphans(prompt, pairs)
        lines = _replayed_lines(prompt)
        # The cut never lands between the two calls or among the results.
        assert not lines or not lines[0].startswith('tool call: {"tool": "beta"'), budget


def test_tool_items_without_call_ids_are_not_split() -> None:
    messages = [
        {"role": "user", "content": "Go."},
        _call(None, tool="first"),
        _call(None, tool="second"),
        _result(None, "first-out"),
        _result(None, "second-out"),
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Next?"},
    ]
    full = ClaudeSDKExecutor._build_prompt(messages, resume_session=False)
    assert isinstance(full, str)
    for budget in range(len(full)):
        prompt = ClaudeSDKExecutor._build_prompt(
            messages, resume_session=False, history_token_budget=budget
        )
        _assert_no_orphans(prompt, {"first-out": "first", "second-out": "second"})


@pytest.mark.parametrize("seed", range(20))
def test_any_budget_keeps_every_replayed_result_with_its_call(seed: int) -> None:
    import random

    rng = random.Random(seed)
    messages: list[dict[str, object]] = []
    pairs: dict[str, str] = {}
    for turn in range(6):
        messages.append({"role": "user", "content": f"ask{turn} " + "u" * rng.randint(0, 90)})
        ids = [f"t{turn}_{n}" for n in range(rng.randint(0, 3))]
        for call_id in ids:
            messages.append(_call(call_id, tool=f"tool_{call_id}"))
        for call_id in rng.sample(ids, len(ids)):
            output = f"out_{call_id} " + "r" * rng.randint(0, 120)
            messages.append(_result(call_id, output))
            pairs[output] = f"tool_{call_id}"
        messages.append(
            {"role": "assistant", "content": f"reply{turn} " + "a" * rng.randint(0, 60)}
        )
    messages.append({"role": "user", "content": "Latest?"})

    for budget in range(0, 700, 3):
        prompt = ClaudeSDKExecutor._build_prompt(
            messages, resume_session=False, history_token_budget=budget
        )
        _assert_no_orphans(prompt, pairs)


def test_cut_points_mark_only_safe_boundaries() -> None:
    prior = [
        {"role": "user", "content": "u"},  # 0
        _call("a"),  # 1
        _call("b"),  # 2: splits a
        _result("a", "x"),  # 3: result, splits b
        _result("b", "y"),  # 4: result
        {"role": "assistant", "content": "ok"},  # 5
        _call(None),  # 6
        _result(None, "z"),  # 7: result
        _call(None),  # 8: inside an id-less run
        {"role": "user", "content": "v"},  # 9
    ]
    assert mod._replay_cut_points(prior) == [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]


def test_unbudgeted_replay_keeps_everything() -> None:
    messages = [
        {"role": "user", "content": "Go."},
        _call("c1"),
        _result("c1", "done"),
        {"role": "user", "content": "Next?"},
    ]
    prompt = ClaudeSDKExecutor._build_prompt(messages, resume_session=False)
    assert isinstance(prompt, str)
    assert "tool result: done" in prompt
    assert _MARKER not in prompt


def _many_short_blocks_message() -> dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            *({"type": "input_text", "text": "x"} for _ in range(900)),
        ],
    }


def test_estimate_covers_the_coalesced_text() -> None:
    rendered = mod._render_prior_message(_many_short_blocks_message())
    coalesced = mod._coalesce_text_blocks(rendered)
    coalesced_bytes = sum(len(str(block["text"]).encode()) for block in coalesced)

    estimate = mod._estimated_replay_tokens(rendered)

    assert estimate >= coalesced_bytes / mod._REPLAYED_HISTORY_BYTES_PER_TOKEN
    assert estimate >= 610


def test_many_short_blocks_do_not_slip_under_the_budget() -> None:
    messages = [
        _many_short_blocks_message(),
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Next?"},
    ]
    prompt = ClaudeSDKExecutor._build_prompt(
        messages, resume_session=False, history_token_budget=25
    )

    assert isinstance(prompt, str)
    assert f"{_MARKER} 1.]" in prompt
    assert "x\nx" not in prompt
    assert "assistant: ok" in prompt
