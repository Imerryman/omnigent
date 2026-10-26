"""The cancellation marker the runner emits is recognized by executors."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from omnigent.runner.session_history import build_session_history
from omnigent.runner.synthetic_markers import CANCELLATION_MARKER_TEXT, is_cancellation_marker


def test_emitted_cancellation_marker_is_recognized() -> None:
    async def scenario() -> list[dict[str, Any]]:
        histories: dict[str, list[dict[str, Any]]] = {"s1": []}

        async def persist(conv_id: str, items: list[dict[str, Any]]) -> None:
            return None

        history = build_session_history(
            _background_tasks=set(),
            _last_server_item_id={},
            _persist_cancellation_items=persist,
            _session_histories=histories,
            _session_spec_cache={},
            server_client=MagicMock(),
        )
        history.append_cancellation_items("s1")
        await asyncio.sleep(0)
        return histories["s1"]

    items = asyncio.run(scenario())
    assert len(items) == 1
    assert is_cancellation_marker(items[0])


def test_is_cancellation_marker_matches_only_the_exact_user_marker() -> None:
    assert is_cancellation_marker({"role": "user", "content": CANCELLATION_MARKER_TEXT})
    assert is_cancellation_marker(
        {"role": "user", "content": [{"type": "text", "text": CANCELLATION_MARKER_TEXT}]}
    )
    assert not is_cancellation_marker({"role": "assistant", "content": CANCELLATION_MARKER_TEXT})
    assert not is_cancellation_marker({"role": "user", "content": "[System: interrupted]"})
    assert not is_cancellation_marker(
        {"role": "user", "content": "please " + CANCELLATION_MARKER_TEXT}
    )
    assert not is_cancellation_marker({"role": "user", "content": None})
