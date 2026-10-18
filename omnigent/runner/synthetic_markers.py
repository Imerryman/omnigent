"""Synthetic user-role messages the runner injects into session history.

Kept free of the runner app's imports so executors (``omnigent.inner``) can
recognize them. The marker text itself is upstream's
``omnigent.inner.executor.INTERRUPTION_MARKER_TEXT`` (#12483), aliased here so
there is one definition.
"""

from __future__ import annotations

from collections.abc import Mapping

from omnigent.inner.executor import INTERRUPTION_MARKER_TEXT

CANCELLATION_MARKER_TEXT = INTERRUPTION_MARKER_TEXT


def is_cancellation_marker(message: Mapping[str, object]) -> bool:
    """
    Return whether ``message`` is the runner's ``[System: interrupted]`` marker.

    The marker is appended as a user-role message after a cancelled turn. It
    closes the abandoned request, so replay logic must not treat user
    messages before it as part of the current input.

    :param message: An inner history message (``role`` + ``content``).
    :returns: ``True`` for the exact cancellation marker, as plain text or as
        a single text block.
    """
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return content == CANCELLATION_MARKER_TEXT
    if isinstance(content, list) and len(content) == 1:
        block = content[0]
        return (
            isinstance(block, dict)
            and block.get("type") in {"text", "input_text"}
            and block.get("text") == CANCELLATION_MARKER_TEXT
        )
    return False
