"""Synthetic user-role messages the runner injects into session history.

Stdlib-only so executors (``omnigent.inner``) can recognize them without
importing the runner app.
"""

from __future__ import annotations

from collections.abc import Mapping

CANCELLATION_MARKER_TEXT = (
    "[System: interrupted]\n"
    "The user interrupted and abandoned their previous request (the user "
    "message immediately before this one). Do not resume or act on that "
    "interrupted request unless the user asks for it again; treat the next "
    "user message as the current instruction. The preceding assistant "
    "message may be incomplete."
)


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
