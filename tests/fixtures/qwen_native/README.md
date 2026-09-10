# qwen-native stream fixtures

Real `qwen` bridge output (`--json-file` NDJSON, qwen **v0.22.0**, stream-json
**protocol 2**), captured from live sessions and reduced for use in
`tests/test_qwen_native_forwarder.py`.

## What is verbatim

Everything the forwarder actually reads:

- envelope fields — `type`, `subtype`, `uuid`, `session_id`, `parent_tool_use_id`
- event ordering, exactly as qwen emitted it
- `message_start` → `message.id`, and the bare `message_stop`
- `assistant` / `user` block **types** and `stop_reason`
- `tool_use` block `id` + `name`, `tool_result` `tool_use_id`

## What was changed

- `content_block_delta` events dropped (streaming noise the forwarder ignores);
  this is why the files are much shorter than the sessions they came from.
- Free text — `text`, `thinking`, `tool_result` content, `tool_use` arguments,
  and `session_start.cwd` — replaced with `<scrubbed …>` placeholders. The
  captured sessions were doing work in unrelated private repositories, and none
  of that content is load-bearing for these tests. qwen's own diagnostic strings
  (e.g. `[API Error: …]`) are kept, since they are product behaviour.

## The three files

| file | what it pins |
| --- | --- |
| `tool_loop_session.ndjson` | A real multi-step tool loop: **7** top-level `message_stop` events, six of them closing `stop_reason="tool_use"` steps. Exactly **one** parent wake is correct. |
| `api_error_session.ndjson` | A turn that failed inside the model call. qwen still emits a proper turn-end `message_stop`, so the wake must fire. |
| `session_end_session.ndjson` | A clean session that ends with `system`/`session_end` after its `message_stop` — the second terminal must not double-wake. |

## Measured facts behind the forwarder's wake logic

Taken over 319k events across 23 recorded sessions:

- a turn is a `message_start` … `message_stop` window and windows **never nest**
  (max concurrent depth 1), so the opening `message.id` names the turn;
- `message_stop` is bare — `{"type": "message_stop"}`, carrying no id — so the
  tool-loop classification must come from the open window, not the stop event;
- `assistant.stop_reason` is only ever `null` (turn end) or `"tool_use"`;
- qwen advertises `result` in `session_start.supported_events` but **never emits
  it** in the interactive TUI mode this forwarder tails (0 occurrences);
- only **2 of 23** sessions carry `system`/`session_end`, and one session ended
  mid-stream with no `message_stop` at all — which is why the forwarder needs a
  process-exit fallback rather than trusting any in-stream terminal.
