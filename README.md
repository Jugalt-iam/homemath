# homemath

*by Jugal Nitin Thakkar*

A small inference layer for OpenAI-compatible chat-completions endpoints. It streams every request and reassembles the answer from chunks, tolerates the schema differences between servers that put the answer in `delta.content`, `delta.reasoning_content`, `choices[0].message.content`, a top-level `message.content` or a top-level `response`, and keeps the model's private thinking channel out of the answer. It classifies each request deterministically into a task class, and that class alone decides whether reasoning mode is enabled and how many output tokens to budget. It fails over from a primary endpoint to a fallback, trims message history to a token budget before sending, and caches responses in memory or in Redis.

## Install

```
pip install homemath
```

Optional extras: `pip install "homemath[redis]"` for the shared cache, `pip install "homemath[test]"` to run the suite.

## Usage

```python
import os
from homemath import homemath_chat, classify_task, strip_thinking

os.environ["LLM_HOST"] = "http://127.0.0.1:8080"
os.environ["LLM_MODEL_PRIMARY"] = "my-model"

messages = [{"role": "user", "content": "Explain why this query is slow."}]

print(classify_task(messages))   # 'general'
answer = homemath_chat(messages) # streams, reassembles, strips <think> blocks
print(answer)
```

Lower-level entry points, when you want the reasoning stream as well as the answer:

```python
from homemath import ollama_chat_stream_dual

result = ollama_chat_stream_dual(
    "http://127.0.0.1:8080/v1/chat/completions",
    {"model": "my-model", "messages": messages},
)
result["content"]   # the answer, one channel only, never concatenated
result["thinking"]  # the model-private thinking stream, verbatim
result["source"]    # which channel the answer came from
```

## Channels: what you get back, and from where

Five fields can carry an answer. They are ranked, the first non-empty one wins outright, and they are never concatenated — a proxy that emits the same answer in two fields would otherwise produce it twice.

| Rank | Field | Notes |
|---|---|---|
| 1 | `choices[0].delta.content` | the normal answer channel |
| 2 | `choices[0].delta.reasoning_content` | reasoning, surfaced as the answer when 1 is empty — see below |
| 3 | `choices[0].message.content` | non-delta shape, some proxies |
| 4 | top-level `message.content` | proxy-wrapped chat shape |
| 5 | top-level `response` | `/api/generate` shape leaking into a chat stream |

`delta.thinking`, `message.thinking` and top-level `thinking` are a separate matter. Those are model-private, accumulate into their own buffer, and are **never** returned as the answer — a stream carrying only those fields returns `""`, pinned by `tests/test_homemath.py:430` (`test_chat_stream_returns_empty_string_when_only_thinking`) and `tests/test_homemath.py:949` (`test_thinking_wins_nothing_when_no_answer_channel`).

**`reasoning_content` is deliberately different.** On vLLM, SGLang and OpenRouter serving a reasoning model, that field carries the model's working. When it is the only thing on the wire, homemath returns it rather than an empty string, because a human reviewing the output — or a downstream judge scoring it — is better served by the model's reasoning than by nothing at all. This is a design choice, pinned by `tests/test_homemath.py:862` (`test_reasoning_content_only_stream`).

The consequence: **`content` may contain reasoning rather than a finished answer.** `source` tells you which happened. If reasoning must not reach your end users, branch on it:

```python
result = ollama_chat_stream_dual(url, payload)
if result["source"] == "delta.reasoning_content":
    # Model produced working, not a finished answer. Send for review,
    # retry, or fall back — rather than rendering it straight to a user.
    ...
```

`ollama_chat_stream` is the content-only convenience wrapper and discards `source`. Use the `_dual` variant whenever that distinction matters to you.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_HOST` | `http://127.0.0.1:8080` | Base URL of the endpoint. Must expose `/v1/chat/completions`. |
| `LLM_HOST_FALLBACK` | falls back to `LLM_HOST` | Optional second endpoint, tried after the first. |
| `LLM_API_KEY` | unset | Bearer token. Leave unset if the endpoint needs no auth. |
| `LLM_MODEL_PRIMARY` | `local-model` | Model id sent to the primary endpoint. |
| `LLM_MODEL_FALLBACK` | falls back to `LLM_MODEL_PRIMARY` | Model id sent to the fallback endpoint. |
| `LLM_TIMEOUT_PRIMARY` | `600` | Per-request timeout in seconds, primary. |
| `LLM_TIMEOUT_FALLBACK` | `60` | Per-request timeout in seconds, fallback. |
| `LLM_MODEL_PROBE_TIMEOUT` | `5` | Timeout for the `/v1/models` availability probe. |
| `LLM_READY` | `false` | Set `true` to skip the startup probe and trust the configuration. |
| `REDIS_URL` | unset | Shared response cache. Unset means in-memory only. |
| `HOMEMATH_TOKEN_BUDGET` | `28000` | Prompt token budget before history is trimmed. |
| `HOMEMATH_SYSTEM_HARD_CAP` | `6000` | Per-system-message token cap. |
| `HOMEMATH_DOMAIN_KEYWORDS` | unset | Regex alternation enabling `TaskClass.DOMAIN`, e.g. `ledger\|invoice`. |

## Classification

`homemath.classifier` turns a message list into a `TaskClass`, and three pure lookups turn that class into the decisions the engine needs. The policy functions never inspect message text — they read the class only, so a keyword cannot reach into a system prompt and switch reasoning on for a background call.

```python
from homemath.classifier import (
    classify, classify_task_class, TaskClass,
    thinking_from_intent, think_style, max_tokens,
)

cls = classify_task_class([{"role": "user", "content": "Why is this failing?"}])
cls                        # TaskClass.DEBUG
thinking_from_intent(cls)  # True  — reasoning mode on
think_style(cls)           # ThinkStyle.DIAGNOSTIC
max_tokens(cls)            # 4096  — the budget the engine sends
```

| Class | Reasoning | Budget | Triggered by |
|---|---|---|---|
| `GREETING` | off | 128 | greeting words, ≤ 8 words |
| `CHAT` | off | 1024 | short turn, nothing else matched |
| `SIMPLE` | off | 512 | extract/format/convert verbs, ≤ 12 words, no complex verb |
| `CONTENT_GEN` | off | 4096 | write/create/draft — fluency, not reasoning |
| `CODE` | on | 4096 | ≥ 2 code keywords across the whole exchange |
| `DOMAIN` | off | 2048 | ≥ 2 `HOMEMATH_DOMAIN_KEYWORDS` hits — recall, not reasoning |
| `JUDGE` | on | 4096 | judge/critique/score/rubric |
| `STRATEGY` | on | 4096 | strategy/roadmap/plan |
| `ANALYSIS` | on | 4096 | analyse/compare/trade-off |
| `DEBUG` | on | 4096 | debug/diagnose/root cause/failing |
| `RESEARCH` | on | 4096 | research/investigate |
| `GENERAL` | off | 2048 | catch-all |

`classify_task` in the top-level package is a coarser adapter over the same pass, returning `greeting | simple | code | domain | general` for callers that switch on a string.

`homemath.policy` lets an upstream caller that has already classified a request bind the decision for its duration, on a `ContextVar` so concurrent requests do not observe each other:

```python
from homemath.policy import set_current_policy, reset_current_policy

token = set_current_policy({"thinking_required": True})
try:
    answer = homemath_chat(messages)   # honours the binding, skips re-derivation
finally:
    reset_current_policy(token)
```

Only `thinking_required` is read by this package; carry whatever else you need alongside it.

## The truncated thinking block

A reasoning model marks its private reasoning with `<think>` and `</think>`. If the stream is cut off partway through — timeout, dropped connection, token limit reached mid-reasoning — the opening tag arrives and the closing tag never does. Naive handling fails in one of two directions. A regex that requires both tags, `<think>.*?</think>`, matches nothing, so the entire unterminated reasoning block survives into the string you hand the user. Dropping everything from the first `<think>` onward is safe but discards a real answer in the case where reasoning completed and the answer followed.

This handles it in two places. First, the tags are a fallback, not the primary mechanism: the stream parser buckets `delta.thinking`, `message.thinking` and top-level `thinking` into a separate accumulator that is never merged into the answer, so on a well-behaved server the private reasoning is never in the answer string to begin with.

Second, for servers that inline the tags into the content channel, `strip_thinking` applies two patterns in order: `<think>[\s\S]*?</think>` removes complete blocks, then `<think>[\s\S]*$` removes an unterminated block through to the end of the string. Complete blocks are removed without touching text that follows them, and an unterminated block is removed entirely rather than leaked. The unterminated case is pinned by `tests/test_homemath.py:113` (`test_strip_thinking_handles_unclosed_block`), the complete case by `tests/test_homemath.py:107`, and the no-tags case by `tests/test_homemath.py:118`, which asserts clean text passes through byte-identical.

The consequence to be aware of: when a `<think>`-tagged block is truncated you get an empty string, not an error. Callers that cannot use an empty answer should check for it and retry.

## Failure behaviour

`HomemathEngine.race` never raises for an inference or routing failure. Every failure path returns `("FAILED", text)` where the text is plain user-safe prose with no operator diagnostics in it — the diagnostics go to the logger at ERROR instead. Callers branch on the `"FAILED"` sentinel, not on catching exceptions.

Provider selection will refuse to downgrade a content, code or domain task to the fallback model once the primary is above its failure threshold, since that is a quality cliff rather than a graceful degradation. That refusal applies only when the fallback is genuinely a different model or endpoint; with `LLM_HOST_FALLBACK` and `LLM_MODEL_FALLBACK` unset both providers are the same model at the same URL, and the request is simply retried there.

## Limitations

- Token counting is an estimate, `len(text) // 4`. There is no tokenizer, so the budget is approximate and wrong for non-English text and for code. It will be off in both directions.
- Only the `requests`-based synchronous transport exists. There is no async client, and calls block the calling thread.
- `HomemathEngine.race` is named for behaviour it no longer has. It is sequential failover: one provider is tried, and the other only if the first fails. Nothing runs concurrently.
- Function and module names use `ollama_` prefixes for historical reasons. They target any OpenAI-compatible endpoint and are not specific to Ollama.
- Provider selection is two fixed slots, primary and fallback. There is no pool, no weighting, and no retry budget beyond the fallback.
- The cache key is a hash of message content only. Temperature, model and token budget are not part of it, so two calls that differ only in those parameters will collide. A shared `REDIS_URL` therefore shares answers across everything pointed at it.
- Creating the engine probes both endpoints' `/v1/models`, so the first call can block for up to `2 × LLM_MODEL_PROBE_TIMEOUT` seconds. Set `LLM_READY=true` to skip it.
- `TaskClass.DOMAIN` does nothing until you configure `HOMEMATH_DOMAIN_KEYWORDS`. There is no built-in subject area.
- The classifier is keyword and regex based. It is deterministic and cheap, and it will misclassify phrasing the keyword banks do not anticipate.
- The test suite mocks the transport entirely. Nothing here has been tested against a live endpoint as part of this package.

## Author

Created and maintained by **Jugal Nitin Thakkar**.

## License

MIT © 2026 Jugal Nitin Thakkar — see [LICENSE](LICENSE).
