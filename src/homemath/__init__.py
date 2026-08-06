"""homemath — a small inference layer for OpenAI-compatible chat endpoints.

Provides:
  - streaming (SSE / JSON-lines) request handling with schema-tolerant parsing
  - separation of the answer channel from the model reasoning channel
  - deterministic task classification driving thinking mode and token budget
  - a priority-aware lock so same-model requests serialize predictably
  - optional response caching (in-memory, or Redis when REDIS_URL is set)

All endpoints, models and credentials come from environment variables.
See .env.example for the full list.
"""
from __future__ import annotations


import os
import re
import time
import hashlib
import heapq
import logging
import requests
import threading
from typing import Optional, List, Tuple, Dict, Any
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("homemath")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

logger = logging.getLogger("homemath")

# Endpoint and model defaults. Every one is overridable by environment
# variable; see .env.example. The defaults point at a loopback address so an
# unconfigured install cannot reach anything but the local machine.
DEFAULT_LLM_HOST: str = "http://127.0.0.1:8080"
DEFAULT_MODEL: str = "local-model"

# Seconds to wait on the /v1/models availability probe.
MODEL_PROBE_TIMEOUT: int = int(os.environ.get("LLM_MODEL_PROBE_TIMEOUT", "5"))

# ── Priority constants for Ollama lock ──
# Workflow-critical calls (blog/email workflow steps) get HIGH priority (1)
# so they acquire the lock before background bus-triggered calls (LOW = 3).
# Lower number = higher priority (heapq is a min-heap).
PRIORITY_HIGH = 1    # Foreground work a user is waiting on
PRIORITY_NORMAL = 2  # Ordinary API calls
PRIORITY_LOW = 3     # Background work that can wait behind everything else


class PriorityLock:
    """Per-model priority-aware lock for Ollama requests.

    Locks are per model. A server can usually run different models
    concurrently; only same-model requests need serializing.

    This unlocks parallel planning + generation, cutting
    pipeline time roughly in half.

    Backward-compatible: acquire()/release() without model param use "_default" key.
    For model-aware locking: acquire(model="my-model")
    """

    def __init__(self):
        self._locks: dict[str, dict] = {}
        self._meta_lock = threading.Lock()

    def _get_state(self, model: str) -> dict:
        if model not in self._locks:
            with self._meta_lock:
                if model not in self._locks:
                    self._locks[model] = {
                        "lock": threading.Lock(),
                        "active": False,
                        "waiters": [],
                        "seq": 0,
                    }
        return self._locks[model]

    def acquire(self, priority: int = PRIORITY_NORMAL, model: str = "_default") -> None:
        """Acquire per-model lock. Lower priority number = served first.

        If the waiting thread is interrupted (KeyboardInterrupt, or a timeout
        raised into it), the waiter cancels itself so a later ``release`` does
        not hand the lock to a thread that is no longer there. Without that,
        one interrupt during a queue wait would strand the lock permanently:
        ``active`` stays True and no one is left to release it.
        """
        state = self._get_state(model)
        waiter = {"event": threading.Event(), "cancelled": False}
        with state["lock"]:
            if not state["active"]:
                state["active"] = True
                return
            state["seq"] += 1
            heapq.heappush(state["waiters"], (priority, state["seq"], waiter))
        try:
            waiter["event"].wait()
        except BaseException:
            with state["lock"]:
                if waiter["event"].is_set():
                    # We were granted the lock in the window before the
                    # interrupt landed, so we own it and must hand it on.
                    self._release_locked(state)
                else:
                    waiter["cancelled"] = True
            raise

    @staticmethod
    def _release_locked(state: dict) -> None:
        """Hand the lock to the next live waiter. Caller must hold state["lock"]."""
        while state["waiters"]:
            _prio, _seq, waiter = heapq.heappop(state["waiters"])
            if not waiter["cancelled"]:
                waiter["event"].set()
                return
        state["active"] = False

    def release(self, model: str = "_default") -> None:
        """Release per-model lock. Wakes the highest-priority live waiter."""
        state = self._get_state(model)
        with state["lock"]:
            self._release_locked(state)

    def queue_depth(self, model: str = "_default") -> int:
        """Return how many live callers are waiting for this model."""
        state = self._get_state(model)
        with state["lock"]:
            return sum(1 for _p, _s, w in state["waiters"] if not w["cancelled"])

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


# ── Per-Model Ollama Chat Lock ──
# Different models run concurrently on Ollama. Only same-model requests serialize.
# Planning + generation = PARALLEL, not sequential.
#
# Import in other modules: from homemath import OLLAMA_CHAT_LOCK
# Backward compatible: acquire()/release() work as before (uses "_default" key).
OLLAMA_CHAT_LOCK = PriorityLock()

# ============================================
# LLM OUTPUT SANITIZATION
# ============================================
# Some reasoning models emit <think>...</think> blocks in the answer channel;
# they must be stripped before the text reaches a caller.
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from LLM output."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


# ============================================
# STREAMING REASSEMBLY
# ============================================
# All calls stream and reassemble from chunks. A single large non-streamed
# response body can be silently truncated by an intermediary with a small MTU,
# which is indistinguishable from an empty answer at the call site. Streaming
# removes that failure mode.
#
# Two endpoint formats:
#   /v1/chat/completions (OpenAI-compat) → SSE "data: {...}" lines
#   /api/generate (native Ollama)        → JSON-lines, one object per line

import json as _json

# ── Context window enforcement ─────────────────────────────────────────────
# Imported lazily inside _call_ollama to avoid circular import risk at module
# load time (context_trimmer has no upstream deps, but the guard is cheap).
try:
    from .context_trimmer import trim_messages as _trim_messages
    _TRIMMER_AVAILABLE = True
except ImportError:
    _TRIMMER_AVAILABLE = False
    logger.warning("context_trimmer not available — token budget enforcement disabled")


def _extract_chunk_tokens(obj: dict) -> dict:
    """Normalize a single streamed JSON chunk into a schema-tolerant token bag.

    This stack can emit any of:
      OpenAI chat completions (SSE):
        choices[0].delta.content             — primary answer channel
        choices[0].delta.reasoning_content   — native reasoning channel emitted
                                                by some servers; treated here as
                                                answer-bearing when content is empty
        choices[0].delta.thinking            — model-private thinking, never returned
        choices[0].message.content           — non-delta message shape emitted by
                                                some proxies mid-stream
        choices[0].finish_reason             — done marker
      Top-level (emitted by some proxies):
        message.content                      — answer-bearing
        message.thinking                     — private, never returned
        response                             — /api/generate token
        thinking                             — private, never returned
        done                                 — done marker

    Returns a dict with per-channel strings so the outer loop can pick the
    winning channel by priority. ``thinking`` is aggregated across all sources
    for observability but MUST NOT be returned to callers.
    """
    out = {
        "content": "",
        "reasoning": "",
        "message_content": "",
        "top_message": "",
        "response": "",
        "thinking": "",
        "done": False,
    }
    if not isinstance(obj, dict):
        return out

    # --- OpenAI chat completions shape ---------------------------------
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice0 = choices[0]
        delta = choice0.get("delta")
        if isinstance(delta, dict):
            c = delta.get("content")
            if isinstance(c, str):
                out["content"] = c
            r = delta.get("reasoning_content")
            if isinstance(r, str):
                out["reasoning"] = r
            t = delta.get("thinking")
            if isinstance(t, str) and t:
                out["thinking"] += t
        msg = choice0.get("message")
        if isinstance(msg, dict):
            mc = msg.get("content")
            if isinstance(mc, str):
                out["message_content"] = mc
            mt = msg.get("thinking")
            if isinstance(mt, str) and mt:
                out["thinking"] += mt
        if choice0.get("finish_reason"):
            out["done"] = True

    # --- Top-level message.content (some proxies) ----------------------
    top_msg = obj.get("message")
    if isinstance(top_msg, dict):
        tmc = top_msg.get("content")
        if isinstance(tmc, str):
            out["top_message"] = tmc
        tmt = top_msg.get("thinking")
        if isinstance(tmt, str) and tmt:
            out["thinking"] += tmt

    # --- Ollama /api/generate shape ------------------------------------
    r = obj.get("response")
    if isinstance(r, str):
        out["response"] = r
    top_t = obj.get("thinking")
    if isinstance(top_t, str) and top_t:
        out["thinking"] += top_t

    if obj.get("done") is True:
        out["done"] = True

    return out


# Channel identifiers used for per-channel accumulators and source labels.
# These MUST match the keys returned by _extract_chunk_tokens() for answer
# channels ("content", "reasoning", "message_content", "top_message", "response").
_CHANNEL_LABELS = {
    "content":         "delta.content",
    "reasoning":       "delta.reasoning_content",
    "message_content": "choices[0].message.content",
    "top_message":     "top-level message.content",
    "response":        "top-level response",
}

# Chat-endpoint priority: OpenAI-compat chat completions. The answer is
# almost always in delta.content.
#
# ``delta.reasoning_content`` sits second by design. On vLLM, SGLang and
# OpenRouter serving a reasoning model, that field carries the model's
# reasoning. Surfacing it when delta.content is empty is deliberate: a
# reasoning-only stream returns the reasoning rather than an empty string, so
# a human reviewing the output — or a caller making a judgement on it — has
# the model's working rather than nothing. ``source`` in the dual-channel
# return names which channel won, so callers can tell the two apart.
#
# ``delta.thinking``/``message.thinking``/top-level ``thinking`` are a
# different matter: those are model-private and never reach the answer.
_CHAT_PRIORITY = ("content", "reasoning", "message_content", "top_message", "response")

# Generate-endpoint priority: /api/generate ships the answer in top-level
# ``response``. We only fall back to chat-shaped channels if a proxy is
# re-wrapping the response in chat-completion events.
_GENERATE_PRIORITY = ("response", "top_message", "content", "reasoning", "message_content")


def _pick_answer(buckets: dict, priority: tuple) -> tuple[str, str]:
    """Pick exactly one channel as the final assembled answer.

    ``buckets`` maps channel name → list of token strings. ``priority`` is an
    ordered tuple of channel names. Returns the first non-empty channel's
    joined text plus a human-readable source label. This is a strict
    pick-one-and-stop selection; channels are never merged, which prevents
    duplication when a proxy emits the same answer in more than one field
    (e.g. both reasoning_content and response, or content and message.content).
    """
    for channel in priority:
        chunks = buckets.get(channel) or []
        if chunks:
            return "".join(chunks), _CHANNEL_LABELS.get(channel, channel)
    return "", "none"


def ollama_chat_stream_dual(url: str, payload: dict, timeout: int = 300,
                            headers: dict | None = None) -> dict:
    """Dual-channel variant of :func:`ollama_chat_stream`.

    Returns ``{"content": str, "thinking": str, "source": str}``.

    Use this when you want both the answer and the model's reasoning stream --
    for example to render reasoning in a side panel while the answer streams.
    ``thinking`` is the raw concatenation of reasoning tokens seen on the wire,
    preserved verbatim. ``strip_thinking`` is deliberately NOT applied here; it
    remains available for callers that want clean text only.

    Answer channels, in priority order. The first non-empty one wins outright;
    channels are never concatenated:
      - choices[0].delta.content                 (primary answer)
      - choices[0].delta.reasoning_content       (native reasoning channel —
                                                  surfaced when content is
                                                  empty, so a reasoning-only
                                                  stream returns the model's
                                                  working rather than nothing)
      - choices[0].message.content               (non-delta, some proxies)
      - top-level message.content                (proxy-wrapped chat shape)
      - top-level response                       (generate shape leaking in)

    ``source`` names the winning channel, so a caller can tell a reasoning
    answer from a content answer and route it for review accordingly.

    Model-private channels, captured into ``thinking`` and never returned as
    the answer:
      - choices[0].delta.thinking
      - choices[0].message.thinking
      - top-level message.thinking
      - top-level thinking

    Observability: emits a single [stream-summary] INFO line per call.
    """
    payload = {**payload, "stream": True}
    resp = requests.post(url, json=payload, headers=headers or {},
                         timeout=timeout, stream=True)
    try:
        return _consume_chat_stream(resp)
    finally:
        # stream=True holds the socket open until the body is fully consumed.
        # An exception from raise_for_status or mid-iteration would otherwise
        # leak it until garbage collection.
        _safe_close(resp)


def _safe_close(resp) -> None:
    """Close a streamed response without masking an in-flight exception."""
    try:
        resp.close()
    except Exception:  # pragma: no cover - close failure must never win
        pass


def _consume_chat_stream(resp) -> dict:
    """Read a chat-completions stream to exhaustion.

    Split out of :func:`ollama_chat_stream_dual` so the socket can be closed in
    a ``finally`` around the whole read.
    """
    resp.raise_for_status()

    # Per-channel accumulators. NEVER merged — one answer channel wins at the end.
    buckets: dict[str, list[str]] = {
        "content": [],
        "reasoning": [],
        "message_content": [],
        "top_message": [],
        "response": [],
    }
    token_counts: dict[str, int] = {k: 0 for k in buckets}
    thinking_buf: list[str] = []  # preserved verbatim, never merged into the answer

    line_count = 0
    empty_chunk_count = 0
    done_seen = False
    unparseable = 0

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        line_count += 1
        # SSE "data: {...}" or bare JSON on a line.
        data_str = line.removeprefix("data: ").strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            done_seen = True
            break
        try:
            obj = _json.loads(data_str)
        except (_json.JSONDecodeError, ValueError):
            unparseable += 1
            # The line body can contain model output, i.e. user data. Count it
            # at WARNING; the payload itself only at DEBUG, which an operator
            # opts into. The [stream-summary] line carries the count either way.
            if unparseable <= 3:
                logger.debug("[stream-debug] unparseable line: %s", data_str[:200])
            continue

        tokens = _extract_chunk_tokens(obj)

        if tokens["thinking"]:
            thinking_buf.append(tokens["thinking"])
            logger.debug("[model-thinking] chat-stream %d chars", len(tokens["thinking"]))

        had_answer_token = False
        for ch in buckets:
            piece = tokens.get(ch) or ""
            if piece:
                buckets[ch].append(piece)
                token_counts[ch] += 1
                had_answer_token = True

        if not had_answer_token:
            empty_chunk_count += 1
            if empty_chunk_count <= 3:
                logger.debug("[stream-debug] empty chunk #%d, keys=%s",
                             empty_chunk_count, list(obj.keys()))

        if tokens["done"]:
            done_seen = True
            # Do not break — usage/finish metadata may arrive after the token
            # carrying finish_reason in some proxy setups.

    assembled, source = _pick_answer(buckets, _CHAT_PRIORITY)
    thinking = "".join(thinking_buf)

    logger.info(
        "[stream-summary] lines=%d content=%d reasoning=%d message=%d "
        "top_message=%d response=%d empty=%d thinking_chars=%d "
        "unparseable=%d done=%s source=%s out_chars=%d",
        line_count, token_counts["content"], token_counts["reasoning"],
        token_counts["message_content"], token_counts["top_message"],
        token_counts["response"], empty_chunk_count, len(thinking),
        unparseable, done_seen, source, len(assembled),
    )

    if not assembled and line_count > 0:
        logger.warning(
            "[stream-debug] %d lines received, 0 answer tokens across "
            "content/reasoning/message/top_message/response channels "
            "(thinking_chars=%d)",
            line_count, len(thinking),
        )

    return {"content": assembled, "thinking": thinking, "source": source}


def ollama_chat_stream(url: str, payload: dict, timeout: int = 300,
                       headers: dict | None = None) -> str:
    """Content-only wrapper around :func:`ollama_chat_stream_dual`.

    Returns the assembled answer with ``<think>...</think>`` blocks stripped.
    Callers that also need the reasoning stream should use
    ``ollama_chat_stream_dual`` instead.
    """
    result = ollama_chat_stream_dual(url, payload, timeout=timeout, headers=headers)
    return strip_thinking(result["content"])


def ollama_generate_stream_dual(url: str, payload: dict, timeout: int = 300,
                                headers: dict | None = None) -> dict:
    """Dual-channel variant of :func:`ollama_generate_stream`.

    Returns ``{"content": str, "thinking": str, "source": str}``.

    Same semantics as :func:`ollama_chat_stream_dual` but with GENERATE-endpoint
    priority (``response`` wins first).

    Accepts ``headers`` so callers can attach an ``Authorization`` header.
    When ``headers`` is None and LLM_API_KEY is set, a bearer header is built
    from it.

    Observability: emits a single [gen-stream-summary] INFO line per call.
    """
    payload = {**payload, "stream": True}
    if headers is None:
        _k = os.environ.get("LLM_API_KEY", "").strip()
        headers = {"Authorization": f"Bearer {_k}"} if _k else {}
    resp = requests.post(url, json=payload, headers=headers,
                         timeout=timeout, stream=True)
    try:
        return _consume_generate_stream(resp)
    finally:
        _safe_close(resp)


def _consume_generate_stream(resp) -> dict:
    """Read an /api/generate stream to exhaustion.

    Split out of :func:`ollama_generate_stream_dual` so the socket can be
    closed in a ``finally`` around the whole read.
    """
    resp.raise_for_status()

    buckets: dict[str, list[str]] = {
        "content": [],
        "reasoning": [],
        "message_content": [],
        "top_message": [],
        "response": [],
    }
    token_counts: dict[str, int] = {k: 0 for k in buckets}
    thinking_buf: list[str] = []

    line_count = 0
    empty_chunk_count = 0
    unparseable = 0
    done_seen = False

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        line_count += 1
        data_str = line.removeprefix("data: ").strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            done_seen = True
            continue
        try:
            obj = _json.loads(data_str)
        except (_json.JSONDecodeError, ValueError):
            unparseable += 1
            continue

        tokens = _extract_chunk_tokens(obj)

        if tokens["thinking"]:
            thinking_buf.append(tokens["thinking"])
            logger.debug("[model-thinking] gen-stream %d chars", len(tokens["thinking"]))

        had_answer_token = False
        for ch in buckets:
            piece = tokens.get(ch) or ""
            if piece:
                buckets[ch].append(piece)
                token_counts[ch] += 1
                had_answer_token = True

        if not had_answer_token:
            empty_chunk_count += 1

        if tokens["done"]:
            done_seen = True

    assembled, source = _pick_answer(buckets, _GENERATE_PRIORITY)
    thinking = "".join(thinking_buf)

    logger.info(
        "[gen-stream-summary] lines=%d response=%d top_message=%d "
        "content=%d reasoning=%d message=%d empty=%d "
        "thinking_chars=%d unparseable=%d done=%s source=%s out_chars=%d",
        line_count, token_counts["response"], token_counts["top_message"],
        token_counts["content"], token_counts["reasoning"],
        token_counts["message_content"], empty_chunk_count,
        len(thinking), unparseable, done_seen, source, len(assembled),
    )

    return {"content": assembled, "thinking": thinking, "source": source}


def ollama_generate_stream(url: str, payload: dict, timeout: int = 300,
                           headers: dict | None = None) -> str:
    """Content-only wrapper around :func:`ollama_generate_stream_dual`.

    Returns the assembled response with ``<think>...</think>`` blocks stripped.

    """
    result = ollama_generate_stream_dual(url, payload, timeout=timeout, headers=headers)
    return strip_thinking(result["content"])


# ============================================
# PROVIDER REGISTRY — OLLAMA LOCAL (PRIMARY + FALLBACK)
# ============================================
# Returned as the content on total failure. Deliberately plain prose with no
# operator diagnostics: callers render this straight to end users.
_USER_FACING_FAILURE = (
    "I can't reach that part of my mind right now — please give me a moment "
    "and try again."
)


class ProviderUnavailable(RuntimeError):
    """No provider is eligible to serve this request.

    Raised by provider selection, not by transport. ``race`` catches this and
    converts it into the documented ``("FAILED", <user-safe text>)`` return, so
    it does not escape the public API.
    """


class LLMProvider:
    """Configuration and failure state for one chat-completions endpoint."""
    def __init__(self, name: str, url: str, api_key: str, model: str,
                 timeout: int = 60, speed_tier: int = 1,
                 backend: str = "ollama"):
        self.name = name
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.speed_tier = speed_tier  # 1=primary, 2=fallback
        self.backend = backend        # transport family; only "ollama" is implemented
        self.consecutive_failures = 0
        self.total_calls = 0
        self.total_successes = 0
        self.avg_latency_ms = 0

    @property
    def available(self) -> bool:
        return self.consecutive_failures < 5

    def record_success(self, latency_ms: float):
        self.consecutive_failures = 0
        self.total_calls += 1
        self.total_successes += 1
        self.avg_latency_ms = (self.avg_latency_ms * 0.8 + latency_ms * 0.2) if self.avg_latency_ms else latency_ms

    def record_failure(self):
        self.consecutive_failures += 1
        self.total_calls += 1


def _check_ollama_models(ollama_base: str) -> List[str]:
    """List model ids advertised by an OpenAI-compatible ``/v1/models`` endpoint.

    Returns an empty list on any failure; callers treat that as "unknown" and
    proceed, rather than blocking startup on a probe.
    """
    try:
        resp = requests.get(f"{ollama_base.rstrip('/')}/v1/models",
                            timeout=MODEL_PROBE_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        return [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    except Exception as e:
        logger.debug("Model probe failed for %s: %s", ollama_base, e)
        return []


def _build_providers() -> List[LLMProvider]:
    """Build the provider list from environment variables.

    Two providers are returned, a primary and a fallback, tried in order by
    :meth:`HomemathEngine.race`. When LLM_HOST_FALLBACK is unset both point at the
    same endpoint, and when LLM_MODEL_FALLBACK is unset both use the same model.
    """
    # Primary endpoint. Override with LLM_HOST.
    llm_base = os.environ.get("LLM_HOST", DEFAULT_LLM_HOST).rstrip("/")
    ollama_primary_url = f"{llm_base}/v1/chat/completions"

    # Secondary endpoint, tried after the primary. Defaults to the primary
    # when LLM_HOST_FALLBACK is unset, so a single-endpoint setup works unchanged.
    ollama_local_base = os.environ.get(
        "LLM_HOST_FALLBACK", llm_base
    ).rstrip("/")
    ollama_fallback_url = f"{ollama_local_base}/v1/chat/completions"

    loaded_models = _check_ollama_models(llm_base)
    loaded_local = _check_ollama_models(ollama_local_base)
    loaded_names = set(m.split(":")[0] for m in loaded_models + loaded_local)
    logger.info("Models advertised by primary endpoint: %s", loaded_models or "NONE")
    logger.info("Models advertised by fallback endpoint: %s", loaded_local or "NONE")

    # Bearer token, when the endpoint requires one. Endpoints that accept
    # unauthenticated requests can leave LLM_API_KEY unset.
    llama_api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not llama_api_key:
        logger.info("LLM_API_KEY not set — sending requests without an Authorization header.")
    # When LLM_HOST_FALLBACK is unset the fallback collapses onto the primary
    # host. Both providers send the same bearer key.
    providers = [
        LLMProvider(
            name="ollama-primary",
            url=ollama_primary_url,
            api_key=llama_api_key or "not-needed",
            model=os.environ.get("LLM_MODEL_PRIMARY", DEFAULT_MODEL),
            timeout=int(os.environ.get("LLM_TIMEOUT_PRIMARY", "600")),
            speed_tier=1,
            backend="ollama",
        ),
        LLMProvider(
            name="ollama-fallback",
            url=ollama_fallback_url,
            api_key=llama_api_key or "not-needed",
            model=os.environ.get("LLM_MODEL_FALLBACK",
                                 os.environ.get("LLM_MODEL_PRIMARY", DEFAULT_MODEL)),
            timeout=int(os.environ.get("LLM_TIMEOUT_FALLBACK", "60")),
            speed_tier=2,
            backend="ollama",
        ),
    ]

    if not loaded_models:
        logger.warning("No Ollama models detected — homemath will retry on each request")
        return providers

    llm_ready = os.environ.get("LLM_READY", "false").lower() == "true"
    if llm_ready:
        return providers

    for p in providers:
        model_base = p.model.split(":")[0]
        if model_base not in loaded_names and p.model not in loaded_models:
            p.consecutive_failures = 3
            logger.info(f"  {p.name} ({p.model}): not loaded yet, deprioritized")
        else:
            logger.info(f"  {p.name} ({p.model}): AVAILABLE")

    if not providers:
        logger.error("No LLM providers configured — homemath cannot operate")

    return providers


# ============================================
# TASK CLASSIFIER (kept for logging/analytics — all tasks route to primary)
# ============================================
_CODE_KEYWORDS = re.compile(
    r'\b(code|sql|query|function|class|script|program|debug|refactor|implement|'
    r'endpoint|api|json|schema|database|migration|regex)\b',
    re.IGNORECASE
)
_SIMPLE_KEYWORDS = re.compile(
    r'\b(extract|format|yes|no|list|summarize|classify|tag|label|parse|convert|'
    r'rewrite|shorten|expand|translate|count|check)\b',
    re.IGNORECASE
)


_CHAT_KEYWORDS = re.compile(
    r'\b(hello|hi|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|'
    r'what\s+can\s+you|who\s+are\s+you|how\s+are\s+you|thanks|thank\s+you|bye|goodbye)\b',
    re.IGNORECASE
)
_COMPLEX_KEYWORDS = re.compile(
    r'\b(write|create|generate|compose|draft|analyze|compare|explain\s+why|'
    r'strategy|research|report|evaluate|audit|plan|design|optimize|'
    r'article|essay)\b',
    re.IGNORECASE
)


def classify_task(messages: list) -> str:
    """Classify task type for provider routing — THIN ADAPTER.

    Canonical classification lives in :mod:`homemath.classifier`. This
    function is the coarse string form kept for callers that switch on a
    string:

        greeting | simple | code | domain | general

    Prefer ``classifier.classify_task_class``, which returns a
    :class:`classifier.TaskClass`.
    """
    try:
        from . import classifier as _c  # local import keeps bootstrap order safe
        return _c.legacy_task_string(_c.classify_task_class(messages))
    except Exception as e:  # pragma: no cover — fallback only if module unloads
        logger.warning("classifier unavailable (%s) — legacy inline path", e)
        # Inline legacy path preserved below for resilience.
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        ).strip()
        text = " ".join(m.get("content", "") for m in messages if m.get("content"))
        if last_user and _CHAT_KEYWORDS.search(last_user) and len(last_user.split()) <= 8:
            return "greeting"
        if (
            last_user
            and len(last_user.split()) <= 12
            and _SIMPLE_KEYWORDS.search(last_user)
            and not _COMPLEX_KEYWORDS.search(last_user)
        ):
            return "simple"
        if len(_CODE_KEYWORDS.findall(text)) >= 2:
            return "code"
        # No domain branch here: domain keywords are caller-configured and live
        # in homemath.classifier. This path only runs if that import failed.
        return "general"


_THINK_KEYWORDS = re.compile(
    r'\b(judge|critique|evaluate|assess|score|grade|'
    r'plan|strategy|roadmap|analyze|reasoning|'
    r'why|trade.?off|decide|choose|compare\s+and|'
    r'debug|root\s+cause|diagnose)\b',
    re.IGNORECASE
)


# ============================================================================
# Convenience adapter. Prefer binding a policy via homemath.policy and letting
# this read it, or calling classifier.thinking_from_intent directly.
# ============================================================================
def needs_thinking(messages: list) -> bool:
    """Decide whether to enable model reasoning mode for this request.

    Policy is a pure function of the classified task
    class. No keyword scans on the policy path. The historical keyword scan —
    which at one point looked at SYSTEM messages and caused phantom/dream
    spurious reasoning on background calls — is structurally unreachable.

    Resolution order:
      1) Bound per-turn policy (`inference_policy.get_current_policy()`) —
         honoured when an upstream caller has already classified.
      2) Unified classifier (`classifier.thinking_from_intent`) — derives the
         decision from the classified TaskClass only, no keyword scan.

    Returns True → thinking enabled ; False → /no_think injected.
    """
    # 1. Bound per-turn policy (upstream classifier may have already decided).
    try:
        from .policy import get_current_policy
        _bound = get_current_policy()
        if _bound is not None:
            return bool(_bound.get("thinking_required", False))
    except Exception:
        pass

    # 2. Pure intent-class policy. No text scan.
    try:
        from . import classifier as _c
        return _c.thinking_from_intent(_c.classify_task_class(messages))
    except Exception as e:  # pragma: no cover — classifier import should never fail
        logger.warning("needs_thinking: classifier unavailable (%s); defaulting False", e)
        return False


# ============================================
# PARALLEL RACING ENGINE
# ============================================
class HomemathEngine:
    """
    Chat engine over one or two OpenAI-compatible endpoints.

    Tries a selected provider first and falls through to the other only on
    failure -- see :meth:`race`, which despite its name is sequential. Caches
    responses in memory, or in Redis when REDIS_URL is set.
    """

    def __init__(self):
        self.providers = _build_providers()
        self._provider_map = {p.name: p for p in self.providers}
        # Bounded in-memory cache, FIFO eviction (not LRU — reads do not
        # promote). Guarded by _cache_lock: race() is called from caller
        # threads and dict mutation here is not atomic across check-then-delete.
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._redis = None
        self._init_redis()
        logger.info(f"homemath initialized: {len(self.providers)} providers (Ollama local mode)")
        for p in self.providers:
            logger.info("  %s: %s (tier %s, backend=%s)", p.name, p.model, p.speed_tier, p.backend)

    def _init_redis(self):
        """Try to connect to Redis for persistent caching."""
        try:
            import redis
            redis_url = os.environ.get("REDIS_URL", "").strip()
            if not redis_url:
                raise RuntimeError("REDIS_URL not set")
            self._redis = redis.from_url(redis_url, decode_responses=True,
                                          socket_connect_timeout=2)
            self._redis.ping()
            logger.info("  Redis cache: connected")
        except Exception:
            logger.info("  Redis cache: unavailable, using in-memory")
            self._redis = None

    def _cache_key(self, messages: list) -> str:
        """Generate cache key from message content."""
        content = "|".join(m.get("content", "") for m in messages)
        return f"homemath:llm:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def _cache_get(self, key: str) -> Optional[str]:
        """Check cache (Redis first, then in-memory)."""
        if self._redis:
            try:
                val = self._redis.get(key)
                if val:
                    self._cache_hits += 1
                    return val
            except Exception as e:
                logger.warning(f"homemath Redis cache read failed for key {key}: {e}")
        with self._cache_lock:
            val = self._cache.get(key)
        if val:
            self._cache_hits += 1
        return val

    def _cache_set(self, key: str, value: str, ttl: int = 3600):
        """Store in cache (both Redis and in-memory).

        Never raises: a cache write is best-effort bookkeeping, and the caller
        has already paid for a real generation by this point. Letting an
        exception out here would discard that answer -- see :meth:`race`.
        """
        try:
            with self._cache_lock:
                self._cache[key] = value
                # Keep in-memory cache bounded. pop(k, None) rather than del so
                # a concurrent eviction of the same key cannot raise KeyError.
                if len(self._cache) > 500:
                    for k in list(self._cache.keys())[:250]:
                        self._cache.pop(k, None)
        except Exception as e:
            logger.warning("homemath in-memory cache write failed for key %s: %s", key, e)
        if self._redis:
            try:
                self._redis.setex(key, ttl, value)
            except Exception as e:
                logger.warning(f"homemath Redis cache write failed for key {key}: {e}")

    def _fallback_is_distinct(self) -> bool:
        """True when the fallback is a different model or a different endpoint.

        With LLM_HOST_FALLBACK and LLM_MODEL_FALLBACK unset, both providers are
        built from the same host and model, so there is no downgrade to refuse.
        """
        primary = self._provider_map.get("ollama-primary")
        fallback = self._provider_map.get("ollama-fallback")
        if not primary or not fallback:
            return False
        return (primary.model, primary.url) != (fallback.model, fallback.url)

    def _select_provider(self, messages: list) -> LLMProvider:
        """Select the best available Ollama provider for this task.

        Routing contract (matches classify_task output):
          greeting/simple → fallback endpoint FIRST (cheaper/faster model)
          code/domain/general → primary endpoint FIRST
        """
        task_type = classify_task(messages)
        primary = self._provider_map.get("ollama-primary")
        fallback = self._provider_map.get("ollama-fallback")

        # Greetings/simple → prefer the fallback endpoint; primary only if it is down.
        if task_type in ("greeting", "simple"):
            if fallback and fallback.available:
                logger.info(f"homemath routing → ollama-fallback [{fallback.model}] ({task_type} task)")
                return fallback
            if primary and primary.available:
                logger.warning("Fallback endpoint unavailable, using primary for %s", task_type)
                return primary
        else:
            # Content / code / domain / general → primary endpoint.
            if primary and primary.available:
                logger.info(f"homemath routing → ollama-primary [{primary.model}] ({task_type} task)")
                return primary
            # Primary is above its failure threshold. Downgrading a content
            # task to a genuinely smaller fallback model is a quality cliff, so
            # it is refused -- but only when the fallback really is a different
            # model. With LLM_HOST_FALLBACK / LLM_MODEL_FALLBACK unset the two
            # providers are the same model at the same URL, and refusing there
            # means refusing to retry the primary at all.
            if fallback and fallback.available:
                if self._fallback_is_distinct():
                    logger.error(
                        "Primary endpoint unavailable and task_type='%s' — refusing to "
                        "downgrade content generation to fallback model '%s'.",
                        task_type, fallback.model,
                    )
                    raise ProviderUnavailable(
                        f"The primary model is unavailable and the fallback model "
                        f"({fallback.model}) is not configured for {task_type} content. "
                        f"Check endpoint connectivity."
                    )
                logger.warning(
                    "Primary above failure threshold; fallback is the same model at "
                    "the same endpoint, so retrying there for %s.", task_type,
                )
                return fallback

        # Both depleted — return first provider regardless (will retry)
        logger.warning(f"homemath: all providers above failure threshold, forcing first provider")
        return self.providers[0]

    def _single_call(self, provider: LLMProvider, messages: list,
                     temperature: float, think: bool = True,
                     priority: int = PRIORITY_NORMAL) -> Tuple[str, str, float]:
        """Execute a single provider call. Dispatches to HF Space or Ollama based on backend type.

        Args:
            priority: Lock priority for Ollama calls. PRIORITY_HIGH for workflow steps,
                      PRIORITY_LOW for bus-triggered background, PRIORITY_NORMAL default.

        Returns (provider_name, content, latency_ms).
        """
        return self._call_ollama(provider, messages, temperature,
                                 think=think, priority=priority)

    @staticmethod
    def _budget_for(messages: list, think: bool) -> int:
        """Output token budget for this request, from its task class.

        Falls back to the old fixed pair (4096 thinking / 512 not) only if the
        classifier cannot be imported.
        """
        try:
            from . import classifier as _c
            return _c.max_tokens(_c.classify_task_class(messages))
        except Exception as e:  # pragma: no cover — classifier import should never fail
            logger.warning("budget: classifier unavailable (%s); using fixed default", e)
            return 4096 if think else 512

    @staticmethod
    def _prepend_no_think(messages: list) -> list:
        """Return a copy with ``/no_think`` on the last user message.

        Content that is not a plain string (OpenAI multimodal part lists) is
        left alone rather than concatenated, which would raise TypeError.
        """
        out = [dict(m) for m in messages]
        for msg in reversed(out):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = "/no_think " + content
                else:
                    logger.debug(
                        "no_think: last user content is %s, not str — left unmodified",
                        type(content).__name__,
                    )
                break
        return out

    def _call_ollama(self, provider: LLMProvider, messages: list,
                     temperature: float, think: bool = True,
                     priority: int = PRIORITY_NORMAL) -> Tuple[str, str, float]:
        """Execute an Ollama provider call via OpenAI-compatible endpoint.

        Acquires OLLAMA_CHAT_LOCK with the given priority to prevent concurrent
        requests from queuing inside Ollama (which serializes inference and adds
        dead wait time). Workflow-critical calls (PRIORITY_HIGH) are served before
        bus-triggered background calls (PRIORITY_LOW).

        When think=False, prepends /no_think to the last user message to suppress
        the model's <think> reasoning block, cutting a large amount of latency
        on reasoning-capable models.

        The output token budget comes from the request's task class
        (:func:`classifier.max_tokens`), not from a fixed pair of constants.

        Returns (provider_name, content, latency_ms).
        """
        start = time.time()
        headers = {"Content-Type": "application/json"}
        if provider.api_key and provider.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {provider.api_key}"

        budget = self._budget_for(messages, think)

        # Conditionally suppress reasoning mode for simple queries
        effective_messages = messages
        if not think:
            effective_messages = self._prepend_no_think(messages)
            logger.info("homemath: thinking DISABLED for this request (simple query)")

        # Enforce the token budget before sending.
        if _TRIMMER_AVAILABLE:
            effective_messages = _trim_messages(effective_messages, caller="homemath._call_ollama")

        # Per-model locking: different models run in parallel (concurrent model requests)
        _lock_model = provider.model or "_default"
        queue_depth = OLLAMA_CHAT_LOCK.queue_depth(model=_lock_model)
        if queue_depth > 0:
            logger.info(f"homemath: waiting for {_lock_model} lock (priority={priority}, queue_depth={queue_depth})")
        OLLAMA_CHAT_LOCK.acquire(priority=priority, model=_lock_model)
        try:
            content = ollama_chat_stream(
                provider.url,
                payload={
                    "model": provider.model,
                    "messages": effective_messages,
                    "temperature": temperature,
                    "max_tokens": budget,
                },
                timeout=provider.timeout,
                headers=headers,
            )
        finally:
            OLLAMA_CHAT_LOCK.release(model=_lock_model)

        if not content and think:
            # Parser's channel-priority fallback produced nothing — the model
            # emitted only thinking-private tokens on all known channels.
            # Retry with /no_think to force a direct answer, keeping the same
            # budget so a long reply is not truncated. Retrying at a reduced
            # budget drops real answers on long generations.
            logger.warning(
                "Empty answer after think pass (all channels empty) — "
                "retrying with /no_think at full budget"
            )
            no_think_msgs = self._prepend_no_think(effective_messages)
            OLLAMA_CHAT_LOCK.acquire(priority=priority, model=_lock_model)
            try:
                content = ollama_chat_stream(
                    provider.url,
                    payload={
                        "model": provider.model,
                        "messages": no_think_msgs,
                        "temperature": temperature,
                        "max_tokens": budget,
                    },
                    timeout=provider.timeout,
                    headers=headers,
                )
            finally:
                OLLAMA_CHAT_LOCK.release(model=_lock_model)
        if not content:
            raise ValueError(
                "Empty response from model after think pass and /no_think retry"
                if think else
                "Empty response from model (/no_think pass)"
            )
        latency = (time.time() - start) * 1000
        provider.record_success(latency)
        return provider.name, content, latency

    def race(self, messages: list, temperature: float = 0.7,
             use_cache: bool = True,
             priority: int = PRIORITY_NORMAL) -> Tuple[str, str]:
        """
        Sequential failover: try the selected provider, fall through to the
        other only on failure.

        The name is historical. This does not issue concurrent requests. When
        providers point at one server that serializes inference, racing only
        doubles queue depth for no throughput gain, and OLLAMA_CHAT_LOCK in
        ``_call_ollama`` serializes same-model calls anyway.

        Reasoning mode is enabled or suppressed per request based on the
        classified task; simple turns skip it, which is a large latency saving
        on reasoning-capable models.

        Args:
            priority: Lock priority. Lower number is served first.

        Returns:
            ``(provider_name, content)``. Never raises for an inference or
            routing failure: total failure returns ``("FAILED", <user-safe
            text>)`` so a caller cannot accidentally show an operator
            diagnostic to an end user.
        """
        # --- Check cache first ---
        cache_key = None
        if use_cache:
            cache_key = self._cache_key(messages)
            cached = self._cache_get(cache_key)
            if cached:
                logger.info(f"homemath cache HIT ({self._cache_hits} total)")
                return "cache", cached
            self._cache_misses += 1

        # Decide whether to enable thinking for this request
        think = needs_thinking(messages)
        if not think:
            logger.info("homemath: needs_thinking=False — will suppress <think> for faster response")

        # Build ordered provider chain: best provider first, then fallbacks.
        # Selection can legitimately find nothing eligible; that is a failure
        # of this call, not an exception for the caller to handle.
        try:
            first = self._select_provider(messages)
        except ProviderUnavailable as e:
            logger.error("HOMEMATH TOTAL FAILURE: no eligible provider — %s", e)
            return "FAILED", _USER_FACING_FAILURE
        provider_chain = [first]
        for p in self.providers:
            if p.name != first.name and p.available:
                provider_chain.append(p)

        # Sequential: try each provider, stop on first success
        for provider in provider_chain:
            try:
                name, content, latency = self._single_call(
                    provider, messages, temperature, think=think, priority=priority
                )
            except Exception as e:
                provider.record_failure()
                logger.warning(f"homemath [{provider.name}] failed: {e} — trying next")
                continue
            # Past this point the generation succeeded and is owed to the
            # caller. Caching is best-effort bookkeeping and must never be able
            # to turn a good answer into a provider failure and a retry.
            logger.info(f"homemath [{name}] {latency:.0f}ms ({len(content)} chars) think={think} priority={priority}")
            if use_cache and cache_key:
                self._cache_set(cache_key, content)
            return name, content

        # ALL providers failed
        logger.error(f"HOMEMATH TOTAL FAILURE: all {len(self.providers)} providers unresponsive")
        return "FAILED", _USER_FACING_FAILURE

    def status(self) -> dict:
        """Return engine status for health check."""
        llm_base = os.environ.get("LLM_HOST", DEFAULT_LLM_HOST).rstrip("/")
        return {
            "engine": "homemath",
            "mode": "ollama_local",
            "llm_url": llm_base,
            "providers": [
                {
                    "name": p.name,
                    "model": p.model,
                    "backend": p.backend,
                    "available": p.available,
                    "speed_tier": p.speed_tier,
                    "total_calls": p.total_calls,
                    "success_rate": f"{p.total_successes/max(p.total_calls,1)*100:.0f}%",
                    "avg_latency_ms": f"{p.avg_latency_ms:.0f}",
                    "consecutive_failures": p.consecutive_failures
                }
                for p in self.providers
            ],
            "cache": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": f"{self._cache_hits/max(self._cache_hits+self._cache_misses,1)*100:.0f}%",
                "in_memory_entries": len(self._cache),
                "redis_connected": self._redis is not None
            }
        }



# ============================================
# GLOBAL INSTANCE
# ============================================
_engine: Optional[HomemathEngine] = None

def get_engine() -> HomemathEngine:
    """Get or create the global homemath engine."""
    global _engine
    if _engine is None:
        _engine = HomemathEngine()
    return _engine


def homemath_chat(messages: list, temperature: float = 0.7, use_cache: bool = True,
               priority: int = PRIORITY_NORMAL) -> str:
    """Drop-in replacement for ollama_chat(). Returns content string.

    Args:
        priority: Lock priority for Ollama calls. PRIORITY_HIGH for workflow
                  steps, PRIORITY_LOW for bus-triggered background calls.
                  Default PRIORITY_NORMAL for direct API calls.
    """
    engine = get_engine()
    provider, content = engine.race(messages, temperature, use_cache, priority=priority)
    return content


# Names that make up the supported surface. Without this, ``import *`` would
# also export os, re, requests, threading and the typing aliases.
__all__ = [
    "__version__",
    # High-level entry points
    "homemath_chat",
    "get_engine",
    "HomemathEngine",
    # Streaming transport
    "ollama_chat_stream",
    "ollama_chat_stream_dual",
    "ollama_generate_stream",
    "ollama_generate_stream_dual",
    "strip_thinking",
    # Classification and policy adapters
    "classify_task",
    "needs_thinking",
    # Providers and errors
    "LLMProvider",
    "ProviderUnavailable",
    # Concurrency
    "PriorityLock",
    "OLLAMA_CHAT_LOCK",
    "PRIORITY_HIGH",
    "PRIORITY_NORMAL",
    "PRIORITY_LOW",
    # Defaults
    "DEFAULT_LLM_HOST",
    "DEFAULT_MODEL",
    "MODEL_PROBE_TIMEOUT",
]
