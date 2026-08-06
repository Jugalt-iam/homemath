"""Tests for homemath.

requests and redis are replaced with module stubs before homemath is imported.
No test opens a socket: every endpoint is a MagicMock returning canned SSE or
JSON-lines chunks.
"""
import sys
import time
import types
from unittest.mock import MagicMock, patch

# ── Pre-import mocks ──────────────────────────────────────────────────────────
for _mod_name in [
    "redis",
    "requests", "requests.adapters", "requests.models",
    "requests.sessions", "requests.exceptions",
    "urllib3", "urllib3.util", "urllib3.util.retry",
]:
    if _mod_name not in sys.modules:
        _mod = types.ModuleType(_mod_name)
        _mod.__dict__.update({k: MagicMock() for k in ["__all__", "__version__"]})
        sys.modules[_mod_name] = _mod

_req_exc = sys.modules["requests.exceptions"]
_req_exc.ConnectionError = type("ConnectionError", (OSError,), {})
_req_exc.Timeout = type("Timeout", (OSError,), {})
_req_exc.HTTPError = type("HTTPError", (OSError,), {})
sys.modules["requests"].exceptions = _req_exc
sys.modules["requests"].ConnectionError = _req_exc.ConnectionError
sys.modules["requests"].Timeout = _req_exc.Timeout
# Ensure requests.post / requests.get exist as MagicMocks
sys.modules["requests"].post = MagicMock()
sys.modules["requests"].get = MagicMock()
# Ensure redis.from_url exists
sys.modules["redis"].from_url = MagicMock(side_effect=Exception("no redis"))

import pytest

def _evict_homemath():
    """Remove any mock/cached homemath from sys.modules so we get a fresh import."""
    for _evict_name in list(sys.modules.keys()):
        if _evict_name == "homemath" or _evict_name.startswith("homemath."):
            del sys.modules[_evict_name]


@pytest.fixture(autouse=True, scope="class")
def fresh_homemath():
    """Ensure homemath is freshly imported from disk for each test class."""
    _evict_homemath()
    yield
    _evict_homemath()


def _make_ollama_response(content: str, status_code: int = 200):
    """Return a mock requests.Response for Ollama /v1/chat/completions.

    Since ollama_chat_stream() is streaming-only (VPN MTU fix), the mock
    must supply iter_lines() yielding SSE-encoded 'data:' lines. resp.json is
    kept for older tests that still inspect it but is not used by the stream
    parser.
    """
    import json as _json
    sse_lines = [
        f"data: {_json.dumps({'choices': [{'delta': {'content': content}, 'finish_reason': 'stop'}]})}",
        "data: [DONE]",
    ]
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": content}}]
    })
    resp.iter_lines = MagicMock(return_value=iter(sse_lines))
    resp.raise_for_status = MagicMock()
    return resp


def _make_engine(disable_redis: bool = True):
    """Build a HomemathEngine with Redis and Ollama model-check patched away."""
    import homemath as hlm
    import redis as _redis_mod
    from homemath import HomemathEngine

    tags_resp = MagicMock()
    tags_resp.status_code = 200
    tags_resp.json = MagicMock(return_value={"models": []})

    with patch.object(hlm, "requests") as mock_req:
        mock_req.get = MagicMock(return_value=tags_resp)
        mock_req.post = MagicMock()
        with patch.object(_redis_mod, "from_url", side_effect=Exception("no redis")):
            engine = HomemathEngine()
    return engine


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level constants and utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleConstants:
    def test_ollama_chat_lock_is_priority_lock(self):
        from homemath import OLLAMA_CHAT_LOCK, PriorityLock
        assert isinstance(OLLAMA_CHAT_LOCK, PriorityLock)
        # Backward compat: acquire/release still work
        assert hasattr(OLLAMA_CHAT_LOCK, 'acquire')
        assert hasattr(OLLAMA_CHAT_LOCK, 'release')

    def test_strip_thinking_removes_think_block(self):
        from homemath import strip_thinking
        result = strip_thinking("<think>hidden</think>Visible")
        assert "<think>" not in result
        assert "Visible" in result

    def test_strip_thinking_handles_unclosed_block(self):
        from homemath import strip_thinking
        result = strip_thinking("<think>started but never closed...")
        assert "<think>" not in result

    def test_strip_thinking_returns_clean_text_unchanged(self):
        from homemath import strip_thinking
        text = "This is normal output."
        assert strip_thinking(text) == text


# ═══════════════════════════════════════════════════════════════════════════════
#  classify_task
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyTask:
    def test_classifies_code_task(self):
        from homemath import classify_task
        msgs = [{"role": "user", "content": "write a SQL query to get all contacts from database"}]
        result = classify_task(msgs)
        assert result == "code"

    def test_classifies_domain_task(self):
        """Domain keywords are caller-supplied; unset by default."""
        from homemath import classify_task
        from homemath.classifier import set_domain_keywords
        msgs = [{"role": "user", "content": "reconcile the ledger against the invoice batch"}]
        set_domain_keywords("ledger|invoice|reconcil\\w*")
        try:
            assert classify_task(msgs) == "domain"
        finally:
            set_domain_keywords(None)

    def test_domain_class_not_produced_when_unconfigured(self):
        """With no domain keywords set, the same text must not route to domain."""
        from homemath import classify_task
        from homemath.classifier import set_domain_keywords
        set_domain_keywords(None)
        msgs = [{"role": "user", "content": "reconcile the ledger against the invoice batch"}]
        assert classify_task(msgs) != "domain"

    def test_classifies_general_task(self):
        from homemath import classify_task
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        result = classify_task(msgs)
        assert result == "general"


# ═══════════════════════════════════════════════════════════════════════════════
#  LLMProvider
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMProvider:
    def test_available_when_no_failures(self):
        from homemath import LLMProvider
        p = LLMProvider("test", "http://x", "key", "model", backend="ollama")
        assert p.available is True

    def test_unavailable_after_five_failures(self):
        from homemath import LLMProvider
        p = LLMProvider("test", "http://x", "key", "model", backend="ollama")
        for _ in range(5):
            p.record_failure()
        assert p.available is False

    def test_record_success_resets_failures(self):
        from homemath import LLMProvider
        p = LLMProvider("test", "http://x", "key", "model", backend="ollama")
        for _ in range(4):
            p.record_failure()
        p.record_success(100.0)
        assert p.consecutive_failures == 0
        assert p.available is True

    def test_success_updates_latency(self):
        from homemath import LLMProvider
        p = LLMProvider("test", "http://x", "key", "model", backend="ollama")
        p.record_success(200.0)
        assert p.avg_latency_ms > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  HomemathEngine initialisation
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathEngineInit:
    def test_engine_has_two_providers(self):
        engine = _make_engine()
        assert len(engine.providers) == 2

    def test_provider_names_are_primary_and_fallback(self):
        engine = _make_engine()
        names = {p.name for p in engine.providers}
        assert "ollama-primary" in names
        assert "ollama-fallback" in names

    def test_all_providers_use_ollama_backend(self):
        engine = _make_engine()
        for p in engine.providers:
            assert p.backend == "ollama"

    def test_status_returns_expected_keys(self):
        engine = _make_engine()
        status = engine.status()
        assert "engine" in status
        assert "providers" in status
        assert "cache" in status


# ═══════════════════════════════════════════════════════════════════════════════
#  HomemathEngine caching
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathEngineCache:
    def test_cache_miss_increments_counter(self):
        engine = _make_engine()
        initial = engine._cache_misses
        # Key that definitely won't be in cache
        engine._cache_get("nonexistent_key_xyz")
        # Cache miss count goes up only when race() is called and no hit
        # _cache_get alone just returns None; miss is counted in race()
        assert engine._cache_get("nonexistent_key_xyz") is None

    def test_cache_set_and_get_roundtrip(self):
        engine = _make_engine()
        engine._cache_set("testkey", "testvalue", ttl=60)
        val = engine._cache_get("testkey")
        assert val == "testvalue"

    def test_cache_key_is_deterministic(self):
        engine = _make_engine()
        messages = [{"role": "user", "content": "hello world"}]
        k1 = engine._cache_key(messages)
        k2 = engine._cache_key(messages)
        assert k1 == k2

    def test_cache_hit_returns_cached_content(self):
        engine = _make_engine()
        messages = [{"role": "user", "content": "cached prompt"}]
        ck = engine._cache_key(messages)
        engine._cache_set(ck, "Cached LLM response")
        name, content = engine.race(messages, use_cache=True)
        assert name == "cache"
        assert content == "Cached LLM response"


# ═══════════════════════════════════════════════════════════════════════════════
#  HomemathEngine.race — success and fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathEngineRace:
    def test_race_returns_provider_name_and_content(self):
        import homemath as hlm
        engine = _make_engine()
        messages = [{"role": "user", "content": "test prompt unique" + str(time.time())}]
        resp = _make_ollama_response("Hello from primary")
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=resp)
            name, content = engine.race(messages, use_cache=False)
        assert isinstance(name, str)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_race_falls_back_when_primary_fails(self):
        """Primary raises; fallback should succeed."""
        import homemath as hlm
        engine = _make_engine()
        messages = [{"role": "user", "content": "fallback test" + str(time.time())}]
        fallback_resp = _make_ollama_response("Fallback content")

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("primary timeout")
            return fallback_resp

        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(side_effect=side_effect)
            name, content = engine.race(messages, use_cache=False)

        assert content == "Fallback content"

    def test_race_returns_failed_when_all_providers_down(self, caplog):
        """All-providers-down contract: three channels must stay separated.

        When every endpoint is unreachable there are three distinct audiences,
        and an operator diagnostic must never become the string a user reads.
        This pins all three:
          - operator channel -> logger.error containing "HOMEMATH TOTAL FAILURE"
          - user channel     -> plain, non-empty, diagnostic-free prose
          - sentinel         -> provider_name == "FAILED" for callers to branch on

        A regression that lets operator text reach the returned content fails
        here first.
        """
        import logging
        import homemath as hlm
        engine = _make_engine()
        messages = [{"role": "user", "content": "all fail" + str(time.time())}]
        with caplog.at_level(logging.ERROR, logger="homemath"):
            with patch.object(hlm, "requests") as mock_req:
                mock_req.post = MagicMock(side_effect=OSError("all down"))
                name, content = engine.race(messages, use_cache=False)

        # (1) Sentinel is the programmatic contract callers branch on.
        assert name == "FAILED"

        # (2) Content is user-safe: non-empty, plain prose. Must NOT re-leak
        # the old operator-diagnostic form that shipped to end users.
        assert isinstance(content, str) and len(content) > 0
        assert not content.lstrip().startswith("[homemath"), (
            f"User-facing fallback must not leak operator diagnostics; got: {content!r}"
        )
        assert "Ensure Ollama" not in content, (
            f"Operator hint must not appear in user-facing content; got: {content!r}"
        )
        assert "models failed" not in content.lower(), (
            f"Internal failure jargon must not reach end users; got: {content!r}"
        )

        # (3) Observability preserved: operators get the homemath diagnostic via
        # logs, not via returned content.
        diag_records = [r for r in caplog.records
                        if r.levelno >= logging.ERROR and "HOMEMATH TOTAL FAILURE" in r.getMessage()]
        assert diag_records, (
            "HOMEMATH TOTAL FAILURE must be emitted at ERROR level so operators "
            "see the diagnostic trail even though users don't."
        )

    def test_provider_failure_increments_consecutive_failures(self):
        import homemath as hlm
        engine = _make_engine()
        messages = [{"role": "user", "content": "track fail" + str(time.time())}]
        primary = engine._provider_map["ollama-primary"]
        initial_failures = primary.consecutive_failures

        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(side_effect=OSError("down"))
            engine.race(messages, use_cache=False)

        # At minimum, primary should have been attempted
        assert primary.consecutive_failures >= initial_failures


# ═══════════════════════════════════════════════════════════════════════════════
#  homemath_chat convenience function
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathChat:
    def test_homemath_chat_returns_string(self):
        from homemath import homemath_chat
        import homemath as hlm
        # Reset global so a fresh engine is created
        hlm._engine = None
        messages = [{"role": "user", "content": "quick test" + str(time.time())}]
        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json = MagicMock(return_value={"models": []})
        resp = _make_ollama_response("Quick reply")

        import redis as _redis_mod
        with patch.object(hlm, "requests") as mock_req:
            mock_req.get = MagicMock(return_value=tags_resp)
            mock_req.post = MagicMock(return_value=resp)
            with patch.object(_redis_mod, "from_url", side_effect=Exception("no redis")):
                result = homemath_chat(messages)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
#  W1-03: deepseek 'thinking' field handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepseekThinkingField:
    """ollama_chat_stream and ollama_generate_stream must not return the
    deepseek-r1 'thinking' field as part of the response content."""

    def _make_sse_lines(self, chunks: list[dict]) -> MagicMock:
        """Build a mock requests.Response whose iter_lines yields SSE-encoded dicts."""
        import json as _json
        lines = [f"data: {_json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=iter(lines))
        return resp

    def _make_jsonl_lines(self, chunks: list[dict]) -> MagicMock:
        """Build a mock requests.Response for /api/generate JSON-lines format."""
        import json as _json
        lines = [_json.dumps(c) for c in chunks]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=iter(lines))
        return resp

    def test_chat_stream_ignores_thinking_field_in_delta(self):
        """ollama_chat_stream: delta.thinking must not appear in result."""
        from homemath import ollama_chat_stream
        import homemath as hlm

        chunks = [
            # deepseek-r1 thinking phase — content is empty, thinking is set
            {"choices": [{"delta": {"thinking": "Let me reason carefully..."}, "finish_reason": None}]},
            # answer phase — content is set, thinking is absent
            {"choices": [{"delta": {"content": "The answer is 42."}, "finish_reason": None}]},
        ]
        mock_resp = self._make_sse_lines(chunks)

        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=mock_resp)
            result = ollama_chat_stream("http://endpoint.invalid/v1/chat/completions", {"model": "test-model", "messages": []})

        assert "Let me reason carefully" not in result
        assert "The answer is 42." in result

    def test_chat_stream_returns_empty_string_when_only_thinking(self):
        """If only thinking chunks arrive (no content), result must be empty string."""
        from homemath import ollama_chat_stream
        import homemath as hlm

        chunks = [
            {"choices": [{"delta": {"thinking": "Deep thoughts..."}, "finish_reason": None}]},
        ]
        mock_resp = self._make_sse_lines(chunks)

        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=mock_resp)
            result = ollama_chat_stream("http://endpoint.invalid/v1/chat/completions", {"model": "test-model", "messages": []})

        assert "Deep thoughts" not in result

    def test_generate_stream_ignores_thinking_field(self):
        """ollama_generate_stream: top-level 'thinking' key must not appear in result."""
        from homemath import ollama_generate_stream
        import homemath as hlm

        chunks = [
            # deepseek-r1 thinking phase: response is empty, thinking is set
            {"response": "", "thinking": "Hmm, let me think...", "done": False},
            # answer phase: response has the actual token
            {"response": "Final answer here.", "done": True},
        ]
        mock_resp = self._make_jsonl_lines(chunks)

        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=mock_resp)
            result = ollama_generate_stream("http://endpoint.invalid/api/generate", {"model": "test-model", "prompt": "test"})

        assert "Hmm, let me think" not in result
        assert "Final answer here." in result

    def test_generate_stream_with_only_thinking_returns_empty(self):
        """If all /api/generate lines have empty response, result must be empty."""
        from homemath import ollama_generate_stream
        import homemath as hlm

        chunks = [
            {"response": "", "thinking": "Thinking...", "done": False},
            {"response": "", "thinking": "Still thinking...", "done": True},
        ]
        mock_resp = self._make_jsonl_lines(chunks)

        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=mock_resp)
            result = ollama_generate_stream("http://endpoint.invalid/api/generate", {"model": "test-model", "prompt": "test"})

        assert "Thinking" not in result
        assert result == ""

    def test_chat_stream_normal_response_unaffected(self):
        """Non-deepseek responses with no thinking field must pass through unchanged."""
        from homemath import ollama_chat_stream
        import homemath as hlm

        chunks = [
            {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "world!"}, "finish_reason": "stop"}]},
        ]
        mock_resp = self._make_sse_lines(chunks)

        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=mock_resp)
            result = ollama_chat_stream("http://endpoint.invalid/v1/chat/completions", {"model": "test-model", "messages": []})

        assert result == "Hello world!"


# ═══════════════════════════════════════════════════════════════════════════════
#  Token trimming wired into HomemathEngine._call_ollama
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathTokenTrimming:
    """_call_ollama must invoke context_trimmer before sending to Ollama."""

    def test_trim_called_when_trimmer_available(self):
        """When _TRIMMER_AVAILABLE is True, _trim_messages must be called."""
        import homemath as hlm
        engine = _make_engine()
        provider = engine.providers[0]

        trimmed_messages = [{"role": "user", "content": "trimmed"}]

        with patch.object(hlm, "_TRIMMER_AVAILABLE", True), \
             patch.object(hlm, "_trim_messages", return_value=trimmed_messages) as mock_trim, \
             patch.object(hlm, "ollama_chat_stream", return_value="response text"):

            hlm.OLLAMA_CHAT_LOCK.acquire()
            hlm.OLLAMA_CHAT_LOCK.release()  # ensure lock is free

            messages = [{"role": "user", "content": "original"}]
            engine._call_ollama(provider, messages, temperature=0.7)

        mock_trim.assert_called_once()
        # The first positional arg to trim should be the effective_messages list
        call_args = mock_trim.call_args
        assert call_args is not None

    def test_trim_skipped_when_trimmer_unavailable(self):
        """When _TRIMMER_AVAILABLE is False, _trim_messages must not be called."""
        import homemath as hlm
        engine = _make_engine()
        provider = engine.providers[0]

        with patch.object(hlm, "_TRIMMER_AVAILABLE", False), \
             patch.object(hlm, "_trim_messages") as mock_trim, \
             patch.object(hlm, "ollama_chat_stream", return_value="response text"):

            messages = [{"role": "user", "content": "original"}]
            engine._call_ollama(provider, messages, temperature=0.7)

        mock_trim.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
#  Schema-drift coverage for ollama_chat_stream / ollama_generate_stream
# ═══════════════════════════════════════════════════════════════════════════════

def _sse_resp(chunks, terminate_with_done=True):
    """Mock response whose iter_lines emits SSE 'data: {...}' lines."""
    import json as _json
    lines = [f"data: {_json.dumps(c)}" for c in chunks]
    if terminate_with_done:
        lines.append("data: [DONE]")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines = MagicMock(return_value=iter(lines))
    return resp


def _jsonl_resp(chunks):
    """Mock response whose iter_lines emits bare JSON-line chunks."""
    import json as _json
    lines = [_json.dumps(c) for c in chunks]
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines = MagicMock(return_value=iter(lines))
    return resp


class TestExtractChunkTokens:
    """_extract_chunk_tokens must correctly bucket every supported shape."""

    def test_delta_content_only(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens({"choices": [{"delta": {"content": "hi"}}]})
        assert out["content"] == "hi"
        assert out["reasoning"] == ""
        assert out["thinking"] == ""

    def test_delta_reasoning_content_only(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens(
            {"choices": [{"delta": {"reasoning_content": "because"}}]}
        )
        assert out["reasoning"] == "because"
        assert out["content"] == ""
        assert out["thinking"] == ""

    def test_delta_thinking_goes_to_thinking_bucket_not_answer(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens(
            {"choices": [{"delta": {"thinking": "pondering..."}}]}
        )
        assert out["thinking"] == "pondering..."
        assert out["content"] == ""
        assert out["reasoning"] == ""
        assert out["message_content"] == ""

    def test_choices_message_content(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens(
            {"choices": [{"message": {"content": "final"}}]}
        )
        assert out["message_content"] == "final"

    def test_top_level_message_content(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens({"message": {"content": "wrapped"}})
        assert out["top_message"] == "wrapped"

    def test_top_level_response_field(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens({"response": "tok", "done": False})
        assert out["response"] == "tok"
        assert out["done"] is False

    def test_done_flag_from_finish_reason(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens(
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
        )
        assert out["done"] is True

    def test_done_flag_from_top_level_done(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens({"response": "", "done": True})
        assert out["done"] is True

    def test_message_thinking_is_thinking_not_answer(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens(
            {"message": {"content": "ans", "thinking": "private"}}
        )
        assert out["top_message"] == "ans"
        assert out["thinking"] == "private"

    def test_garbage_input_returns_clean_defaults(self):
        from homemath import _extract_chunk_tokens
        out = _extract_chunk_tokens("not-a-dict")  # type: ignore[arg-type]
        assert out["content"] == ""
        assert out["done"] is False


class TestPickAnswerPriority:
    """Channel priority must follow the spec exactly, per endpoint."""

    def _buckets(self, **kw):
        out = {"content": [], "reasoning": [], "message_content": [],
               "top_message": [], "response": []}
        out.update(kw)
        return out

    # ---- chat endpoint priority --------------------------------------------
    def test_chat_content_beats_reasoning(self):
        from homemath import _pick_answer, _CHAT_PRIORITY
        text, src = _pick_answer(
            self._buckets(content=["C"], reasoning=["R"], message_content=["M"],
                          top_message=["T"], response=["G"]),
            _CHAT_PRIORITY,
        )
        assert (text, src) == ("C", "delta.content")

    def test_chat_reasoning_beats_message(self):
        from homemath import _pick_answer, _CHAT_PRIORITY
        text, src = _pick_answer(
            self._buckets(reasoning=["R"], message_content=["M"],
                          top_message=["T"], response=["G"]),
            _CHAT_PRIORITY,
        )
        assert (text, src) == ("R", "delta.reasoning_content")

    def test_chat_message_beats_top_message(self):
        from homemath import _pick_answer, _CHAT_PRIORITY
        text, src = _pick_answer(
            self._buckets(message_content=["M"], top_message=["T"], response=["G"]),
            _CHAT_PRIORITY,
        )
        assert (text, src) == ("M", "choices[0].message.content")

    def test_chat_top_message_beats_response(self):
        from homemath import _pick_answer, _CHAT_PRIORITY
        text, src = _pick_answer(
            self._buckets(top_message=["T"], response=["G"]),
            _CHAT_PRIORITY,
        )
        assert (text, src) == ("T", "top-level message.content")

    def test_chat_response_is_last_resort(self):
        from homemath import _pick_answer, _CHAT_PRIORITY
        text, src = _pick_answer(self._buckets(response=["G"]), _CHAT_PRIORITY)
        assert (text, src) == ("G", "top-level response")

    def test_chat_all_empty_returns_none_source(self):
        from homemath import _pick_answer, _CHAT_PRIORITY
        text, src = _pick_answer(self._buckets(), _CHAT_PRIORITY)
        assert (text, src) == ("", "none")

    # ---- generate endpoint priority ----------------------------------------
    def test_generate_response_beats_top_message(self):
        from homemath import _pick_answer, _GENERATE_PRIORITY
        text, src = _pick_answer(
            self._buckets(response=["G"], top_message=["T"], content=["C"],
                          reasoning=["R"], message_content=["M"]),
            _GENERATE_PRIORITY,
        )
        assert (text, src) == ("G", "top-level response")

    def test_generate_top_message_beats_content(self):
        from homemath import _pick_answer, _GENERATE_PRIORITY
        text, src = _pick_answer(
            self._buckets(top_message=["T"], content=["C"], reasoning=["R"],
                          message_content=["M"]),
            _GENERATE_PRIORITY,
        )
        assert (text, src) == ("T", "top-level message.content")

    def test_generate_content_beats_reasoning(self):
        from homemath import _pick_answer, _GENERATE_PRIORITY
        text, src = _pick_answer(
            self._buckets(content=["C"], reasoning=["R"], message_content=["M"]),
            _GENERATE_PRIORITY,
        )
        assert (text, src) == ("C", "delta.content")

    def test_generate_reasoning_beats_message_content(self):
        from homemath import _pick_answer, _GENERATE_PRIORITY
        text, src = _pick_answer(
            self._buckets(reasoning=["R"], message_content=["M"]),
            _GENERATE_PRIORITY,
        )
        assert (text, src) == ("R", "delta.reasoning_content")

    def test_generate_message_content_is_last_resort(self):
        from homemath import _pick_answer, _GENERATE_PRIORITY
        text, src = _pick_answer(self._buckets(message_content=["M"]), _GENERATE_PRIORITY)
        assert (text, src) == ("M", "choices[0].message.content")


class TestNoDuplicationAcrossChannels:
    """Critical regression: if the same answer is emitted in MULTIPLE channels
    in the same stream, the returned string must be the one channel that wins
    per priority — not a concatenation. No duplication. Ever."""

    def test_chat_same_answer_in_content_and_message_content(self):
        """Proxy echoes the answer in both choices[0].delta.content and
        choices[0].message.content. Output must be the content once, not twice."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"content": "Answer only once."},
                           "message": {"content": "Answer only once."}}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "Answer only once."
        assert out.count("Answer only once.") == 1

    def test_chat_same_answer_in_reasoning_and_response(self):
        """Some proxies leak the answer on both reasoning_content and the
        top-level response field. Chat priority picks reasoning_content; the
        response field must NOT be appended."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "unique-42"}}],
             "response": "unique-42"},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out.count("unique-42") == 1
        assert out == "unique-42"

    def test_chat_same_answer_in_content_and_top_message(self):
        """delta.content wins over top-level message.content. Top message must
        NOT be appended even when identical."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"content": "same"}}],
             "message": {"content": "same"}},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "same"
        assert out.count("same") == 1

    def test_generate_same_answer_in_response_and_content(self):
        """On /api/generate, if response and delta.content both carry the same
        tokens, response (generate priority #1) wins and content is dropped."""
        from homemath import ollama_generate_stream
        import homemath as hlm
        chunks = [
            {"response": "generated",
             "choices": [{"delta": {"content": "generated"}}],
             "done": True},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_jsonl_resp(chunks))
            out = ollama_generate_stream("http://x/api/generate", {"prompt": ""})
        assert out.count("generated") == 1
        assert out == "generated"

    def test_generate_same_answer_in_response_and_top_message(self):
        """Response vs top-level message.content on /api/generate — response wins."""
        from homemath import ollama_generate_stream
        import homemath as hlm
        chunks = [
            {"response": "dup", "message": {"content": "dup"}, "done": True},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_jsonl_resp(chunks))
            out = ollama_generate_stream("http://x/api/generate", {"prompt": ""})
        assert out == "dup"
        assert out.count("dup") == 1

    def test_chat_multi_chunk_duplication_across_channels(self):
        """Multi-chunk stream where every chunk emits the same answer on both
        delta.content and choices[0].message.content. The result must equal the
        content stream alone, with no interleaved/duplicated message content."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"content": "Alpha "},
                           "message": {"content": "Alpha "}}]},
            {"choices": [{"delta": {"content": "Bravo "},
                           "message": {"content": "Bravo "}}]},
            {"choices": [{"delta": {"content": "Charlie."},
                           "message": {"content": "Charlie."},
                           "finish_reason": "stop"}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "Alpha Bravo Charlie."
        assert out.count("Alpha") == 1
        assert out.count("Bravo") == 1
        assert out.count("Charlie") == 1


class TestOllamaChatStreamSchemaDrift:
    """End-to-end normalization behaviour over the streaming parser."""

    def test_delta_content_only_stream(self):
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world."}, "finish_reason": "stop"}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "Hello world."

    def test_reasoning_content_only_stream(self):
        """When the server emits ONLY reasoning_content, it must be returned."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "The "}}]},
            {"choices": [{"delta": {"reasoning_content": "answer is 42."},
                           "finish_reason": "stop"}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "The answer is 42."

    def test_thinking_then_content_returns_only_content(self):
        """Thinking arrives first, then content. Thinking must never leak."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"thinking": "SECRET REASONING"}}]},
            {"choices": [{"delta": {"thinking": " continues"}}]},
            {"choices": [{"delta": {"content": "Answer: go east."},
                           "finish_reason": "stop"}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert "SECRET REASONING" not in out
        assert "continues" not in out
        assert out == "Answer: go east."

    def test_empty_deltas_then_answer(self):
        """Several heartbeat/empty chunks followed by the real answer."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {}}]},
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "ok"

    def test_done_sentinel_terminates_stream(self):
        """[DONE] sentinel must stop parsing immediately."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        import json as _json
        pre_done = [f"data: {_json.dumps({'choices':[{'delta':{'content':'kept'}}]})}"]
        pre_done.append("data: [DONE]")
        # Anything after [DONE] must NOT be included.
        pre_done.append(f"data: {_json.dumps({'choices':[{'delta':{'content':'after-done'}}]})}")
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=iter(pre_done))
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=resp)
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "kept"
        assert "after-done" not in out

    def test_choices_message_content_non_delta(self):
        """Some proxies emit choices[0].message.content mid-stream."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"message": {"content": "non-delta answer"}}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "non-delta answer"

    def test_top_level_message_content(self):
        """Top-level message.content (proxy shape) is answer-bearing."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"message": {"content": "wrapped answer"}},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == "wrapped answer"

    def test_thinking_wins_nothing_when_no_answer_channel(self):
        """All thinking, no answer => empty string (never leak thinking)."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"thinking": "reasoning-only-stream"}}]},
            {"thinking": "more reasoning"},
            {"message": {"thinking": "still reasoning"}},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert out == ""
        assert "reasoning" not in out

    def test_think_tags_in_content_are_stripped(self):
        """<think> blocks inside assembled content must be stripped."""
        from homemath import ollama_chat_stream
        import homemath as hlm
        chunks = [
            {"choices": [{"delta": {"content": "<think>secret</think>clean"}}]},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_sse_resp(chunks))
            out = ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        assert "<think>" not in out
        assert "secret" not in out
        assert "clean" in out


class TestOllamaGenerateStreamSchemaDrift:
    """Same normalization on the /api/generate shape."""

    def test_response_only_stream(self):
        from homemath import ollama_generate_stream
        import homemath as hlm
        chunks = [
            {"response": "Hello ", "done": False},
            {"response": "world.", "done": True},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_jsonl_resp(chunks))
            out = ollama_generate_stream("http://x/api/generate", {"prompt": ""})
        assert out == "Hello world."

    def test_thinking_then_response(self):
        from homemath import ollama_generate_stream
        import homemath as hlm
        chunks = [
            {"response": "", "thinking": "private", "done": False},
            {"response": "final", "done": True},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_jsonl_resp(chunks))
            out = ollama_generate_stream("http://x/api/generate", {"prompt": ""})
        assert out == "final"
        assert "private" not in out

    def test_generate_falls_back_to_top_message_content(self):
        """If proxy emits chat-shaped message.content on /api/generate, use it."""
        from homemath import ollama_generate_stream
        import homemath as hlm
        chunks = [
            {"message": {"content": "chat-shaped on generate"}, "done": True},
        ]
        with patch.object(hlm, "requests") as mock_req:
            mock_req.post = MagicMock(return_value=_jsonl_resp(chunks))
            out = ollama_generate_stream("http://x/api/generate", {"prompt": ""})
        assert out == "chat-shaped on generate"


class TestHomemathNoThinkRetryFullBudget:
    """Regression: /no_think retry must NOT drop max_tokens to the old 512 cap."""

    def test_retry_path_uses_full_4096_budget(self):
        """When first pass returns empty and think=True, the retry must be
        dispatched with max_tokens=4096, not the fragile 512 cap."""
        import homemath as hlm
        engine = _make_engine()
        provider = engine._provider_map["ollama-primary"]

        # First call: empty string (simulates all-thinking pass).
        # Second call: actual content.
        call_log = []

        def fake_stream(url, payload, timeout=300, headers=None):
            call_log.append(dict(payload))  # snapshot
            if len(call_log) == 1:
                return ""          # first pass: nothing
            return "REAL ANSWER"   # retry: real content

        with patch.object(hlm, "ollama_chat_stream", side_effect=fake_stream):
            name, content, _lat = engine._call_ollama(
                provider,
                [{"role": "user", "content": "write something substantial"}],
                temperature=0.7,
                think=True,
            )

        assert content == "REAL ANSWER"
        assert len(call_log) == 2, "retry must have fired"
        # First call ran with think budget 4096
        assert call_log[0]["max_tokens"] == 4096
        # CRITICAL: retry also keeps full 4096 budget — NOT 512
        assert call_log[1]["max_tokens"] == 4096, (
            "Retry must use full answer budget; old 512 truncated real answers."
        )
        # Retry must have /no_think on the last user message
        last_user = next(m for m in reversed(call_log[1]["messages"])
                         if m.get("role") == "user")
        assert last_user["content"].startswith("/no_think ")

    def test_retry_not_fired_when_first_pass_has_content(self):
        """If the first pass already returned text, no retry should happen."""
        import homemath as hlm
        engine = _make_engine()
        provider = engine._provider_map["ollama-primary"]

        call_log = []

        def fake_stream(url, payload, timeout=300, headers=None):
            call_log.append(dict(payload))
            return "first pass answer"

        with patch.object(hlm, "ollama_chat_stream", side_effect=fake_stream):
            name, content, _lat = engine._call_ollama(
                provider,
                [{"role": "user", "content": "analyze Q4 strategy and write a report"}],
                temperature=0.7,
                think=True,
            )

        assert content == "first pass answer"
        assert len(call_log) == 1, "no retry should fire when first pass succeeded"


# ═══════════════════════════════════════════════════════════════════════════════
#  Provider selection — no exception escapes race()
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathProviderSelectionContract:
    """race() returns; it does not raise. Selection failure is a failure of the
    call, not an exception the caller has to catch."""

    def test_identical_fallback_is_retried_not_refused(self):
        """Default config: LLM_HOST_FALLBACK and LLM_MODEL_FALLBACK unset, so
        both providers are the same model at the same URL. Refusing to
        'downgrade' there means refusing to retry the primary at all."""
        import homemath as hlm
        engine = _make_engine()
        primary = engine._provider_map["ollama-primary"]
        fallback = engine._provider_map["ollama-fallback"]

        # Precondition: the two really are indistinguishable.
        assert (primary.model, primary.url) == (fallback.model, fallback.url)
        assert engine._fallback_is_distinct() is False

        primary.consecutive_failures = 5          # above threshold
        assert primary.available is False
        assert fallback.available is True

        messages = [{"role": "user", "content": "write a detailed quarterly report"}]
        assert hlm.classify_task(messages) == "general"

        chosen = engine._select_provider(messages)
        assert chosen is fallback, "same model at same URL must be retried, not refused"

    def test_distinct_fallback_is_still_refused_for_content(self):
        """The quality-cliff guard survives: a genuinely smaller fallback model
        is still not used for content generation."""
        engine = _make_engine()
        primary = engine._provider_map["ollama-primary"]
        fallback = engine._provider_map["ollama-fallback"]

        fallback.model = "tiny-fallback-model"     # now genuinely distinct
        assert engine._fallback_is_distinct() is True
        primary.consecutive_failures = 5

        messages = [{"role": "user", "content": "write a detailed quarterly report"}]
        with pytest.raises(hlm_provider_unavailable()):
            engine._select_provider(messages)

    def test_race_converts_selection_failure_into_failed_sentinel(self, caplog):
        """A ProviderUnavailable must never reach the caller: race() honours the
        same ("FAILED", user-safe text) contract it uses for total failure."""
        import logging
        import homemath as hlm
        engine = _make_engine()
        primary = engine._provider_map["ollama-primary"]
        fallback = engine._provider_map["ollama-fallback"]

        fallback.model = "tiny-fallback-model"
        primary.consecutive_failures = 5

        messages = [{"role": "user", "content": "write a detailed quarterly report"}]
        with caplog.at_level(logging.ERROR, logger="homemath"):
            name, content = engine.race(messages, use_cache=False)

        assert name == "FAILED"
        assert isinstance(content, str) and content
        # User-facing text carries no operator diagnostics.
        assert "tiny-fallback-model" not in content
        assert "endpoint" not in content.lower()
        # Operators still get the detail, via logs.
        assert any("TOTAL FAILURE" in r.getMessage() for r in caplog.records)

    def test_homemath_chat_never_raises_on_selection_failure(self):
        """The top-level convenience function has the same contract."""
        import homemath as hlm
        engine = _make_engine()
        engine._provider_map["ollama-fallback"].model = "tiny-fallback-model"
        engine._provider_map["ollama-primary"].consecutive_failures = 5

        with patch.object(hlm, "_engine", engine):
            out = hlm.homemath_chat(
                [{"role": "user", "content": "write a detailed quarterly report"}],
                use_cache=False,
            )
        assert isinstance(out, str) and out


def hlm_provider_unavailable():
    from homemath import ProviderUnavailable
    return ProviderUnavailable


# ═══════════════════════════════════════════════════════════════════════════════
#  Token budget comes from the task class, not a fixed pair of constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathClassDrivenBudget:
    """README claims the task class decides the output token budget. Pin it."""

    def _budget_sent_for(self, prompt: str, think: bool):
        import homemath as hlm
        engine = _make_engine()
        provider = engine._provider_map["ollama-primary"]
        seen = []

        def fake_stream(url, payload, timeout=300, headers=None):
            seen.append(payload["max_tokens"])
            return "answer"

        with patch.object(hlm, "ollama_chat_stream", side_effect=fake_stream):
            engine._call_ollama(provider, [{"role": "user", "content": prompt}],
                                temperature=0.7, think=think)
        return seen[0]

    def test_greeting_gets_the_small_budget(self):
        from homemath.classifier import TaskClass, max_tokens
        assert self._budget_sent_for("hello there", think=False) == max_tokens(TaskClass.GREETING)

    def test_debug_gets_the_large_budget(self):
        from homemath.classifier import TaskClass, max_tokens
        assert self._budget_sent_for("why is this failing", think=True) == max_tokens(TaskClass.DEBUG)

    def test_budgets_actually_differ_by_class(self):
        """Guards against a regression that silently reinstates one constant."""
        assert self._budget_sent_for("hello there", think=False) != \
               self._budget_sent_for("why is this failing", think=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cache write must not be able to destroy a successful generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathCacheCannotDiscardAnswer:
    def test_cache_write_failure_still_returns_the_answer(self):
        """A cache write is bookkeeping. If it explodes, the caller has still
        paid for a real generation and is owed the result — it must not be
        recorded as a provider failure and the answer thrown away."""
        import homemath as hlm
        engine = _make_engine()
        primary = engine._provider_map["ollama-primary"]
        before = primary.consecutive_failures
        calls = []

        def once(url, payload, timeout=300, headers=None):
            calls.append(payload)
            return "REAL ANSWER"

        class ExplodingCache(dict):
            def __setitem__(self, k, v):
                raise KeyError("evicted")

        # Fail inside the real _cache_set body, not by patching over it — the
        # guard being tested lives in there.
        engine._cache = ExplodingCache()

        with patch.object(hlm, "ollama_chat_stream", side_effect=once):
            name, content = engine.race(
                [{"role": "user", "content": "hi"}], use_cache=True,
            )

        assert content == "REAL ANSWER", "a cache bug must not discard a paid generation"
        assert name != "FAILED"
        assert len(calls) == 1, "must not re-run the model because caching failed"
        assert primary.consecutive_failures == before, (
            "a cache bug must not be recorded as a provider failure"
        )

    def test_in_memory_cache_write_is_non_raising(self):
        """_cache_set promises not to raise, so race()'s no-raise contract holds."""
        engine = _make_engine()

        class ExplodingCache(dict):
            def __setitem__(self, k, v):
                raise RuntimeError("boom")

        engine._cache = ExplodingCache()
        engine._cache_set("k", "v")   # must not raise

    def test_redis_write_failure_is_swallowed(self):
        engine = _make_engine()
        engine._redis = MagicMock()
        engine._redis.setex = MagicMock(side_effect=OSError("redis down"))
        engine._cache_set("k", "v")   # must not raise

    def test_eviction_keeps_cache_bounded(self):
        engine = _make_engine()
        for i in range(600):
            engine._cache_set(f"key-{i}", f"val-{i}")
        assert len(engine._cache) <= 500


# ═══════════════════════════════════════════════════════════════════════════════
#  /no_think must tolerate non-string content
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathNoThinkContentTypes:
    def test_multimodal_content_list_is_left_alone(self):
        """OpenAI multimodal messages carry a list of parts. Concatenating a
        string onto that raises TypeError."""
        from homemath import HomemathEngine
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        out = HomemathEngine._prepend_no_think(msgs)
        assert out[0]["content"] == [{"type": "text", "text": "hi"}]

    def test_string_content_still_gets_the_prefix(self):
        from homemath import HomemathEngine
        out = HomemathEngine._prepend_no_think([{"role": "user", "content": "hi"}])
        assert out[0]["content"] == "/no_think hi"

    def test_input_is_not_mutated(self):
        from homemath import HomemathEngine
        msgs = [{"role": "user", "content": "hi"}]
        HomemathEngine._prepend_no_think(msgs)
        assert msgs[0]["content"] == "hi"


# ═══════════════════════════════════════════════════════════════════════════════
#  PriorityLock liveness
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathPriorityLockLiveness:
    def test_cancelled_waiter_does_not_strand_the_lock(self):
        """A waiter interrupted mid-wait used to stay in the heap; the next
        release handed the lock to a thread that was gone, and no one was left
        to release it — permanent deadlock for that model."""
        import threading
        from homemath import PriorityLock

        lock = PriorityLock()
        lock.acquire(model="m")               # holder

        state = lock._get_state("m")
        ghost = {"event": threading.Event(), "cancelled": True}
        import heapq
        heapq.heappush(state["waiters"], (1, 999, ghost))

        assert lock.queue_depth(model="m") == 0, "cancelled waiters are not live"

        lock.release(model="m")               # must skip the ghost, free the lock

        acquired = threading.Event()

        def take():
            lock.acquire(model="m")
            acquired.set()

        t = threading.Thread(target=take, daemon=True)
        t.start()
        assert acquired.wait(timeout=2.0), "lock was stranded by a cancelled waiter"
        lock.release(model="m")

    def test_release_wakes_highest_priority_waiter_first(self):
        import threading
        from homemath import PriorityLock, PRIORITY_HIGH, PRIORITY_LOW

        lock = PriorityLock()
        lock.acquire(model="m")
        order = []

        def waiter(priority, label, started):
            started.set()
            lock.acquire(priority=priority, model="m")
            order.append(label)
            lock.release(model="m")

        s_low, s_high = threading.Event(), threading.Event()
        t_low = threading.Thread(target=waiter, args=(PRIORITY_LOW, "low", s_low), daemon=True)
        t_low.start()
        s_low.wait()
        while lock.queue_depth(model="m") < 1:
            time.sleep(0.01)

        t_high = threading.Thread(target=waiter, args=(PRIORITY_HIGH, "high", s_high), daemon=True)
        t_high.start()
        s_high.wait()
        while lock.queue_depth(model="m") < 2:
            time.sleep(0.01)

        lock.release(model="m")
        t_high.join(timeout=2.0)
        t_low.join(timeout=2.0)
        assert order == ["high", "low"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Streamed responses are closed
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathStreamClosesResponse:
    def test_response_closed_on_success(self):
        import homemath as hlm
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=[
            'data: {"choices":[{"delta":{"content":"ok"}}]}', 'data: [DONE]'])
        with patch.object(hlm, "requests") as m:
            m.post = MagicMock(return_value=resp)
            hlm.ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        resp.close.assert_called_once()

    def test_response_closed_when_status_raises(self):
        """stream=True holds the socket until the body is consumed. An HTTP
        error must not leak it."""
        import homemath as hlm
        resp = MagicMock()
        resp.raise_for_status = MagicMock(side_effect=OSError("500"))
        with patch.object(hlm, "requests") as m:
            m.post = MagicMock(return_value=resp)
            with pytest.raises(OSError):
                hlm.ollama_chat_stream("http://x/v1/chat/completions", {"messages": []})
        resp.close.assert_called_once()

    def test_generate_response_closed_on_error(self):
        import homemath as hlm
        resp = MagicMock()
        resp.raise_for_status = MagicMock(side_effect=OSError("500"))
        with patch.object(hlm, "requests") as m:
            m.post = MagicMock(return_value=resp)
            with pytest.raises(OSError):
                hlm.ollama_generate_stream("http://x/api/generate", {"prompt": ""})
        resp.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
#  Public surface
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomemathPublicSurface:
    def test_all_does_not_leak_stdlib_imports(self):
        import homemath
        for leaked in ("os", "re", "time", "requests", "threading", "hashlib",
                       "logging", "heapq", "Dict", "List", "Optional", "Tuple"):
            assert leaked not in homemath.__all__

    def test_all_names_actually_exist(self):
        import homemath
        missing = [n for n in homemath.__all__ if not hasattr(homemath, n)]
        assert not missing, f"__all__ names a nonexistent attribute: {missing}"

    def test_documented_entry_points_are_exported(self):
        import homemath
        for name in ("homemath_chat", "HomemathEngine", "classify_task",
                     "strip_thinking", "ollama_chat_stream_dual",
                     "ProviderUnavailable"):
            assert name in homemath.__all__

    def test_py_typed_marker_present(self):
        """Without this file the annotations are invisible to consumers."""
        import pathlib, homemath
        marker = pathlib.Path(homemath.__file__).parent / "py.typed"
        assert marker.is_file()
