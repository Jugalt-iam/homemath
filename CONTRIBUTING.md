# Contributing

Thanks for taking a look. This is a small library and the bar for a change is
mainly: does it come with a test that fails without it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip        # editable installs need pip >= 21.3
pip install -e ".[test,redis]"
pytest -q
```

The pip upgrade is not optional on older setups. Editable installs of a
`pyproject.toml`-only project use PEP 660, and pip below 21.3 fails with
"editable mode currently requires a setuptools-based build" — which reads like a
problem with this project but is just an old pip. The macOS system Python ships
one old enough to hit this.

The install itself is required too: `homemath` lives under `src/`, so a bare
`pytest` in a fresh checkout cannot import it.

The suite runs in well under a second and opens no sockets — `requests` and
`redis` are replaced with module stubs before `homemath` is imported, and every
endpoint is a mock returning canned SSE or JSON-lines chunks. If a change you
make needs a live endpoint to test, that is worth discussing in an issue first.

## What a change needs

- **A test.** New behaviour gets a test; a bug fix gets a test that fails
  before the fix. Tests live in `tests/test_homemath.py`, grouped in classes by
  the area under test.
- **Python 3.9 compatibility.** CI runs 3.9 through 3.13. `from __future__ import
  annotations` is already in place in the modules that need it — if you add a
  new module using `X | Y` annotations, add the import too.
- **No new required dependencies.** `requests` is the only hard dependency and
  that is deliberate. Optional integrations go behind an extra, imported lazily,
  and degrade to a working default when absent (see how Redis is handled in
  `_init_redis`).

## Things worth knowing before you change them

- **The keyword banks are only reachable from `classify_*`.** The policy
  functions in `classifier.py` (`thinking_from_intent`, `think_style`,
  `max_tokens`) read the `TaskClass` and never inspect message text. This is
  deliberate and load-bearing: a keyword scan reachable from the policy layer
  will eventually match a word inside a system prompt and switch on reasoning
  for a background call that did not need it. Keep `_KW_` names out of policy
  functions.
- **Answer channels are picked, never merged.** `_pick_answer` takes the first
  non-empty channel in priority order and stops. Concatenating channels
  duplicates the answer whenever a proxy emits it in two fields at once.
- **`race` does not raise.** Every failure path returns
  `("FAILED", <user-safe text>)` so a caller cannot render an operator
  diagnostic to an end user. There is a test pinning all three channels of that
  contract; if you add a failure mode, route it through the same return.

## Reporting a bug

Include the server you are pointing at (Ollama, vLLM, SGLang, llama.cpp, a
proxy), and if the problem is a parsing one, a few raw stream lines. Set
`logging.getLogger("homemath").setLevel(logging.DEBUG)` — the `[stream-summary]`
line reports per-channel token counts and which channel won, which is usually
enough to identify a schema mismatch.
