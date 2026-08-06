# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1]

Added `__version__` to the package.

Initial release.

### Added
- Streaming request handling for OpenAI-compatible `/v1/chat/completions` and
  native `/api/generate`, with schema-tolerant chunk parsing across five answer
  channel shapes.
- Separation of the model-private thinking channel (`delta.thinking`,
  `message.thinking`, top-level `thinking`) from the answer, with `source` on
  the dual-channel return naming which channel produced the answer.
- Deterministic keyword classification into a `TaskClass`, driving both
  reasoning mode and the per-request output token budget.
- Per-model priority lock so same-model requests serialize predictably while
  different models proceed in parallel.
- Prompt trimming to a token budget, preserving system messages and the most
  recent turns.
- Response caching in memory, or in Redis when `REDIS_URL` is set.
- `py.typed` marker: type annotations are exported to consumers.

[Unreleased]: https://github.com/Jugalt-iam/homemath/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Jugalt-iam/homemath/releases/tag/v0.1.0
