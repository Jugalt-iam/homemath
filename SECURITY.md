# Security Policy

## Supported versions

This project is pre-1.0. Only the latest release receives fixes.

## Reporting a vulnerability

Please report privately rather than opening a public issue: use
[GitHub's private vulnerability reporting](https://github.com/Jugalt-iam/homemath/security/advisories/new)
on this repository.

Expect an acknowledgement within a week. If the report is valid, a fix and an
advisory follow; you will be credited unless you prefer otherwise.

## Scope and threat model

This library makes outbound HTTP requests to an endpoint **you** configure and
returns model output to **your** code. Worth knowing:

- **`LLM_HOST` is trusted by construction.** Requests go wherever it points,
  carrying `LLM_API_KEY` as a bearer token. Do not point it at a host you do not
  control or trust. `requests` strips the `Authorization` header on a
  cross-host redirect, so a redirect alone does not leak the key, but the
  response body is still parsed and returned to your caller.
- **Model output is not sanitized.** Answer text is returned as-is; `<think>`
  block stripping is a formatting step, not a security boundary. If you render
  output as HTML or feed it to a downstream tool, escape and validate it there.
- **Reasoning may be returned as the answer.** When a server emits
  `delta.reasoning_content` and no `delta.content`, that reasoning is returned
  as the answer by design, so a human or a downstream judge has the model's
  working rather than an empty string. `source` on the dual-channel return names
  the winning channel. If reasoning must never reach your end users, branch on
  `source` — see the README.
- **The cache key is a hash of message content only.** It does not include
  temperature, model, or the calling identity. A shared `REDIS_URL` across
  tenants means identical prompts share a cached answer. Use a per-tenant Redis
  database, or leave `REDIS_URL` unset for a per-process cache.
- **Logs can contain prompt and response data.** The `[stream-summary]` line at
  INFO carries counts only, but DEBUG logs raw stream lines. Do not enable DEBUG
  in production if prompts carry sensitive data.

Reports about a misconfigured `LLM_HOST` reaching an attacker-controlled server
are out of scope — that endpoint is a trusted input by design.
