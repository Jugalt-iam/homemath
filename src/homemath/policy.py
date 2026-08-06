"""Per-request inference policy, carried on a ContextVar.

An upstream caller that has already decided how a request should be handled can
bind a policy for the duration of that request. Anything further down the stack
reads the bound policy instead of re-deriving the decision from message text.

The policy is an open mapping. The only key this package reads is
``thinking_required`` (bool); callers may carry anything else they need.

    from homemath.policy import set_current_policy, get_current_policy

    token = set_current_policy({"thinking_required": True, "budget": 4096})
    try:
        ...
    finally:
        reset_current_policy(token)

A ContextVar is used rather than a module global so concurrent requests --
threads, or asyncio tasks -- do not observe each other's policy.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Dict, Optional

_CURRENT_POLICY: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "homemath_current_policy", default=None,
)


def get_current_policy() -> Optional[Dict[str, Any]]:
    """Return the policy bound to this context, or ``None`` if unbound."""
    return _CURRENT_POLICY.get()


def set_current_policy(policy: Optional[Dict[str, Any]]) -> Token:
    """Bind ``policy`` to the current context. Returns a reset token."""
    return _CURRENT_POLICY.set(policy)


def reset_current_policy(token: Token) -> None:
    """Unbind, restoring whatever policy was in place before ``token``."""
    _CURRENT_POLICY.reset(token)


__all__ = [
    "get_current_policy",
    "set_current_policy",
    "reset_current_policy",
]
