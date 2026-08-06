"""Token budget enforcement for a chat message list.

Keeps the assembled prompt inside a configured token budget. The strategy
preserves two things:
  1. System messages -- always kept, truncated only if they alone overflow.
  2. The most recent user/assistant turns -- kept from the tail.

Middle history turns are dropped first. If the system messages alone exceed
SYSTEM_HARD_CAP tokens, each is truncated to fit.

Token estimate: len(text) // 4, a rough approximation for English that needs
no tokenizer and no external dependency. It is an estimate, not a count --
see the limitations section of the README.

Usage:
    from homemath.context_trimmer import trim_messages

    messages = trim_messages(messages)
    messages = trim_messages(messages, budget=2000, caller="my_component")
"""

import logging
import os

logger = logging.getLogger("homemath.context_trimmer")

# Maximum tokens allowed in the combined prompt. Sized for a 32K context
# window, leaving roughly 4K for the answer. Override per call, or set
# HOMEMATH_TOKEN_BUDGET.
DEFAULT_TOKEN_BUDGET: int = int(os.environ.get("HOMEMATH_TOKEN_BUDGET", "28000"))

# If a single system message exceeds this many tokens, truncate it.
SYSTEM_HARD_CAP: int = int(os.environ.get("HOMEMATH_SYSTEM_HARD_CAP", "6000"))

# Minimum characters we keep per system message when hard-capping
# (avoids reducing a system message to nothing).
SYSTEM_MIN_CHARS: int = 200


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters for English.

    One approximation shared by every caller, so budget arithmetic is
    consistent even though it is not exact.
    """
    return max(1, len(text) // 4)


def _messages_tokens(messages: list[dict]) -> int:
    """Estimate total token count for a list of chat messages."""
    return sum(_estimate_tokens(m.get("content", "")) for m in messages)


def trim_messages(
    messages: list[dict],
    budget: int = DEFAULT_TOKEN_BUDGET,
    caller: str = "unknown",
) -> list[dict]:
    """Trim a messages list to stay within *budget* tokens.

    Strategy:
      1. Split messages into system vs. conversation (user/assistant) turns.
      2. Truncate each system message if it alone exceeds SYSTEM_HARD_CAP.
      3. If system messages + ALL conversation already fit → return unchanged.
      4. Otherwise drop conversation turns from the front (oldest first) until
         the combined count is under budget.  Always keep at least the last
         2 conversation turns so the model has immediate context.

    Args:
        messages: OpenAI-style list of {"role": ..., "content": ...} dicts.
        budget: Maximum token count (default HOMEMATH_TOKEN_BUDGET, 28000).
        caller: Name of the calling module, used in log messages only.

    Returns:
        A new list (never mutates input).  Guaranteed to be non-empty as long
        as the input is non-empty.
    """
    if not messages:
        return messages

    # --- Step 1: separate system messages from conversation turns ---
    system_msgs: list[dict] = []
    convo_msgs: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(dict(msg))   # shallow copy — we may truncate content
        else:
            convo_msgs.append(msg)          # keep refs; convo is never mutated here

    # --- Step 2: truncate oversized system messages ---
    for sm in system_msgs:
        content = sm.get("content", "")
        if _estimate_tokens(content) > SYSTEM_HARD_CAP:
            cap_chars = SYSTEM_HARD_CAP * 4
            sm["content"] = content[:max(cap_chars, SYSTEM_MIN_CHARS)]
            logger.warning(
                "[context_trimmer] [%s] system message truncated from %d to %d chars",
                caller, len(content), len(sm["content"]),
            )

    system_tokens = _messages_tokens(system_msgs)

    # --- Step 3: fast path — already under budget ---
    total_tokens = system_tokens + _messages_tokens(convo_msgs)
    if total_tokens <= budget:
        return system_msgs + convo_msgs

    # --- Step 4: drop oldest conversation turns from the front ---
    remaining_budget = budget - system_tokens
    if remaining_budget <= 0:
        # System messages alone fill the budget; keep only the last user message
        # so the model knows what it's being asked.
        last_user = next(
            (m for m in reversed(convo_msgs) if m.get("role") == "user"), None
        )
        kept = [last_user] if last_user else []
        original_tokens = total_tokens
        kept_tokens = system_tokens + _messages_tokens(kept)
        logger.warning(
            "[context_trimmer] [%s] system msgs alone exceed budget (%d > %d tokens). "
            "Kept only last user message. Dropped from %d to %d tokens.",
            caller, system_tokens, budget, original_tokens, kept_tokens,
        )
        return system_msgs + kept

    # Keep as many turns from the tail as fit, but always keep ≥ 2 turns.
    kept_convo: list[dict] = []
    tokens_used: int = 0
    for msg in reversed(convo_msgs):
        msg_tokens = _estimate_tokens(msg.get("content", ""))
        if tokens_used + msg_tokens <= remaining_budget:
            kept_convo.insert(0, msg)
            tokens_used += msg_tokens
        elif len(kept_convo) < 2:
            # Must keep at least 2 turns; truncate the content to fit
            available_chars = max((remaining_budget - tokens_used) * 4, SYSTEM_MIN_CHARS)
            truncated = dict(msg)
            truncated["content"] = msg.get("content", "")[:available_chars]
            kept_convo.insert(0, truncated)
            tokens_used += _estimate_tokens(truncated["content"])
            logger.warning(
                "[context_trimmer] [%s] conversation message truncated to %d chars to meet minimum-2 requirement",
                caller, len(truncated["content"]),
            )
        else:
            break

    dropped = len(convo_msgs) - len(kept_convo)
    if dropped > 0:
        logger.warning(
            "[context_trimmer] [%s] dropped %d of %d conversation turns "
            "(%d tokens → %d tokens, budget=%d)",
            caller,
            dropped, len(convo_msgs),
            total_tokens, system_tokens + tokens_used,
            budget,
        )

    return system_msgs + kept_convo
