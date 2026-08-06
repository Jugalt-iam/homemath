"""Deterministic request classification driving inference policy.

One classification pass produces a :class:`TaskClass`, and three pure lookup
functions turn that class into the three decisions the caller needs:

    thinking_from_intent(cls) -> bool   # enable model reasoning mode
    think_style(cls)          -> ThinkStyle
    max_tokens(cls)           -> int    # output token budget

The policy functions never inspect message text. They read the class only.
That is deliberate: a keyword scan reachable from the policy layer will
eventually match a keyword inside a system prompt and enable reasoning mode on
a background call that did not need it. Keeping the keyword banks reachable
only from ``classify_*`` makes that failure mode unreachable by construction,
and greppable -- no ``_KW_`` name appears inside a policy function.

Reasoning-relevant keywords are scanned on the most recent USER turn only.
Corpus-level scans (across system and assistant turns) are used only for the
code and domain classes, where grounding across the whole exchange helps and
neither resulting class enables reasoning mode on its own.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TaskClass(str, Enum):
    """Routing category -- drives thinking policy and token budget."""

    GREETING = "greeting"
    CHAT = "chat"                # short conversational / factual
    SIMPLE = "simple"            # short one-shot extract/format/classify
    CONTENT_GEN = "content_gen"  # long-form writing -- fluency, not reasoning
    CODE = "code"                # code / SQL / schema -- analytical
    DOMAIN = "domain"            # caller-defined subject area, see set_domain_keywords
    JUDGE = "judge"              # critique / score / rubric
    STRATEGY = "strategy"        # plan / roadmap
    ANALYSIS = "analysis"        # compare / evaluate / audit
    DEBUG = "debug"              # diagnose / root-cause
    GENERAL = "general"          # catch-all
    RESEARCH = "research"        # open-ended multi-source gathering


class ThinkStyle(str, Enum):
    """System-prompt prefix family. ``NONE`` means skip the prefix entirely."""

    NONE = "none"
    FLUENCY = "fluency"
    EVALUATIVE = "evaluative"
    HIERARCHICAL = "hierarchical"
    ANALYTICAL = "analytical"
    DIAGNOSTIC = "diagnostic"


# ---------------------------------------------------------------------------
# Policy maps -- pure lookups, no text inspection
# ---------------------------------------------------------------------------

_THINKING_MAP: Mapping[TaskClass, bool] = {
    TaskClass.GREETING:    False,
    TaskClass.CHAT:        False,
    TaskClass.SIMPLE:      False,
    TaskClass.CONTENT_GEN: False,   # fluency, not reasoning
    TaskClass.CODE:        True,
    TaskClass.DOMAIN:      False,   # recall, not reasoning
    TaskClass.JUDGE:       True,
    TaskClass.STRATEGY:    True,
    TaskClass.ANALYSIS:    True,
    TaskClass.DEBUG:       True,
    TaskClass.GENERAL:     False,   # conservative default
    TaskClass.RESEARCH:    True,
}

_STYLE_MAP: Mapping[TaskClass, ThinkStyle] = {
    TaskClass.GREETING:    ThinkStyle.NONE,
    TaskClass.CHAT:        ThinkStyle.NONE,
    TaskClass.SIMPLE:      ThinkStyle.NONE,
    TaskClass.CONTENT_GEN: ThinkStyle.FLUENCY,
    TaskClass.CODE:        ThinkStyle.ANALYTICAL,
    TaskClass.DOMAIN:      ThinkStyle.NONE,
    TaskClass.JUDGE:       ThinkStyle.EVALUATIVE,
    TaskClass.STRATEGY:    ThinkStyle.HIERARCHICAL,
    TaskClass.ANALYSIS:    ThinkStyle.ANALYTICAL,
    TaskClass.DEBUG:       ThinkStyle.DIAGNOSTIC,
    TaskClass.GENERAL:     ThinkStyle.NONE,
    TaskClass.RESEARCH:    ThinkStyle.ANALYTICAL,
}

_MAX_TOKENS_MAP: Mapping[TaskClass, int] = {
    TaskClass.GREETING:     128,
    TaskClass.CHAT:         1024,
    TaskClass.SIMPLE:       512,
    TaskClass.CONTENT_GEN:  4096,
    TaskClass.CODE:         4096,
    TaskClass.DOMAIN:       2048,
    TaskClass.JUDGE:        4096,
    TaskClass.STRATEGY:     4096,
    TaskClass.ANALYSIS:     4096,
    TaskClass.DEBUG:        4096,
    TaskClass.GENERAL:      2048,
    TaskClass.RESEARCH:     4096,
}


def thinking_from_intent(task_class: TaskClass) -> bool:
    """Return the thinking flag for a TaskClass. Pure -- never reads messages."""
    return _THINKING_MAP[task_class]


def think_style(task_class: TaskClass) -> ThinkStyle:
    """Return the ThinkStyle for a TaskClass. Pure."""
    return _STYLE_MAP[task_class]


def max_tokens(task_class: TaskClass) -> int:
    """Return the output token budget for a TaskClass. Pure."""
    return _MAX_TOKENS_MAP[task_class]


# ---------------------------------------------------------------------------
# Keyword banks -- referenced ONLY from classify_* below
# ---------------------------------------------------------------------------

_KW_GREETING = re.compile(
    r'\b(hello|hi|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|'
    r'what\s+can\s+you|who\s+are\s+you|how\s+are\s+you|thanks|thank\s+you|bye|'
    r'goodbye)\b',
    re.IGNORECASE,
)
_KW_SIMPLE = re.compile(
    r'\b(extract|format|yes|no|list|summarize|classify|tag|label|parse|convert|'
    r'rewrite|shorten|expand|translate|count|check)\b',
    re.IGNORECASE,
)
_KW_COMPLEX = re.compile(
    r'\b(write|create|generate|compose|draft|analyze|compare|explain\s+why|'
    r'strategy|research|report|evaluate|audit|plan|design|optimize|'
    r'article|essay)\b',
    re.IGNORECASE,
)
_KW_CODE = re.compile(
    r'\b(code|sql|query|function|class|script|program|debug|refactor|implement|'
    r'endpoint|api|json|schema|database|migration|regex)\b',
    re.IGNORECASE,
)
_KW_JUDGE = re.compile(
    r'\b(judge|critique|evaluate|assess|score|grade|rubric)\b',
    re.IGNORECASE,
)
_KW_STRATEGY = re.compile(
    r'\b(strategy|roadmap|plan|launch|goal\s+tree)\b',
    re.IGNORECASE,
)
_KW_ANALYSIS = re.compile(
    r'\b(analy[sz]e|compare|contrast|trade.?off|pros\s+and\s+cons|versus|vs\.)\b',
    re.IGNORECASE,
)
_KW_DEBUG = re.compile(
    r'\b(debug|diagnose|root\s+cause|why\s+(is|does|did)|broken|failing|error|stack\s+trace)\b',
    re.IGNORECASE,
)
_KW_RESEARCH = re.compile(
    r'\b(research|investigate|find\s+out|gather\s+information|what\s+are\s+the\s+latest)\b',
    re.IGNORECASE,
)

# Domain keywords are caller-supplied. Empty by default, so TaskClass.DOMAIN is
# never produced until the caller configures it -- there is no built-in subject
# area. Set HOMEMATH_DOMAIN_KEYWORDS, or call set_domain_keywords() at runtime.
_KW_DOMAIN: Optional["re.Pattern"] = None

# Minimum corpus hits before the domain class is assigned.
DOMAIN_HIT_THRESHOLD: int = 2


def set_domain_keywords(pattern: Optional[str]) -> None:
    """Configure the keyword set for :attr:`TaskClass.DOMAIN`.

    ``pattern`` is a regex alternation body without delimiters, for example
    ``"invoicing|ledger|reconciliation"``. Pass ``None`` or an empty string to
    disable the domain class entirely, which is the default.
    """
    global _KW_DOMAIN
    if not pattern or not pattern.strip():
        _KW_DOMAIN = None
        return
    _KW_DOMAIN = re.compile(r'\b(?:' + pattern.strip() + r')\b', re.IGNORECASE)


set_domain_keywords(os.environ.get("HOMEMATH_DOMAIN_KEYWORDS"))


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """Immutable classification record consumed by the policy layer."""

    task_class: TaskClass
    confidence: float

    @property
    def thinking(self) -> bool:
        return thinking_from_intent(self.task_class)

    @property
    def style(self) -> ThinkStyle:
        return think_style(self.task_class)

    @property
    def budget(self) -> int:
        return max_tokens(self.task_class)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _last_user_text(messages: Sequence[Mapping[str, object]]) -> str:
    for m in reversed(messages):
        if isinstance(m, Mapping) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


def _joined_text(messages: Sequence[Mapping[str, object]]) -> str:
    parts = []
    for m in messages:
        if isinstance(m, Mapping):
            c = m.get("content")
            if isinstance(c, str) and c:
                parts.append(c)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def classify_task_class(messages: Sequence[Mapping[str, object]]) -> TaskClass:
    """Classify a chat history into a :class:`TaskClass`.

    Decision order (first match wins):
      1. Greeting -- short user turn with greeting keywords.
      2. Simple -- short user turn, simple verbs only, no complex verbs.
      3. Debug / Judge / Analysis / Strategy / Research -- explicit user intent.
      4. Code -- at least two code keywords across the whole exchange.
      5. Domain -- at least DOMAIN_HIT_THRESHOLD domain keywords across the
         whole exchange, when domain keywords are configured.
      6. Content generation -- a complex verb in the user turn.
      7. Chat or General fallback, by user turn length.

    Steps 1-3 and 6 read the latest USER turn only. Steps 4-5 read the whole
    exchange, and neither of the classes they produce enables reasoning mode.
    """
    last_user = _last_user_text(messages)
    if not last_user:
        return TaskClass.CHAT

    user_words = len(last_user.split())

    if _KW_GREETING.search(last_user) and user_words <= 8:
        return TaskClass.GREETING

    if (
        user_words <= 12
        and _KW_SIMPLE.search(last_user)
        and not _KW_COMPLEX.search(last_user)
    ):
        return TaskClass.SIMPLE

    if _KW_DEBUG.search(last_user):
        return TaskClass.DEBUG
    if _KW_JUDGE.search(last_user):
        return TaskClass.JUDGE
    if _KW_ANALYSIS.search(last_user):
        return TaskClass.ANALYSIS
    if _KW_STRATEGY.search(last_user):
        return TaskClass.STRATEGY
    if _KW_RESEARCH.search(last_user):
        return TaskClass.RESEARCH

    # Corpus-level: a code-heavy prompt should not route to content generation
    # merely because it opens with "write".
    corpus = _joined_text(messages)
    if len(_KW_CODE.findall(corpus)) >= 2:
        return TaskClass.CODE

    if _KW_DOMAIN is not None and len(_KW_DOMAIN.findall(corpus)) >= DOMAIN_HIT_THRESHOLD:
        return TaskClass.DOMAIN

    if _KW_COMPLEX.search(last_user):
        return TaskClass.CONTENT_GEN

    if user_words <= 6:
        return TaskClass.CHAT
    return TaskClass.GENERAL


def classify(
    user_message: str,
    history: Optional[Iterable[Mapping[str, object]]] = None,
) -> Classification:
    """Classify a single user message, optionally with prior history.

    ``history`` is optional. When present the corpus-level checks (code,
    domain) have material to work with; when absent only the user turn is
    inspected. Confidence is advisory and is never gated on by policy.
    """
    msgs: list = list(history) if history else []
    if user_message:
        msgs = msgs + [{"role": "user", "content": user_message}]
    task = classify_task_class(msgs)
    confidence = 0.9 if task is not TaskClass.GENERAL else 0.5
    return Classification(task_class=task, confidence=confidence)


def legacy_task_string(task: TaskClass) -> str:
    """Collapse a TaskClass onto the coarse string taxonomy.

    Classes with no coarse equivalent collapse onto ``"general"``.
    """
    if task is TaskClass.GREETING:
        return "greeting"
    if task is TaskClass.SIMPLE:
        return "simple"
    if task is TaskClass.CODE:
        return "code"
    if task is TaskClass.DOMAIN:
        return "domain"
    return "general"


__all__ = [
    "TaskClass",
    "ThinkStyle",
    "Classification",
    "DOMAIN_HIT_THRESHOLD",
    "classify",
    "classify_task_class",
    "set_domain_keywords",
    "thinking_from_intent",
    "think_style",
    "max_tokens",
    "legacy_task_string",
]
