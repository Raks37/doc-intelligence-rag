"""
Guardrails for free-text RAG questions.

Rule-based, not a model call -- the same "explicit, auditable rules over
another opaque decision" philosophy as app/confidence.py (whose docstring
argues confidence scoring itself should be explainable; here, refusals
should be too). Critically, this also doesn't depend on the small local
model reliably following a safety instruction it might simply ignore --
the same reason app/confidence.py's exception thresholds are plain code,
not another LLM call.

Two independent layers, mirroring the belt-and-suspenders PII pattern
already used elsewhere in this codebase
(AuthorizationCertificateExtraction.mask_account_for_pii in app/schemas.py,
and app/audit.py's separate _redact() pass on top of it):

1. check_question() -- runs BEFORE retrieval or any LLM call. Blocks
   explicit requests for raw sensitive data and prompt-injection-shaped
   phrasing. A blocked question short-circuits the whole /ask flow: no
   retrieval, no LLM call, and the refusal reason is itself audit-logged.

2. redact_answer() -- runs on EVERY generated answer, regardless of
   whether the question was flagged, because a legitimately-phrased
   question ("what account is this for?") could still cause the model to
   surface a digit run straight out of a retrieved chunk.
"""
from __future__ import annotations
import re

MAX_QUESTION_CHARS = 500

BLOCKED_INTENT_PATTERNS = [
    r"\bfull\s+account\s+number\b",
    r"\bunmask\b",
    r"\bignore\s+(the\s+|all\s+)?(previous|prior|above)\s+instructions?\b",
    r"\bsystem\s+prompt\b",
    r"\breveal\s+your\s+(instructions|prompt)\b",
    r"\bssn\b",
    r"\bsocial\s+security\b",
    r"\btax\s+id\b",
]

# Anything that looks like an account number (8+ consecutive digits) gets
# masked, mirroring mask_account_for_pii's "****" + last-4 style exactly.
ACCOUNT_NUMBER_PATTERN = re.compile(r"\b\d{8,}\b")


def check_question(question: str) -> tuple[bool, str | None]:
    """Returns (allowed, reason). reason is None when allowed=True."""
    question = question or ""
    if len(question) > MAX_QUESTION_CHARS:
        return False, "question exceeds max length"

    lowered = question.lower()
    for pattern in BLOCKED_INTENT_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"question matched a blocked pattern ({pattern})"

    return True, None


def redact_answer(answer: str) -> tuple[str, bool]:
    """Masks any 8+ digit run in `answer` as ****<last 4 digits>. Returns
    (redacted_text, was_redacted)."""
    redacted, count = ACCOUNT_NUMBER_PATTERN.subn(
        lambda m: f"****{m.group()[-4:]}", answer or ""
    )
    return redacted, count > 0
