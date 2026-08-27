"""
Requirement: "Understanding of PII handling, model explainability, and audit
controls."

Every pipeline step writes a structured, append-only audit event: what
happened, what the model saw, what it decided, and why. In a real deployment
this would go to a write-once store (e.g. S3 with object lock, or an audit DB
table) -- for the demo we write structured JSON lines locally, which is enough
to show a reviewer "here's exactly why the system extracted/flagged this."
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "audit_log.jsonl"
logger = logging.getLogger("audit")


def log_pipeline_step(document_id: str, step: str, details: dict) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "document_id": document_id,
        "step": step,
        "details": _redact(details),
    }
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    logger.info("audit: %s | %s", document_id, step)


def _redact(details: dict) -> dict:
    """Belt-and-suspenders PII scrub for anything that lands in logs, on top of
    the field-level masking already done in the Pydantic schema validators."""
    redacted = json.loads(json.dumps(details, default=str))
    _redact_recursive(redacted)
    return redacted


SENSITIVE_KEYS = {"ssn", "tax_id", "date_of_birth", "full_account_number"}


def _redact_recursive(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in SENSITIVE_KEYS:
                obj[k] = "***REDACTED***"
            else:
                _redact_recursive(v)
    elif isinstance(obj, list):
        for item in obj:
            _redact_recursive(item)
