"""
Requirement #4: "Generate field-level and record-level confidence scores."

Field-level confidence is produced at extraction time (template regex confidence,
or LLM self-reported confidence -- see extractors/). This module rolls those up
into a RECORD-level score and flags exceptions, using explicit, auditable rules
rather than another opaque model call -- confidence scoring itself should be
explainable.
"""
from app.schemas import ExtractedField, ExceptionRecord, ExceptionSeverity

# Fields that MUST be present/high-confidence for the record to auto-pass.
CRITICAL_FIELDS = {"entity_name", "account_number", "authorization_scope"}

AUTO_APPROVE_THRESHOLD = 0.85
MANUAL_REVIEW_THRESHOLD = 0.60


def compute_record_confidence(fields: list[ExtractedField]) -> float:
    """Weighted average: critical fields count double. A document with a
    perfect date but a missing account number should NOT score well."""
    total_weight = 0.0
    weighted_sum = 0.0
    for f in fields:
        weight = 2.0 if f.field_name in CRITICAL_FIELDS else 1.0
        weighted_sum += f.confidence * weight
        total_weight += weight
    return round(weighted_sum / total_weight, 3) if total_weight else 0.0


def flag_exceptions(document_id: str, fields: list[ExtractedField]) -> list[ExceptionRecord]:
    """Requirement: 'exception handling'. Turns low-confidence / missing
    critical fields into actionable, routable exception records instead of
    silently shipping bad data downstream."""
    exceptions: list[ExceptionRecord] = []

    for f in fields:
        if f.field_name in CRITICAL_FIELDS and (f.value is None or f.confidence < MANUAL_REVIEW_THRESHOLD):
            exceptions.append(ExceptionRecord(
                document_id=document_id,
                severity=ExceptionSeverity.HIGH,
                field_name=f.field_name,
                reason=f"Critical field '{f.field_name}' missing or low-confidence "
                       f"({f.confidence}).",
                suggested_action="Route to manual review queue before Contact Master load.",
            ))
        elif f.value is not None and MANUAL_REVIEW_THRESHOLD <= f.confidence < AUTO_APPROVE_THRESHOLD:
            exceptions.append(ExceptionRecord(
                document_id=document_id,
                severity=ExceptionSeverity.MEDIUM,
                field_name=f.field_name,
                reason=f"Field '{f.field_name}' extracted with moderate confidence ({f.confidence}).",
                suggested_action="Spot-check recommended; auto-approve with flag.",
            ))

    return exceptions
