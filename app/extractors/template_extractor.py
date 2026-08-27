"""
Template-based ("document scraping") extraction for structured forms.

Requirement #2: "Implement document scraping / template-based extraction for
structured forms." This is the deterministic, high-confidence path: when a
document matches a known layout (e.g. a specific bank's standard Authorization
Certificate template), we extract by label proximity / regex rather than
paying for an LLM call. This is cheaper, faster, fully explainable (no
hallucination risk), and should be preferred whenever a template match is
possible -- LLM extraction is the fallback for everything else.
"""
import re
from app.schemas import (
    ExtractedField, ExtractionMethod, AuthorizationCertificateExtraction,
)

# Each pattern captures a labeled value on a known template. In production you'd
# maintain one of these per known source-system template (versioned, testable).
FIELD_PATTERNS = {
    "entity_name": r"(?:Entity Name|Company Name|Account Holder)\s*[:\-]\s*(.+)",
    "account_number": r"(?:Account (?:No\.?|Number))\s*[:\-]\s*([\w\-]+)",
    "effective_date": r"(?:Effective Date)\s*[:\-]\s*([\d/\-]+)",
    "expiration_date": r"(?:Expir\w+ Date)\s*[:\-]\s*([\d/\-]+)",
    "authorization_scope": r"(?:Authoriz\w+ Scope|Authority Limit)\s*[:\-]\s*(.+)",
    "governing_entity_type": r"(?:Entity Type)\s*[:\-]\s*(.+)",
}

SIGNATORY_PATTERN = r"(?:Authorized Signatory|Signatory Name)\s*[:\-]\s*(.+?)(?:,\s*Title\s*[:\-]\s*(.+))?\n"


def looks_like_known_template(text: str) -> bool:
    """Cheap heuristic: does this text contain enough of our known labels to be
    worth the deterministic path? If not, route to the LLM extractor instead."""
    hits = sum(1 for pattern in FIELD_PATTERNS.values() if re.search(pattern, text, re.IGNORECASE))
    return hits >= 3


def extract_with_template(text: str) -> tuple[AuthorizationCertificateExtraction, list[ExtractedField]]:
    fields: list[ExtractedField] = []
    values: dict = {}

    for field_name, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            values[field_name] = value
            fields.append(ExtractedField(
                field_name=field_name,
                value=value,
                # Deterministic regex match on a known template = high confidence.
                # Still not 1.0 -- OCR noise or template drift can still corrupt a match.
                confidence=0.95,
                extraction_method=ExtractionMethod.TEMPLATE,
                source_snippet=match.group(0),
            ))
        else:
            fields.append(ExtractedField(
                field_name=field_name, value=None, confidence=0.0,
                extraction_method=ExtractionMethod.TEMPLATE,
            ))

    signatories = []
    for m in re.finditer(SIGNATORY_PATTERN, text, re.IGNORECASE):
        signatories.append({"name": m.group(1).strip(), "title": (m.group(2) or "").strip() or None})

    record = AuthorizationCertificateExtraction(
        entity_name=values.get("entity_name"),
        account_number=values.get("account_number"),
        effective_date=values.get("effective_date"),
        expiration_date=values.get("expiration_date"),
        authorization_scope=values.get("authorization_scope"),
        governing_entity_type=values.get("governing_entity_type"),
        authorized_signatories=[{"name": s["name"], "title": s["title"]} for s in signatories],
    )
    return record, fields
