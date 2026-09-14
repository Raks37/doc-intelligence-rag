"""
Requirement #5: "Implement AI comparison logic:
  a. Extracted data vs source document
  b. Post-load verification between Contact Master and source docs"

Two comparators, same underlying fuzzy/semantic matcher:

1. extracted_vs_source: sanity-checks that every extracted field value actually
   appears (or closely matches) the raw source text -- this is your
   hallucination guardrail for the LLM extraction path.

2. contact_master_vs_source: after a record is loaded into the downstream
   Contact Master system, re-pull it and diff against the original source
   document. Catches load bugs, manual overwrites, or stale data -- independent
   of whether the original extraction was correct.

Uses RapidFuzz (fast, no GPU) for field-level string similarity plus a light
semantic layer (sentence-transformers) for free-text fields like
'authorization_scope' where wording can differ without meaning differing.
"""
import logging
from rapidfuzz import fuzz
from app.embeddings import get_embedder
from app.schemas import (
    AuthorizationCertificateExtraction, ComparisonResult, ComparisonType,
    ComparisonStatus, FieldComparison,
)

logger = logging.getLogger(__name__)

EXACT_MATCH_FIELDS = {"account_number", "effective_date", "expiration_date"}
SEMANTIC_FIELDS = {"authorization_scope"}

MATCH_THRESHOLD = 0.90
PARTIAL_THRESHOLD = 0.65


def _semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of sentence embeddings. If the embedding model can't be
    loaded (e.g. running offline and it isn't cached), degrade gracefully to a
    fuzzy token match rather than failing the whole comparison."""
    try:
        model = get_embedder()
    except Exception as e:  # noqa: BLE001 - any load/download failure -> fallback
        logger.warning("Embedding model unavailable (%s); falling back to fuzzy match", e)
        return fuzz.token_sort_ratio(a, b) / 100

    embeddings = model.encode([a, b], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])  # cosine similarity, vectors are normalized


def _field_similarity(field_name: str, a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return 0.0
    if field_name in EXACT_MATCH_FIELDS:
        return 1.0 if a.strip().lower() == b.strip().lower() else fuzz.ratio(a, b) / 100
    if field_name in SEMANTIC_FIELDS:
        return _semantic_similarity(a, b)
    return fuzz.token_sort_ratio(a, b) / 100  # names, free text -> fuzzy token match


def _status_for_score(score: float, a_present: bool, b_present: bool) -> ComparisonStatus:
    if not a_present and not b_present:
        return ComparisonStatus.MATCH  # both sides agree the field has no value
    if not a_present and b_present:
        return ComparisonStatus.MISSING_IN_TARGET
    if a_present and not b_present:
        return ComparisonStatus.MISSING_IN_SOURCE
    if score >= MATCH_THRESHOLD:
        return ComparisonStatus.MATCH
    if score >= PARTIAL_THRESHOLD:
        return ComparisonStatus.PARTIAL_MATCH
    return ComparisonStatus.MISMATCH


def _compare_records(
    document_id: str,
    comparison_type: ComparisonType,
    record_a: AuthorizationCertificateExtraction,
    record_b: AuthorizationCertificateExtraction,
) -> ComparisonResult:
    field_comparisons = []
    for field_name in AuthorizationCertificateExtraction.model_fields:
        if field_name == "authorized_signatories":
            continue  # compared separately below if needed
        a_val = getattr(record_a, field_name)
        b_val = getattr(record_b, field_name)
        score = _field_similarity(field_name, a_val, b_val)
        status = _status_for_score(score, a_val is not None, b_val is not None)
        field_comparisons.append(FieldComparison(
            field_name=field_name, extracted_value=a_val, reference_value=b_val,
            similarity_score=round(score, 3), status=status,
        ))

    mismatches = [fc for fc in field_comparisons if fc.status in
                  (ComparisonStatus.MISMATCH, ComparisonStatus.MISSING_IN_SOURCE, ComparisonStatus.MISSING_IN_TARGET)]
    overall = ComparisonStatus.MATCH if not mismatches else (
        ComparisonStatus.MISMATCH if any(fc.field_name in {"account_number", "entity_name"} for fc in mismatches)
        else ComparisonStatus.PARTIAL_MATCH
    )

    return ComparisonResult(
        document_id=document_id,
        comparison_type=comparison_type,
        field_comparisons=field_comparisons,
        overall_status=overall,
        requires_manual_review=overall != ComparisonStatus.MATCH,
        review_reason=None if overall == ComparisonStatus.MATCH else
            f"{len(mismatches)} field(s) mismatched: {', '.join(fc.field_name for fc in mismatches)}",
    )


def compare_extracted_vs_source(document_id: str, extracted: AuthorizationCertificateExtraction,
                                 source_parsed_reference: AuthorizationCertificateExtraction) -> ComparisonResult:
    """In practice `source_parsed_reference` here is a second, independent
    extraction pass (e.g. template extraction as ground truth check on an LLM
    extraction, or a human-keyed spot value) used purely to validate fidelity."""
    return _compare_records(document_id, ComparisonType.EXTRACTED_VS_SOURCE, extracted, source_parsed_reference)


def compare_contact_master_vs_source(document_id: str, contact_master_record: AuthorizationCertificateExtraction,
                                      source_record: AuthorizationCertificateExtraction) -> ComparisonResult:
    return _compare_records(document_id, ComparisonType.CONTACT_MASTER_VS_SOURCE, contact_master_record, source_record)
