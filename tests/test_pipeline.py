"""
Deterministic unit tests -- none of these call an LLM, so they run instantly
and are good to show in an interview as "here's how I'd test this in CI."
"""
from app.schemas import ExtractedField, ExtractionMethod, AuthorizationCertificateExtraction
from app.confidence import compute_record_confidence, flag_exceptions
from app.extractors.template_extractor import looks_like_known_template, extract_with_template
from app.comparison import compare_extracted_vs_source

STRUCTURED_TEXT = """
AUTHORIZATION CERTIFICATE
Entity Name: Meridian Capital Partners LLC
Account Number: 5551234821
Effective Date: 01/15/2026
Expiration Date: 01/15/2028
Authorization Scope: Wire transfers up to $500,000
"""


def test_template_detection():
    assert looks_like_known_template(STRUCTURED_TEXT) is True
    assert looks_like_known_template("Dear Sir, please find attached...") is False


def test_template_extraction_fields():
    record, fields = extract_with_template(STRUCTURED_TEXT)
    assert record.entity_name == "Meridian Capital Partners LLC"
    assert record.account_number == "****4821"  # masked by validator
    assert all(f.extraction_method == ExtractionMethod.TEMPLATE for f in fields)


def test_record_confidence_weighting():
    fields = [
        ExtractedField(field_name="entity_name", value="Acme", confidence=1.0,
                        extraction_method=ExtractionMethod.TEMPLATE),
        ExtractedField(field_name="account_number", value=None, confidence=0.0,
                        extraction_method=ExtractionMethod.TEMPLATE),
        ExtractedField(field_name="effective_date", value="01/01/2026", confidence=0.9,
                        extraction_method=ExtractionMethod.TEMPLATE),
    ]
    score = compute_record_confidence(fields)
    # critical field (account_number) missing should pull the score down hard
    assert score < 0.6


def test_exception_flagging_on_missing_critical_field():
    fields = [
        ExtractedField(field_name="account_number", value=None, confidence=0.0,
                        extraction_method=ExtractionMethod.TEMPLATE),
    ]
    exceptions = flag_exceptions("doc-1", fields)
    assert len(exceptions) == 1
    assert exceptions[0].severity == "high"


def test_comparison_exact_match():
    a = AuthorizationCertificateExtraction(entity_name="Acme LLC", account_number="1234",
                                            effective_date="01/01/2026")
    b = AuthorizationCertificateExtraction(entity_name="Acme LLC", account_number="1234",
                                            effective_date="01/01/2026")
    result = compare_extracted_vs_source("doc-1", a, b)
    assert result.overall_status == "match"
    assert result.requires_manual_review is False


def test_comparison_flags_account_mismatch():
    a = AuthorizationCertificateExtraction(entity_name="Acme LLC", account_number="1234")
    b = AuthorizationCertificateExtraction(entity_name="Acme LLC", account_number="9999")
    result = compare_extracted_vs_source("doc-1", a, b)
    assert result.overall_status == "mismatch"
    assert result.requires_manual_review is True
