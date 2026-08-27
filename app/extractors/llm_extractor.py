"""
LLM-based extraction for unstructured / semi-structured / non-standard documents.

Requirement #3: "Develop AI/LLM-based extraction for unstructured and
semi-structured documents." Uses LangChain's structured-output binding against
the same Pydantic schema the template extractor produces, so downstream code
(confidence scoring, comparison) doesn't care which path a document took.

We also ask the model to self-report a per-field confidence and cite the
supporting text span -- this is the "field-level confidence score" requirement,
and the span is what makes the extraction explainable/auditable rather than a
black box.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.schemas import (
    ExtractedField, ExtractionMethod, AuthorizationCertificateExtraction,
)
from app.llm import get_chat_model


class _LLMFieldExtraction(BaseModel):
    """Internal schema: forces the model to give a value, its own confidence,
    and the supporting quote for every field, in one shot."""
    entity_name: str | None = None
    entity_name_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    entity_name_source: str | None = None

    account_number: str | None = None
    account_number_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    account_number_source: str | None = None

    effective_date: str | None = None
    effective_date_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    effective_date_source: str | None = None

    expiration_date: str | None = None
    expiration_date_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    expiration_date_source: str | None = None

    authorization_scope: str | None = None
    authorization_scope_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    authorization_scope_source: str | None = None

    governing_entity_type: str | None = None
    signatory_names: list[str] = Field(default_factory=list)
    signatory_titles: list[str] = Field(default_factory=list)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract structured fields from financial-services authorization documents. "
     "For every field: (1) extract the value only if it's actually present -- never guess "
     "or hallucinate a plausible-looking value, (2) give a confidence 0-1 reflecting how "
     "explicit/unambiguous the source text is, (3) quote the exact source sentence/phrase "
     "that supports your extraction. If a field is absent, leave it null with confidence 0."),
    ("human", "Document text:\n\n{text}"),
])


def extract_with_llm(text: str, is_ocr: bool) -> tuple[AuthorizationCertificateExtraction, list[ExtractedField]]:
    model = get_chat_model().with_structured_output(_LLMFieldExtraction)
    chain = EXTRACTION_PROMPT | model
    result: _LLMFieldExtraction = chain.invoke({"text": text})

    method = ExtractionMethod.OCR_LLM if is_ocr else ExtractionMethod.LLM
    # OCR-sourced text is noisier -- discount every field's confidence a bit
    # rather than trusting the model's self-reported number blindly.
    ocr_penalty = 0.85 if is_ocr else 1.0

    field_defs = [
        ("entity_name", result.entity_name, result.entity_name_confidence, result.entity_name_source),
        ("account_number", result.account_number, result.account_number_confidence, result.account_number_source),
        ("effective_date", result.effective_date, result.effective_date_confidence, result.effective_date_source),
        ("expiration_date", result.expiration_date, result.expiration_date_confidence, result.expiration_date_source),
        ("authorization_scope", result.authorization_scope, result.authorization_scope_confidence, result.authorization_scope_source),
    ]

    fields = [
        ExtractedField(
            field_name=name, value=value,
            confidence=round((conf or 0.0) * ocr_penalty, 3),
            extraction_method=method, source_snippet=source,
        )
        for name, value, conf, source in field_defs
    ]

    record = AuthorizationCertificateExtraction(
        entity_name=result.entity_name,
        account_number=result.account_number,
        effective_date=result.effective_date,
        expiration_date=result.expiration_date,
        authorization_scope=result.authorization_scope,
        governing_entity_type=result.governing_entity_type,
        authorized_signatories=[
            {"name": n, "title": (result.signatory_titles[i] if i < len(result.signatory_titles) else None)}
            for i, n in enumerate(result.signatory_names)
        ],
    )
    return record, fields
