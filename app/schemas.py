"""
Core Pydantic schemas for the Document Intelligence pipeline.

These are the data contracts that flow between every stage:
ingestion -> classification -> extraction -> confidence scoring -> comparison -> audit.

Using Pydantic (not raw dicts) gives us:
- Guaranteed structure for LLM structured-output parsing (LangChain .with_structured_output)
- Type safety across microservice boundaries (these models double as the JSON schema
  you'd publish on a Kafka topic or expose via REST)
- Free validation (e.g. confidence must be 0-1, dates must parse) which feeds directly
  into "rule-based validation" from the JD
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 1. Document classification
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    AUTHORIZATION_CERTIFICATE = "authorization_certificate"
    NON_STANDARD = "non_standard"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Which extraction strategy produced a field. Drives confidence weighting."""
    TEMPLATE = "template_based"   # regex / positional extraction from a known form layout
    LLM = "llm_based"             # LangChain structured extraction for unstructured text
    OCR_LLM = "ocr_llm_based"     # scanned doc -> OCR text -> LLM extraction (lowest trust)


class ClassificationResult(BaseModel):
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Short explanation for audit/explainability")
    is_scanned_image: bool = Field(
        description="True if the PDF had no extractable text layer and required OCR"
    )


# ---------------------------------------------------------------------------
# 2. Extraction output
# ---------------------------------------------------------------------------

class AuthorizedSignatory(BaseModel):
    name: str
    title: Optional[str] = None
    signature_present: bool = False


class ExtractedField(BaseModel):
    """A single field with its own provenance and confidence -- this is what
    lets you do FIELD-LEVEL confidence scoring, not just document-level."""
    field_name: str
    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: ExtractionMethod
    source_snippet: Optional[str] = Field(
        default=None, description="Raw text span the value was extracted from (for audit trail)"
    )


class AuthorizationCertificateExtraction(BaseModel):
    """The structured record we want out of an Authorization Certificate.
    This is the schema handed to the LLM as a structured-output target,
    and also the schema stored/compared downstream."""
    entity_name: Optional[str] = None
    account_number: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    authorized_signatories: list[AuthorizedSignatory] = Field(default_factory=list)
    authorization_scope: Optional[str] = Field(
        default=None, description="e.g. 'wire transfers up to $500,000'"
    )
    governing_entity_type: Optional[str] = None

    @field_validator("account_number")
    @classmethod
    def mask_account_for_pii(cls, v: Optional[str]) -> Optional[str]:
        """PII handling: never store full account numbers in plaintext downstream.
        Keep last 4 digits only, in line with audit/PII controls in the JD."""
        if v and len(v) > 4:
            return f"****{v[-4:]}"
        return v


class ExtractionRecord(BaseModel):
    """The full extraction result for one document: every field + provenance."""
    document_id: str
    document_type: DocumentType
    fields: list[ExtractedField]
    record: AuthorizationCertificateExtraction
    record_level_confidence: float = Field(ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 3. Comparison / validation
# ---------------------------------------------------------------------------

class ComparisonStatus(str, Enum):
    MATCH = "match"
    PARTIAL_MATCH = "partial_match"
    MISMATCH = "mismatch"
    MISSING_IN_SOURCE = "missing_in_source"
    MISSING_IN_TARGET = "missing_in_target"


class FieldComparison(BaseModel):
    field_name: str
    extracted_value: Optional[str]
    reference_value: Optional[str]
    similarity_score: float = Field(ge=0.0, le=1.0)
    status: ComparisonStatus


class ComparisonType(str, Enum):
    EXTRACTED_VS_SOURCE = "extracted_vs_source"       # did the LLM/template hallucinate?
    CONTACT_MASTER_VS_SOURCE = "contact_master_vs_source"  # post-load verification


class ComparisonResult(BaseModel):
    document_id: str
    comparison_type: ComparisonType
    field_comparisons: list[FieldComparison]
    overall_status: ComparisonStatus
    requires_manual_review: bool
    review_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# 4. Exception handling / workflow
# ---------------------------------------------------------------------------

class ExceptionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExceptionRecord(BaseModel):
    document_id: str
    severity: ExceptionSeverity
    reason: str
    field_name: Optional[str] = None
    suggested_action: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 5. Pipeline state (used by LangGraph)
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    document_id: str
    file_path: str
    raw_text: Optional[str] = None
    is_scanned: Optional[bool] = None
    classification: Optional[ClassificationResult] = None
    extraction: Optional[ExtractionRecord] = None
    extracted_vs_source: Optional[ComparisonResult] = None
    contact_master_vs_source: Optional[ComparisonResult] = None
    exceptions: list[ExceptionRecord] = Field(default_factory=list)
    status: str = "ingested"


# ---------------------------------------------------------------------------
# 6. RAG Q&A (chunking / retrieval / answer generation)
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity, or "
                          "RapidFuzz ratio if the embedding model was unavailable")


class RAGAnswer(BaseModel):
    """Result of asking a free-text question about a processed document.
    `retrieved_chunks` carries full chunk text + score (not just indices) so
    a caller -- the chat front end included -- can show *why* this answer
    was given, not just what it was."""
    document_id: str
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool = Field(description="Model's self-report of whether the "
                            "answer is actually supported by the retrieved context")
    retrieval_method: str = Field(description="'semantic' | 'fuzzy_fallback' | 'blocked'")
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    blocked: bool = Field(default=False, description="True if a guardrail "
                           "refused the question before any retrieval/LLM call")
    block_reason: Optional[str] = None
    pii_redacted: bool = Field(default=False, description="True if the "
                                "answer contained an account-number-shaped "
                                "value that was masked before returning it")
    answered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
