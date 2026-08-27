"""
LangGraph orchestration.

Why LangGraph over a plain function chain: the pipeline has a genuine branch
(template path vs LLM path) and needs a shared, typed state object threaded
through every node -- exactly LangGraph's sweet spot, and it's explicitly
called out in the JD's required skills. Each node is a small, independently
testable function, which also makes the pipeline easy to explain step-by-step
in an interview / demo.

Flow:
  ingest -> classify -> [route] -> extract_template | extract_llm
         -> score_confidence -> compare_extracted_vs_source
         -> flag_exceptions -> audit_log
"""
from langgraph.graph import StateGraph, END

from app.schemas import (
    PipelineState, DocumentType, AuthorizationCertificateExtraction, ExtractionRecord,
)
from app.pdf_parser import parse_pdf
from app.classifier import classify_document
from app.extractors.template_extractor import looks_like_known_template, extract_with_template
from app.extractors.llm_extractor import extract_with_llm
from app.confidence import compute_record_confidence, flag_exceptions
from app.comparison import compare_extracted_vs_source
from app.audit import log_pipeline_step


def node_ingest(state: PipelineState) -> dict:
    parsed = parse_pdf(state.file_path, state.document_id)
    log_pipeline_step(state.document_id, "ingest", {
        "pages": len(parsed.pages), "is_scanned": parsed.is_scanned, "chars": len(parsed.full_text),
    })
    return {"raw_text": parsed.full_text, "is_scanned": parsed.is_scanned, "status": "parsed"}


def node_classify(state: PipelineState) -> dict:
    result = classify_document(state.raw_text, state.is_scanned)
    log_pipeline_step(state.document_id, "classify", result.model_dump())
    return {"classification": result, "status": "classified"}


def route_extraction(state: PipelineState) -> str:
    """Template extraction only for known, non-scanned Authorization
    Certificates with a recognizable layout. Everything else -> LLM."""
    if (state.classification.document_type == DocumentType.AUTHORIZATION_CERTIFICATE
            and not state.is_scanned
            and looks_like_known_template(state.raw_text)):
        return "extract_template"
    return "extract_llm"


def node_extract_template(state: PipelineState) -> dict:
    record, fields = extract_with_template(state.raw_text)
    return _finalize_extraction(state, record, fields)


def node_extract_llm(state: PipelineState) -> dict:
    record, fields = extract_with_llm(state.raw_text, is_ocr=state.is_scanned)
    return _finalize_extraction(state, record, fields)


def _finalize_extraction(state: PipelineState, record: AuthorizationCertificateExtraction, fields) -> dict:
    record_conf = compute_record_confidence(fields)
    extraction = ExtractionRecord(
        document_id=state.document_id, document_type=state.classification.document_type,
        fields=fields, record=record, record_level_confidence=record_conf,
    )
    log_pipeline_step(state.document_id, "extract", {
        "record_confidence": record_conf,
        "fields": {f.field_name: {"value": f.value, "confidence": f.confidence} for f in fields},
    })
    return {"extraction": extraction, "status": "extracted"}


def node_score_and_flag(state: PipelineState) -> dict:
    exceptions = flag_exceptions(state.document_id, state.extraction.fields)
    log_pipeline_step(state.document_id, "exceptions", {"count": len(exceptions)})
    return {"exceptions": exceptions, "status": "scored"}


def node_validate_vs_source(state: PipelineState) -> dict:
    """Independent second extraction pass (template, if it matches) used as a
    ground-truth check against the primary extraction -- catches LLM
    hallucination. Falls back to a no-op comparison if no template match exists."""
    if looks_like_known_template(state.raw_text):
        reference_record, _ = extract_with_template(state.raw_text)
        comparison = compare_extracted_vs_source(state.document_id, state.extraction.record, reference_record)
        log_pipeline_step(state.document_id, "compare_extracted_vs_source", {
            "overall_status": comparison.overall_status, "requires_review": comparison.requires_manual_review,
        })
        return {"extracted_vs_source": comparison, "status": "validated"}
    return {"status": "validated"}


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", node_ingest)
    graph.add_node("classify", node_classify)
    graph.add_node("extract_template", node_extract_template)
    graph.add_node("extract_llm", node_extract_llm)
    graph.add_node("score_and_flag", node_score_and_flag)
    graph.add_node("validate_vs_source", node_validate_vs_source)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "classify")
    graph.add_conditional_edges("classify", route_extraction, {
        "extract_template": "extract_template", "extract_llm": "extract_llm",
    })
    graph.add_edge("extract_template", "score_and_flag")
    graph.add_edge("extract_llm", "score_and_flag")
    graph.add_edge("score_and_flag", "validate_vs_source")
    graph.add_edge("validate_vs_source", END)

    return graph.compile()


pipeline = build_pipeline()


def run_pipeline(document_id: str, file_path: str) -> PipelineState:
    initial_state = PipelineState(document_id=document_id, file_path=file_path)
    result = pipeline.invoke(initial_state.model_dump())
    return PipelineState(**result)
