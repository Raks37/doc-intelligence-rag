"""
Requirement: "REST APIs / microservices" + "system integrations".

Thin FastAPI layer over the LangGraph pipeline. Each endpoint corresponds to a
JD responsibility so it's easy to demo requirement-by-requirement:

  POST /documents/process              -> ingest, classify, extract, score (#1-4)
  POST /documents/{id}/compare-contact-master  -> post-load verification (#5b)
  GET  /documents/{id}/exceptions       -> exception queue for workflow platform
  GET  /audit/{id}                      -> explainability / audit trail
  POST /documents/{id}/index            -> on-demand RAG chunk+embed indexing
  POST /documents/{id}/ask              -> free-text Q&A over the document (RAG)
"""
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.graph_pipeline import run_pipeline
from app.schemas import AuthorizationCertificateExtraction
from app.comparison import compare_contact_master_vs_source
from app.rag import build_index, answer_question, DocumentIndex
from app.kafka_producer import (
    publish_event, TOPIC_EXTRACTION_COMPLETE, TOPIC_EXCEPTIONS,
    TOPIC_COMPARISON_RESULT, TOPIC_QA_ANSWERED,
)
from app.audit import AUDIT_LOG_PATH

app = FastAPI(
    title="Document Intelligence & Validation API",
    description="Ingestion, extraction, confidence scoring, and comparison for "
                "Middle Office authorization workflows.",
    version="0.1.0",
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONTACT_MASTER_PATH = Path(__file__).resolve().parent.parent / "data" / "contact_master.json"

# In-memory store for the demo; swap for Postgres/DynamoDB in production.
_STATE_STORE: dict[str, dict] = {}

# Holds live DocumentIndex objects (incl. numpy embedding matrices) per
# document -- unlike _STATE_STORE this isn't JSON-safe, so it's kept in its
# own store rather than folded in. Same "swap for a real vector DB in
# production" caveat applies.
_RAG_INDEX_STORE: dict[str, DocumentIndex] = {}


@app.post("/documents/process")
async def process_document(file: UploadFile = File(...)):
    document_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"
    file_path.write_bytes(await file.read())

    state = run_pipeline(document_id=document_id, file_path=str(file_path))
    _STATE_STORE[document_id] = state.model_dump()

    publish_event(TOPIC_EXTRACTION_COMPLETE, key=document_id, payload=state.extraction.model_dump())
    if state.exceptions:
        for exc in state.exceptions:
            publish_event(TOPIC_EXCEPTIONS, key=document_id, payload=exc.model_dump())

    return JSONResponse(state.model_dump(mode="json"))


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    if document_id not in _STATE_STORE:
        raise HTTPException(404, "Document not found")
    return _STATE_STORE[document_id]


@app.get("/documents/{document_id}/exceptions")
async def get_exceptions(document_id: str):
    state = _STATE_STORE.get(document_id)
    if not state:
        raise HTTPException(404, "Document not found")
    return state["exceptions"]


@app.post("/documents/{document_id}/compare-contact-master")
async def compare_contact_master(document_id: str, contact_master_id: str):
    """Requirement #5b: post-load verification between Contact Master and source docs."""
    state = _STATE_STORE.get(document_id)
    if not state:
        raise HTTPException(404, "Document not found")

    cm_data = json.loads(CONTACT_MASTER_PATH.read_text())
    if contact_master_id not in cm_data:
        raise HTTPException(404, "Contact Master record not found")

    contact_master_record = AuthorizationCertificateExtraction(**cm_data[contact_master_id])
    source_record = AuthorizationCertificateExtraction(**state["extraction"]["record"])

    result = compare_contact_master_vs_source(document_id, contact_master_record, source_record)
    publish_event(TOPIC_COMPARISON_RESULT, key=document_id, payload=result.model_dump())
    return result.model_dump(mode="json")


@app.post("/documents/{document_id}/index")
async def index_document(document_id: str):
    """On-demand chunk + embed -- not part of /documents/process, so a
    document nobody ever questions never pays this cost."""
    state = _STATE_STORE.get(document_id)
    if not state:
        raise HTTPException(404, "Document not found")

    index = build_index(document_id, state["raw_text"] or "")
    _RAG_INDEX_STORE[document_id] = index
    return {
        "document_id": document_id,
        "chunk_count": len(index.chunks),
        "embedding_backend": "semantic" if index.embeddings is not None else "fuzzy_fallback",
    }


@app.post("/documents/{document_id}/ask")
async def ask_document(document_id: str, question: str):
    """Free-text RAG Q&A. Auto-indexes on first call if /index wasn't
    called explicitly first."""
    state = _STATE_STORE.get(document_id)
    if not state:
        raise HTTPException(404, "Document not found")

    index = _RAG_INDEX_STORE.get(document_id)
    if index is None:
        index = build_index(document_id, state["raw_text"] or "")
        _RAG_INDEX_STORE[document_id] = index

    answer = answer_question(document_id, index, question)
    publish_event(TOPIC_QA_ANSWERED, key=document_id, payload=answer.model_dump())
    return answer.model_dump(mode="json")


@app.get("/audit/{document_id}")
async def get_audit_trail(document_id: str):
    """Requirement: model explainability and audit controls."""
    if not AUDIT_LOG_PATH.exists():
        return []
    events = [json.loads(line) for line in AUDIT_LOG_PATH.read_text().splitlines()]
    return [e for e in events if e["document_id"] == document_id]


@app.get("/health")
async def health():
    return {"status": "ok"}
