"""
RAG (retrieval-augmented generation) Q&A over a processed document.

Chunk -> embed -> retrieve -> answer, each step logged to the audit trail
(app/audit.py) the same way every other pipeline stage is, so "why did it
answer that?" has as concrete a paper trail as "why did it extract that?".

Indexing is deliberately NOT a node in app/graph_pipeline.py's LangGraph --
it's triggered on demand (build_index(), called from app/main.py's
/documents/{id}/index endpoint, or lazily on first /ask) so a document that
nobody ever questions never pays the embedding cost. This mirrors how
compare_contact_master_vs_source is already a separate, on-demand step
distinct from the main /documents/process pipeline.

Retrieval degrades gracefully exactly like app/comparison.py's
_semantic_similarity: if the embedding model can't be loaded, fall back to
RapidFuzz keyword matching over the raw chunk text instead of failing.

Answer generation follows the same structured-output shape as every other
LLM call site in this codebase (app/classifier.py, app/extractors/
llm_extractor.py): a private Pydantic schema, a ChatPromptTemplate, and
get_chat_model().with_structured_output(...) -- there is no plain "answer in
prose" call anywhere here, by design (app/llm.py's local-model wrapper only
exposes structured output, which is also what keeps a small model's answer
schema-checked rather than trusted at face value).

Guardrails (app/rag_guardrails.py) run before retrieval (refusing
inappropriate/manipulative questions outright, cheaply, with no model call)
and after generation (redacting anything that looks like a leaked account
number from the final answer, regardless of what the model produced).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.audit import log_pipeline_step
from app.chunker import chunk_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS
from app.embeddings import embed_texts, EMBEDDING_MODEL_NAME
from app.llm import get_chat_model
from app.rag_guardrails import check_question, redact_answer
from app.schemas import DocumentChunk, RetrievedChunk, RAGAnswer

logger = logging.getLogger(__name__)

RAG_TOP_K = 3
MAX_CONTEXT_CHARS = 3000


@dataclass
class DocumentIndex:
    """Live, in-process index for one document -- NOT JSON-safe (holds a
    numpy embedding matrix when available), so it's kept out of app/main.py's
    _STATE_STORE and lives in its own store there instead."""
    document_id: str
    chunks: list[DocumentChunk]
    embeddings: "np.ndarray | None"  # None => fuzzy fallback


# ---------------------------------------------------------------------------
# Indexing: chunk, then embed -- two separate, separately-logged steps.
# ---------------------------------------------------------------------------

def build_index(document_id: str, text: str) -> DocumentIndex:
    raw_chunks = chunk_text(text)
    chunks = [
        DocumentChunk(document_id=document_id, chunk_index=c.chunk_index, text=c.text)
        for c in raw_chunks
    ]
    log_pipeline_step(document_id, "rag_chunk", {
        "chunk_count": len(chunks),
        "chunk_size_chars": CHUNK_SIZE_CHARS,
        "overlap_chars": CHUNK_OVERLAP_CHARS,
        "chunk_char_lengths": [len(c.text) for c in chunks],
    })

    embeddings = None
    backend = "fuzzy_fallback"
    if chunks:
        try:
            embeddings = embed_texts([c.text for c in chunks])
            backend = "semantic"
        except Exception as e:  # noqa: BLE001 - any load/download failure -> fallback
            logger.warning("Embedding model unavailable (%s); RAG will use fuzzy retrieval", e)

    log_pipeline_step(document_id, "rag_embed", {
        "backend": backend,
        "embedding_model": EMBEDDING_MODEL_NAME if backend == "semantic" else None,
        "chunk_count": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings is not None else None,
    })

    return DocumentIndex(document_id=document_id, chunks=chunks, embeddings=embeddings)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _rank_chunks(chunks: list[DocumentChunk], embeddings, query_vector, top_k: int) -> list[RetrievedChunk]:
    """Pure: cosine similarity (vectors already normalized, so this is just a
    dot product) between the query and every chunk, sorted descending."""
    scored = [
        (chunks[i], float(embeddings[i] @ query_vector))
        for i in range(len(chunks))
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [RetrievedChunk(chunk=c, score=round(s, 4)) for c, s in scored[:top_k]]


def _fuzzy_retrieve(chunks: list[DocumentChunk], question: str, top_k: int) -> list[RetrievedChunk]:
    """Pure, deterministic: RapidFuzz's WRatio (its general-purpose "best of
    several strategies" score) between the question and each chunk's raw
    text -- same library app/comparison.py already uses as its own
    semantic-similarity fallback. WRatio specifically (not token_set_ratio)
    because it weighs partial/substring alignment, which matters here: a
    short question is being matched against a much longer chunk, not two
    similar-length strings. Lowercased first -- RapidFuzz is case-sensitive,
    and "Account Number" in the source text should still match a lowercase
    "account number" in the question."""
    question_lower = question.lower()
    scored = [
        (c, fuzz.WRatio(question_lower, c.text.lower()) / 100)
        for c in chunks
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [RetrievedChunk(chunk=c, score=round(s, 4)) for c, s in scored[:top_k]]


def _semantic_retrieve(index: DocumentIndex, question: str, top_k: int) -> list[RetrievedChunk]:
    query_vector = embed_texts([question])[0]
    return _rank_chunks(index.chunks, index.embeddings, query_vector, top_k)


def retrieve(index: DocumentIndex, question: str, top_k: int = RAG_TOP_K) -> list[RetrievedChunk]:
    if not index.chunks:
        return []

    if index.embeddings is not None:
        try:
            retrieved = _semantic_retrieve(index, question, top_k)
            backend = "semantic"
            reasoning = (
                f"ranked all {len(index.chunks)} chunk(s) by cosine similarity to the "
                f"question's embedding; kept the top {len(retrieved)}"
            )
        except Exception as e:  # noqa: BLE001 - embedding the question failed mid-flight
            logger.warning("Question embedding failed (%s); falling back to fuzzy retrieval", e)
            retrieved = _fuzzy_retrieve(index.chunks, question, top_k)
            backend = "fuzzy_fallback"
            reasoning = (
                f"question embedding failed; ranked all {len(index.chunks)} chunk(s) by "
                f"keyword match instead; kept the top {len(retrieved)}"
            )
    else:
        retrieved = _fuzzy_retrieve(index.chunks, question, top_k)
        backend = "fuzzy_fallback"
        reasoning = (
            f"embedding model unavailable at index time; ranked all {len(index.chunks)} "
            f"chunk(s) by keyword match; kept the top {len(retrieved)}"
        )

    log_pipeline_step(index.document_id, "rag_retrieve", {
        "question": question,
        "backend": backend,
        "top_k": top_k,
        "candidates_considered": len(index.chunks),
        "retrieved": [
            {"chunk_index": r.chunk.chunk_index, "score": r.score} for r in retrieved
        ],
        "reasoning": reasoning,
    })
    return retrieved


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def _build_context(retrieved: list[RetrievedChunk]) -> str:
    """Pure: numbers each retrieved chunk so the model (and the audit log)
    can refer back to a specific excerpt, truncating at MAX_CONTEXT_CHARS
    rather than blowing the local model's 2048-token context window."""
    parts = []
    total = 0
    for r in retrieved:
        piece = f"[Excerpt {r.chunk.chunk_index}] {r.chunk.text}"
        if total + len(piece) > MAX_CONTEXT_CHARS:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


class _RAGAnswerLLM(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    grounded: bool = False


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a financial-services authorization document "
     "using ONLY the context excerpts provided below -- never information from "
     "outside them. If the answer isn't present in the excerpts, say so plainly "
     "and set grounded=false rather than guessing. Give a confidence 0-1 "
     "reflecting how directly the excerpts support your answer."),
    ("human", "Context excerpts:\n\n{context}\n\nQuestion: {question}"),
])


def _call_llm_for_answer(question: str, retrieved: list[RetrievedChunk]) -> tuple[str, float, bool]:
    context = _build_context(retrieved)
    if not context:
        return "I don't have any indexed content to answer that from.", 0.0, False

    model = get_chat_model().with_structured_output(_RAGAnswerLLM)
    chain = ANSWER_PROMPT | model
    result: _RAGAnswerLLM = chain.invoke({"context": context, "question": question})
    return result.answer, result.confidence, result.grounded


# ---------------------------------------------------------------------------
# Top-level orchestrator -- this is the only function app/main.py calls.
# ---------------------------------------------------------------------------

def answer_question(document_id: str, index: DocumentIndex, question: str) -> RAGAnswer:
    allowed, reason = check_question(question)
    if not allowed:
        log_pipeline_step(document_id, "rag_guardrail_blocked", {
            "question": question, "reason": reason,
        })
        return RAGAnswer(
            document_id=document_id, question=question,
            answer="I can't help with that request.", confidence=0.0, grounded=False,
            retrieval_method="blocked", blocked=True, block_reason=reason,
        )

    retrieved = retrieve(index, question)
    raw_answer, confidence, grounded = _call_llm_for_answer(question, retrieved)
    final_answer, was_redacted = redact_answer(raw_answer)

    backend = "semantic" if index.embeddings is not None else "fuzzy_fallback"
    log_pipeline_step(document_id, "rag_answer", {
        "question": question,
        "grounded": grounded,
        "confidence": confidence,
        "retrieved_chunk_indices": [r.chunk.chunk_index for r in retrieved],
        "pii_redacted": was_redacted,
    })

    return RAGAnswer(
        document_id=document_id, question=question, answer=final_answer,
        confidence=confidence, grounded=grounded, retrieval_method=backend,
        retrieved_chunks=retrieved, pii_redacted=was_redacted,
    )
