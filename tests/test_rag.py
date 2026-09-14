"""
Deterministic unit tests for chunking and retrieval-ranking -- like
tests/test_pipeline.py, none of these call a real LLM or the real embedding
model. Only pure functions (chunk_text, _rank_chunks, _fuzzy_retrieve,
_build_context) and retrieve()'s empty-chunks short-circuit (which never
touches I/O) are exercised. build_index, _semantic_retrieve, and
_call_llm_for_answer are impure boundary calls -- left untested here, same
as extract_with_llm/classify_document are left untested in test_pipeline.py.
"""
import numpy as np

from app.chunker import chunk_text
from app.rag import _rank_chunks, _fuzzy_retrieve, _build_context, retrieve, DocumentIndex
from app.schemas import DocumentChunk, RetrievedChunk


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("Hello world")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Hello world"


def test_chunk_text_splits_long_text_and_overlaps():
    text = ("A" * 60) + "\n\n" + ("B" * 60) + "\n\n" + ("C" * 60)
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert len(chunks) == 3
    # each chunk after the first should start with the overlap tail of the one before it
    assert chunks[1].text.startswith(chunks[0].text[-20:])
    assert chunks[2].text.startswith(chunks[1].text[-20:])


def test_rank_chunks_orders_by_cosine_similarity():
    chunks = [
        DocumentChunk(document_id="doc-1", chunk_index=0, text="irrelevant"),
        DocumentChunk(document_id="doc-1", chunk_index=1, text="relevant"),
        DocumentChunk(document_id="doc-1", chunk_index=2, text="somewhat relevant"),
    ]
    embeddings = np.array([
        [0.0, 1.0],  # orthogonal to the query -> lowest score
        [1.0, 0.0],  # parallel to the query -> highest score
        [0.7, 0.7],  # partial alignment -> medium score
    ])
    query_vector = np.array([1.0, 0.0])

    ranked = _rank_chunks(chunks, embeddings, query_vector, top_k=2)

    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_index == 1
    assert ranked[0].score > ranked[1].score


def test_fuzzy_retrieve_finds_keyword_match_without_embeddings():
    chunks = [
        DocumentChunk(document_id="doc-1", chunk_index=0, text="The weather today is sunny and warm."),
        DocumentChunk(document_id="doc-1", chunk_index=1, text="Account Number: 5551234821, held by Acme LLC."),
    ]
    ranked = _fuzzy_retrieve(chunks, "What is the account number?", top_k=1)
    assert ranked[0].chunk.chunk_index == 1


def test_retrieve_empty_chunks_returns_empty_list():
    index = DocumentIndex(document_id="doc-1", chunks=[], embeddings=None)
    assert retrieve(index, "anything?") == []


def test_build_context_stops_at_char_budget():
    retrieved = [
        RetrievedChunk(
            chunk=DocumentChunk(document_id="doc-1", chunk_index=i, text="x" * 1000),
            score=1.0 - i * 0.1,
        )
        for i in range(4)
    ]
    context = _build_context(retrieved)
    assert context.count("[Excerpt") == 2
    assert "[Excerpt 3]" not in context
