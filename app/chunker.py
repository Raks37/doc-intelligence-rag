"""
Text chunking for RAG retrieval.

Splits a document's flattened raw_text (app/pdf_parser.py's
ParsedDocument.full_text -- already just one "\\n\\n"-joined string by the
time it reaches this module) into overlapping windows small enough to retrieve
a handful of at once and still fit the local model's context window.

Hand-rolled rather than reaching for a library (e.g. LangChain's
RecursiveCharacterTextSplitter, even though langchain is already a
dependency here): this is deterministic string logic with no ambiguity to
resolve, in the same spirit as app/confidence.py's explicit, auditable rules
over another opaque dependency for something this simple.

Sizing is driven by the local model's context window (LLM_N_CTX=2048,
LLM_MAX_TOKENS=512 in app/llm.py -- roughly 4 chars/token for English text):
CHUNK_SIZE_CHARS=800 is ~200 tokens per chunk; app/rag.py retrieves
RAG_TOP_K=3 of them (~600 tokens of context), leaving comfortable room under
2048 for the system/question prompt plus the 512-token completion.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 120

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class TextChunk:
    chunk_index: int
    text: str


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Greedily packs paragraphs into ~chunk_size windows, carrying `overlap`
    trailing characters into the next window so a fact split across a chunk
    boundary still appears whole in at least one chunk. A single paragraph
    longer than chunk_size is sliced with fixed-size, overlapping windows
    instead (it has no internal boundary to split on)."""
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]

    windows: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if buffer:
                windows.append(buffer)
                buffer = ""
            windows.extend(_slice_fixed(paragraph, chunk_size, overlap))
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        windows.append(buffer)
        carry = buffer[-overlap:] if overlap else ""
        buffer = f"{carry}\n\n{paragraph}" if carry else paragraph

    if buffer:
        windows.append(buffer)

    return [TextChunk(chunk_index=i, text=w) for i, w in enumerate(windows)]


def _slice_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    stride = max(chunk_size - overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, len(text), stride)]
