"""
Shared sentence-embedding model loader.

Extracted out of app/comparison.py so both the field-level semantic
comparison (`compare_extracted_vs_source` / `compare_contact_master_vs_source`)
and the RAG retriever (app/rag.py) share one cached model load and one
"what if it can't load" story, instead of each keeping a private copy of the
same three lines. Callers are responsible for catching load failures (e.g.
running offline with the model not yet cached) and degrading gracefully --
this module doesn't hide that decision from them.
"""
from functools import lru_cache

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]):
    """Returns a normalized embedding matrix (one row per input text), so
    cosine similarity between any two rows is just a dot product."""
    return get_embedder().encode(texts, normalize_embeddings=True)
