# Document Intelligence & Validation — Demo

A working prototype covering every requirement in the AI Developer – Document
Intelligence & Validation.

## What it covers, mapped to the JD

|Requirement | Where it lives |
|---|---|
| Ingestion & classification pipeline | `app/pdf_parser.py`, `app/classifier.py` |
| Template-based extraction (structured forms) | `app/extractors/template_extractor.py` |
| LLM extraction (unstructured/semi-structured) | `app/extractors/llm_extractor.py` |
| Field-level + record-level confidence scoring | `app/confidence.py`, `_LLMFieldExtraction` in `llm_extractor.py` |
| Extracted vs source comparison | `app/comparison.py` → `compare_extracted_vs_source` |
| Contact Master vs source (post-load verification) | `app/comparison.py` → `compare_contact_master_vs_source` |
| LangGraph orchestration | `app/graph_pipeline.py` |
| REST API / microservice | `app/main.py` |
| Kafka / JSON schema mapping | `app/kafka_producer.py` |
| PII handling | `AuthorizationCertificateExtraction.mask_account_for_pii`, `app/audit.py` |
| Explainability / audit controls | `app/audit.py`, `source_snippet` on every extracted field |
| Exception handling | `app/confidence.py` → `flag_exceptions` |
| Free-text Q&A over a document (RAG) | `app/chunker.py`, `app/embeddings.py`, `app/rag.py` → `POST /documents/{id}/index`, `POST /documents/{id}/ask` |
| Guardrails (question filtering + PII redaction) | `app/rag_guardrails.py` |


## Architecture

```
PDF Upload
   │
   ▼
[Ingest: pdfplumber + OCR fallback] ──► raw_text, is_scanned
   │
   ▼
[Classify: LLM structured output] ──► Authorization Certificate | Non-standard
   │
   ├── known template? ──► [Template Extractor: regex/positional] ─┐
   │                                                                 │
   └── else ──────────────► [LLM Extractor: LangChain + Pydantic] ──┤
                                                                      ▼
                                                    [Confidence Scoring: field + record level]
                                                                      │
                                                                      ▼
                                                    [Exception Flagging: severity-routed]
                                                                      │
                                                                      ▼
                                            [Compare: extracted vs source (2nd-pass validation)]
                                                                      │
                                                                      ▼
                                        [Publish to Kafka: extraction.completed, exceptions.raised]
                                                                      │
                                                                      ▼
                                                          [Audit log: every step, every decision]

Separately, on demand:
[Contact Master record] ──► [Compare: Contact Master vs source] ──► post-load verification

[raw_text] ──► [Chunk + Embed: app/chunker.py + app/embeddings.py]
            ──► [Retrieve top-k (guardrail-checked): app/rag.py]
            ──► [LLM answer, PII-redacted] ──► grounded Q&A
```


## Run it locally

Runs fully offline with a local GGUF model — no API key required.

### 1. Environment

```powershell
cd doc-intelligence-rag
py -3.13 -m venv .venv --system-site-packages   # reuses globally-installed torch / llama-cpp-python if present
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

```powershell
copy .env.example .env
```

Edit `.env` and point `LLM_MODEL_PATH` at your `.gguf` file. The default is a
Phi-2 build:

```
LLM_PROVIDER=local
LLM_MODEL_PATH=C:\path\to\phi-2.Q4_K_M.gguf
LLM_N_CTX=2048
```

> **Phi-2 is a 2.7B base model with a 2048-token context.** The pipeline runs
> and returns schema-valid JSON (output is grammar-constrained), but extraction
> accuracy on free-text documents is modest — expect the LLM path to leave
> fields empty, which the pipeline then routes to the exception queue. The
> template path (structured forms) does not use the LLM and is unaffected. Long
> documents will exceed the context window. For much better results, still local,
> run a bigger model
> behind an OpenAI-compatible server (e.g. LM Studio) and set in `.env`:
> `LLM_PROVIDER=openai`, `OPENAI_BASE_URL=http://localhost:1234/v1`,
> `OPENAI_API_KEY=lm-studio`, `LLM_MODEL=<loaded model>` (needs
> `pip install langchain-openai`).

### 3. Generate sample documents

```powershell
python data/generate_sample_docs.py
```

Creates `data/sample_docs/authorization_certificate_structured.pdf` (triggers the
template path — no LLM call) and `authorization_letter_unstructured.pdf` (triggers
the LLM path).

### 4. Run the tests

```powershell
pytest -q
```

The tests are deterministic and never call an LLM.

### 5. Start the API

```powershell
uvicorn app.main:app --reload

```

Open http://127.0.0.1:8000/docs for Swagger UI, or use curl:

```bash
# Full pipeline (returns the complete PipelineState)
curl -F "file=@data/sample_docs/authorization_certificate_structured.pdf" \
     http://127.0.0.1:8000/documents/process

# Using the id from the response above:
curl http://127.0.0.1:8000/documents/<id>
curl http://127.0.0.1:8000/documents/<id>/exceptions
curl http://127.0.0.1:8000/audit/<id>
curl -X POST "http://127.0.0.1:8000/documents/<id>/compare-contact-master?contact_master_id=CM-1001"
```

First request is slow (the GGUF model loads on demand). Kafka is disabled by
default — events are logged as `[KAFKA-SIMULATED]`. The audit trail is written to
`data/audit_log.jsonl`.

### 6. Ask questions about a document (RAG)

Chunking, embedding, retrieval, and answer generation are all on-demand — a
document nobody ever questions never pays the embedding cost. Indexing runs
automatically the first time you ask; you can also trigger it explicitly:

```bash
curl -X POST http://127.0.0.1:8000/documents/<id>/index

curl -X POST "http://127.0.0.1:8000/documents/<id>/ask?question=Who%20are%20the%20authorized%20signatories%3F"
```

Every step — chunking, embedding, retrieval (with its ranked candidates and
the reasoning behind which ones were kept), and the final answer — is logged
to the same audit trail as the rest of the pipeline
(`curl http://127.0.0.1:8000/audit/<id>`). Two guardrails run on every
question: an input filter blocks off-topic or manipulative questions before
any retrieval or LLM call happens (try `question=What%20is%20the%20full%20account%20number%3F`
to see it refuse), and an output filter masks any account-number-shaped
value out of the final answer regardless of what the model produced. Same
Phi-2/2048-token-context caveat as extraction applies here too — answer
quality (and the model's own `grounded` self-report) is modest with the
default local model.

### 7. Chat with it

`chat_app.py` is a Gradio chatbot front end: drag a PDF into the chat box and
it shows every stage of the pipeline's reasoning — classification rationale,
per-field extraction with confidence bars and source snippets, exceptions
raised, the extracted-vs-source comparison table, and the full audit trail —
laid out as one chat reply. After uploading, type `compare CM-1001` (or
`CM-1002`) to run post-load verification against a Contact Master record,
`audit` to re-print the audit trail, or `ask <question>` (e.g.
`ask what is the authorization scope?`) to ask it something — the reply
shows the answer, a grounded/confidence indicator, and which retrieved
excerpts (with scores) it was based on.

It's a plain HTTP client of the API above (`requests` calls to
`/documents/process`, `/documents/{id}/compare-contact-master`,
`/documents/{id}/ask`, `/audit/{id}`) — not a reimplementation of the
pipeline — so it needs the API
server running (step 5) and lives in its **own venv**, separate from
`.venv`:

```powershell
py -3.13 -m venv .venv-chat
.\.venv-chat\Scripts\Activate.ps1
pip install -r requirements-chat.txt
python chat_app.py
```

Open http://127.0.0.1:7860. (Separate venv because `gradio` requires
`huggingface-hub>=1.16`, which conflicts with the `transformers`/
`tokenizers` pins `requirements.txt` needs for the optional
semantic-comparison fallback in `app/comparison.py`; since `chat_app.py`
only talks HTTP to the API, it never needs `transformers` at all.)

