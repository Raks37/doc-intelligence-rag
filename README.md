# Document Intelligence & Validation — Demo

A working prototype covering every requirement in the AI Developer – Document
Intelligence & Validation JD, built to demo in an interview.

## What it covers, mapped to the JD

| JD requirement | Where it lives |
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

