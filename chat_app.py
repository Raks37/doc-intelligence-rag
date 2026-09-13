"""
Gradio chatbot front end for the Document Intelligence & Validation API.

This is a thin HTTP client, not a re-implementation of the pipeline: every
answer in the chat comes straight from the existing FastAPI backend
(`app/main.py`), so "the bot calling this codebase in the backend" is
literal, not a figure of speech. Deliberately kept in its own virtualenv
(`.venv-chat`, just `gradio` + `requests`) rather than importing `app.*`
directly -- `gradio` needs `huggingface-hub>=1.16`, which conflicts with the
`transformers`/`tokenizers` pins the main pipeline venv (`.venv`) uses for
the optional semantic-comparison fallback in `app/comparison.py`. Two
processes, two venvs, one REST boundary between them -- the same boundary a
real caller of this API would sit behind.

Run:
    1. In the main venv, in one terminal:   uvicorn app.main:app --reload
    2. In .venv-chat, in another terminal:  python chat_app.py
    Opens http://127.0.0.1:7860

Env:
    BACKEND_URL   base URL of the FastAPI server (default http://127.0.0.1:8000)
"""
import os
import json

import gradio as gr
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 300  # local GGUF inference can be slow on CPU

# Per-Gradio-session bookkeeping: which document/contact-master ids has this
# browser tab produced, so free-text follow-up commands can refer to "it".
_SESSION_STATE: dict = {}


def _confidence_bar(score: float) -> str:
    filled = round(max(0.0, min(1.0, score)) * 5)
    return "\U0001f7e9" * filled + "⬜" * (5 - filled)


def _fmt_field(f: dict) -> str:
    value = f["value"] if f["value"] is not None else "—"
    lines = [
        f"- **{f['field_name']}**: `{value}`  \n"
        f"  confidence {f['confidence']:.2f} {_confidence_bar(f['confidence'])}"
        f"  _(method: {f['extraction_method']})_"
    ]
    if f.get("source_snippet"):
        lines.append(f"  > source: “{f['source_snippet']}”")
    return "\n".join(lines)


def _fmt_comparison(comp: dict, label: str, left_col: str, right_col: str) -> str:
    rows = "\n".join(
        f"| {fc['field_name']} | {fc['extracted_value'] or '—'} | {fc['reference_value'] or '—'} "
        f"| {fc['similarity_score']:.2f} | {fc['status']} |"
        for fc in comp["field_comparisons"]
    )
    header = (
        f"**{label}** — overall: **{comp['overall_status']}**"
        f"  ·  needs manual review: **{comp['requires_manual_review']}**"
    )
    if comp.get("review_reason"):
        header += f"\n> {comp['review_reason']}"
    table = f"| field | {left_col} | {right_col} | similarity | status |\n|---|---|---|---|---|\n{rows}"
    return f"{header}\n\n{table}"


def _fmt_pipeline_result(filename: str, state: dict) -> str:
    parts = [f"\U0001f4ce **{filename}** → document id `{state['document_id']}`"]

    parts.append(
        f"### 1️⃣ Ingest\n"
        f"{len(state['raw_text'])} characters extracted · scanned image: **{state['is_scanned']}**"
    )

    c = state["classification"]
    parts.append(
        f"### 2️⃣ Classify\n"
        f"**{c['document_type']}**  (confidence {c['confidence']:.2f})\n"
        f"> reasoning: _{c['reasoning']}_"
    )

    e = state["extraction"]
    method = e["fields"][0]["extraction_method"] if e["fields"] else "n/a"
    field_lines = "\n".join(_fmt_field(f) for f in e["fields"])
    parts.append(
        f"### 3️⃣ Extract  _(method: {method})_\n{field_lines}\n\n"
        f"**Record-level confidence: {e['record_level_confidence']:.2f} "
        f"{_confidence_bar(e['record_level_confidence'])}**"
    )

    if state["exceptions"]:
        exc_lines = "\n".join(
            f"- \U0001f534 **{x['severity'].upper()}** `{x['field_name']}` — {x['reason']}\n"
            f"  → _{x['suggested_action']}_"
            for x in state["exceptions"]
        )
        parts.append(f"### 4️⃣ Exceptions\n{exc_lines}")
    else:
        parts.append("### 4️⃣ Exceptions\n✅ none raised")

    if state.get("extracted_vs_source"):
        parts.append(
            "### 5️⃣ Compare extracted vs. source\n"
            + _fmt_comparison(state["extracted_vs_source"], "extracted vs. source", "extracted", "source")
        )
    else:
        parts.append(
            "### 5️⃣ Compare extracted vs. source\n"
            "_(skipped — no known template available to cross-check this extraction against)_"
        )

    parts.append(
        f"### 6️⃣ Audit trail\n"
        f"```json\n{json.dumps(state.get('_audit', []), indent=2)}\n```"
    )

    return "\n\n".join(parts)


def _process_upload(file_path: str) -> str:
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as fh:
        resp = requests.post(
            f"{BACKEND_URL}/documents/process",
            files={"file": (filename, fh, "application/pdf")},
            timeout=REQUEST_TIMEOUT,
        )
    if resp.status_code != 200:
        return f"❌ Backend returned {resp.status_code}:\n```\n{resp.text}\n```"

    state = resp.json()
    document_id = state["document_id"]

    # The backend's single worker blocks its event loop for the full duration of
    # a synchronous pipeline run, so a request queued right behind a slow LLM
    # extraction can sit waiting well past a short timeout even though the
    # audit endpoint itself is cheap. Match REQUEST_TIMEOUT rather than using a
    # short one here.
    audit_resp = requests.get(f"{BACKEND_URL}/audit/{document_id}", timeout=REQUEST_TIMEOUT)
    state["_audit"] = audit_resp.json() if audit_resp.status_code == 200 else []

    _SESSION_STATE["last_document_id"] = document_id
    _SESSION_STATE["last_filename"] = filename

    footer = (
        "\n\n---\n\U0001f4a1 Type `compare <contact-master-id>` (e.g. `compare CM-1001`) "
        "to run post-load verification of this document against a Contact Master record, "
        "or `audit` to re-print its audit trail."
    )
    return _fmt_pipeline_result(filename, state) + footer


def _handle_compare_command(cm_id: str) -> str:
    document_id = _SESSION_STATE.get("last_document_id")
    if not document_id:
        return "Upload a PDF first, then I can compare it against a Contact Master record."

    resp = requests.post(
        f"{BACKEND_URL}/documents/{document_id}/compare-contact-master",
        params={"contact_master_id": cm_id},
        timeout=30,
    )
    if resp.status_code != 200:
        return f"❌ Backend returned {resp.status_code}:\n```\n{resp.text}\n```"

    comp = resp.json()
    return _fmt_comparison(comp, f"Contact Master `{cm_id}` vs. source", "contact master", "source")


def _handle_audit_command() -> str:
    document_id = _SESSION_STATE.get("last_document_id")
    if not document_id:
        return "Upload a PDF first, then I can show its audit trail."
    resp = requests.get(f"{BACKEND_URL}/audit/{document_id}", timeout=30)
    if resp.status_code != 200:
        return f"❌ Backend returned {resp.status_code}:\n```\n{resp.text}\n```"
    return f"### Audit trail for `{document_id}`\n```json\n{json.dumps(resp.json(), indent=2)}\n```"


def chat_fn(message: dict, history: list) -> str:
    files = message.get("files") or []
    text = (message.get("text") or "").strip()

    if files:
        try:
            results = [_process_upload(f) for f in files]
        except requests.exceptions.ConnectionError:
            return (
                f"❌ Can't reach the backend at `{BACKEND_URL}`. "
                f"Start it first with `uvicorn app.main:app --reload` (in the main `.venv`)."
            )
        return "\n\n---\n\n".join(results)

    if text.lower().startswith("compare"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: `compare <contact-master-id>`, e.g. `compare CM-1001`."
        try:
            return _handle_compare_command(parts[1].strip())
        except requests.exceptions.ConnectionError:
            return f"❌ Can't reach the backend at `{BACKEND_URL}`."

    if text.lower() == "audit":
        try:
            return _handle_audit_command()
        except requests.exceptions.ConnectionError:
            return f"❌ Can't reach the backend at `{BACKEND_URL}`."

    return (
        "\U0001f4ce Attach a PDF (Authorization Certificate) to run it through the full pipeline: "
        "ingest → classify → extract → confidence-score → flag exceptions → "
        "compare vs. source → audit log. I'll show every stage's reasoning and output.\n\n"
        "After that, try `compare CM-1001` or `audit`."
    )


demo = gr.ChatInterface(
    fn=chat_fn,
    multimodal=True,
    title="Document Intelligence & Validation — Chatbot",
    description=(
        f"Backend: `{BACKEND_URL}` (start with `uvicorn app.main:app --reload`). "
        "Upload an Authorization Certificate PDF and I'll run it through the full "
        "LangGraph pipeline, showing classification reasoning, per-field extraction "
        "with confidence and source snippets, exceptions, source comparison, and the "
        "audit trail."
    ),
)

if __name__ == "__main__":
    demo.launch()
