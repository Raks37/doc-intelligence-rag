"""
Document classification.

Requirement #1: "Build document ingestion and classification pipelines for
Authorization Certificates and non-standard documents."

This is a routing decision: known, templated bank forms go down the cheap/fast
TEMPLATE extraction path; anything irregular goes down the LLM extraction path.
We use a lightweight LLM call with structured output rather than training a
classifier from scratch -- appropriate for a 1-day demo, and defensible in an
interview as "fast to ship, upgrade to a fine-tuned classifier once you have
labeled volume."
"""
from langchain_core.prompts import ChatPromptTemplate
from app.schemas import ClassificationResult
from app.llm import get_chat_model

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You classify financial-services middle-office documents. "
     "'authorization_certificate' = a formal bank/entity document naming authorized "
     "signatories and their transaction authority (often has a standard letterhead/template). "
     "'non_standard' = anything else (letters, emails, ad-hoc PDFs, board resolutions in "
     "free-text form, etc). Be concise in your reasoning."),
    ("human", "Document text (first 2000 chars):\n\n{text}"),
])


def classify_document(raw_text: str, is_scanned: bool) -> ClassificationResult:
    model = get_chat_model().with_structured_output(ClassificationResult)
    chain = CLASSIFY_PROMPT | model
    result: ClassificationResult = chain.invoke({"text": raw_text[:2000]})
    result.is_scanned_image = is_scanned
    return result
