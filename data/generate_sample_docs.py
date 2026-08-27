"""
Generates two demo PDFs so you don't need real bank documents for the interview:

1. sample_docs/authorization_certificate_structured.pdf
   -- matches the FIELD_PATTERNS in app/extractors/template_extractor.py,
      so it triggers the TEMPLATE extraction path.

2. sample_docs/authorization_letter_unstructured.pdf
   -- free-text authorization letter, no consistent labels, triggers the LLM
      extraction path.

Run: python data/generate_sample_docs.py
Requires: pip install reportlab
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT_DIR = Path(__file__).resolve().parent / "sample_docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_structured_pdf():
    path = OUT_DIR / "authorization_certificate_structured.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    lines = [
        "AUTHORIZATION CERTIFICATE",
        "",
        "Entity Name: Meridian Capital Partners LLC",
        "Entity Type: Limited Liability Company",
        "Account Number: 5551234821",
        "Effective Date: 01/15/2026",
        "Expiration Date: 01/15/2028",
        "Authorization Scope: Wire transfers up to $500,000 and account maintenance requests",
        "",
        "Authorized Signatory: Jane R. Whitfield, Title: Chief Financial Officer",
        "Authorized Signatory: Marcus T. Delgado, Title: Managing Partner",
        "",
        "This certificate is issued under the authority of the entity's governing documents",
        "and remains valid until the expiration date listed above unless revoked in writing.",
    ]
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    print(f"Wrote {path}")


def make_unstructured_pdf():
    path = OUT_DIR / "authorization_letter_unstructured.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    text = c.beginText(72, 750)
    text.setFont("Helvetica", 11)
    paragraphs = [
        "To Whom It May Concern,",
        "",
        "This letter confirms that Northbridge Family Office, a Delaware trust,",
        "has authorized the following individuals to act on its behalf for",
        "transactions on account ending in 7743:",
        "",
        "  - Elena Marchetti, serving as Trustee",
        "  - David Osei, serving as Investment Director",
        "",
        "Both individuals are permitted to authorize outgoing wire transfers not",
        "exceeding two hundred fifty thousand dollars per transaction, effective",
        "immediately as of March 3, 2026, through March 3, 2027, at which point",
        "this authorization must be renewed.",
        "",
        "Please contact our office with any questions regarding this authorization.",
        "",
        "Sincerely,",
        "Northbridge Family Office, Legal Department",
    ]
    for line in paragraphs:
        text.textLine(line)
    c.drawText(text)
    c.save()
    print(f"Wrote {path}")


if __name__ == "__main__":
    make_structured_pdf()
    make_unstructured_pdf()
