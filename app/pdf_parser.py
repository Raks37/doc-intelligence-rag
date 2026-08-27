"""
PDF ingestion layer.

Covers requirement #1/#2: "document ingestion pipelines" + "document scraping /
template-based extraction for structured forms".

Strategy:
1. Try native text-layer extraction with pdfplumber (fast, high-confidence, works
   for digitally-generated PDFs -- most Authorization Certificates from banks).
2. If no usable text layer is found (scanned/faxed doc), fall back to OCR via
   pytesseract on rasterized pages (pdf2image). OCR-derived text is flagged
   `is_scanned=True` so downstream confidence scoring can discount it.
3. Also extract per-page word-level bounding boxes -- this is what powers
   template-based (positional) extraction for known form layouts, vs. handing
   raw text to an LLM for unstructured docs.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

import pdfplumber

logger = logging.getLogger(__name__)

MIN_CHARS_FOR_TEXT_LAYER = 40  # below this, treat page as scanned/image-only


@dataclass
class PageContent:
    page_number: int
    text: str
    words: list[dict] = field(default_factory=list)  # [{text, x0, x1, top, bottom}, ...]
    is_scanned: bool = False


@dataclass
class ParsedDocument:
    document_id: str
    file_path: str
    pages: list[PageContent]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def is_scanned(self) -> bool:
        return any(p.is_scanned for p in self.pages)


def parse_pdf(file_path: str, document_id: str) -> ParsedDocument:
    pages: list[PageContent] = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            words = page.extract_words() or []

            if len(text) < MIN_CHARS_FOR_TEXT_LAYER:
                logger.info("Page %d has no reliable text layer, falling back to OCR", i)
                text, words = _ocr_page(file_path, i)
                pages.append(PageContent(page_number=i, text=text, words=words, is_scanned=True))
            else:
                pages.append(PageContent(page_number=i, text=text, words=words, is_scanned=False))

    return ParsedDocument(document_id=document_id, file_path=file_path, pages=pages)


def _ocr_page(file_path: str, page_index: int) -> tuple[str, list[dict]]:
    """OCR fallback for scanned/image-only pages.

    Requires system packages: `poppler-utils` (for pdf2image) and `tesseract-ocr`.
    On Ubuntu: sudo apt-get install poppler-utils tesseract-ocr
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(file_path, first_page=page_index + 1, last_page=page_index + 1, dpi=300)
        image = images[0]
        text = pytesseract.image_to_string(image)

        # word-level boxes with confidence, useful for template extraction on scans
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = [
            {
                "text": ocr_data["text"][i],
                "x0": ocr_data["left"][i],
                "x1": ocr_data["left"][i] + ocr_data["width"][i],
                "top": ocr_data["top"][i],
                "bottom": ocr_data["top"][i] + ocr_data["height"][i],
                "ocr_confidence": int(ocr_data["conf"][i]) / 100 if ocr_data["conf"][i] != "-1" else 0.0,
            }
            for i in range(len(ocr_data["text"]))
            if ocr_data["text"][i].strip()
        ]
        return text, words
    except Exception as e:
        logger.error("OCR failed for page %d: %s", page_index, e)
        return "", []
