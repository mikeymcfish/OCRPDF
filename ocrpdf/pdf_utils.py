"""PDF helper utilities for the DeepSeek OCR demo."""

from __future__ import annotations

import io
from typing import List

import pypdfium2 as pdfium
from PIL import Image

DEFAULT_RENDER_SCALE = 300 / 72  # Render at ~300 DPI.


def pdf_to_images(pdf_bytes: bytes, scale: float = DEFAULT_RENDER_SCALE) -> List[Image.Image]:
    """Convert a PDF (in bytes) to a list of Pillow images.

    Args:
        pdf_bytes: Raw bytes of the PDF document.
        scale: Rendering scale factor. The default renders pages at
            approximately 300 DPI, which works well for OCR.

    Returns:
        A list of :class:`PIL.Image.Image` objects representing each page.

    Raises:
        ValueError: If the provided data cannot be parsed as a PDF.
    """

    try:
        pdf_doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    except pdfium.PdfiumError as exc:  # pragma: no cover - depends on library internals
        raise ValueError("The provided file is not a valid PDF document.") from exc

    images: List[Image.Image] = []
    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        pil_image = page.render(scale=scale).to_pil()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        images.append(pil_image)
        page.close()  # Explicitly release resources.

    pdf_doc.close()
    return images
