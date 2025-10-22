"""Gradio application for running DeepSeek OCR on PDF documents."""

from __future__ import annotations

from typing import List, Optional

import gradio as gr

from ocrpdf import DeepSeekOCRClient, DeepSeekOCRError, pdf_to_images


def extract_text_from_pdf(pdf_bytes: bytes, client: Optional[DeepSeekOCRClient] = None) -> str:
    """Convert a PDF into text by delegating to the DeepSeek OCR service."""

    if not pdf_bytes:
        return "No data received. Please upload a valid PDF file."

    client = client or DeepSeekOCRClient()

    try:
        pages = pdf_to_images(pdf_bytes)
    except ValueError as exc:
        return f"Failed to read PDF: {exc}"

    if not pages:
        return "The supplied PDF does not contain any pages."

    combined_sections: List[str] = []

    for index, page in enumerate(pages, start=1):
        try:
            text = client.recognize(page)
        except DeepSeekOCRError as exc:
            combined_sections.append(f"### Page {index}\nError: {exc}")
            continue

        if text:
            combined_sections.append(f"### Page {index}\n{text.strip()}")
        else:
            combined_sections.append(f"### Page {index}\n(No text returned)")

    return "\n\n".join(combined_sections)


def process_pdf_file(pdf_file: gr.File) -> str:
    if pdf_file is None:
        return "Please upload a PDF file to begin."

    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    return extract_text_from_pdf(pdf_bytes)


DESCRIPTION = """
Upload a PDF file and the app will render each page into an image, send the
images to the `deepseek-ai/DeepSeek-OCR` model hosted on Hugging Face, and
combine the resulting text.

To authenticate against the Hugging Face Inference API, set the
`HF_API_TOKEN` environment variable before launching the app. You can also
override the target endpoint by setting `HF_INFERENCE_ENDPOINT`.
"""

iface = gr.Interface(
    fn=process_pdf_file,
    inputs=gr.File(label="PDF document", file_types=[".pdf"]),
    outputs=gr.Markdown(label="Recognized text"),
    title="DeepSeek OCR PDF Demo",
    description=DESCRIPTION.strip(),
    allow_flagging="never",
)


if __name__ == "__main__":
    iface.launch()
