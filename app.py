"""Gradio application for running DeepSeek OCR on PDF documents."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from ocrpdf import DeepSeekOCRClient, DeepSeekOCRError, pdf_to_images


StateDict = Dict[str, Any]


def _initial_state() -> StateDict:
    return {
        "source_name": None,
        "source_kind": None,
        "temp_dir": None,
        "zip_path": None,
        "log": [],
        "image_paths": [],
    }


def _ensure_state(state: Optional[StateDict]) -> StateDict:
    if not state:
        return _initial_state()

    # Guarantee expected keys exist even when the object came from a previous
    # version of the UI state structure.
    defaults = _initial_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    return state


def _clear_state(state: StateDict) -> StateDict:
    temp_dir = state.get("temp_dir")
    if isinstance(temp_dir, str):
        shutil.rmtree(temp_dir, ignore_errors=True)
    return _initial_state()


def _reset_temp_dir(state: StateDict) -> Path:
    if state.get("temp_dir"):
        shutil.rmtree(state["temp_dir"], ignore_errors=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="ocrpdf_pages_"))
    state["temp_dir"] = str(temp_dir)
    return temp_dir


def _infer_upload_name(upload: object, fallback: str) -> str:
    for attr in ("orig_name", "name"):
        value = getattr(upload, attr, None)
        if isinstance(value, str) and value:
            return Path(value).name
    return fallback


def _read_upload_bytes(upload: object, *, fallback_name: str) -> Tuple[bytes, str]:
    if upload is None:
        raise ValueError("No file provided.")

    display_name = _infer_upload_name(upload, fallback_name)

    if isinstance(upload, (bytes, bytearray)):
        return bytes(upload), display_name

    if hasattr(upload, "read"):
        file_obj = upload
        try:
            file_obj.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        data = file_obj.read()
        return data, display_name

    path_candidate = getattr(upload, "name", None)
    if not isinstance(path_candidate, str):
        path_candidate = getattr(upload, "value", None)

    if isinstance(path_candidate, str) and Path(path_candidate).exists():
        with open(path_candidate, "rb") as handle:
            return handle.read(), display_name

    if isinstance(upload, str) and Path(upload).exists():
        with open(upload, "rb") as handle:
            return handle.read(), display_name

    raise ValueError("Unable to read the uploaded file.")


def _render_pdf_to_disk(pdf_bytes: bytes, state: StateDict) -> Tuple[List[str], str, List[str]]:
    logs: List[str] = []

    try:
        images = pdf_to_images(pdf_bytes)
    except ValueError as exc:
        raise ValueError(f"Failed to read PDF: {exc}") from exc

    if not images:
        raise ValueError("The supplied PDF does not contain any pages.")

    temp_dir = _reset_temp_dir(state)
    image_paths: List[str] = []

    for index, image in enumerate(images, start=1):
        page_path = temp_dir / f"page_{index:03d}.png"
        image.save(page_path)
        image_paths.append(str(page_path))

    logs.append(f"Rendered {len(image_paths)} page image(s) into {temp_dir}.")

    zip_path = temp_dir / "pages.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in image_paths:
            zip_file.write(path, arcname=Path(path).name)

    logs.append(f"Packaged page images into archive: {zip_path.name}.")

    state["image_paths"] = image_paths
    state["zip_path"] = str(zip_path)

    return image_paths, str(zip_path), logs


def _extract_zip_to_disk(zip_bytes: bytes, state: StateDict, zip_name: str) -> Tuple[List[str], str, List[str]]:
    logs: List[str] = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            temp_dir = _reset_temp_dir(state)
            image_paths: List[str] = []

            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                filename = Path(member).name
                suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
                if suffix not in {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"}:
                    continue
                target_path = temp_dir / filename
                with archive.open(member) as source, open(target_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                image_paths.append(str(target_path))
    except zipfile.BadZipFile as exc:
        raise ValueError("Failed to read images: invalid ZIP archive.") from exc

    if not image_paths:
        raise ValueError("The ZIP archive does not contain any supported images.")

    logs.append(f"Extracted {len(image_paths)} image file(s) from '{zip_name}'.")

    temp_dir = Path(state["temp_dir"])
    zip_path = temp_dir / "pages.zip"
    with open(zip_path, "wb") as handle:
        handle.write(zip_bytes)

    logs.append(f"Cached original archive as {zip_path.name}.")

    state["image_paths"] = image_paths
    state["zip_path"] = str(zip_path)

    return image_paths, str(zip_path), logs


def _append_logs(state: StateDict, new_logs: List[str]) -> str:
    log_list = state.get("log")
    if not isinstance(log_list, list):
        log_list = []
    log_list.extend(new_logs)
    state["log"] = log_list
    return "\n".join(log_list)


def render_pdf(
    pdf_file: object, images_zip: object, state: Optional[StateDict]
) -> Tuple[List[str], Optional[str], str, str, StateDict]:
    state = _ensure_state(state)

    if pdf_file is None and images_zip is None:
        log_text = _append_logs(
            state,
            ["No PDF or image archive uploaded. Please provide a file before rendering."],
        )
        state["image_paths"] = []
        state["zip_path"] = None
        return [], None, "", log_text, state

    try:
        if pdf_file is not None:
            source_kind = "pdf"
            file_bytes, source_name = _read_upload_bytes(pdf_file, fallback_name="uploaded.pdf")
        else:
            source_kind = "zip"
            file_bytes, source_name = _read_upload_bytes(images_zip, fallback_name="images.zip")
    except ValueError as exc:
        log_text = _append_logs(state, [f"Error: {exc}"])
        return [], None, "", log_text, state

    if state.get("source_name") != source_name or state.get("source_kind") != source_kind:
        # Starting over with a new document clears prior logs and artifacts.
        state = _clear_state(state)
        state["source_name"] = source_name
        state["source_kind"] = source_kind
        descriptor = "PDF" if source_kind == "pdf" else "ZIP archive"
        log_lines = [f"Loaded {descriptor} '{source_name}' ({len(file_bytes)} bytes)."]
    else:
        descriptor = "PDF" if state.get("source_kind") == "pdf" else "ZIP archive"
        log_lines = [f"Re-processing existing {descriptor} '{source_name}'."]

    try:
        if source_kind == "pdf":
            image_paths, zip_path, new_logs = _render_pdf_to_disk(file_bytes, state)
        else:
            image_paths, zip_path, new_logs = _extract_zip_to_disk(file_bytes, state, source_name)
    except ValueError as exc:
        log_lines.append(f"Error: {exc}")
        log_text = _append_logs(state, log_lines)
        return [], None, "", log_text, state

    log_lines.extend(new_logs)
    log_text = _append_logs(state, log_lines)

    return image_paths, zip_path, "", log_text, state


def run_ocr(
    pdf_file: object, images_zip: object, state: Optional[StateDict]
) -> Tuple[str, str, StateDict]:
    state = _ensure_state(state)

    log_lines: List[str] = []

    if pdf_file is not None or images_zip is not None:
        try:
            if pdf_file is not None:
                source_kind = "pdf"
                file_bytes, source_name = _read_upload_bytes(pdf_file, fallback_name="uploaded.pdf")
            else:
                source_kind = "zip"
                file_bytes, source_name = _read_upload_bytes(images_zip, fallback_name="images.zip")
        except ValueError as exc:
            log_text = _append_logs(state, [f"Error: {exc}"])
            return "", log_text, state

        if state.get("source_name") != source_name or state.get("source_kind") != source_kind:
            state = _clear_state(state)
            state["source_name"] = source_name
            state["source_kind"] = source_kind
            descriptor = "PDF" if source_kind == "pdf" else "ZIP archive"
            log_lines.append(
                f"Loaded {descriptor} '{source_name}' ({len(file_bytes)} bytes) for OCR."
            )
        else:
            descriptor = "PDF" if state.get("source_kind") == "pdf" else "ZIP archive"
            log_lines.append(f"Reprocessing existing {descriptor} '{source_name}'.")

        try:
            if source_kind == "pdf":
                image_paths, zip_path, render_logs = _render_pdf_to_disk(file_bytes, state)
            else:
                image_paths, zip_path, render_logs = _extract_zip_to_disk(
                    file_bytes, state, source_name
                )
        except ValueError as exc:
            log_lines.append(f"Error: {exc}")
            log_text = _append_logs(state, log_lines)
            return "", log_text, state
        state["zip_path"] = zip_path
        log_lines.extend(render_logs)
    else:
        image_paths_raw = state.get("image_paths")
        image_paths = (
            [str(path) for path in image_paths_raw]
            if isinstance(image_paths_raw, list)
            else []
        )
        if not image_paths:
            log_text = _append_logs(
                state,
                [
                    "No PDF or image archive uploaded. Please provide a file before running OCR.",
                ],
            )
            return "", log_text, state
        log_lines.append(
            f"Running OCR on cached {len(image_paths)} image file(s) from previous upload."
        )

    image_paths_raw = state.get("image_paths")
    image_paths = [str(path) for path in image_paths_raw] if isinstance(image_paths_raw, list) else []

    client = DeepSeekOCRClient()

    combined_sections: List[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        page_name = Path(image_path).name
        log_lines.append(f"Recognizing text on page {index}: {page_name}.")
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                text = client.recognize(rgb_image)
        except DeepSeekOCRError as exc:
            log_lines.append(f"Page {index} error: {exc}")
            combined_sections.append(f"### Page {index}\nError: {exc}")
            continue

        text = text.strip()
        if text:
            log_lines.append(f"Page {index} OCR complete ({len(text)} characters).")
            combined_sections.append(f"### Page {index}\n{text}")
        else:
            log_lines.append(f"Page {index} OCR complete (no text detected).")
            combined_sections.append(f"### Page {index}\n(No text returned)")

    if not combined_sections:
        log_lines.append("OCR finished but no text was extracted.")
        result = "No text extracted from the document."
    else:
        log_lines.append("OCR finished successfully.")
        result = "\n\n".join(combined_sections)

    log_text = _append_logs(state, log_lines)
    return result, log_text, state


DESCRIPTION = """
Upload a PDF file or a ZIP archive of page images, render each page into an
image archive, and optionally run the `deepseek-ai/DeepSeek-OCR` model locally
to retrieve the recognized text.

The application downloads the model weights automatically the first time it is
used (they are cached for subsequent runs). To use the Hugging Face Inference
API instead of local execution, set the `HF_API_TOKEN` environment variable (and
optionally `DEEPSEEK_OCR_MODE=remote`). You can also override the target
endpoint by setting `HF_INFERENCE_ENDPOINT`.
"""


with gr.Blocks(title="DeepSeek OCR PDF Demo") as iface:
    gr.Markdown(DESCRIPTION.strip())

    state = gr.State(_initial_state())

    pdf_input = gr.File(label="PDF document", file_types=[".pdf"], file_count="single")
    zip_input = gr.File(
        label="Images archive (zip)", file_types=[".zip"], file_count="single"
    )

    with gr.Row():
        render_button = gr.Button("Render to Images", variant="secondary")
        ocr_button = gr.Button("Run OCR", variant="primary")

    image_gallery = gr.Gallery(
        label="Rendered pages",
        show_label=True,
        columns=3,
        preview=True,
    )

    zip_output = gr.File(label="Download page images (zip)")
    text_output = gr.Markdown(label="Recognized text")
    log_output = gr.Textbox(label="Action log", lines=12)

    render_button.click(
        render_pdf,
        inputs=[pdf_input, zip_input, state],
        outputs=[image_gallery, zip_output, text_output, log_output, state],
    )

    ocr_button.click(
        run_ocr,
        inputs=[pdf_input, zip_input, state],
        outputs=[text_output, log_output, state],
    )


if __name__ == "__main__":
    iface.launch(share=True)
