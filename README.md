# DeepSeek OCR PDF Demo

This repository contains a minimal [Gradio](https://www.gradio.app/) application
that converts every page of an uploaded PDF into an image, sends the images to
[`deepseek-ai/DeepSeek-OCR`](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
via the Hugging Face Inference API, and combines the recognized text into a
single Markdown document.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The application depends on `gradio`, `pypdfium2`, `pillow`, and `requests`.
   Rendering PDFs into images requires native libraries that `pypdfium2`
   bundles for common platforms.

2. **Configure authentication**

   The app uses the Hugging Face Inference API. Set an API token in the
   `HF_API_TOKEN` environment variable before launching the app:

   ```bash
   export HF_API_TOKEN="hf_your_token_here"
   ```

   If you are using a custom inference endpoint, override the default URL by
   setting `HF_INFERENCE_ENDPOINT`.

3. **Run the Gradio interface**

   ```bash
   python app.py
   ```

   A local Gradio interface will launch. Upload a PDF file to begin the OCR
   process. The recognized text is presented one page at a time in the output
   panel.

## Notes

- The Hugging Face endpoint may return a `503` status while the model is
  loading. The client retries automatically using the estimated wait time
  returned by the API.
- Large PDFs can take a while to process because each page is rendered at
  approximately 300 DPI before being sent to the OCR model.
