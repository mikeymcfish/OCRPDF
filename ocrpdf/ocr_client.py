"""Thin wrapper around the Hugging Face Inference API for DeepSeek OCR."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from PIL import Image

__all__ = ["DeepSeekOCRClient", "DeepSeekOCRError"]


class DeepSeekOCRError(RuntimeError):
    """Raised when the DeepSeek OCR API returns an error response."""


@dataclass
class DeepSeekOCRClient:
    """Client for the ``deepseek-ai/DeepSeek-OCR`` Hugging Face model."""

    api_url: str = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-OCR"
    api_token: Optional[str] = None
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 3.0

    def __post_init__(self) -> None:
        if self.api_url == "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-OCR":
            self.api_url = os.environ.get("HF_INFERENCE_ENDPOINT", self.api_url)
        if self.api_token is None:
            self.api_token = os.environ.get("HF_API_TOKEN")

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "image/png"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _parse_response(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text

        if isinstance(payload, list):
            texts = [item.get("generated_text", "") for item in payload if isinstance(item, dict)]
            combined = "\n".join(text for text in texts if text)
            return combined.strip()

        if isinstance(payload, dict):
            if "generated_text" in payload:
                return str(payload.get("generated_text", "")).strip()
            if "error" in payload:
                raise DeepSeekOCRError(str(payload["error"]))

        return str(payload).strip()

    def recognize(self, image: Image.Image) -> str:
        """Run OCR on a Pillow image and return the generated text."""

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()

        headers = self._headers()

        attempt = 0
        while True:
            attempt += 1
            response = requests.post(self.api_url, headers=headers, data=data, timeout=self.timeout)

            if response.status_code == 200:
                return self._parse_response(response)

            if response.status_code == 503 and attempt <= self.retry_attempts:
                # Model is loading. Hugging Face returns an estimated wait time in the payload.
                try:
                    payload = response.json()
                    wait_time = float(payload.get("estimated_time", self.retry_delay))
                except Exception:  # pragma: no cover - best effort logging only
                    wait_time = self.retry_delay
                time.sleep(wait_time)
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:  # pragma: no cover - simple wrapper
                raise DeepSeekOCRError(str(exc)) from exc

            raise DeepSeekOCRError(f"Unexpected response: {response.text}")
