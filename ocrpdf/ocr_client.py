"""Utilities for running the DeepSeek OCR model locally or via an API."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests
from PIL import Image

__all__ = ["DeepSeekOCRClient", "DeepSeekOCRError"]


class DeepSeekOCRError(RuntimeError):
    """Raised when the DeepSeek OCR API returns an error response."""


@dataclass
class DeepSeekOCRClient:
    """Client for running the ``deepseek-ai/DeepSeek-OCR`` model.

    The client can operate in two modes:

    ``"local"``
        Uses :mod:`transformers` to download the model weights from the
        Hugging Face Hub (respecting the HF cache) and runs inference locally.

    ``"remote"``
        Calls the Hugging Face Inference API, matching the behaviour of the
        original implementation.

    The default mode is ``"auto"`` which prefers local inference unless an
    explicit API token or custom endpoint is configured. The mode can also be
    overridden via the ``DEEPSEEK_OCR_MODE`` environment variable.
    """

    api_url: str = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-OCR"
    api_token: Optional[str] = None
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 3.0
    mode: str = "auto"
    model_id: str = "deepseek-ai/DeepSeek-OCR"
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    generation_kwargs: Optional[Dict[str, Any]] = None

    _backend: str = field(init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _generation_kwargs: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_url == "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-OCR":
            self.api_url = os.environ.get("HF_INFERENCE_ENDPOINT", self.api_url)
        if self.api_token is None:
            self.api_token = os.environ.get("HF_API_TOKEN")

        mode_override = os.environ.get("DEEPSEEK_OCR_MODE")
        if mode_override:
            self.mode = mode_override

        mode = self.mode.lower()
        if mode not in {"auto", "local", "remote"}:
            raise ValueError("mode must be 'auto', 'local', or 'remote'")

        if mode == "auto":
            # Prefer local inference unless the user provided remote
            # credentials or a custom endpoint.
            if self.api_token or self.api_url != "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-OCR":
                backend = "remote"
            else:
                backend = "local"
        else:
            backend = mode

        self._backend = backend

        if self._backend == "local":
            try:
                self._initialise_local_backend()
            except DeepSeekOCRError:
                if mode == "auto" and (
                    self.api_token
                    or self.api_url
                    != "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-OCR"
                ):
                    # Fall back to the remote backend if local execution is not
                    # possible but remote credentials were supplied explicitly.
                    self._backend = "remote"
                else:
                    raise

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "image/png"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    # ------------------------------------------------------------------
    # Local backend helpers
    # ------------------------------------------------------------------
    def _initialise_local_backend(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise DeepSeekOCRError(
                "Local OCR mode requires the 'torch' and 'transformers' packages. "
                "Install them with `pip install torch transformers`."
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        dtype = None
        if self.torch_dtype:
            try:
                dtype = getattr(torch, self.torch_dtype)
            except AttributeError as exc:  # pragma: no cover - config error
                raise DeepSeekOCRError(f"Unsupported torch dtype: {self.torch_dtype}") from exc
        elif self._device.type == "cuda":  # pragma: no cover - depends on hardware
            dtype = torch.float16

        if self._device.type == "cpu":
            dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        model = None
        errors = []

        for candidate, model_type_name in (
            ("vision2seq", "AutoModelForVision2Seq"),
            ("causal-lm", "AutoModelForCausalLM"),
        ):
            try:
                if candidate == "vision2seq":
                    model = AutoModelForVision2Seq.from_pretrained(
                        self.model_id, **model_kwargs
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_id, **model_kwargs
                    )
                break
            except Exception as exc:  # pragma: no cover - depends on model layout
                errors.append((model_type_name, exc))

        if model is None:
            message = "Unable to load local DeepSeek OCR model."
            if errors:
                details = ", ".join(f"{name}: {err}" for name, err in errors)
                message = f"{message} Tried loaders failed with: {details}"
            raise DeepSeekOCRError(message)

        self._model = model.to(self._device)
        self._model.eval()

        # Default generation parameters tuned for OCR-style outputs.
        defaults = {"max_new_tokens": 512, "num_beams": 1}
        if self.generation_kwargs:
            defaults.update(self.generation_kwargs)
        self._generation_kwargs = defaults

        self._torch = torch

    def _recognize_local(self, image: Image.Image) -> str:
        if not self._processor or not self._model:
            raise DeepSeekOCRError("Local OCR backend is not initialised.")

        torch = self._torch

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {
            key: (value.to(self._device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **self._generation_kwargs)

        texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        if not texts:
            return ""
        return texts[0].strip()

    # ------------------------------------------------------------------
    # Remote backend helpers
    # ------------------------------------------------------------------
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

    def _recognize_remote(self, image: Image.Image) -> str:
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

    def recognize(self, image: Image.Image) -> str:
        """Run OCR on a Pillow image and return the generated text."""

        if self._backend == "local":
            return self._recognize_local(image)

        return self._recognize_remote(image)
