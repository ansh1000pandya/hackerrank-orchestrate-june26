"""
VLM client layer: a provider-agnostic interface plus concrete backends.

Design goal: swapping models/providers (Gemini -> Claude -> GPT-4o, or
comparing two strategies for the evaluation report) should mean changing
one constructor call, not rewriting the pipeline.

Includes:
- on-disk response caching keyed by (provider, model, prompt hash, image
  hashes) so repeated dev runs don't re-spend API calls
- retry with exponential backoff on 429 / transient errors, since free
  tiers have low RPM
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class VLMResponse:
    text: str
    usage: VLMUsage
    cached: bool = False
    latency_s: float = 0.0


class VLMClient(ABC):
    """Common interface every provider backend implements."""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def _call(self, system_prompt: str, user_prompt: str, image_paths: list[str]) -> VLMResponse:
        ...

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        cache: "ResponseCache | None" = None,
        max_retries: int = 5,
    ) -> VLMResponse:
        cache_key = None
        if cache is not None:
            cache_key = cache.make_key(self.name, self.model, system_prompt, user_prompt, image_paths)
            hit = cache.get(cache_key)
            if hit is not None:
                return VLMResponse(text=hit["text"], usage=VLMUsage(**hit["usage"]), cached=True)

        delay = 1.0
        last_err = None
        for attempt in range(max_retries):
            try:
                start = time.time()
                resp = self._call(system_prompt, user_prompt, image_paths)
                resp.latency_s = time.time() - start
                if cache is not None and cache_key is not None:
                    cache.set(
                        cache_key,
                        {
                            "text": resp.text,
                            "usage": {
                                "input_tokens": resp.usage.input_tokens,
                                "output_tokens": resp.usage.output_tokens,
                            },
                        },
                    )
                return resp
            except RateLimitError as e:
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            except TransientAPIError as e:
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 15.0)
        raise RuntimeError(f"VLM call failed after {max_retries} retries: {last_err}")


class RateLimitError(Exception):
    pass


class TransientAPIError(Exception):
    pass


MAX_IMAGE_DIMENSION = 1568  # generous for damage assessment; well within all providers' practical limits


def _encode_image(path: str) -> tuple[str, str]:
    """
    Encode an image file as base64 + a MIME type that matches its ACTUAL
    content, not its filename extension, downscaling oversized images.

    This dataset has two real data-quality issues, confirmed by scanning
    every file in images/ with PIL:
    1. ~40% of files named `*.jpg` are actually WEBP, PNG, or AVIF bytes.
       Trusting the extension sends a wrong Content-Type to the provider.
       Gemini's SDK tolerated this silently; Groq's API correctly rejects
       it with "invalid image data" (400).
    2. At least one source photo is ~7900x5900px (~47 megapixels, 5.4MB
       on disk) -- re-encoding that at full resolution produces an
       ~11.7MB base64 payload that exceeds Groq's per-request size limit
       (413 Request Entity Too Large).

    Fix: detect the real format with PIL, downscale anything larger than
    MAX_IMAGE_DIMENSION on its longest side (damage/issue assessment does
    not need full sensor resolution), and re-encode every image as a
    genuine, reasonably-sized JPEG before sending it to any provider.
    """
    from PIL import Image
    import io

    with Image.open(path) as im:
        im = im.convert("RGB")  # strips alpha/palette modes that JPEG can't hold, normalizes WEBP/PNG/AVIF alike
        if max(im.size) > MAX_IMAGE_DIMENSION:
            im.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        raw = buf.getvalue()

    data = base64.b64encode(raw).decode("utf-8")
    return data, "image/jpeg"


class GeminiClient(VLMClient):
    """
    Backend for Google's Gemini API (free tier, vision-capable).

    Reads the API key from the GEMINI_API_KEY (or GOOGLE_API_KEY) env var.
    Never hardcode keys -- per AGENTS.md / hackathon rules and basic
    security hygiene.
    """

    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key_env: str = "GEMINI_API_KEY"):
        self.model = model
        api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"No Gemini API key found in env var {api_key_env} or GOOGLE_API_KEY. "
                "Set it before running, e.g.: export GEMINI_API_KEY=your_key_here"
            )
        self._api_key = api_key
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai --break-system-packages"
            ) from e
        genai.configure(api_key=self._api_key)
        self._genai = genai
        self._client = genai.GenerativeModel(model_name=self.model)

    def _call(self, system_prompt: str, user_prompt: str, image_paths: list[str]) -> VLMResponse:
        import google.api_core.exceptions as gexc

        parts: list = [system_prompt + "\n\n" + user_prompt]
        for p in image_paths:
            data, mime = _encode_image(p)
            parts.append({"mime_type": mime, "data": base64.b64decode(data)})

        try:
            result = self._client.generate_content(
                parts,
                generation_config={
                    "temperature": 0.0,  # deterministic where possible, per AGENTS.md
                    "response_mime_type": "application/json",
                },
            )
        except gexc.ResourceExhausted as e:
            raise RateLimitError(str(e)) from e
        except (gexc.ServiceUnavailable, gexc.DeadlineExceeded, gexc.InternalServerError) as e:
            raise TransientAPIError(str(e)) from e

        text = result.text
        usage = VLMUsage()
        meta = getattr(result, "usage_metadata", None)
        if meta is not None:
            usage = VLMUsage(
                input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            )
        return VLMResponse(text=text, usage=usage)


class GroqClient(VLMClient):
    """
    Backend for Groq's API (free tier, vision-capable via Llama 4 Scout/Maverick).

    Groq's free tier has a much higher requests-per-day ceiling than
    Gemini's current Flash free tier, at the cost of using an open-source
    vision model (Llama 4 Scout/Maverick) instead of Gemini. Supports up
    to 5 images per request and native JSON mode, both of which this
    dataset's claims (max 3 images each) fit comfortably within.

    Reads the API key from the GROQ_API_KEY env var. Never hardcode keys.
    """

    name = "groq"

    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                 api_key_env: str = "GROQ_API_KEY"):
        self.model = model
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No Groq API key found in env var {api_key_env}. "
                "Set it before running, e.g.: $env:GROQ_API_KEY=\"your_key_here\" (PowerShell) "
                "or export GROQ_API_KEY=your_key_here (bash)."
            )
        self._api_key = api_key
        try:
            from groq import Groq
        except ImportError as e:
            raise RuntimeError(
                "groq package not installed. Run: pip install groq --break-system-packages"
            ) from e
        self._client = Groq(api_key=api_key)

    def _call(self, system_prompt: str, user_prompt: str, image_paths: list[str]) -> VLMResponse:
        import groq as groq_module

        content: list = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            data, mime = _encode_image(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })

        try:
            result = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,  # deterministic where possible, per AGENTS.md
                response_format={"type": "json_object"},
            )
        except groq_module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except (groq_module.APIConnectionError, groq_module.InternalServerError,
                groq_module.APITimeoutError) as e:
            raise TransientAPIError(str(e)) from e

        text = result.choices[0].message.content
        usage = VLMUsage()
        if result.usage is not None:
            usage = VLMUsage(
                input_tokens=result.usage.prompt_tokens or 0,
                output_tokens=result.usage.completion_tokens or 0,
            )
        return VLMResponse(text=text, usage=usage)


class MockClient(VLMClient):
    """
    Deterministic offline stand-in with no network calls.

    Used for (a) unit-testing the pipeline plumbing without burning API
    quota, and (b) as a documented fallback strategy in the evaluation
    report comparing "no VLM / rule-only" vs "VLM-backed" approaches.
    It intentionally returns conservative, low-confidence answers rather
    than guessing, since it cannot actually see the images.
    """

    name = "mock"
    model = "mock-rule-only"

    def _call(self, system_prompt: str, user_prompt: str, image_paths: list[str]) -> VLMResponse:
        payload = {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "Mock client cannot inspect images; insufficient by default.",
            "risk_flags": ["manual_review_required"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "No vision model available in mock mode.",
            "supporting_image_ids": ["none"],
            "valid_image": bool(image_paths),
            "severity": "unknown",
        }
        return VLMResponse(text=json.dumps(payload), usage=VLMUsage(input_tokens=0, output_tokens=0))


class ResponseCache:
    """Simple on-disk JSON cache, one file per cache key."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
    ) -> str:
        h = hashlib.sha256()
        h.update(provider.encode())
        h.update(model.encode())
        h.update(system_prompt.encode())
        h.update(user_prompt.encode())
        for p in image_paths:
            # Hash file bytes, not just the path, so edited/replaced images invalidate cache.
            with open(p, "rb") as f:
                h.update(f.read())
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: dict) -> None:
        self._path(key).write_text(json.dumps(value), encoding="utf-8")


def build_client(provider: str, model: str | None = None) -> VLMClient:
    """Factory so callers (main.py, evaluation/main.py) pick a backend by name/config."""
    if provider == "gemini":
        return GeminiClient(model=model or "gemini-2.5-flash")
    if provider == "groq":
        return GroqClient(model=model or "meta-llama/llama-4-scout-17b-16e-instruct")
    if provider == "mock":
        return MockClient()
    raise ValueError(f"Unknown provider: {provider}")
