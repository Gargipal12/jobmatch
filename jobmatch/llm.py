"""Talking to an LLM and getting back *validated* structured data.

LLMs sometimes return malformed or incomplete JSON. Never trust the raw text:
parse it into a pydantic model (which checks field names and types), and if that
fails, ask again with the validation error attached. After a few failures, raise
LLMError so the caller can fall back gracefully instead of crashing.

Network/API trouble is a separate problem: a busy server (503) or rate limit (429)
is temporary, so we wait and retry; a wrong model name (404) or bad key (401) is
permanent, so we fail at once.

Two clients share one tiny interface, generate(prompt, schema) -> raw text:
- GeminiClient: the real thing (needs GEMINI_API_KEY).
- ScriptedLLM: returns canned replies, for tests and offline demos.
"""
import os
import re
import sys
import time
from typing import Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class LLMError(Exception):
    """The LLM call failed or never produced valid output."""


class LLM(Protocol):
    def generate(self, prompt: str, schema: type[BaseModel]) -> str: ...


_FENCES = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _clean(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences even when told not to."""
    return _FENCES.sub("", (text or "").strip()).strip()


def ask(llm: LLM, prompt: str, schema: type[T], retries: int = 2) -> T:
    """Call the LLM and return a validated `schema` object, retrying on bad output."""
    last_error = ""
    for attempt in range(retries + 1):
        if attempt == 0:
            full_prompt = prompt
        else:
            full_prompt = (
                f"{prompt}\n\nYour previous reply was rejected: {last_error}\n"
                "Reply again with valid JSON only."
            )
        raw = llm.generate(full_prompt, schema)
        try:
            return schema.model_validate_json(_clean(raw))
        except ValidationError as e:
            last_error = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'reply'}: {err['msg']}"
                for err in e.errors()[:3]
            )
    raise LLMError(
        f"no valid {schema.__name__} after {retries + 1} attempts (last error: {last_error})"
    )


# ---- retrying temporary API errors ------------------------------------------

_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_WORDS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")


def _is_transient(error: Exception) -> bool:
    """True for errors that usually go away if we wait (busy server, rate limit)."""
    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code in _TRANSIENT_CODES
    return any(word in str(error) for word in _TRANSIENT_WORDS)


def call_with_retries(
    fn: Callable[[], R],
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[float, Exception], None]] = None,
) -> R:
    """Run fn(); on a temporary error wait 2s, 4s, 8s... and try again.
    Permanent errors (wrong model, bad key) are raised immediately."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1 or not _is_transient(e):
                raise
            delay = base_delay * 2**attempt
            if on_retry:
                on_retry(delay, e)
            sleep(delay)
    raise AssertionError("unreachable")


class GeminiClient:
    """Gemini via the google-genai SDK. Set GEMINI_API_KEY in your environment.

    The model name comes from GEMINI_MODEL (or the model= argument). Model names
    change over time, so check Google AI Studio for a current one if the default
    is rejected.
    """

    DEFAULT_MODEL = "gemini-3.8-flash"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Get a free key at aistudio.google.com, then in "
                'PowerShell run:  $env:GEMINI_API_KEY="your-key"'
            )
        from google import genai  # imported lazily so tests run without the SDK
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL

    def _config(self, schema: type[BaseModel]):
        t = self._types
        try:
            return t.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
                temperature=0,
            )
        except Exception:  # older SDK versions use response_schema instead
            return t.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            )

    @staticmethod
    def _announce(delay: float, error: Exception) -> None:
        print(f"  Gemini is busy, retrying in {delay:.0f}s...", file=sys.stderr)

    def generate(self, prompt: str, schema: type[BaseModel]) -> str:
        config = self._config(schema)
        try:
            resp = call_with_retries(
                lambda: self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                ),
                on_retry=self._announce,
            )
            return resp.text or ""
        except Exception as e:
            raise LLMError(f"Gemini call failed: {e}") from e


class ScriptedLLM:
    """Returns replies from a list, in order. An item can be a string, an
    Exception (raised), or a function(prompt, schema) -> string."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: type[BaseModel]) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(prompt, schema) if callable(item) else item