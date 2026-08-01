"""Shared OpenRouter client configuration for LLM calls."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"

_CLIENT = None


def get_llm_model():
    """Return the configured OpenRouter model slug."""
    return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)


def _openrouter_headers():
    headers = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def get_llm_client():
    """Create the OpenAI-compatible client pointed at OpenRouter."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Add it to .env or export it in your shell."
        )

    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_openrouter_headers(),
    )
    return _CLIENT


def _content_to_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "".join(parts).strip()
    return ""


def extract_chat_response_text(response):
    """Return message text from an OpenAI-compatible chat response."""
    if not getattr(response, "choices", None):
        raise ValueError("LLM response contained no choices")

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    text = _content_to_text(content)
    if text:
        return text

    finish_reason = getattr(choice, "finish_reason", None)
    refusal = getattr(message, "refusal", None)
    raise ValueError(
        "LLM response had empty message content "
        f"(finish_reason={finish_reason}, refusal={refusal!r})"
    )


def usage_to_dict(response):
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def create_chat_completion_text(messages, temperature=0, max_retries=2):
    """Call OpenRouter and return non-empty text plus token usage."""
    model = get_llm_model()
    last_error = None
    for attempt in range(int(max_retries) + 1):
        response = get_llm_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        try:
            return extract_chat_response_text(response), usage_to_dict(response)
        except ValueError as exc:
            last_error = exc
            if attempt >= int(max_retries):
                break
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"OpenRouter returned no text after retries: {last_error}")
