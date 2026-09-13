"""Shared OpenRouter client configuration for LLM calls."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .matcher_checkpoint import utc_now


load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"

# OpenRouter reserves credit against max_tokens, not against actual usage. Left
# unset it defaults to the model's full output ceiling (65,535 for gemini-2.5-flash),
# so a request is rejected with HTTP 402 whenever the remaining balance cannot cover
# that ceiling -- even though the matcher answers in a single token. Callers pass a
# realistic cap instead; see MATCHER_MAX_TOKENS / PROFILE_MAX_TOKENS below.
DEFAULT_MAX_TOKENS = 512

# Reasoning models spend their budget on hidden thinking tokens before emitting any
# content, so a cap sized for the visible answer truncates them: deepseek-v4-flash
# burns 33-80 reasoning tokens on a yes/no matcher prompt and returns nothing under
# max_tokens=16. That budget is also unstable across identical requests at
# temperature=0, so no fixed cap is safe while reasoning is on. Disabling it keeps
# every model on the single-token answer path the caps below assume, and keeps
# results comparable across model switches. Set ENABLE_LLM_REASONING=1 to study the
# reasoning variant deliberately -- raise MATCHER_MAX_TOKENS well above 128 if you do.
ENABLE_REASONING_ENV = "ENABLE_LLM_REASONING"

# Restricts which upstream providers OpenRouter may route to, as a comma-separated
# list of provider names (e.g. "DeepInfra"). OpenRouter spreads one model across
# dozens of independent hosts that differ in quantization and in which request
# parameters they honour, so leaving this unset means a single run is served by a
# shifting mix of them.
#
# Leave it unset for normal runs. Measured on Fodors-Zagat matcher prompts, 4 workers,
# extrapolated to the full 2,665-pair candidate set:
#
#   unset                       ~33 min   DeepInfra ~80% of calls anyway
#   pinned, allow_fallbacks     ~52 min   same ~80/20 provider mix as unset
#   pinned, no fallbacks       ~345 min   DeepInfra 100%, single calls stalling 300s
#
# Hard pinning is the only setting that actually fixes the provider as a variable, and
# it costs a 10x slowdown: with no fallback available, every rate-limited call waits out
# the retry instead of being served elsewhere. Pinning with fallbacks buys nothing, since
# the resulting provider mix matches what unpinned routing already produces. Truncation
# from providers that ignore reasoning.enabled=false is handled by MATCHER_MAX_TOKENS and
# the retry in create_chat_completion_text, not by pinning.
PROVIDER_ORDER_ENV = "OPENROUTER_PROVIDER_ORDER"

# The matcher replies with one label token. The cap is nonetheless sized for a full
# reasoning preamble: a minority of OpenRouter's providers ignore reasoning.enabled=false
# and emit thinking tokens anyway (measured at 1/250 calls, provider Phala, truncated
# at 8). Since billing follows actual usage rather than the cap, the only cost of the
# headroom is a larger credit reservation, while the cost of a truncation is a lost
# pair scored as a non-match.
MATCHER_MAX_TOKENS = 256
# Attribute-importance profiling returns a small JSON object whose length scales with
# schema width: 50 completion tokens on DBLP-ACM's 4 attributes, 133 on
# Amazon-Walmart's 13. 512 keeps a wide margin over the widest schema in use.
PROFILE_MAX_TOKENS = 512

# A truncation retry raises the cap instead of repeating the same request. A provider
# that ignores reasoning.enabled=false needs room the caller never budgeted for, and
# re-sending the identical cap just re-truncates -- which is how one call in 13,080
# exhausted its retries and voided a whole 25-minute DBLP-ACM run. Escalating covers
# that host while leaving the common case reserving MATCHER_MAX_TOKENS.
TRUNCATION_RETRY_FACTOR = 4
TRUNCATION_RETRY_CEILING = 4096

# Truncation gets its own attempt budget, separate from the general retry count, because
# escalating the cap did not by itself stop the failures: a run still lost one call in
# 13,080 even after reaching the ceiling. Each attempt also re-rolls the upstream
# provider, so the budget is really insurance against repeatedly drawing a host that
# ignores reasoning.enabled=false. The pipeline refuses to record a run containing any
# failed call, so a single unlucky pair discards the whole run -- and the wider the
# candidate set, the more certain that becomes: at the observed rate an 88,296-pair
# Amazon-Walmart run would expect several. Retries here are far cheaper than that.
TRUNCATION_MAX_ATTEMPTS = 6
EMPTY_RESPONSE_MAX_RETRIES = 5

_CLIENT = None


class TruncatedResponseError(RuntimeError):
    """Raised when max_tokens cut off the response before any content arrived."""


class CompletionError(RuntimeError):
    """A failed completion with the usage and responses observed before failure."""

    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage


def get_llm_model():
    """Return the configured OpenRouter model slug."""
    return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)


def reasoning_enabled():
    """Whether hidden reasoning tokens are allowed; off unless explicitly requested."""
    return os.getenv(ENABLE_REASONING_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def provider_order():
    """Return the configured upstream provider allow-list, or [] to let OpenRouter route."""
    raw = os.getenv(PROVIDER_ORDER_ENV, "")
    return [name.strip() for name in raw.split(",") if name.strip()]


def request_routing_options():
    """Build the OpenRouter-specific request fields shared by every call."""
    options = {} if reasoning_enabled() else {"reasoning": {"enabled": False}}
    providers = provider_order()
    if providers:
        # allow_fallbacks stays off so a pinned run cannot silently drift onto another
        # host mid-run; a provider outage should surface as an error, not as results
        # quietly produced somewhere else.
        options["provider"] = {"order": providers, "allow_fallbacks": False}
    return options


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
    if finish_reason == "length":
        raise TruncatedResponseError(
            "LLM response was cut off by max_tokens before producing any content. "
            "Raise the caller's max_tokens, or pin a provider that honours "
            f"reasoning.enabled=false via {PROVIDER_ORDER_ENV}."
        )
    raise ValueError(
        "LLM response had empty message content "
        f"(finish_reason={finish_reason}, refusal={refusal!r})"
    )


def usage_to_dict(response):
    """Token counts, plus which upstream actually served the call.

    OpenRouter routes one model slug across several providers, and the mix is not fixed
    over time: replaying 69,087 byte-identical Amazon-Walmart prompts a fortnight apart
    moved the Yes count from 1,589 to 3,250 with the slug, the prompt, the request
    parameters and this code all unchanged. Nothing in the response was being recorded
    that could name the cause, so it could only be inferred. `provider` is a
    non-standard field OpenRouter adds to the completion (e.g. "Sail Research"), and
    `model` is the slug it resolved; recording both turns that inference into an
    observation. Absent on an OpenAI-compatible endpoint that is not OpenRouter, hence
    the empty-string default rather than a hard read.

    The two string keys sit alongside the token counts because every aggregation site
    iterates over a fixed set of token keys rather than over this dict, so adding
    non-numeric entries here cannot land in a sum.
    """
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "provider": str(getattr(response, "provider", "") or ""),
        "served_model": str(getattr(response, "model", "") or ""),
    }


def summarize_attempts(attempts):
    last = attempts[-1] if attempts else {}
    return {
        **{key: sum(a.get(key, 0) for a in attempts)
           for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "provider": last.get("provider", ""),
        "served_model": last.get("served_model", ""),
        "attempts": attempts,
    }


def create_chat_completion_text(messages, temperature=0, max_retries=EMPTY_RESPONSE_MAX_RETRIES,
                                max_tokens=DEFAULT_MAX_TOKENS):
    """Call OpenRouter and return non-empty text plus token usage.

    max_tokens is always sent: omitting it makes OpenRouter reserve credit against
    the model's full output ceiling, which fails with HTTP 402 on a low balance
    regardless of how little the call actually needs. Reasoning is disabled unless
    ENABLE_LLM_REASONING is set, so the caps stay valid on reasoning models.
    """
    model = get_llm_model()
    extra_body = request_routing_options()
    attempt_max_tokens = int(max_tokens)
    empty_attempts_left = int(max_retries)
    truncation_attempts_left = TRUNCATION_MAX_ATTEMPTS - 1
    delay_step = 0
    last_error = None
    attempts = []
    while True:
        started_at = utc_now()
        try:
            response = get_llm_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=attempt_max_tokens,
                extra_body=extra_body,
            )
        except Exception as exc:
            # Transport retries inside the SDK do not expose per-attempt token usage.
            attempts.append({"started_at": started_at, "finished_at": utc_now(),
                             "max_tokens": attempt_max_tokens, "error": str(exc),
                             "usage_reported": False})
            raise CompletionError(str(exc), summarize_attempts(attempts)) from exc
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        attempts.append({
            **usage_to_dict(response),
            "started_at": started_at,
            "finished_at": utc_now(),
            "max_tokens": attempt_max_tokens,
            "finish_reason": getattr(choice, "finish_reason", None),
            "raw_response": _content_to_text(getattr(getattr(choice, "message", None), "content", None)),
            "usage_reported": getattr(response, "usage", None) is not None,
        })
        try:
            text = extract_chat_response_text(response)
            return text, summarize_attempts(attempts)
        except TruncatedResponseError as exc:
            # Each retry both raises the cap and re-rolls the upstream provider, which
            # are the two things that can fix this.
            last_error = exc
            attempts[-1]["error"] = str(exc)
            if truncation_attempts_left <= 0:
                break
            truncation_attempts_left -= 1
            attempt_max_tokens = min(
                attempt_max_tokens * TRUNCATION_RETRY_FACTOR, TRUNCATION_RETRY_CEILING
            )
        except ValueError as exc:
            # A blank body is a different failure; the cap is not the problem, so the
            # cap is left alone and the ordinary retry budget applies.
            last_error = exc
            attempts[-1]["error"] = str(exc)
            if empty_attempts_left <= 0:
                break
            empty_attempts_left -= 1
        delay_step += 1
        time.sleep(0.5 * delay_step)

    raise CompletionError(f"OpenRouter returned no text after retries: {last_error}",
                          summarize_attempts(attempts))
