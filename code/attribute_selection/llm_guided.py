import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from collections import defaultdict

from ..llm_client import PROFILE_MAX_TOKENS, create_chat_completion_text, get_llm_model

ATTRIBUTE_PROMPT = (
    "You are an expert in data integration and entity resolution.\n"
    "Below are two records and whether they refer to the same real-world entity.\n\n"
    "Record A: {record_a}\n"
    "Record B: {record_b}\n"
    "Match: {label}\n\n"
    "Available attributes: {attributes}\n\n"
    "Question: Which fields were most influential in deciding whether these two records match?\n"
    "Select the smallest sufficient subset of attributes. Prefer fields that provide decisive "
    "match or non-match evidence, and do not include weakly useful or redundant fields.\n"
    "Return only a JSON list of attribute names. Do not include scores, explanations, or extra keys.\n"
    "Only include attributes from the available attributes list.\n"
    "Example: [\"title\", \"authors\", \"year\"]"
)

ADAPTIVE_PROFILE_PROMPT = (
    "You are an expert in entity resolution.\n\n"
    "Your task is to analyze a pair of records and identify which attributes are useful "
    "for deciding whether the two records refer to the same real-world entity.\n\n"
    "Important:\n"
    "- Score each attribute independently based on how useful it is for the match/non-match decision.\n"
    "- Use only the allowed integer scores: 0, 1, 2, 3.\n"
    "- Do not score an attribute highly just because the values are identical if both values are empty, "
    "missing, generic, or uninformative.\n"
    "- An attribute can be important because it supports a match or because it provides strong evidence "
    "for a non-match.\n"
    "- Return JSON only.\n\n"
    "Scoring guide:\n"
    "0 = not useful, missing, generic, or irrelevant\n"
    "1 = weak evidence\n"
    "2 = useful supporting evidence\n"
    "3 = highly important or decisive evidence\n\n"
    "Attributes:\n"
    "{attributes}\n\n"
    "Record A:\n"
    "{record_a_json}\n\n"
    "Record B:\n"
    "{record_b_json}\n\n"
    "Return JSON in this exact format:\n"
    "{\n"
    "  \"attribute_importance\": {\n"
    "    \"attribute_name_1\": 0,\n"
    "    \"attribute_name_2\": 0\n"
    "  }\n"
    "}"
)


# Same framing and same guardrails as ADAPTIVE_PROFILE_PROMPT above -- only the answer
# format differs, so a run-to-run comparison isolates the response format rather than
# confounding it with a reworded task. The ordinal variant asks for a calibrated score
# per attribute, which the logs show the LLM does not supply: across every recorded run
# DBLP-ACM used score 0 on 1% of judgements (a 4-point scale collapsed to 3) and
# Fodors-Zagat used score 3 on 55%. This variant asks instead for the decision the
# pipeline actually needs -- a subset -- and lets the model weigh attributes against
# each other in one judgement, which is where redundancy between attributes (an address
# adding nothing once a phone number agrees) becomes visible.
DECISIVE_PROFILE_PROMPT = (
    "You are an expert in entity resolution.\n\n"
    "Your task is to analyze a pair of records and identify which attributes are decisive "
    "for deciding whether the two records refer to the same real-world entity.\n\n"
    "Important:\n"
    "- Return the smallest subset of attributes that is sufficient to make the decision.\n"
    "- Weigh the attributes against each other: omit an attribute that adds nothing once "
    "the attributes you already selected are known.\n"
    "- Do not select an attribute just because the values are identical if both values are "
    "empty, missing, generic, or uninformative.\n"
    "- An attribute can be decisive because it supports a match or because it provides strong "
    "evidence for a non-match.\n"
    "- Select at least one attribute, and only names from the attribute list below.\n"
    "- Return JSON only.\n\n"
    "Attributes:\n"
    "{attributes}\n\n"
    "Record A:\n"
    "{record_a_json}\n\n"
    "Record B:\n"
    "{record_b_json}\n\n"
    "Return JSON in this exact format:\n"
    "{\n"
    "  \"decisive_attributes\": [\"attribute_name_1\", \"attribute_name_2\"]\n"
    "}"
)

# How profiling asks the LLM for per-attribute importance. "ordinal" is the original
# 0-3 score; "decisive" asks for the subset directly.
PROFILE_SCORING_MODES = ("ordinal", "decisive")
DEFAULT_PROFILE_SCORING = "ordinal"


def _clean_record_for_prompt(record):
    return {
        k: ('' if pd.isna(v) else str(v))
        for k, v in record.items()
    }


def _strip_json_fences(content: str) -> str:
    return content.replace("```json", "").replace("```", "").strip()


def _load_json_with_common_repairs(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    if content.startswith("{{") and content.endswith("}}"):
        try:
            return json.loads(content[1:-1].strip())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON response", content, 0)


def parse_important_attribute_list(content: str, attributes: list) -> list:
    """Parse a Step-1 LLM response into an ordered list of valid attributes."""
    content = _strip_json_fences(content)
    valid_attributes = set(attributes)
    try:
        payload = _load_json_with_common_repairs(content)
    except json.JSONDecodeError:
        print(f"  Warning: could not parse LLM attribute list response: {content}")
        return []

    if not isinstance(payload, list):
        print(f"  Warning: LLM attribute response was not a list: {content}")
        return []

    selected = []
    seen = set()
    for item in payload:
        attr = str(item).strip()
        if attr in valid_attributes and attr not in seen:
            selected.append(attr)
            seen.add(attr)
    return selected


def parse_adaptive_attribute_importance(content: str, attributes: list) -> dict:
    """Parse adaptive profiling JSON into integer scores keyed by attribute."""
    content = _strip_json_fences(content)
    try:
        payload = _load_json_with_common_repairs(content)
    except json.JSONDecodeError:
        print(f"  Warning: could not parse adaptive LLM response: {content}")
        return {attr: 0 for attr in attributes}

    raw_scores = payload.get("attribute_importance", {})
    if not isinstance(raw_scores, dict):
        print(f"  Warning: adaptive response missing attribute_importance: {content}")
        return {attr: 0 for attr in attributes}

    scores = {}
    for attr in attributes:
        try:
            score = int(round(float(raw_scores.get(attr, 0))))
        except (TypeError, ValueError):
            score = 0
        scores[attr] = max(0, min(3, score))
    return scores


def parse_decisive_attribute_set(content: str, attributes: list) -> dict:
    """Parse a decisive-subset response into 1.0/0.0 scores keyed by attribute.

    Returned in the same shape as parse_adaptive_attribute_importance so both scoring
    modes feed the selector identically; only the value range differs. A response that
    names nothing valid yields all zeros, which normalizes to a uniform vector -- the
    same neutral fallback an unparseable ordinal response produces.
    """
    content = _strip_json_fences(content)
    try:
        payload = _load_json_with_common_repairs(content)
    except json.JSONDecodeError:
        print(f"  Warning: could not parse decisive LLM response: {content}")
        return {attr: 0.0 for attr in attributes}

    if isinstance(payload, dict):
        chosen = payload.get("decisive_attributes", [])
    else:
        # A bare list is a plausible deviation from the requested envelope.
        chosen = payload

    if not isinstance(chosen, list):
        print(f"  Warning: decisive response was not a list: {content}")
        return {attr: 0.0 for attr in attributes}

    valid = set(attributes)
    selected = {str(item).strip() for item in chosen}
    unknown = selected - valid
    if unknown:
        print(f"  Warning: decisive response named unknown attributes {sorted(unknown)}")
    return {attr: (1.0 if attr in selected else 0.0) for attr in attributes}


# ── 0. Profiling response cache ──
#
# Step-1.5 trains its whole selector on one small sample -- 20 pairs on Fodors-Zagat --
# so the profiling responses have far more leverage over the run than any single matcher
# answer does. They are also not stable: re-running the identical prompts at
# temperature=0 changed 8 of 20 pairs' scores and 13% of individual cells, because
# OpenRouter serves each call from a different upstream host. That makes an A/B over
# selection policy unreadable, since the two runs differ in the policy *and* in what the
# selector was trained on. Caching by prompt holds the profiling fixed so a policy
# comparison varies only the policy.
#
# It does not suppress the variation being studied elsewhere: a different seed samples
# different profile pairs, and a different scoring mode builds a different prompt, so
# both still miss the cache and issue fresh calls.

PROFILE_CACHE_DIR = Path(os.getenv("PROFILE_CACHE_DIR", "cache/default/profiling"))
USE_PROFILE_CACHE = os.getenv("PROFILE_DISABLE_CACHE", "").lower() not in {"1", "true", "yes"}


def set_profile_cache_dir(cache_dir):
    """Set the directory holding cached Step-1.5 profiling responses."""
    global PROFILE_CACHE_DIR
    PROFILE_CACHE_DIR = Path(cache_dir)
    PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def set_profile_cache_enabled(enabled):
    """Enable or disable profiling cache reads and writes."""
    global USE_PROFILE_CACHE
    USE_PROFILE_CACHE = bool(enabled)


def profile_cache_key(prompt, model, scoring):
    """Hash everything that determines the response: model, scoring mode, exact prompt."""
    payload = json.dumps({"model": model, "scoring": scoring, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_profile_cache(key):
    if not USE_PROFILE_CACHE:
        return None
    cache_file = PROFILE_CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        # A half-written cache entry must not take down a run; re-query instead.
        return None


def _save_profile_cache(key, data):
    if not USE_PROFILE_CACHE:
        return
    PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_CACHE_DIR / f"{key}.json", "w") as handle:
        json.dump(data, handle, indent=2)


def profiling_completion(prompt, scoring):
    """Return (content, usage) for a profiling prompt, reusing a cached response if present.

    usage carries cache_hit so callers can report how much of a run was replayed. Token
    counts on a hit are the ones the original call reported, matching how the matcher
    cache accounts for reuse.
    """
    model = get_llm_model()
    key = profile_cache_key(prompt, model, scoring)

    cached = _load_profile_cache(key)
    if cached is not None:
        usage = dict(cached.get("usage", {}))
        usage["cache_hit"] = True
        return cached.get("content", ""), usage

    content, usage = create_chat_completion_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=PROFILE_MAX_TOKENS,
    )
    _save_profile_cache(key, {
        "model": model,
        "scoring": scoring,
        "prompt": prompt,
        "content": content,
        "usage": usage,
    })
    usage = dict(usage)
    usage["cache_hit"] = False
    return content, usage


# ── 1. Query LLM for field importance per pair ──

def query_llm_field_importance(idx_a, idx_b, label, df_A, df_B):
    """
    Ask LLM which fields were most influential for a single pair.
    Returns an ordered list of important attribute names.

    Parameters
    ----------
    idx_a, idx_b : positional indices into df_A, df_B
    label        : 1 = match, 0 = non-match
    """
    attributes, _ = query_llm_field_importance_with_usage(idx_a, idx_b, label, df_A, df_B)
    return attributes


def query_llm_field_importance_with_usage(idx_a, idx_b, label, df_A, df_B):
    """
    Ask LLM which fields were most influential for a single pair.
    Returns (ordered attribute list, usage_dict).
    """
    recA = {k: ('' if pd.isna(v) else str(v))
            for k, v in df_A.iloc[idx_a].to_dict().items()}
    recB = {k: ('' if pd.isna(v) else str(v))
            for k, v in df_B.iloc[idx_b].to_dict().items()}
    attributes = [c for c in df_A.columns if c in df_B.columns]

    prompt = (ATTRIBUTE_PROMPT
              .replace("{record_a}", str(recA))
              .replace("{record_b}", str(recB))
              .replace("{attributes}", json.dumps(attributes, ensure_ascii=False))
              .replace("{label}", "Yes" if label == 1 else "No"))

    content, usage_dict = create_chat_completion_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=PROFILE_MAX_TOKENS,
    )
    return parse_important_attribute_list(content, attributes), usage_dict


def query_llm_adaptive_attribute_importance(idx_a, idx_b, df_A, df_B):
    """
    Ask LLM for independent integer 0-3 attribute usefulness scores.
    Used only by adaptive post-blocking profiling.
    """
    scores, _ = query_llm_adaptive_attribute_importance_with_usage(idx_a, idx_b, df_A, df_B)
    return scores


def query_llm_adaptive_attribute_importance_with_usage(idx_a, idx_b, df_A, df_B):
    """
    Ask LLM for independent integer 0-3 attribute usefulness scores.
    Returns (scores, usage_dict) for adaptive post-blocking profiling.
    """
    attributes = [c for c in df_A.columns if c in df_B.columns]
    recA = _clean_record_for_prompt(df_A.iloc[idx_a].to_dict())
    recB = _clean_record_for_prompt(df_B.iloc[idx_b].to_dict())

    prompt = (ADAPTIVE_PROFILE_PROMPT
              .replace("{attributes}", json.dumps(attributes, ensure_ascii=False))
              .replace("{record_a_json}", json.dumps(recA, ensure_ascii=False))
              .replace("{record_b_json}", json.dumps(recB, ensure_ascii=False)))

    content, usage_dict = profiling_completion(prompt, "ordinal")
    return parse_adaptive_attribute_importance(content, attributes), usage_dict


def query_llm_decisive_attributes_with_usage(idx_a, idx_b, df_A, df_B):
    """
    Ask LLM for the decisive attribute subset for one pair.
    Returns (scores, usage_dict) with 1.0 for selected attributes and 0.0 otherwise.
    """
    attributes = [c for c in df_A.columns if c in df_B.columns]
    recA = _clean_record_for_prompt(df_A.iloc[idx_a].to_dict())
    recB = _clean_record_for_prompt(df_B.iloc[idx_b].to_dict())

    prompt = (DECISIVE_PROFILE_PROMPT
              .replace("{attributes}", json.dumps(attributes, ensure_ascii=False))
              .replace("{record_a_json}", json.dumps(recA, ensure_ascii=False))
              .replace("{record_b_json}", json.dumps(recB, ensure_ascii=False)))

    content, usage_dict = profiling_completion(prompt, "decisive")
    return parse_decisive_attribute_set(content, attributes), usage_dict


def query_llm_profile_importance_with_usage(idx_a, idx_b, df_A, df_B, scoring=DEFAULT_PROFILE_SCORING):
    """Dispatch adaptive profiling to the configured scoring mode."""
    if scoring == "ordinal":
        return query_llm_adaptive_attribute_importance_with_usage(idx_a, idx_b, df_A, df_B)
    if scoring == "decisive":
        return query_llm_decisive_attributes_with_usage(idx_a, idx_b, df_A, df_B)
    raise ValueError(
        f"Unknown profile scoring mode {scoring!r}; expected one of {PROFILE_SCORING_MODES}"
    )


# ── 2. Aggregate importance scores across all sampled pairs ──

def aggregate_importance(responses: list, cols: list) -> list:
    """
    Aggregate important-attribute lists across all LLM responses.
    Uses selection frequency per field — fields not listed get score 0.

    Parameters
    ----------
    responses : list of attribute-name lists from LLM
    cols      : all candidate attribute columns
    """
    counts = defaultdict(int)
    for response in responses:
        seen = set()
        for attr in response:
            if attr in cols and attr not in seen:
                counts[attr] += 1
                seen.add(attr)

    denominator = len(responses) if responses else 1
    aggregated = {
        col: counts[col] / denominator
        for col in cols
    }

    # rank by selection frequency descending
    ranked = sorted(aggregated.items(), key=lambda x: -x[1])
    return ranked


# ── 3. Select attributes above threshold ──

def select_top_attributes(ranked: list, threshold: float = 0.3, top_k: int = None) -> list:
    """
    Keep attributes selected in at least ``threshold`` fraction of LLM responses,
    optionally capped to the highest-frequency ``top_k`` attributes.

    Parameters
    ----------
    ranked    : list of (attr, score) from aggregate_importance
    threshold : minimum response frequency to retain attribute
    top_k     : maximum number of ranked attributes to retain
    """
    selected = [attr for attr, score in ranked if score >= threshold]
    if top_k is not None:
        selected = selected[: int(top_k)]
    top_k_text = "all" if top_k is None else str(top_k)
    print(
        f"\nSelected attributes "
        f"(response frequency >= {threshold}, top_k={top_k_text}): {selected}"
    )
    return selected


# ── 4. Full pipeline ──

def llm_guided_selection(df_A, df_B, labeled_pairs, threshold=0.3, top_k=None, return_summary=False):
    """
    Full LLM-guided attribute selection pipeline.

    Parameters
    ----------
    df_A, df_B     : DataFrames with all attributes
    labeled_pairs  : sampled list of (idxA, idxB, label)
    threshold      : minimum response frequency to retain attribute
    top_k          : maximum number of ranked attributes to retain
    return_summary : when True, also return Step-1 token accounting
    """
    cols = [c for c in df_A.columns if c in df_B.columns]
    print(f"Candidate attributes: {cols}")
    sample = list(labeled_pairs)

    # query LLM for each sampled pair
    responses = []
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for idx, (idx_a, idx_b, label) in enumerate(sample):
        print(f"  Querying pair {idx+1}/{len(sample)} "
              f"({'match' if label == 1 else 'non-match'})...")
        attributes, usage = query_llm_field_importance_with_usage(idx_a, idx_b, label, df_A, df_B)
        responses.append(attributes)
        for key in token_usage:
            token_usage[key] += int(usage.get(key, 0))
        print(f"  → {attributes}")

    # aggregate scores across all responses
    ranked = aggregate_importance(responses, cols)

    print("\n[LLM-Guided Attribute Frequency Ranking]")
    for attr, score in ranked:
        print(f"  {attr:<20} : {score:.4f}")

    # select attributes above threshold
    selected = select_top_attributes(ranked, threshold=threshold, top_k=top_k)
    selection_summary = {
        "selection_token_usage": token_usage,
        "selection_pair_count": len(sample),
    }

    if return_summary:
        return df_A[selected].copy(), df_B[selected].copy(), ranked, selection_summary
    return df_A[selected].copy(), df_B[selected].copy(), ranked
