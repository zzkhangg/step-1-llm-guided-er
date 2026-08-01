import json
import hashlib
from pathlib import Path
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from .llm_client import create_chat_completion_text, get_llm_model

PROMPT = """You are an entity resolution system.

Decide whether two records refer to the same real-world entity.

Use robust entity-resolution reasoning:
- Ignore minor formatting differences, punctuation, capitalization, spacing, and token order.
- Treat common abbreviations and expanded forms as potentially equivalent when the meaning is clear.
- Treat missing values as unknown, not as evidence of mismatch.
- Do not require exact equality across all fields.
- Prefer agreement on distinctive identifiers, names/titles, model numbers, addresses, dates, or other high-information fields.
- Treat exact or near-exact model numbers, SKUs, UPCs, ISBNs, and manufacturer part numbers as strong match evidence, even when spacing, punctuation, or token grouping differs.
- Return No when there is a clear contradiction on important fields, or when the shared evidence is too weak.

Record A:
{record_a}

Record B:
{record_b}

Answer with exactly one word: Yes or No."""

# ---------------------------------------
# Setup
# ---------------------------------------
CACHE_DIR = Path(os.getenv("MATCHER_CACHE_DIR", "cache/default"))
USE_CACHE = os.getenv("MATCHER_DISABLE_CACHE", "").lower() not in {"1", "true", "yes"}


def set_cache_dir(cache_dir):
    """Set the cache directory used by pairwise LLM matching."""
    global CACHE_DIR
    CACHE_DIR = Path(cache_dir)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def set_cache_enabled(enabled):
    """Enable or disable pairwise LLM matching cache reads and writes."""
    global USE_CACHE
    USE_CACHE = bool(enabled)

# ---------------------------------------
# Cache utilities
# ---------------------------------------
def prompt_hash(prompt_template=None):
    """Return a stable hash for the matcher prompt template."""
    if prompt_template is None:
        prompt_template = PROMPT
    return hashlib.sha256(prompt_template.encode()).hexdigest()


def record_pair_hash(recA, recB, model, prompt_template=None):
    if prompt_template is None:
        prompt_template = PROMPT
    rec_str = json.dumps(
        {
            "model": model,
            "prompt_hash": prompt_hash(prompt_template),
            "records": [recA, recB],
        },
        sort_keys=True,
    )
    return hashlib.sha256(rec_str.encode()).hexdigest()

def load_from_cache(pair_hash):
    if not USE_CACHE:
        return None
    cache_file = CACHE_DIR / f"{pair_hash}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None

def save_to_cache(pair_hash, data):
    if not USE_CACHE:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{pair_hash}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def clean_record(rec):
    cleaned = {}
    for k, v in rec.items():
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, float) and math.isnan(v):
            cleaned[k] = ""
        else:
            cleaned[k] = str(v)
    return cleaned

def infer_pair(i, j, df_A, df_B, selected_attributes=None):
    """
    Single API call for one record pair using the PROMPT template.
    selected_attributes optionally condenses this post-blocking pair before
    prompting the LLM.
    Returns dict: {indexA, indexB, answer, input_tokens}
    """
    selected_attributes_list = list(selected_attributes) if selected_attributes is not None else list(df_A.columns)
    selected_attributes_json = json.dumps(selected_attributes_list, ensure_ascii=False)

    recA_raw = df_A.iloc[i].to_dict()
    recB_raw = df_B.iloc[j].to_dict()
    if selected_attributes is not None:
        recA_raw = {attr: recA_raw.get(attr, "") for attr in selected_attributes}
        recB_raw = {attr: recB_raw.get(attr, "") for attr in selected_attributes}

    recA = clean_record(recA_raw)
    recB = clean_record(recB_raw)
    model = get_llm_model()
    active_prompt_hash = prompt_hash()
    pair_hash = record_pair_hash(recA, recB, model)

    # Check cache
    cached = load_from_cache(pair_hash)
    if cached:
        return {
            "indexA": i,
            "indexB": j,
            "answer": cached.get("answer"),   # fallback if missing
            "prompt_tokens": cached.get("prompt_tokens", 0),
            "completion_tokens": cached.get("completion_tokens", 0),
            "total_tokens": cached.get("total_tokens", cached.get("prompt_tokens", 0)),
            "cache_hit": True,
            "llm_model": model,
            "llm_error": "",
            "prompt_hash": cached.get("prompt_hash", active_prompt_hash),
            "selected_attributes": selected_attributes_json,
            "selected_attribute_count": len(selected_attributes_list),
        }

    # Build prompt
    prompt = (PROMPT
          .replace("{record_a}", json.dumps(recA, ensure_ascii=False))
          .replace("{record_b}", json.dumps(recB, ensure_ascii=False)))

    content, usage = create_chat_completion_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer  = "Yes" if content.lower().startswith("yes") else "No"

    data = {
        "prompt": prompt,
        "model": model,
        "prompt_hash": active_prompt_hash,
        "answer": answer,
        "raw_response": content,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }

    save_to_cache(pair_hash, data)

    result = {
        "indexA": i,
        "indexB": j,
        "answer": answer,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cache_hit": False,
        "llm_model": model,
        "llm_error": "",
        "prompt_hash": active_prompt_hash,
        "selected_attributes": selected_attributes_json,
        "selected_attribute_count": len(selected_attributes_list),
    }
    return result

# ---------------------------------------
# Concurrent pairwise inference
# ---------------------------------------
def infer_candidates_pairwise(df_A, df_B, candidate_pairs, max_workers=8, selected_attributes_by_pair=None):
    """
    Run one API call per candidate pair concurrently.
    selected_attributes_by_pair can be keyed by (idxA, idxB) or candidate-pair
    ordinal index to condense each post-blocking pair before LLM matching.
    Returns a DataFrame with columns: indexA, indexB, answer, input_tokens
    """
    results = []
    total = len(candidate_pairs)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for pair_index, (i, j) in enumerate(candidate_pairs):
            selected_attributes = None
            if selected_attributes_by_pair is not None:
                if isinstance(selected_attributes_by_pair, dict):
                    selected_attributes = selected_attributes_by_pair.get((i, j))
                    if selected_attributes is None:
                        selected_attributes = selected_attributes_by_pair.get(pair_index)
                else:
                    selected_attributes = selected_attributes_by_pair[pair_index]
            futures[executor.submit(infer_pair, i, j, df_A, df_B, selected_attributes)] = (
                i,
                j,
                selected_attributes,
            )
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                i, j, selected_attributes = futures[future]
                selected_attributes_list = (
                    list(selected_attributes)
                    if selected_attributes is not None
                    else list(df_A.columns)
                )
                results.append({
                    "indexA": i,
                    "indexB": j,
                    "answer": "Error",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cache_hit": False,
                    "llm_model": get_llm_model(),
                    "llm_error": str(e),
                    "prompt_hash": prompt_hash(),
                    "selected_attributes": json.dumps(selected_attributes_list, ensure_ascii=False),
                    "selected_attribute_count": len(selected_attributes_list),
                })
                print(f"  Error on pair ({i}, {j}): {e}")
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    return pd.DataFrame(results)
