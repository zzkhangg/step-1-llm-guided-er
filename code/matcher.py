import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = (
    "You are an expert in data integration and entity resolution. "
    "Your task is to determine whether the following two records refer to the exact same real-world entity.\n"
    "Record A: {record_a}\n"
    "Record B: {record_b}\n"
    "Question: Do these records match? Please answer strictly with 'Yes' or 'No'."
)

# ---------------------------------------
# Setup
# ---------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=API_KEY)

CACHE_DIR = Path("cache/Fodors-Zagat")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Cache utilities
# ---------------------------------------
def record_pair_hash(recA, recB):
    rec_str = json.dumps([recA, recB], sort_keys=True)
    return hashlib.sha256(rec_str.encode()).hexdigest()

def load_from_cache(pair_hash):
    cache_file = CACHE_DIR / f"{pair_hash}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None

def save_to_cache(pair_hash, data):
    cache_file = CACHE_DIR / f"{pair_hash}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

def infer_pair(i, j, df_A, df_B):
    """
    Single API call for one record pair using the PROMPT template.
    Returns dict: {indexA, indexB, answer, input_tokens}
    """
    recA = df_A.iloc[i].to_dict()
    recB = df_B.iloc[j].to_dict()
    pair_hash = record_pair_hash(recA, recB)

    # Check cache
    cached = load_from_cache(pair_hash)
    if cached:
        return {
            "indexA": i,
            "indexB": j,
            "answer": cached.get("answer"),   # fallback if missing
            "prompt_tokens": cached.get("prompt_tokens")  # fallback
        }

    # Build prompt
    prompt = (PROMPT
          .replace("{record_a}", json.dumps(recA, ensure_ascii=False))
          .replace("{record_b}", json.dumps(recB, ensure_ascii=False)))

    # Single API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    answer  = "Yes" if content.lower().startswith("yes") else "No"

    usage = response.usage

    data = {
        "prompt": prompt,
        "answer": answer,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }

    save_to_cache(pair_hash, data)

    result = {
        "indexA": i,
        "indexB": j,
        "answer": answer,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }
    return result

# ---------------------------------------
# Concurrent pairwise inference
# ---------------------------------------
def infer_candidates_pairwise(df_A, df_B, candidate_pairs, max_workers=8):
    """
    Run one API call per candidate pair concurrently.
    Returns a DataFrame with columns: indexA, indexB, answer, input_tokens
    """
    results = []
    total = len(candidate_pairs)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(infer_pair, i, j, df_A, df_B): (i, j)
            for i, j in candidate_pairs
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                i, j = futures[future]
                print(f"  Error on pair ({i}, {j}): {e}")
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    return pd.DataFrame(results)