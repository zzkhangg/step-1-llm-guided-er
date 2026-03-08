import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os
from constants import PROMPT
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
# ---------------------------------------
# Setup
# ---------------------------------------
load_dotenv()  # load .env file
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=API_KEY)

CACHE_DIR = Path("cache/Itunes-Amazon")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------
# Utility functions
# ---------------------------------------
def record_pair_hash(recA, recB):
    """Generate a unique hash for caching each record pair."""
    rec_str = json.dumps([recA, recB], sort_keys=True)
    return hashlib.sha256(rec_str.encode()).hexdigest()


def load_from_cache(pair_hash):
    cache_file = CACHE_DIR / f"{pair_hash}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def save_to_cache(pair_hash, data):
    """Save cache including input prompt, answer, and input tokens."""
    cache_file = CACHE_DIR / f"{pair_hash}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------
# LLM Inference for batch of pairs
# ---------------------------------------
def infer_pairs_batch(pairs, df_A, df_B):
    """
    Infer multiple record pairs in a single API call.
    Returns a list of results: {"indexA", "indexB", "answer", "input_tokens"}
    """
    # Check cache first
    uncached_pairs = []
    results = []

    for i, j in pairs:
        recA = df_A.iloc[i].to_dict()
        recB = df_B.iloc[j].to_dict()
        pair_hash = record_pair_hash(recA, recB)
        cached = load_from_cache(pair_hash)
        if cached:
            results.append({
                "indexA": i,
                "indexB": j,
                "answer": cached["answer"],
                "input_tokens": cached["input_tokens"]
            })
        else:
            uncached_pairs.append((i, j, recA, recB, pair_hash))

    if not uncached_pairs:
        return results

    # Build prompt for all uncached pairs in batch
    prompt_lines = []
    for idx, (i, j, recA, recB, pair_hash) in enumerate(uncached_pairs, 1):
        prompt_lines.append(
            f"Pair {idx}:\nRecord A: {recA}\nRecord B: {recB}"
        )
    batch_prompt = (
        "You are an expert in data integration and entity resolution.\n"
        "For each pair below, answer strictly 'Yes' or 'No':\n\n"
        + "\n\n".join(prompt_lines) +
        "\n\nAnswer format: 1. Yes/No, 2. Yes/No, ..."
    )

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": batch_prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    answers = [ans.strip().split(".")[-1].strip() for ans in content.split("\n") if ans.strip()]
    if len(answers) != len(uncached_pairs):
        print("Warning: GPT returned fewer/more answers than expected")
    
    # Assign answers to uncached pairs and save cache
    for (i, j, recA, recB, pair_hash), ans in zip(uncached_pairs, answers):
        ans_norm = "Yes" if ans.lower().startswith("yes") else "No"
        cache_data = {
            "prompt": batch_prompt,
            "answer": ans_norm,
            "input_tokens": response.usage.prompt_tokens
        }
        save_to_cache(pair_hash, cache_data)
        results.append({
            "indexA": i,
            "indexB": j,
            "answer": ans_norm,
            "input_tokens": response.usage.prompt_tokens
        })

    return results


# ---------------------------------------
# Batch inference with concurrency and batching
# ---------------------------------------
def infer_candidates_batch(df_A, df_B, candidate_pairs, batch_size=5, max_workers=8):
    """
    Infer candidate pairs using batches and concurrency.
    Returns a DataFrame with columns: indexA, indexB, answer, input_tokens
    """
    results = []
    total_pairs = len(candidate_pairs)
    processed_pairs = 0

    # Split candidate_pairs into batches
    batches = [candidate_pairs[i:i + batch_size] for i in range(0, total_pairs, batch_size)]

    def worker(batch):
        res = infer_pairs_batch(batch, df_A, df_B)
        nonlocal processed_pairs
        processed_pairs += len(batch)
        print(f"[Progress] Compared {processed_pairs}/{total_pairs} pairs ({processed_pairs/total_pairs*100:.2f}%)")
        return res

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(worker, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            results.extend(future.result())

    return pd.DataFrame(results)