import json
import pandas as pd
import numpy as np
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ATTRIBUTE_PROMPT = (
    "You are an expert in data integration and entity resolution.\n"
    "Below are two records and whether they refer to the same real-world entity.\n\n"
    "Record A: {record_a}\n"
    "Record B: {record_b}\n"
    "Match: {label}\n\n"
    "Question: Which fields were most influential in deciding whether these two records match?\n"
    "Return a JSON object with field names as keys and importance score (0.0 to 1.0) as values.\n"
    "Only include fields that exist in the records.\n"
    "Example: {{\"title\": 0.9, \"authors\": 0.8, \"year\": 0.6, \"venue\": 0.1}}\n"
    "Return JSON only, no explanation."
)


# ── 1. Sample representative pairs ──

def sample_pairs(labeled_pairs, n_pos=10, n_neg=10, seed=42):
    """
    Sample n_pos positive and n_neg negative pairs from labeled pairs.
    Balanced sample gives LLM both match and non-match examples.

    Parameters
    ----------
    labeled_pairs : list of (idxA, idxB, label)
    n_pos         : number of positive pairs to sample
    n_neg         : number of negative pairs to sample
    seed          : random seed for reproducibility
    """
    rng = np.random.RandomState(seed)

    pos_pairs = [(a, b, l) for a, b, l in labeled_pairs if l == 1]
    neg_pairs = [(a, b, l) for a, b, l in labeled_pairs if l == 0]

    pos_sample = [pos_pairs[i] for i in
                  rng.choice(len(pos_pairs),
                             min(n_pos, len(pos_pairs)),
                             replace=False)]
    neg_sample = [neg_pairs[i] for i in
                  rng.choice(len(neg_pairs),
                             min(n_neg, len(neg_pairs)),
                             replace=False)]

    sample = pos_sample + neg_sample
    rng.shuffle(sample)

    print(f"Sampled {len(pos_sample)} positive + {len(neg_sample)} negative pairs")
    return sample


# ── 2. Query LLM for field importance per pair ──

def query_llm_field_importance(idx_a, idx_b, label, df_A, df_B):
    """
    Ask LLM which fields were most influential for a single pair.
    Returns dict of {field: importance_score}.

    Parameters
    ----------
    idx_a, idx_b : positional indices into df_A, df_B
    label        : 1 = match, 0 = non-match
    """
    recA = {k: ('' if pd.isna(v) else str(v))
            for k, v in df_A.iloc[idx_a].to_dict().items()}
    recB = {k: ('' if pd.isna(v) else str(v))
            for k, v in df_B.iloc[idx_b].to_dict().items()}

    prompt = (ATTRIBUTE_PROMPT
              .replace("{record_a}", str(recA))
              .replace("{record_b}", str(recB))
              .replace("{label}", "Yes" if label == 1 else "No"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # strip markdown code fences if present
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        scores = json.loads(content)
        return {k: float(v) for k, v in scores.items()}
    except json.JSONDecodeError:
        print(f"  Warning: could not parse LLM response: {content}")
        return {}


# ── 3. Aggregate importance scores across all sampled pairs ──

def aggregate_importance(responses: list, cols: list) -> list:
    """
    Aggregate field importance scores across all LLM responses.
    Uses mean score per field — fields not mentioned get score 0.

    Parameters
    ----------
    responses : list of {field: score} dicts from LLM
    cols      : all candidate attribute columns
    """
    score_accumulator = defaultdict(list)

    for response in responses:
        for col in cols:
            score = response.get(col, 0.0)
            score_accumulator[col].append(score)

    aggregated = {
        col: np.mean(scores) if scores else 0.0
        for col, scores in score_accumulator.items()
    }

    # rank by importance descending
    ranked = sorted(aggregated.items(), key=lambda x: -x[1])
    return ranked


# ── 4. Select attributes above threshold ──

def select_top_attributes(ranked: list, threshold: float = 0.3) -> list:
    """
    Keep attributes whose mean importance score exceeds threshold.

    Parameters
    ----------
    ranked    : list of (attr, score) from aggregate_importance
    threshold : minimum mean importance score to retain attribute
    """
    selected = [attr for attr, score in ranked if score >= threshold]
    print(f"\nSelected attributes (score >= {threshold}): {selected}")
    return selected


# ── 5. Full pipeline ──

def llm_guided_selection(df_A, df_B, labeled_pairs,
                          n_pos=10, n_neg=10,
                          threshold=0.3, seed=42):
    """
    Full LLM-guided attribute selection pipeline.

    Parameters
    ----------
    df_A, df_B     : DataFrames with all attributes
    labeled_pairs  : list of (idxA, idxB, label) — from build_labeled_pairs()
    n_pos          : positive pairs to sample for LLM queries
    n_neg          : negative pairs to sample for LLM queries
    threshold      : minimum importance score to retain attribute
    seed           : random seed for sampling reproducibility
    """
    cols = [c for c in df_A.columns if c in df_B.columns]
    print(f"Candidate attributes: {cols}")

    # sample representative pairs
    sample = sample_pairs(labeled_pairs, n_pos=n_pos, n_neg=n_neg, seed=seed)

    # query LLM for each sampled pair
    responses = []
    for idx, (idx_a, idx_b, label) in enumerate(sample):
        print(f"  Querying pair {idx+1}/{len(sample)} "
              f"({'match' if label == 1 else 'non-match'})...")
        scores = query_llm_field_importance(idx_a, idx_b, label, df_A, df_B)
        responses.append(scores)
        print(f"  → {scores}")

    # aggregate scores across all responses
    ranked = aggregate_importance(responses, cols)

    print("\n[LLM-Guided Attribute Importance Ranking]")
    for attr, score in ranked:
        print(f"  {attr:<20} : {score:.4f}")

    # select attributes above threshold
    selected = select_top_attributes(ranked, threshold=threshold)

    return df_A[selected].copy(), df_B[selected].copy(), ranked