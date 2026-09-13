import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from .lsh import create_random_planes, query_lsh_fast
from .utils import build_id_maps, build_gt_set
from .loader import load_data
from .attribute_selection import (
    DEFAULT_RATIO_LAMBDA,
    DEFAULT_SELECTION_POLICY,
    SELECTION_POLICIES,
    HybridAttributeSelector,
    attribute_evidence_mask,
    attribute_blankness,
    compute_importance_proxy_scores,
    compute_attribute_features,
    compute_numeric_scales,
    flatten_pair_features,
    heuristic_selection,
    infer_attribute_roles,
    manual_selection,
    select_diverse_profile_pairs,
    select_importance_based_profile_pairs,
    select_profile_pairs,
    select_random_profile_pairs,
    select_stratified_diverse_profile_pairs,
)
from .attribute_selection.llm_guided import (
    DEFAULT_PROFILE_SCORING,
    PROFILE_SCORING_MODES,
    llm_guided_selection,
    query_llm_profile_importance_with_usage,
    set_profile_cache_dir,
    set_profile_cache_enabled,
)
from .attribute_selection.supervised import (
    select_top_attributes as supervised_select_top_attributes,
    train_attribute_selector,
)
from .matcher import infer_candidates_pairwise, set_cache_dir, set_cache_enabled
from .embeddings import embed_dataframe_sbert
from .llm_client import get_llm_model, provider_order, reasoning_enabled
from .preprocessing import normalize_dataframe_records


DEFAULT_PROFILE_SAMPLER = "similarity_stratified"
DEFAULT_PROFILE_MAX_WORKERS = 4

DATASET_CONFIGS = {
    "Fodors-Zagat": {
        "base_path": "datasets/Fodors-Zagat",
        "table_a": "tableA.csv",
        "table_b": "tableB.csv",
        "ground_truth": "gold.csv",
        "id_a_col": "id",
        "id_b_col": "id",
        "gt_id_a_col": "ltable_id",
        "gt_id_b_col": "rtable_id",
        "encoding": "utf-8",
        "cache_dir": "cache/Fodors-Zagat",
        "blocking": {"num_tables": 5, "num_planes": 8, "num_flips": 1, "top_k": 5},
        "adaptive": {
            "profile_sampler": DEFAULT_PROFILE_SAMPLER,
            "profile_sample_size": 20,
            "profile_max_workers": DEFAULT_PROFILE_MAX_WORKERS,
            "top_k_retrieval": 5,
            "min_k_attributes": 2,
            "max_k_attributes": 5,
            "cumulative_importance_threshold": 0.8,
            "required_attribute_roles": ["identity"],
        },
    },
    "Amazon-Walmart": {
        "base_path": "datasets/Amazon-Walmart",
        "table_a": "amazon.csv",
        "table_b": "walmart.csv",
        "ground_truth": "gold.tsv",
        "id_a_col": "id",
        "id_b_col": "id",
        "gt_id_a_col": "id1",
        "gt_id_b_col": "id2",
        "encoding": "utf-8",
        "cache_dir": "cache/Amazon-Walmart",
        "blocking": {"num_tables": 10, "num_planes": 4, "num_flips": 1, "top_k": 4},
        "adaptive": {
            "profile_sampler": DEFAULT_PROFILE_SAMPLER,
            "profile_sample_size": 50,
            "profile_max_workers": DEFAULT_PROFILE_MAX_WORKERS,
            "top_k_retrieval": 5,
            "min_k_attributes": 2,
            "max_k_attributes": 5,
            "cumulative_importance_threshold": 0.8,
            "required_attribute_roles": ["identity"],
        },
    },
    "DBLP-ACM": {
        "base_path": "datasets/DBLP-ACM",
        "table_a": "DBLP.csv",
        "table_b": "ACM.csv",
        "ground_truth": "gold.csv",
        "id_a_col": "id",
        "id_b_col": "id",
        "gt_id_a_col": "idDBLP",
        "gt_id_b_col": "idACM",
        "encoding": "latin1",
        "cache_dir": "cache/DBLP-ACM",
        "blocking": {"num_tables": 15, "num_planes": 8, "num_flips": 1, "top_k": 5},
        "adaptive": {
            "profile_sampler": DEFAULT_PROFILE_SAMPLER,
            "profile_sample_size": 50,
            "profile_max_workers": DEFAULT_PROFILE_MAX_WORKERS,
            "top_k_retrieval": 5,
            "min_k_attributes": 2,
            "max_k_attributes": 5,
            "cumulative_importance_threshold": 0.8,
            "required_attribute_roles": ["identity"],
        },
    },
    # Held-out dataset. Every setting below was fixed before the first run and
    # copied from DBLP-ACM rather than tuned here, so that nothing on this
    # dataset has been chosen by looking at its results. See
    # docs/heldout_preregistration_2026-09-04.md.
    "DBLP-Scholar": {
        "base_path": "datasets/DBLP-Scholar",
        "table_a": "tableA.csv",
        "table_b": "tableB.csv",
        "ground_truth": "gold.csv",
        "id_a_col": "id",
        "id_b_col": "id",
        "gt_id_a_col": "ltable_id",
        "gt_id_b_col": "rtable_id",
        "encoding": "utf-8",
        "cache_dir": "cache/DBLP-Scholar",
        "blocking": {"num_tables": 15, "num_planes": 8, "num_flips": 1, "top_k": 5},
        "adaptive": {
            "profile_sampler": DEFAULT_PROFILE_SAMPLER,
            "profile_sample_size": 50,
            "profile_max_workers": DEFAULT_PROFILE_MAX_WORKERS,
            "top_k_retrieval": 5,
            "min_k_attributes": 2,
            "max_k_attributes": 5,
            "cumulative_importance_threshold": 0.8,
            "required_attribute_roles": ["identity"],
        },
    },
}

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EXPERIMENT_MODES = ("full", "step1", "step15")
TUPLE_STRATEGIES = ("none", "manual", "heuristic", "llm_guided", "supervised")
PROFILE_SAMPLERS = (
    "random",
    "similarity_stratified",
    "diversity",
    "stratified_diversity",
    "importance_based",
)
ADAPTIVE_USE_SEMANTIC_FEATURES = False
RANDOM_SEED = 42
LOG_ROOT = Path("logs")


def compute_candidate_pair_scores(candidate_pairs, tableA_vectors, tableB_vectors):
    """Compute cosine-style scores for already-normalized blocked pairs."""
    return np.asarray([
        float(np.dot(tableA_vectors[i], tableB_vectors[j]))
        for i, j in candidate_pairs
    ])


def check_api_errors(result_df, allow_api_errors=False):
    """Return the failed-call count, refusing to proceed unless explicitly overridden.

    A failed LLM call is recorded as ``answer="Error"`` and scored as a non-match, so a
    run that lost its API budget still produces a complete, plausible-looking summary:
    F1 of 0.0000 if every call failed, some intermediate value if only some did. Nine
    such runs were mistaken for results before this check existed, so the default is to
    raise rather than write numbers nobody should trust.
    """
    if "answer" not in result_df:
        return 0
    error_count = int((result_df["answer"] == "Error").sum())
    if not error_count:
        return 0

    total = len(result_df)
    share = error_count / total if total else 0.0
    message = (
        f"{error_count}/{total} LLM calls failed ({share:.1%}). Failed calls are scored "
        "as non-matches, so any metrics from this run are meaningless."
    )
    if not allow_api_errors:
        raise RuntimeError(
            message + " Nothing has been written to logs/. Re-run once the cause is "
            "resolved, or pass --allow-api-errors to record the run anyway."
        )
    print(f"\n[WARNING] {message}")
    print("[WARNING] --allow-api-errors was set: the run will be written with "
          "metrics_invalid=true. Do not quote its numbers.")
    return error_count


def build_step1_labeled_pairs(gt_set, candidate_pairs, n_pos, n_neg, seed=RANDOM_SEED):
    """Build a balanced Step-1 sample from known matches and blocked hard negatives."""
    positives = [(idx_a, idx_b, 1) for idx_a, idx_b in gt_set]
    negatives = [
        (idx_a, idx_b, 0)
        for idx_a, idx_b in candidate_pairs
        if (idx_a, idx_b) not in gt_set
    ]

    rng = np.random.RandomState(seed)
    if n_pos is not None and n_pos < len(positives):
        indices = rng.choice(len(positives), size=n_pos, replace=False)
        positives = [positives[i] for i in indices]
    if n_neg is not None and n_neg < len(negatives):
        indices = rng.choice(len(negatives), size=n_neg, replace=False)
        negatives = [negatives[i] for i in indices]

    labeled_pairs = positives + negatives
    rng.shuffle(labeled_pairs)

    print("\n[Step 1] Global attribute selection sample")
    print(f"  Positives: {len(positives)}")
    print(f"  Negatives: {len(negatives)}")
    print(f"  Total    : {len(labeled_pairs)}")
    return labeled_pairs


def parse_manual_attributes(value):
    if not value:
        return []
    return [attr.strip() for attr in value.split(",") if attr.strip()]


def drop_excluded_attributes(df_a, df_b, excluded_attributes):
    """Drop excluded attributes from both tables before blocking and selection."""
    excluded = list(dict.fromkeys(excluded_attributes or []))
    if not excluded:
        return df_a, df_b, []

    missing = [attr for attr in excluded if attr not in df_a.columns or attr not in df_b.columns]
    if missing:
        raise ValueError(f"Excluded attributes are not present in both tables: {missing}")

    return (
        df_a.drop(columns=excluded).copy(),
        df_b.drop(columns=excluded).copy(),
        excluded,
    )


def apply_global_tuple_strategy(
    df_a,
    df_b,
    strategy,
    gt_set,
    candidate_pairs,
    threshold,
    top_k,
    n_pos,
    n_neg,
    manual_attributes=None,
    seed=RANDOM_SEED,
):
    """Apply one Step-1 global tuple-condensation strategy."""
    strategy = strategy or "none"
    summary = {
        "strategy": strategy,
        "input_attributes": df_a.columns.tolist(),
        "input_attribute_count": len(df_a.columns),
    }

    if strategy == "none":
        summary.update({
            "selected_attributes": df_a.columns.tolist(),
            "selected_attribute_count": len(df_a.columns),
        })
        return df_a, df_b, summary

    if strategy == "manual":
        selected = list(manual_attributes or [])
        missing = [attr for attr in selected if attr not in df_a.columns or attr not in df_b.columns]
        if missing:
            raise ValueError(f"Manual attributes are not present in both tables: {missing}")
        if not selected:
            raise ValueError("Manual tuple strategy requires --manual-attributes attr1,attr2,...")
        out_a, out_b = manual_selection(df_a, df_b, selected)
        ranked = [(attr, float(len(selected) - idx)) for idx, attr in enumerate(selected)]

    elif strategy == "heuristic":
        out_a, out_b = heuristic_selection(df_a, df_b)
        selected = out_a.columns.tolist()
        ranked = [(attr, float(len(selected) - idx)) for idx, attr in enumerate(selected)]

    elif strategy == "llm_guided":
        labeled_pairs = build_step1_labeled_pairs(
            gt_set,
            candidate_pairs,
            n_pos=n_pos,
            n_neg=n_neg,
            seed=seed,
        )
        # llm_guided thresholds a response frequency -- the fraction of profiled pairs in
        # which the LLM named the attribute -- which is already independent of schema
        # width, so it keeps the fixed 0.3 default. Only the supervised strategy, which
        # thresholds a share normalised across attributes, needs the relative bar.
        out_a, out_b, ranked, selection_summary = llm_guided_selection(
            df_a,
            df_b,
            labeled_pairs,
            threshold=0.3 if threshold is None else threshold,
            top_k=top_k,
            return_summary=True,
        )
        selected = out_a.columns.tolist()
        summary["labeled_pair_count"] = len(labeled_pairs)
        summary.update(selection_summary)

    elif strategy == "supervised":
        labeled_pairs = build_step1_labeled_pairs(
            gt_set,
            candidate_pairs,
            n_pos=n_pos,
            n_neg=n_neg,
            seed=seed,
        )
        cols = [col for col in df_a.columns if col in df_b.columns]
        _, _, ranked, _ = train_attribute_selector(df_a, df_b, labeled_pairs, cols)
        selected = supervised_select_top_attributes(ranked, threshold=threshold, top_k=top_k)
        out_a = df_a[selected].copy()
        out_b = df_b[selected].copy()
        summary["labeled_pair_count"] = len(labeled_pairs)

    else:
        raise ValueError(f"Unknown tuple strategy: {strategy}")

    if not selected:
        raise RuntimeError(
            f"Tuple strategy '{strategy}' selected no attributes. "
            "Lower --tuple-threshold or choose a different strategy."
        )

    # Every strategy above emits its subset in importance-ranked order, and the
    # subsequent df[selected] reprojection makes that the column order the matcher
    # serializes. Restore table-column order so that Step-1, Step-1.5 and no-selection
    # runs present the same fields in the same sequence and stay comparable; the
    # ranking itself is preserved in summary["ranked_attributes"].
    canonical_position = {attr: pos for pos, attr in enumerate(df_a.columns)}
    selected = sorted(selected, key=lambda attr: canonical_position.get(attr, len(canonical_position)))
    out_a = out_a[selected].copy()
    out_b = out_b[selected].copy()

    summary.update({
        "threshold": threshold,
        "top_k": top_k,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "selected_attributes": selected,
        "selected_attribute_count": len(selected),
        "ranked_attributes": ranked,
        "canonical_attribute_order": True,
    })
    print(f"\n[Step 1] Strategy: {strategy}")
    print("[Step 1] Selected global attributes:", selected)
    return out_a, out_b, summary


def run_lsh_blocking(df_a, df_b, model, blocking, seed=RANDOM_SEED, label="records",
                     model_name=None):
    """Embed two tables and run LSH blocking for the active attribute view."""
    table_a_vectors = embed_dataframe_sbert(df_a, model, batch_size=256, model_name=model_name)
    table_b_vectors = embed_dataframe_sbert(df_b, model, batch_size=256, model_name=model_name)

    print(f"TableA {label} vectors shape:", table_a_vectors.shape)
    print(f"TableB {label} vectors shape:", table_b_vectors.shape)

    planes_list = create_random_planes(
        num_tables=blocking["num_tables"],
        num_planes=blocking["num_planes"],
        dim=table_a_vectors.shape[1],
        seed=seed,
    )
    candidate_pairs = query_lsh_fast(
        table_a_vectors,
        table_b_vectors,
        planes_list,
        num_flips=blocking["num_flips"],
        top_k=blocking["top_k"],
    )
    pair_similarity_scores = compute_candidate_pair_scores(
        candidate_pairs,
        table_a_vectors,
        table_b_vectors,
    )
    return table_a_vectors, table_b_vectors, candidate_pairs, pair_similarity_scores


def pair_feature_vector(df_A, df_B, idx_a, idx_b, attributes, embedding_model=None, numeric_scales=None):
    """Build the flattened adaptive feature vector for one candidate pair."""
    feature_dict = compute_attribute_features(
        df_A.iloc[idx_a].to_dict(),
        df_B.iloc[idx_b].to_dict(),
        attributes,
        embedding_model=embedding_model,
        numeric_scales=numeric_scales,
    )
    return flatten_pair_features(feature_dict, attributes)


def normalize_importance_vector(importance, attributes):
    """Clamp profiling scores to non-negative values and normalize to sum 1."""
    raw = {attr: max(0.0, float(importance.get(attr, 0.0))) for attr in attributes}
    total = sum(raw.values())
    if total > 0:
        normalized = {attr: value / total for attr, value in raw.items()}
    else:
        fallback = 1.0 / len(attributes) if attributes else 0.0
        normalized = {attr: fallback for attr in attributes}
    return raw, normalized


def aligned_pair_similarity_scores(candidate_pairs, pair_similarity_scores):
    """Return candidate-pair similarity scores aligned with candidate_pairs."""
    pairs = list(candidate_pairs)
    if isinstance(pair_similarity_scores, dict):
        return np.asarray(
            [pair_similarity_scores.get(pair, pair_similarity_scores.get(tuple(pair), 0.0)) for pair in pairs],
            dtype=float,
        )

    scores = np.asarray(pair_similarity_scores, dtype=float)
    if scores.shape[0] != len(pairs):
        raise ValueError("pair_similarity_scores must align with candidate_pairs")
    return scores


def _pairwise_distance_stats(feature_matrix):
    if len(feature_matrix) < 2:
        return {"min": 0.0, "mean": 0.0}

    features = np.asarray(feature_matrix, dtype=float)
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    pairwise = distances[np.triu_indices(len(features), k=1)]
    return {
        "min": float(pairwise.min()) if len(pairwise) else 0.0,
        "mean": float(pairwise.mean()) if len(pairwise) else 0.0,
    }


def summarize_profile_sampling(
    candidate_pairs,
    pair_similarity_scores,
    pair_feature_matrix,
    profile_indices,
    profile_sampler,
    requested_sample_size,
):
    """Build label-free diagnostics for comparing profile samplers."""
    pairs = list(candidate_pairs)
    selected_indices = [int(idx) for idx in profile_indices]
    scores = np.clip(aligned_pair_similarity_scores(pairs, pair_similarity_scores), 0.0, 1.0)
    selected_scores = scores[selected_indices] if selected_indices else np.asarray([], dtype=float)
    importance_proxy_scores = compute_importance_proxy_scores(
        pairs,
        scores,
        pair_feature_matrix,
    )
    selected_importance_scores = (
        importance_proxy_scores[selected_indices]
        if selected_indices
        else np.asarray([], dtype=float)
    )

    candidate_strata_counts = {"low": 0, "medium": 0, "high": 0}
    selected_strata_counts = {"low": 0, "medium": 0, "high": 0}
    if len(scores):
        q_low, q_high = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])

        def stratum_name(score):
            if score <= q_low:
                return "low"
            if score >= q_high:
                return "high"
            return "medium"

        candidate_strata_counts = dict(collections.Counter(stratum_name(score) for score in scores))
        selected_strata_counts = dict(collections.Counter(stratum_name(score) for score in selected_scores))

    selected_features = (
        np.asarray(pair_feature_matrix, dtype=float)[selected_indices]
        if selected_indices
        else np.empty((0, 0), dtype=float)
    )
    diversity = _pairwise_distance_stats(selected_features)

    return {
        "profile_sampler": profile_sampler,
        "label_usage": "ground_truth_not_used_for_sampling",
        "candidate_pair_count": len(pairs),
        "requested_profile_pair_count": int(requested_sample_size),
        "profile_pair_count": len(selected_indices),
        "candidate_similarity_strata_counts": candidate_strata_counts,
        "profile_similarity_strata_counts": selected_strata_counts,
        "profile_similarity_min": float(selected_scores.min()) if len(selected_scores) else 0.0,
        "profile_similarity_mean": float(selected_scores.mean()) if len(selected_scores) else 0.0,
        "profile_similarity_max": float(selected_scores.max()) if len(selected_scores) else 0.0,
        "candidate_importance_proxy_mean": (
            float(importance_proxy_scores.mean()) if len(importance_proxy_scores) else 0.0
        ),
        "profile_importance_proxy_min": (
            float(selected_importance_scores.min()) if len(selected_importance_scores) else 0.0
        ),
        "profile_importance_proxy_mean": (
            float(selected_importance_scores.mean()) if len(selected_importance_scores) else 0.0
        ),
        "profile_importance_proxy_max": (
            float(selected_importance_scores.max()) if len(selected_importance_scores) else 0.0
        ),
        "profile_feature_distance_min": diversity["min"],
        "profile_feature_distance_mean": diversity["mean"],
    }


def select_adaptive_profile_pairs(
    candidate_pairs,
    pair_similarity_scores,
    pair_feature_matrix,
    sample_size,
    profile_sampler=DEFAULT_PROFILE_SAMPLER,
    seed=RANDOM_SEED,
):
    """Select Step-1.5 profile pairs without using ground-truth labels."""
    sampler = profile_sampler or DEFAULT_PROFILE_SAMPLER
    if sampler == "random":
        profile_pairs, profile_indices = select_random_profile_pairs(
            candidate_pairs,
            sample_size=sample_size,
            seed=seed,
        )
    elif sampler == "similarity_stratified":
        profile_pairs, profile_indices = select_profile_pairs(
            candidate_pairs,
            pair_similarity_scores,
            sample_size=sample_size,
            seed=seed,
        )
    elif sampler == "diversity":
        profile_pairs, profile_indices = select_diverse_profile_pairs(
            candidate_pairs,
            pair_feature_matrix,
            sample_size=sample_size,
            seed=seed,
        )
    elif sampler == "stratified_diversity":
        profile_pairs, profile_indices = select_stratified_diverse_profile_pairs(
            candidate_pairs,
            pair_similarity_scores,
            pair_feature_matrix,
            sample_size=sample_size,
            seed=seed,
        )
    elif sampler == "importance_based":
        profile_pairs, profile_indices = select_importance_based_profile_pairs(
            candidate_pairs,
            pair_similarity_scores,
            pair_feature_matrix,
            sample_size=sample_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown profile sampler: {sampler}")

    sampling_summary = summarize_profile_sampling(
        candidate_pairs,
        pair_similarity_scores,
        pair_feature_matrix,
        profile_indices,
        sampler,
        sample_size,
    )
    print(f"\n[Adaptive Profiling] Profile sampler: {sampler}")
    print(f"  Label usage       : {sampling_summary['label_usage']}")
    print(f"  Profile pairs     : {len(profile_pairs)}/{len(candidate_pairs)}")
    print(f"  Similarity strata : {sampling_summary['profile_similarity_strata_counts']}")
    print(f"  Importance proxy  : mean={sampling_summary['profile_importance_proxy_mean']:.4f}")
    print(f"  Feature diversity : mean distance={sampling_summary['profile_feature_distance_mean']:.4f}")
    return profile_pairs, profile_indices, sampling_summary


def profile_adaptive_pair(
    df_A,
    df_B,
    idx_a,
    idx_b,
    profile_idx,
    candidate_index,
    attributes,
    feature_vector,
    similarity_score,
    profile_scoring=DEFAULT_PROFILE_SCORING,
):
    importance, usage = query_llm_profile_importance_with_usage(
        idx_a, idx_b, df_A, df_B, scoring=profile_scoring
    )
    raw_importance, normalized_importance = normalize_importance_vector(importance, attributes)
    return {
        "cache_hit": bool(usage.get("cache_hit", False)),
        "profile_index": profile_idx,
        "candidate_index": candidate_index,
        "indexA": idx_a,
        "indexB": idx_b,
        "feature_vector": feature_vector,
        "raw_importance": raw_importance,
        "normalized_importance": normalized_importance,
        "similarity_score": float(similarity_score),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "profile_error": "",
    }


def build_adaptive_attribute_map(
    df_A,
    df_B,
    candidate_pairs,
    pair_similarity_scores,
    embedding_model=None,
    profile_sample_size=20,
    profile_sampler=DEFAULT_PROFILE_SAMPLER,
    profile_max_workers=DEFAULT_PROFILE_MAX_WORKERS,
    top_k_retrieval=5,
    top_k_attributes=None,
    min_k_attributes=2,
    max_k_attributes=5,
    evidence_mask_mode="drop",
    cumulative_importance_threshold=0.8,
    required_attribute_roles=None,
    attribute_roles=None,
    seed=RANDOM_SEED,
    selection_policy=DEFAULT_SELECTION_POLICY,
    ratio_lambda=DEFAULT_RATIO_LAMBDA,
    profile_scoring=DEFAULT_PROFILE_SCORING,
):
    """
    Profile a small post-blocking sample with the LLM, train the adaptive
    selector, then transfer pair-specific attribute choices to all candidates.
    """
    if profile_scoring not in PROFILE_SCORING_MODES:
        raise ValueError(
            f"Unknown profile scoring mode {profile_scoring!r}; expected one of {PROFILE_SCORING_MODES}"
        )
    attributes = [col for col in df_A.columns if col in df_B.columns]
    if not attributes or not candidate_pairs:
        return {}, None, {}
    inferred_attribute_roles = infer_attribute_roles(attributes)
    if attribute_roles:
        inferred_attribute_roles.update({
            attr: role for attr, role in attribute_roles.items() if attr in inferred_attribute_roles
        })
    numeric_scales = compute_numeric_scales(df_A, df_B, attributes)

    pair_similarity_scores = aligned_pair_similarity_scores(candidate_pairs, pair_similarity_scores)

    print("\n[Adaptive Profiling] Computing label-free pair features")
    pair_feature_matrix = np.asarray([
        pair_feature_vector(
            df_A,
            df_B,
            idx_a,
            idx_b,
            attributes,
            embedding_model=embedding_model,
            numeric_scales=numeric_scales,
        )
        for idx_a, idx_b in candidate_pairs
    ])

    profile_pairs, profile_indices, sampling_summary = select_adaptive_profile_pairs(
        candidate_pairs,
        pair_similarity_scores,
        pair_feature_matrix,
        sample_size=profile_sample_size,
        profile_sampler=profile_sampler,
        seed=seed,
    )
    if not profile_pairs:
        return {}, None, {}

    profile_max_workers = max(1, int(profile_max_workers))
    print(
        f"\n[Adaptive Profiling] Profiling {len(profile_pairs)} post-blocking pairs "
        f"with {profile_max_workers} workers"
    )

    profile_results = []
    profiling_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    with ThreadPoolExecutor(max_workers=profile_max_workers) as executor:
        futures = {}
        for profile_idx, (idx_a, idx_b) in enumerate(profile_pairs, start=1):
            candidate_index = profile_indices[profile_idx - 1]
            print(f"  Queueing profile pair {profile_idx}/{len(profile_pairs)}...")
            futures[executor.submit(
                profile_adaptive_pair,
                df_A,
                df_B,
                idx_a,
                idx_b,
                profile_idx,
                candidate_index,
                attributes,
                pair_feature_matrix[candidate_index],
                pair_similarity_scores[candidate_index],
                profile_scoring,
            )] = (profile_idx, idx_a, idx_b, candidate_index)

        completed_profiles = 0
        for future in as_completed(futures):
            profile_idx, idx_a, idx_b, candidate_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raw_importance, normalized_importance = normalize_importance_vector({}, attributes)
                result = {
                    "profile_index": profile_idx,
                    "candidate_index": candidate_index,
                    "indexA": idx_a,
                    "indexB": idx_b,
                    "feature_vector": pair_feature_matrix[candidate_index],
                    "raw_importance": raw_importance,
                    "normalized_importance": normalized_importance,
                    "similarity_score": float(pair_similarity_scores[candidate_index]),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cache_hit": False,
                    "profile_error": str(exc),
                }
                print(f"  Error profiling pair {profile_idx}/{len(profile_pairs)} ({idx_a}, {idx_b}): {exc}")

            profile_results.append(result)
            completed_profiles += 1
            print(
                f"  Completed profile pair {result['profile_index']}/{len(profile_pairs)} "
                f"({completed_profiles}/{len(profile_pairs)})"
            )

    profile_results.sort(key=lambda row: row["profile_index"])

    profile_features = []
    profile_importance_vectors = []
    raw_profile_importance_vectors = []
    profiling_rows = []
    for result in profile_results:
        for key in profiling_token_usage:
            profiling_token_usage[key] += int(result.get(key, 0))

        raw_importance = result["raw_importance"]
        normalized_importance = result["normalized_importance"]
        profile_features.append(result["feature_vector"])
        raw_profile_importance_vectors.append(raw_importance)
        profile_importance_vectors.append(normalized_importance)
        profiling_rows.append({
            "profile_index": result["profile_index"],
            "candidate_index": result["candidate_index"],
            "indexA": result["indexA"],
            "indexB": result["indexB"],
            "similarity_score": result["similarity_score"],
            "raw_importance": raw_importance,
            "normalized_importance": normalized_importance,
            "normalized_sum": sum(normalized_importance.values()),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "profile_error": result["profile_error"],
        })
        print(f"    Pair {result['profile_index']} raw importance: {raw_importance}")
        print(f"    Pair {result['profile_index']} normalized importance sum: {sum(normalized_importance.values()):.4f}")

    profiling_error_count = sum(1 for row in profiling_rows if row["profile_error"])
    profile_cache_hits = sum(1 for result in profile_results if result.get("cache_hit"))

    all_scores = [
        int(score)
        for importance in raw_profile_importance_vectors
        for score in importance.values()
    ]
    score_counts = collections.Counter(all_scores)
    # Counts how many attributes each profiled pair marked as carrying any evidence.
    # Comparable across scoring modes, unlike the 0/3 mass, which only means something
    # on the ordinal scale.
    marked_per_pair = [
        sum(1 for score in importance.values() if float(score) > 0)
        for importance in raw_profile_importance_vectors
    ]
    print(f"\n[Adaptive Profiling] Importance score distribution ({profile_scoring})")
    print(f"  Raw counts: {dict(sorted(score_counts.items()))}")
    if profile_scoring == "ordinal":
        print(
            "  0/3 mass: "
            f"{score_counts.get(0, 0) + score_counts.get(3, 0)}/{len(all_scores)} "
            f"({((score_counts.get(0, 0) + score_counts.get(3, 0)) / len(all_scores)) if all_scores else 0.0:.2%})"
        )
    print(
        "  Attributes marked per pair: "
        f"mean={np.mean(marked_per_pair):.2f} of {len(attributes)}"
        if marked_per_pair else "  Attributes marked per pair: n/a"
    )
    print("\n[Adaptive Profiling] Token usage")
    print(f"  Prompt tokens     : {profiling_token_usage['prompt_tokens']}")
    print(f"  Completion tokens : {profiling_token_usage['completion_tokens']}")
    print(f"  Total tokens      : {profiling_token_usage['total_tokens']}")
    print(f"  Profiling errors  : {profiling_error_count}/{len(profiling_rows)}")
    print(f"  Cache hits        : {profile_cache_hits}/{len(profiling_rows)}")

    selector = HybridAttributeSelector(
        attributes=attributes,
        top_k_attributes=top_k_attributes,
        min_k_attributes=min_k_attributes,
        max_k_attributes=max_k_attributes,
        cumulative_importance_threshold=cumulative_importance_threshold,
        selection_policy=selection_policy,
        ratio_lambda=ratio_lambda,
        required_attribute_roles=required_attribute_roles,
        attribute_roles=inferred_attribute_roles,
        top_k_retrieval=top_k_retrieval,
        fusion_strategy="weighted_average",
        predictor_weight=0.5,
        retrieval_weight=0.5,
    )
    selector.fit(np.asarray(profile_features), profile_importance_vectors)

    # The selector emits its subset ranked by predicted importance, and that order
    # reaches the matcher prompt verbatim (matcher.infer_pair rebuilds each record
    # by iterating the list). Since predicted importance shifts with the profile
    # sample, leaving it alone makes every run vary prompt field order alongside
    # field choice -- and on DBLP-ACM the order effect is the larger of the two
    # (permuting a fixed subset flips 5.4% of answers, versus 0.1% for an identical
    # prompt). Canonicalizing to table-column order keeps the subset intact and
    # leaves attribute choice as the only thing a reseed varies.
    canonical_position = {attr: pos for pos, attr in enumerate(attributes)}

    # The mask decides which attributes may be *chosen*; whether a masked-out attribute is
    # still *shown* is a separate question. Dropping it saves tokens but also hides from the
    # matcher that the field exists and is unknown for this pair -- and the matcher prompt
    # already instructs it to read a blank as unknown rather than as a mismatch, an
    # instruction that can only apply to a field the matcher can see. "keep" therefore shows
    # the attributes the unmasked ranking would have taken, blank side included, alongside
    # the ones actually selected; "drop" shows only the selection. Selection itself is
    # identical under both, so the two differ purely in what reaches the prompt.
    # A third mode follows from measuring the two kinds of blank separately. Re-running the
    # keep-vs-drop comparison split by blank type (25,526 affected Amazon-Walmart pairs, both
    # runs on one day) gives opposite answers: on the 12,894 pairs whose restored attribute was
    # blank on one side, drop is right 999 times against keep's 504 (p=7e-38); on the 12,560
    # where it was blank on both, keep is right 233 times against drop's 156 (p=1e-4). Showing
    # a half-filled field invites agreement it has not earned; showing that neither side has
    # the field is a genuine reason to doubt, and those pairs match at 0.11%. "both_blank"
    # keeps the second and drops the first.
    mask_mode = str(evidence_mask_mode).lower()
    show_masked = mask_mode in ("keep", "both_blank")
    selected_attributes_by_pair = {}
    selection_sizes = []
    display_sizes = []
    masked_shown_pairs = 0
    print("\n[Adaptive Transfer] Selecting attributes for all candidate pairs")
    shown = {"drop": "removed", "keep": "shown blank",
             "both_blank": "shown blank only when blank on both sides"}[mask_mode]
    print(f"[Adaptive Transfer] Masked attributes are {shown} in the matcher prompt")
    for pair_idx, (idx_a, idx_b) in enumerate(candidate_pairs):
        feature_vector = pair_feature_matrix[pair_idx]
        record_a = df_A.iloc[idx_a].to_dict()
        record_b = df_B.iloc[idx_b].to_dict()
        evidence_mask = attribute_evidence_mask(record_a, record_b, attributes)
        selection = selector.select_attributes(feature_vector, evidence_mask=evidence_mask)
        selected = list(selection["selected_attributes"])
        selection_sizes.append(len(selected))

        display = list(selected)
        if show_masked:
            # What the same rule would have taken with no mask applied. Only the attributes
            # the mask actually removed are added back, so a pair with no blanks is
            # unaffected and every mode shares its cache entry.
            unmasked = selector.select_attributes(feature_vector)["selected_attributes"]
            extra = [a for a in unmasked if a not in selected]
            if mask_mode == "both_blank":
                kind = dict(zip(attributes, attribute_blankness(record_a, record_b, attributes)))
                extra = [a for a in extra if kind.get(a) == "both_blank"]
            if extra:
                display.extend(extra)
                masked_shown_pairs += 1
        display = sorted(
            display, key=lambda attr: canonical_position.get(attr, len(canonical_position))
        )
        selected_attributes_by_pair[(idx_a, idx_b)] = display
        display_sizes.append(len(display))

    print(
        "Adaptive attributes per pair: "
        f"min={min(selection_sizes)}, "
        f"mean={np.mean(selection_sizes):.2f}, "
        f"max={max(selection_sizes)}"
    )
    if show_masked:
        print(
            "Attributes shown per pair (selection plus masked-out fields): "
            f"min={min(display_sizes)}, mean={np.mean(display_sizes):.2f}, "
            f"max={max(display_sizes)}; {masked_shown_pairs} pairs show at least one "
            f"masked attribute ({masked_shown_pairs / len(candidate_pairs) * 100:.1f}%)"
        )
    print(f"Profiled candidate indices: {profile_indices}")

    profiling_summary = {
        "selection_policy": {
            "top_k_attributes": top_k_attributes,
            "min_k_attributes": min_k_attributes,
            "evidence_mask_mode": evidence_mask_mode,
            "max_k_attributes": max_k_attributes,
            "cumulative_importance_threshold": cumulative_importance_threshold,
            "selection_policy": selection_policy,
            "ratio_lambda": ratio_lambda,
            "required_attribute_roles": list(required_attribute_roles or []),
            "attribute_roles": inferred_attribute_roles,
            "canonical_attribute_order": True,
        },
        "profile_sampler": profile_sampler,
        "profile_scoring": profile_scoring,
        "profile_max_workers": profile_max_workers,
        "sampling": sampling_summary,
        "profile_pairs": profile_pairs,
        "profile_indices": profile_indices,
        "score_counts": dict(sorted(score_counts.items())),
        # Only interpretable on the ordinal scale; left null for other scoring modes
        # so it cannot be read as a 0-3 statistic by mistake.
        "zero_three_mass": (
            score_counts.get(0, 0) + score_counts.get(3, 0)
            if profile_scoring == "ordinal" else None
        ),
        "score_total": len(all_scores),
        "marked_per_pair_mean": float(np.mean(marked_per_pair)) if marked_per_pair else 0.0,
        "marked_per_pair_min": int(min(marked_per_pair)) if marked_per_pair else 0,
        "marked_per_pair_max": int(max(marked_per_pair)) if marked_per_pair else 0,
        "profiling_token_usage": profiling_token_usage,
        "profiling_error_count": profiling_error_count,
        # Anything below the profiled-pair count means part of the sample was re-queried,
        # so this run's selector was not trained on the same responses as the run it is
        # being compared against.
        "profile_cache_hits": profile_cache_hits,
        "profiling_rows": profiling_rows,
        "selection_size_min": min(selection_sizes) if selection_sizes else 0,
        "selection_size_mean": float(np.mean(selection_sizes)) if selection_sizes else 0.0,
        "selection_size_max": max(selection_sizes) if selection_sizes else 0,
    }
    return selected_attributes_by_pair, selector, profiling_summary

def load_configured_dataset(config):
    base_path = config["base_path"]
    encoding = config.get("encoding", "utf-8")
    df_a = load_data(os.path.join(base_path, config["table_a"]), encoding=encoding)
    df_b = load_data(os.path.join(base_path, config["table_b"]), encoding=encoding)
    df_gt = load_data(os.path.join(base_path, config["ground_truth"]), encoding=encoding)

    print(f"Table A raw shape: {df_a.shape}")
    print(f"Table B raw shape: {df_b.shape}")

    id_a_to_pos, id_b_to_pos = build_id_maps(
        df_a,
        df_b,
        config["id_a_col"],
        config["id_b_col"],
    )
    gt_set = build_gt_set(
        df_gt,
        id_a_to_pos,
        id_b_to_pos,
        config["gt_id_a_col"],
        config["gt_id_b_col"],
    )

    df_a = df_a.drop(columns=[config["id_a_col"]]).copy()
    df_b = df_b.drop(columns=[config["id_b_col"]]).copy()
    if config.get("normalize_records", True):
        normalization_roles = config.get("normalization_roles")
        df_a = normalize_dataframe_records(df_a, normalization_roles=normalization_roles)
        df_b = normalize_dataframe_records(df_b, normalization_roles=normalization_roles)
    return df_a, df_b, gt_set


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_run_logs(dataset_name, result_df, summary, experiment_mode):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / dataset_name / "runs" / f"{run_id}_{experiment_mode}"
    suffix = 1
    while log_dir.exists():
        suffix += 1
        log_dir = LOG_ROOT / dataset_name / "runs" / f"{run_id}_{experiment_mode}_{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = log_dir / "final_results.csv"
    summary_path = log_dir / "run_summary.json"

    csv_df = result_df.copy()
    if "attempts" in csv_df:
        csv_df["attempts"] = csv_df["attempts"].map(json.dumps)
    csv_df.to_csv(result_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print(f"\nSaved final pair results to {result_path}")
    print(f"Saved run summary to {summary_path}")


@contextmanager
def timed_stage(timings, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
        print(f"[Timing] {name}: {elapsed:.2f}s")


def run_pipeline(
    dataset_name,
    experiment_mode="step15",
    model_name=DEFAULT_MODEL_NAME,
    tuple_strategy=None,
    tuple_threshold=None,
    tuple_top_k=None,
    tuple_n_pos=10,
    tuple_n_neg=10,
    manual_attributes=None,
    exclude_attributes=None,
    profile_sampler=None,
    profile_sample_size=None,
    use_matcher_cache=True,
    use_profile_cache=True,
    step1_reblock=True,
    random_seed=RANDOM_SEED,
    blocking_seed=None,
    allow_api_errors=False,
    selection_policy=DEFAULT_SELECTION_POLICY,
    ratio_lambda=DEFAULT_RATIO_LAMBDA,
    profile_scoring=DEFAULT_PROFILE_SCORING,
    min_k_attributes=None,
    max_k_attributes=None,
    evidence_mask_mode="drop",
    max_candidate_pairs=None,
    candidate_sample_seed=7,
    matcher_cache_dir=None,
    cumulative_threshold=None,
    matcher_checkpoint_dir=None,
):
    # Blocking keeps its own seed so that repeated-seed experiments vary only the
    # sampling stages and compare samplers over an identical candidate set.
    blocking_seed = RANDOM_SEED if blocking_seed is None else int(blocking_seed)
    random_seed = int(random_seed)
    if matcher_checkpoint_dir and use_matcher_cache:
        raise ValueError("--matcher-checkpoint-dir requires --disable-matcher-cache")
    config = DATASET_CONFIGS[dataset_name]
    blocking = config["blocking"]
    adaptive = config["adaptive"]
    profile_sampler = profile_sampler or adaptive.get("profile_sampler", DEFAULT_PROFILE_SAMPLER)
    resolved_profile_sample_size = (
        adaptive["profile_sample_size"]
        if profile_sample_size is None
        else int(profile_sample_size)
    )
    # The floor on how many attributes a pair keeps, and the cumulative-importance
    # target that normally decides the count, are both overridable per run. They set
    # how aggressively Step 1.5 may condense: with the defaults (floor 2, target 0.8)
    # a pair can never be reduced to a single attribute, even when the profiling
    # signal concentrates on one -- which a global baseline restricted to one
    # attribute is free to do, making the two not directly comparable.
    resolved_min_k = (
        adaptive.get("min_k_attributes", 2) if min_k_attributes is None else int(min_k_attributes)
    )
    # The ceiling is resolved the same way, with one extra option the floor does not
    # need: a non-positive value removes the cap entirely. That matters for "cumulative"
    # on a wide schema, where the target is only reachable by keeping more attributes
    # than the default cap of 5 allows, so the cap -- not the target -- ends up deciding
    # every count. Uncapping lets the rule express what it actually wants to select.
    if max_k_attributes is None:
        resolved_max_k = adaptive.get("max_k_attributes", 5)
    else:
        resolved_max_k = int(max_k_attributes)
        if resolved_max_k <= 0:
            resolved_max_k = None
    resolved_cumulative_threshold = (
        adaptive.get("cumulative_importance_threshold", 0.8)
        if cumulative_threshold is None
        else float(cumulative_threshold)
    )
    timings = {}
    pipeline_start = time.perf_counter()

    print(f"\n=== Running ER pipeline: {dataset_name} ===")
    print(f"Experiment mode: {experiment_mode}")
    print(f"Embedding model: {model_name}")
    print(f"LLM model: {get_llm_model()}")
    if tuple_strategy is None:
        tuple_strategy = "llm_guided" if experiment_mode == "step1" else "none"
    if experiment_mode == "full" and tuple_strategy != "none":
        raise ValueError("--mode full cannot use global tuple condensation; use --mode step1 instead.")
    print(f"Global tuple strategy: {tuple_strategy}")
    print(f"Blocking hyperparameters: {blocking}")
    if experiment_mode == "step15":
        print(f"Adaptive hyperparameters: {adaptive}")
        print(f"Adaptive profile sampler: {profile_sampler}")
        print(f"Adaptive profile scoring: {profile_scoring}")
        print(f"Adaptive profile sample size: {resolved_profile_sample_size}")
    print(f"Random seed: {random_seed} (blocking seed: {blocking_seed})")
    # A repeat run needs fresh calls, which --disable-matcher-cache achieves by neither
    # reading nor writing. That also means a run killed at 99% loses every answer it paid
    # for. Pointing each repeat at its own cache directory gives the same freshness -- the
    # directory starts empty -- while still saving as it goes, so a restart replays instead
    # of re-billing.
    if matcher_cache_dir:
        config = {**config, "cache_dir": matcher_cache_dir}
    set_cache_dir(config["cache_dir"])
    set_cache_enabled(use_matcher_cache)
    print(f"Matcher cache dir: {config['cache_dir']}")
    print(f"Matcher cache enabled: {use_matcher_cache}")
    if matcher_checkpoint_dir:
        print(f"Matcher checkpoint dir: {matcher_checkpoint_dir}")
    # Kept separate from the matcher cache: --disable-matcher-cache exists to time
    # matching honestly, whereas the profiling cache exists to hold the training sample
    # fixed across runs being compared. Disabling one should not disable the other.
    profile_cache_dir = os.path.join(config["cache_dir"], "profiling")
    set_profile_cache_dir(profile_cache_dir)
    set_profile_cache_enabled(use_profile_cache)
    if experiment_mode == "step15":
        print(f"Profiling cache dir: {profile_cache_dir}")
        print(f"Profiling cache enabled: {use_profile_cache}")

    with timed_stage(timings, "load_data"):
        df_a, df_b, gt_set = load_configured_dataset(config)
        df_a, df_b, excluded_attributes = drop_excluded_attributes(df_a, df_b, exclude_attributes)
    if excluded_attributes:
        print("Excluded attributes:", excluded_attributes)

    with timed_stage(timings, "load_embedding_model"):
        model = SentenceTransformer(model_name)
        np.random.seed(random_seed)

    with timed_stage(timings, "blocking_full_attributes"):
        table_a_vectors, table_b_vectors, candidate_pairs, pair_similarity_scores = run_lsh_blocking(
            df_a,
            df_b,
            model,
            blocking,
            model_name=model_name,
            seed=blocking_seed,
            label="full-attribute",
        )
    print("Number of full-attribute candidate pairs after top-k:", len(candidate_pairs))

    global_tuple_summary = {}
    if experiment_mode in ("step1", "step15") and tuple_strategy != "none":
        with timed_stage(timings, "global_tuple_selection"):
            df_a, df_b, global_tuple_summary = apply_global_tuple_strategy(
                df_a,
                df_b,
                tuple_strategy,
                gt_set,
                candidate_pairs,
                threshold=tuple_threshold,
                top_k=tuple_top_k,
                n_pos=tuple_n_pos,
                n_neg=tuple_n_neg,
                manual_attributes=manual_attributes,
                seed=random_seed,
            )
        # Re-blocking is Step 1's original semantics: if the claim is that only these
        # attributes matter, the whole pipeline including candidate generation should
        # run on them. It also confounds the comparison against Step 1.5, which leaves
        # blocking untouched. Measured on DBLP-ACM with the supervised strategy, dropping
        # to [title, authors] replaced 8,786 of 13,080 candidates; those replacements
        # contained no ground-truth match at all yet drew 600 false positives, more than
        # half of that run's total. Skipping the re-block holds the candidate set fixed
        # so that attribute choice is the only variable.
        if step1_reblock:
            with timed_stage(timings, "blocking_condensed_attributes"):
                table_a_vectors, table_b_vectors, candidate_pairs, pair_similarity_scores = run_lsh_blocking(
                    df_a,
                    df_b,
                    model,
                    blocking,
                    seed=blocking_seed,
                    label=f"{tuple_strategy}-condensed",
                    model_name=model_name,
                )
            print("Number of condensed candidate pairs after top-k:", len(candidate_pairs))
        else:
            print("Skipping re-block: keeping the full-attribute candidate set of "
                  f"{len(candidate_pairs)} pairs")

    print("Blocking attributes used:", df_a.columns.tolist())
    if experiment_mode == "step15":
        print("Adaptive candidate attributes:", df_a.columns.tolist())

    total_pairs = table_a_vectors.shape[0] * table_b_vectors.shape[0]
    candidate_pairs_count = len(candidate_pairs)
    reduction_percentage = (total_pairs - candidate_pairs_count) / total_pairs * 100

    print(f"Reduction percentage: {reduction_percentage:.2f}%")

    cand_set = set(candidate_pairs)
    found = sum(1 for pair in gt_set if pair in cand_set)
    pc = found / len(gt_set) if gt_set else 0.0
    print(f"True matches:      {len(gt_set)}")
    print(f"Found in blocking: {found}")
    print(f"Pair Completeness: {pc:.4f}")
    print(f"Candidates:        {len(candidate_pairs)}")

    # Restricting the run to a fixed random slice of the candidate set. This exists to make
    # repetition affordable: measuring how much the matcher's answers move between two runs
    # of one configuration needs many runs, not many pairs, and on Amazon-Walmart a full
    # candidate set is 88,296 calls per run.
    #
    # The slice is drawn from its own seed, independent of the blocking seed, so that every
    # configuration and every repeat sees the *same* pairs. A slice redrawn per run would mix
    # sampling noise into the very quantity being measured.
    #
    # Ground truth is restricted to the matches inside the slice, so precision and recall
    # describe this sub-experiment rather than the dataset. Pair completeness above is
    # reported on the full blocking output and is unaffected. Levels from a sliced run are
    # therefore not comparable with a full run -- only spreads across repeats are.
    candidate_subset_summary = None
    if max_candidate_pairs is not None and 0 < int(max_candidate_pairs) < len(candidate_pairs):
        n_before, gt_before = len(candidate_pairs), len(gt_set)
        score_lookup = dict(
            zip(candidate_pairs,
                aligned_pair_similarity_scores(candidate_pairs, pair_similarity_scores))
        )
        rng = np.random.RandomState(candidate_sample_seed)
        picked = rng.choice(n_before, size=int(max_candidate_pairs), replace=False)
        picked.sort()  # preserve blocking order, which downstream ordinal indexing assumes
        candidate_pairs = [candidate_pairs[i] for i in picked]
        pair_similarity_scores = {pair: score_lookup[pair] for pair in candidate_pairs}
        gt_set = {pair for pair in gt_set if pair in set(candidate_pairs)}
        candidate_subset_summary = {
            "max_candidate_pairs": int(max_candidate_pairs),
            "candidate_sample_seed": int(candidate_sample_seed),
            "candidate_pairs_before": n_before,
            "candidate_pairs_after": len(candidate_pairs),
            "ground_truth_before": gt_before,
            "ground_truth_in_subset": len(gt_set),
        }
        print(f"\n[Subset] Sampled {len(candidate_pairs)} of {n_before} candidate pairs "
              f"(seed {candidate_sample_seed})")
        print(f"[Subset] Ground-truth matches inside the subset: {len(gt_set)} of {gt_before}")
        print("[Subset] Metrics below describe this subset only and are not comparable "
              "with a full-candidate run.")

    selected_attributes_by_pair = None
    profiling_summary = {}
    if experiment_mode == "step15":
        adaptive_embedding_model = model if ADAPTIVE_USE_SEMANTIC_FEATURES else None
        with timed_stage(timings, "adaptive_profiling_and_transfer"):
            selected_attributes_by_pair, _, profiling_summary = build_adaptive_attribute_map(
                df_a,
                df_b,
                candidate_pairs,
                pair_similarity_scores,
                embedding_model=adaptive_embedding_model,
                profile_sample_size=resolved_profile_sample_size,
                profile_sampler=profile_sampler,
                profile_max_workers=adaptive.get("profile_max_workers", DEFAULT_PROFILE_MAX_WORKERS),
                top_k_retrieval=adaptive["top_k_retrieval"],
                top_k_attributes=adaptive.get("top_k_attributes"),
                min_k_attributes=resolved_min_k,
                max_k_attributes=resolved_max_k,
                evidence_mask_mode=evidence_mask_mode,
                cumulative_importance_threshold=resolved_cumulative_threshold,
                selection_policy=selection_policy,
                ratio_lambda=ratio_lambda,
                profile_scoring=profile_scoring,
                required_attribute_roles=adaptive.get("required_attribute_roles"),
                attribute_roles=adaptive.get("attribute_roles"),
                seed=random_seed,
            )

    with timed_stage(timings, "matching"):
        result_df = infer_candidates_pairwise(
            df_a,
            df_b,
            candidate_pairs,
            max_workers=16,
            selected_attributes_by_pair=selected_attributes_by_pair,
            checkpoint_dir=matcher_checkpoint_dir,
        )

    api_error_count = check_api_errors(result_df, allow_api_errors)

    with timed_stage(timings, "metrics"):
        y_pred = []
        y_true = []
        for _, row in result_df.iterrows():
            i, j = int(row["indexA"]), int(row["indexB"])
            y_pred.append(1 if row["answer"] == "Yes" else 0)
            y_true.append(1 if (i, j) in gt_set else 0)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_gt_matches = len(gt_set)
        end_to_end_precision = tp / (tp + fp) if (tp + fp) else 0.0
        end_to_end_recall = tp / total_gt_matches if total_gt_matches else 0.0
        end_to_end_f1 = (
            2 * end_to_end_precision * end_to_end_recall / (end_to_end_precision + end_to_end_recall)
            if (end_to_end_precision + end_to_end_recall)
            else 0.0
        )

    total_prompt_tokens = result_df["prompt_tokens"].sum()
    total_completion_tokens = result_df["completion_tokens"].sum()
    total_tokens = result_df["total_tokens"].sum()
    cache_hits = int(result_df["cache_hit"].sum()) if "cache_hit" in result_df else 0
    checkpoint_hits = int(result_df["checkpoint_hit"].sum()) if "checkpoint_hit" in result_df else 0
    attempt_rows = [a for attempts in result_df.get("attempts", []) for a in attempts]
    retry_tokens = sum(int(a.get("total_tokens", 0)) for a in attempt_rows if a.get("error"))
    response_times = [a[key] for a in attempt_rows for key in ("started_at", "finished_at") if a.get(key)]

    # Which upstream served this run. One model slug is routed across several providers
    # whose answers differ, and that mix is the leading explanation for two runs of one
    # configuration disagreeing on 5% of pairs. Counted over the calls actually issued:
    # a cache hit replays an earlier run's provider and would otherwise be attributed to
    # this one.
    provider_counts = {}
    if "provider" in result_df:
        fresh = result_df[~result_df["cache_hit"].astype(bool)] if "cache_hit" in result_df else result_df
        provider_counts = {
            str(k): int(v)
            for k, v in fresh["provider"].fillna("").replace("", "unreported").value_counts().items()
        }

    profiling_token_usage = profiling_summary.get("profiling_token_usage", {})
    profiling_prompt_tokens = int(profiling_token_usage.get("prompt_tokens", 0))
    profiling_completion_tokens = int(profiling_token_usage.get("completion_tokens", 0))
    profiling_total_tokens = int(profiling_token_usage.get("total_tokens", 0))
    step1_token_usage = global_tuple_summary.get("selection_token_usage", {})
    step1_prompt_tokens = int(step1_token_usage.get("prompt_tokens", 0))
    step1_completion_tokens = int(step1_token_usage.get("completion_tokens", 0))
    step1_total_tokens = int(step1_token_usage.get("total_tokens", 0))
    pipeline_prompt_tokens = int(total_prompt_tokens) + profiling_prompt_tokens + step1_prompt_tokens
    pipeline_completion_tokens = int(total_completion_tokens) + profiling_completion_tokens + step1_completion_tokens
    pipeline_total_tokens = int(total_tokens) + profiling_total_tokens + step1_total_tokens

    if provider_counts:
        issued = sum(provider_counts.values())
        shares = ", ".join(
            f"{name} {count} ({count / issued:.1%})"
            for name, count in sorted(provider_counts.items(), key=lambda kv: -kv[1])
        )
        print(f"\nServed by (of {issued} pair answers issued for this draw, including resumed pairs): {shares}")

    print("\n--- Token Usage ---")
    print(f"Step 1 prompt tokens       : {step1_prompt_tokens}")
    print(f"Step 1 completion tokens   : {step1_completion_tokens}")
    print(f"Step 1 total tokens        : {step1_total_tokens}")
    print(f"Profiling prompt tokens     : {profiling_prompt_tokens}")
    print(f"Profiling completion tokens : {profiling_completion_tokens}")
    print(f"Profiling total tokens      : {profiling_total_tokens}")
    print(f"Matching prompt tokens      : {total_prompt_tokens}")
    print(f"Matching completion tokens  : {total_completion_tokens}")
    print(f"Matching total tokens       : {total_tokens}")
    print(f"Pipeline prompt tokens      : {pipeline_prompt_tokens}")
    print(f"Pipeline completion tokens  : {pipeline_completion_tokens}")
    print(f"Pipeline total tokens       : {pipeline_total_tokens}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"Candidate precision : {precision:.4f}")
    print(f"Candidate recall    : {recall:.4f}")
    print(f"Candidate F1        : {f1:.4f}")
    print(f"End-to-end precision: {end_to_end_precision:.4f}")
    print(f"End-to-end recall   : {end_to_end_recall:.4f}")
    print(f"End-to-end F1       : {end_to_end_f1:.4f}")
    print(f"Cache hits: {cache_hits}/{len(result_df)}")
    print(f"Checkpoint hits: {checkpoint_hits}/{len(result_df)}")
    print(f"Reported tokens from failed attempts (included above): {retry_tokens}")
    print(f"LLM API errors: {api_error_count}/{len(result_df)}")

    cand_set = set(zip(result_df["indexA"], result_df["indexB"]))
    in_cands = sum(1 for pair in gt_set if pair in cand_set)
    found_llm = tp

    print(f"\nTrue matches total         : {total_gt_matches}")
    print(f"True matches in candidates : {in_cands}  (blocking recall)")
    print(f"True matches found by LLM  : {found_llm}  (end-to-end TP)")
    timings["total_before_logging"] = time.perf_counter() - pipeline_start

    summary = {
        "dataset": dataset_name,
        # The exact invocation, so a pre-registered run can be replayed verbatim
        # rather than reconstructed from the flags that happen to be summarised.
        "invocation": " ".join(shlex.quote(a) for a in sys.argv),
        "experiment_mode": experiment_mode,
        "model_name": model_name,
        "llm_model": get_llm_model(),
        "random_seed": random_seed,
        "blocking_seed": blocking_seed,
        "config": config,
        "attribute_selection": {
            "mode": experiment_mode,
            "global_tuple_strategy": tuple_strategy,
            "profile_sampler": profile_sampler if experiment_mode == "step15" else None,
            "profile_scoring": profile_scoring if experiment_mode == "step15" else None,
            "profile_sample_size": resolved_profile_sample_size if experiment_mode == "step15" else None,
            "attributes_used": df_a.columns.tolist(),
            "attribute_count": len(df_a.columns),
            "excluded_attributes": excluded_attributes,
            "step1_reblock": bool(step1_reblock) if experiment_mode == "step1" else None,
            "global_tuple_condensation": global_tuple_summary,
        },
        "runtime_controls": {
            "matcher_cache_enabled": bool(use_matcher_cache),
            "matcher_checkpoint_dir": str(matcher_checkpoint_dir) if matcher_checkpoint_dir else None,
            "profile_cache_enabled": bool(use_profile_cache),
            # Reasoning changes what the LLM does per call, so runs recorded under
            # different settings are not comparable.
            "llm_reasoning_enabled": bool(reasoning_enabled()),
            # Empty means OpenRouter chose the upstream host per call, so the run was
            # served by a mix of providers that differ in quantization.
            "llm_provider_order": provider_order(),
        },
        "blocking": {
            "total_pairs": total_pairs,
            "candidate_pairs": candidate_pairs_count,
            "reduction_percentage": reduction_percentage,
            "gt_matches": len(gt_set),
            "found_matches": found,
            "pair_completeness": pc,
            # Present only on sliced runs. Its presence is what marks a run's metrics as
            # describing a subset, so anything reading these summaries must check for it
            # before comparing levels against a full run.
            "candidate_subset": candidate_subset_summary,
        },
        "adaptive": profiling_summary,
        "matching": {
            "num_result_pairs": len(result_df),
            "prompt_tokens": int(total_prompt_tokens),
            "completion_tokens": int(total_completion_tokens),
            "total_tokens": int(total_tokens),
            "cache_hits": cache_hits,
            "checkpoint_hits": checkpoint_hits,
            "providers": provider_counts,
            "attempt_providers": dict(collections.Counter(
                a.get("provider") or "unreported" for a in attempt_rows
            )),
            "request_attempts": len(attempt_rows),
            "failed_attempts": sum(bool(a.get("error")) for a in attempt_rows),
            "reported_failed_attempt_tokens": retry_tokens,
            "unreported_usage_attempts": sum(not a.get("usage_reported", True) for a in attempt_rows),
            "requests_started_at": min(response_times) if response_times else None,
            "requests_finished_at": max(response_times) if response_times else None,
            "api_errors": api_error_count,
            # Only ever true when --allow-api-errors was passed: without it a run with
            # any API error raises before reaching this point. Downstream readers should
            # treat these metrics as unusable.
            "metrics_invalid": bool(api_error_count),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "candidate_precision": float(precision),
            "candidate_recall": float(recall),
            "candidate_f1": float(f1),
            "end_to_end_precision": float(end_to_end_precision),
            "end_to_end_recall": float(end_to_end_recall),
            "end_to_end_f1": float(end_to_end_f1),
            "true_matches_in_candidates": in_cands,
            "true_matches_found_by_llm": int(found_llm),
        },
        "token_usage": {
            "step1_prompt_tokens": step1_prompt_tokens,
            "step1_completion_tokens": step1_completion_tokens,
            "step1_total_tokens": step1_total_tokens,
            "profiling_prompt_tokens": profiling_prompt_tokens,
            "profiling_completion_tokens": profiling_completion_tokens,
            "profiling_total_tokens": profiling_total_tokens,
            "matching_prompt_tokens": int(total_prompt_tokens),
            "matching_completion_tokens": int(total_completion_tokens),
            "matching_total_tokens": int(total_tokens),
            "pipeline_prompt_tokens": pipeline_prompt_tokens,
            "pipeline_completion_tokens": pipeline_completion_tokens,
            "pipeline_total_tokens": pipeline_total_tokens,
        },
        "timing": {
            **{key: float(value) for key, value in timings.items()},
        },
    }
    log_mode = f"{experiment_mode}_{profile_sampler}" if experiment_mode == "step15" else experiment_mode
    # Non-default variants go in the directory name so runs that differ only by a flag
    # stay distinguishable without opening every run_summary.json.
    if experiment_mode == "step15" and profile_scoring != DEFAULT_PROFILE_SCORING:
        log_mode = f"{log_mode}_{profile_scoring}"
    if experiment_mode == "step15" and selection_policy != DEFAULT_SELECTION_POLICY:
        log_mode = f"{log_mode}_{selection_policy}"
    if experiment_mode == "step15" and min_k_attributes is not None:
        log_mode = f"{log_mode}_mink{resolved_min_k}"
    if experiment_mode == "step15" and cumulative_threshold is not None:
        log_mode = f"{log_mode}_thr{resolved_cumulative_threshold:g}"
    if experiment_mode == "step15" and evidence_mask_mode != "drop":
        log_mode = f"{log_mode}_mask{evidence_mask_mode}"
    if candidate_subset_summary is not None:
        log_mode = f"{log_mode}_sub{candidate_subset_summary['candidate_pairs_after']}"
    if experiment_mode == "step1" and not step1_reblock:
        log_mode = f"{log_mode}_noreblock"
    if random_seed != RANDOM_SEED:
        log_mode = f"{log_mode}_seed{random_seed}"
    with timed_stage(timings, "write_logs"):
        write_run_logs(dataset_name, result_df, summary, log_mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Run configurable ER experiments for one configured dataset.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="DBLP-ACM",
        help="Dataset configuration to run.",
    )
    parser.add_argument(
        "--mode",
        choices=EXPERIMENT_MODES,
        default="step15",
        help=(
            "Experiment mode: full = full attributes, "
            "step1 = one global LLM-guided attribute subset, "
            "step15 = adaptive per-pair attribute selection."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name used for blocking embeddings.",
    )
    parser.add_argument(
        "--tuple-strategy",
        choices=TUPLE_STRATEGIES,
        default=None,
        help=(
            "Global tuple-condensation strategy. Defaults to llm_guided for --mode step1 "
            "and none for --mode full/step15."
        ),
    )
    parser.add_argument(
        "--manual-attributes",
        default="",
        help="Comma-separated attributes for --tuple-strategy manual.",
    )
    parser.add_argument(
        "--exclude-attributes",
        default="",
        help="Comma-separated attributes to drop before blocking, selection, and matching.",
    )
    parser.add_argument(
        "--profile-sampler",
        choices=PROFILE_SAMPLERS,
        default=None,
        help=(
            "Step-1.5 profiling pair sampler. Use random as the baseline; "
            "the default is similarity_stratified."
        ),
    )
    parser.add_argument(
        "--profile-sample-size",
        type=int,
        default=None,
        help="Override the dataset default number of Step-1.5 profile pairs.",
    )
    parser.add_argument(
        "--tuple-threshold",
        type=float,
        default=None,
        help=(
            "Absolute importance cut-off for the llm_guided or supervised tuple "
            "strategies. Left unset, the supervised strategy uses a bar of 1.2 times the "
            "uniform share 1/m instead, which is scale-free: a fixed cut-off selects "
            "nothing once the schema is wide enough that no attribute's normalised score "
            "can reach it."
        ),
    )
    parser.add_argument(
        "--tuple-top-k",
        type=int,
        default=None,
        help="Maximum number of attributes to keep for llm_guided or supervised tuple strategies.",
    )
    parser.add_argument(
        "--tuple-n-pos",
        type=int,
        default=10,
        help="Positive labeled pairs sampled for global tuple selection.",
    )
    parser.add_argument(
        "--tuple-n-neg",
        type=int,
        default=10,
        help="Hard negative pairs sampled for global tuple selection.",
    )
    parser.add_argument(
        "--step1-threshold",
        type=float,
        default=0.3,
        help="Deprecated alias for --tuple-threshold.",
    )
    parser.add_argument(
        "--step1-n-pos",
        type=int,
        default=10,
        help="Deprecated alias for --tuple-n-pos.",
    )
    parser.add_argument(
        "--step1-n-neg",
        type=int,
        default=10,
        help="Deprecated alias for --tuple-n-neg.",
    )
    parser.add_argument(
        "--disable-matcher-cache",
        action="store_true",
        help="Disable matcher cache reads and writes for controlled runtime experiments.",
    )
    parser.add_argument(
        "--step1-no-reblock",
        action="store_true",
        help=(
            "Keep the full-attribute candidate set after Step-1 condensation instead of "
            "re-blocking on the condensed attributes. Re-blocking is the default and the "
            "original semantics, but it changes which pairs are compared -- on DBLP-ACM it "
            "replaced two thirds of the candidate set with pairs containing no true match "
            "-- so a like-for-like comparison against Step 1.5 needs this flag."
        ),
    )
    parser.add_argument(
        "--disable-profile-cache",
        action="store_true",
        help=(
            "Disable the Step-1.5 profiling cache. On by default, and separate from the "
            "matcher cache: identical profiling prompts are not stable across runs "
            "(8/20 pairs changed on a re-query), so caching them is what keeps a "
            "selection-policy or scoring A/B varying only the thing under test. Disable "
            "it to measure that instability instead of controlling for it."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default=DEFAULT_SELECTION_POLICY,
        help=(
            "Step-1.5 rule for how many attributes to keep. 'cumulative' (default) sums "
            "importance up to a fixed target and is not scale-free -- it saturates at "
            "max_k on wide schemas. 'ratio' keeps attributes carrying at least "
            "--ratio-lambda times the uniform share and adapts to schema width. "
            "'gap' cuts at the largest drop in the ranked importance profile."
        ),
    )
    parser.add_argument(
        "--ratio-lambda",
        type=float,
        default=DEFAULT_RATIO_LAMBDA,
        help="Multiple of the uniform share 1/n required by --selection-policy ratio.",
    )
    parser.add_argument(
        "--profile-scoring",
        choices=PROFILE_SCORING_MODES,
        default=DEFAULT_PROFILE_SCORING,
        help=(
            "How Step-1.5 profiling asks the LLM for per-attribute importance. "
            "'ordinal' (default) requests a 0-3 score per attribute. 'decisive' requests "
            "the decisive attribute subset directly, which lets the model weigh "
            "attributes against each other instead of scoring each in isolation."
        ),
    )
    parser.add_argument(
        "--min-k-attributes",
        type=int,
        default=None,
        help=(
            "Minimum attributes Step 1.5 keeps per pair (default 2, from the dataset "
            "config). Set to 1 to let a pair be condensed to a single attribute, which "
            "is what a global baseline selecting one attribute is allowed to do."
        ),
    )
    parser.add_argument(
        "--max-k-attributes",
        type=int,
        default=None,
        help=(
            "Maximum attributes Step 1.5 keeps per pair (default 5, from the dataset "
            "config). Pass 0 or a negative value to remove the cap. On a wide schema "
            "the cap is what stops --selection-policy cumulative from reaching its "
            "target, so uncapping is required to test that policy there."
        ),
    )
    parser.add_argument(
        "--matcher-cache-dir",
        default=None,
        help=(
            "Override the matcher response cache directory. Give each repeat of a "
            "configuration its own directory to get fresh calls without losing them if the "
            "run dies: an empty directory forces real calls, and answers are still saved."
        ),
    )
    parser.add_argument(
        "--matcher-checkpoint-dir",
        default=None,
        help=(
            "Durably save and resume one independent matcher draw by pair index. "
            "Requires --disable-matcher-cache. Use a different directory for each draw; "
            "data, pair selections and request settings must match when resuming."
        ),
    )
    parser.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=None,
        help=(
            "Run on a random slice of this many candidate pairs instead of all of them, so "
            "that repeating a configuration is affordable. Ground truth is restricted to the "
            "matches inside the slice, so metrics describe the slice and their levels are not "
            "comparable with a full run; spreads across repeats are."
        ),
    )
    parser.add_argument(
        "--candidate-sample-seed",
        type=int,
        default=7,
        help=(
            "Seed for --max-candidate-pairs. Held fixed across runs on purpose: every "
            "configuration and repeat must see the same pairs, or sampling noise is measured "
            "alongside matcher noise."
        ),
    )
    parser.add_argument(
        "--evidence-mask-mode",
        choices=["drop", "keep", "both_blank"],
        default="drop",
        help=(
            "What to do with an attribute the evidence mask rules out because it is blank "
            "on at least one side. 'drop' (default) removes it from the matcher prompt. "
            "'keep' still bars it from being selected but shows it, blank, alongside the "
            "selection, so the matcher can read it as unknown rather than not knowing the "
            "field exists. 'both_blank' shows it only when neither side has the field, which "
            "is the case that predicts a non-match; a field only one side is missing is still "
            "removed. Selection is identical under all three."
        ),
    )
    parser.add_argument(
        "--cumulative-threshold",
        type=float,
        default=None,
        help=(
            "Importance mass --selection-policy cumulative must reach before it stops "
            "adding attributes (default 0.8). This, not --min-k-attributes, is usually "
            "what fixes the count: a pair keeps one attribute only if that attribute "
            "alone carries the whole target."
        ),
    )
    parser.add_argument(
        "--allow-api-errors",
        action="store_true",
        help=(
            "Write the run even if some LLM calls failed. Off by default: failed calls "
            "are scored as non-matches, so the metrics would be meaningless. The run "
            "summary is marked metrics_invalid when this is used."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=(
            "Random seed for the sampling stages (Step-1.5 profile-pair selection, "
            "Step-1 labeled-pair sampling). Blocking is unaffected, so repeated seeds "
            "compare samplers over an identical candidate set. Runs with a non-default "
            "seed are logged to a directory suffixed with _seed<N>."
        ),
    )
    parser.add_argument(
        "--blocking-seed",
        type=int,
        default=None,
        help=(
            "Seed for the LSH random hyperplanes. Defaults to 42 regardless of --seed. "
            "Change this only to measure blocking variance, since it alters the "
            "candidate-pair set."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tuple_threshold = args.tuple_threshold
    tuple_n_pos = args.tuple_n_pos
    tuple_n_neg = args.tuple_n_neg
    if args.step1_threshold != 0.3 and args.tuple_threshold is None:
        tuple_threshold = args.step1_threshold
    if args.step1_n_pos != 10 and args.tuple_n_pos == 10:
        tuple_n_pos = args.step1_n_pos
    if args.step1_n_neg != 10 and args.tuple_n_neg == 10:
        tuple_n_neg = args.step1_n_neg

    run_pipeline(
        args.dataset,
        experiment_mode=args.mode,
        model_name=args.embedding_model,
        tuple_strategy=args.tuple_strategy,
        tuple_threshold=tuple_threshold,
        tuple_top_k=args.tuple_top_k,
        tuple_n_pos=tuple_n_pos,
        tuple_n_neg=tuple_n_neg,
        manual_attributes=parse_manual_attributes(args.manual_attributes),
        exclude_attributes=parse_manual_attributes(args.exclude_attributes),
        profile_sampler=args.profile_sampler,
        profile_sample_size=args.profile_sample_size,
        use_matcher_cache=not args.disable_matcher_cache,
        use_profile_cache=not args.disable_profile_cache,
        step1_reblock=not args.step1_no_reblock,
        random_seed=args.seed,
        blocking_seed=args.blocking_seed,
        allow_api_errors=args.allow_api_errors,
        selection_policy=args.selection_policy,
        ratio_lambda=args.ratio_lambda,
        profile_scoring=args.profile_scoring,
        min_k_attributes=args.min_k_attributes,
        max_k_attributes=args.max_k_attributes,
        evidence_mask_mode=args.evidence_mask_mode,
        max_candidate_pairs=args.max_candidate_pairs,
        candidate_sample_seed=args.candidate_sample_seed,
        matcher_cache_dir=args.matcher_cache_dir,
        cumulative_threshold=args.cumulative_threshold,
        matcher_checkpoint_dir=args.matcher_checkpoint_dir,
    )


if __name__ == "__main__":
    main()
