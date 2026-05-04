import argparse
import collections
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from .lsh import create_random_planes, query_lsh_fast
from .utils import build_id_maps, build_gt_set
from .loader import load_data
from .attribute_selection import (
    HybridAttributeSelector,
    compute_attribute_features,
    flatten_pair_features,
    infer_attribute_roles,
    select_profile_pairs,
)
from .attribute_selection.llm_guided import query_llm_adaptive_attribute_importance_with_usage
from .matcher import infer_candidates_pairwise, set_cache_dir
from .embeddings import embed_dataframe_sbert
from .preprocessing import normalize_dataframe_records


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
            "profile_sample_size": 20,
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
        "blocking": {"num_tables": 10, "num_planes": 4, "num_flips": 1, "top_k": 5},
        "adaptive": {
            "profile_sample_size": 50,
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
            "profile_sample_size": 50,
            "top_k_retrieval": 5,
            "min_k_attributes": 2,
            "max_k_attributes": 5,
            "cumulative_importance_threshold": 0.8,
            "required_attribute_roles": ["identity"],
        },
    },
}

MODEL_NAME = "all-mpnet-base-v2"
ENABLE_ADAPTIVE_TRANSFER = True
ADAPTIVE_USE_SEMANTIC_FEATURES = False
RANDOM_SEED = 42
LOG_ROOT = Path("logs")


def compute_candidate_pair_scores(candidate_pairs, tableA_vectors, tableB_vectors):
    """Compute cosine-style scores for already-normalized blocked pairs."""
    return np.asarray([
        float(np.dot(tableA_vectors[i], tableB_vectors[j]))
        for i, j in candidate_pairs
    ])


def pair_feature_vector(df_A, df_B, idx_a, idx_b, attributes, embedding_model=None):
    """Build the flattened adaptive feature vector for one candidate pair."""
    feature_dict = compute_attribute_features(
        df_A.iloc[idx_a].to_dict(),
        df_B.iloc[idx_b].to_dict(),
        attributes,
        embedding_model=embedding_model,
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


def select_balanced_profile_pairs(candidate_pairs, pair_similarity_scores, gt_set, sample_size, seed=RANDOM_SEED):
    """Sample profiled pairs with an explicit positive/negative balance."""
    pairs = list(candidate_pairs)
    if sample_size <= 0 or not pairs:
        return [], []

    scores = list(pair_similarity_scores)
    pair_to_score = {pair: scores[idx] for idx, pair in enumerate(pairs)}
    positives = [pair for pair in pairs if pair in gt_set]
    negatives = [pair for pair in pairs if pair not in gt_set]

    target = min(int(sample_size), len(pairs))
    target_pos = min(len(positives), target // 2)
    target_neg = min(len(negatives), target - target_pos)

    # Fill any shortage from the other class so sample_size is still respected.
    remaining = target - target_pos - target_neg
    if remaining > 0:
        extra_pos = min(len(positives) - target_pos, remaining)
        target_pos += extra_pos
        remaining -= extra_pos
    if remaining > 0:
        extra_neg = min(len(negatives) - target_neg, remaining)
        target_neg += extra_neg

    pos_pairs, _ = select_profile_pairs(
        positives,
        [pair_to_score[pair] for pair in positives],
        sample_size=target_pos,
        seed=seed,
    )
    neg_pairs, _ = select_profile_pairs(
        negatives,
        [pair_to_score[pair] for pair in negatives],
        sample_size=target_neg,
        seed=seed,
    )

    profile_pairs = pos_pairs + neg_pairs
    rng = np.random.RandomState(seed)
    rng.shuffle(profile_pairs)

    pair_to_indices = collections.defaultdict(list)
    for idx, pair in enumerate(pairs):
        pair_to_indices[pair].append(idx)
    profile_indices = [pair_to_indices[pair].pop(0) for pair in profile_pairs]

    print("\n[Adaptive Profiling] Balanced profile sample")
    print(f"  Candidate positives: {len(positives)}")
    print(f"  Candidate negatives: {len(negatives)}")
    print(f"  Profile positives  : {sum(1 for pair in profile_pairs if pair in gt_set)}")
    print(f"  Profile negatives  : {sum(1 for pair in profile_pairs if pair not in gt_set)}")

    return profile_pairs, profile_indices


def build_adaptive_attribute_map(
    df_A,
    df_B,
    candidate_pairs,
    pair_similarity_scores,
    gt_set,
    embedding_model=None,
    profile_sample_size=20,
    top_k_retrieval=5,
    top_k_attributes=None,
    min_k_attributes=2,
    max_k_attributes=5,
    cumulative_importance_threshold=0.8,
    required_attribute_roles=None,
    attribute_roles=None,
    seed=RANDOM_SEED,
):
    """
    Profile a small post-blocking sample with the LLM, train the adaptive
    selector, then transfer pair-specific attribute choices to all candidates.
    """
    attributes = [col for col in df_A.columns if col in df_B.columns]
    if not attributes or not candidate_pairs:
        return {}, None, {}
    inferred_attribute_roles = infer_attribute_roles(attributes)
    if attribute_roles:
        inferred_attribute_roles.update({
            attr: role for attr, role in attribute_roles.items() if attr in inferred_attribute_roles
        })

    profile_pairs, profile_indices = select_balanced_profile_pairs(
        candidate_pairs,
        pair_similarity_scores,
        gt_set,
        sample_size=profile_sample_size,
        seed=seed,
    )
    if not profile_pairs:
        return {}, None, {}

    print(f"\n[Adaptive Profiling] Profiling {len(profile_pairs)} post-blocking pairs")

    profile_features = []
    profile_importance_vectors = []
    raw_profile_importance_vectors = []
    profiling_rows = []
    profiling_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for profile_idx, (idx_a, idx_b) in enumerate(profile_pairs, start=1):
        label = 1 if (idx_a, idx_b) in gt_set else 0
        print(
            f"  Profiling pair {profile_idx}/{len(profile_pairs)} "
            f"({'match' if label == 1 else 'non-match'})..."
        )

        feature_vector = pair_feature_vector(
            df_A,
            df_B,
            idx_a,
            idx_b,
            attributes,
            embedding_model=embedding_model,
        )
        importance, usage = query_llm_adaptive_attribute_importance_with_usage(idx_a, idx_b, df_A, df_B)
        raw_importance, normalized_importance = normalize_importance_vector(importance, attributes)
        for key in profiling_token_usage:
            profiling_token_usage[key] += int(usage.get(key, 0))

        profile_features.append(feature_vector)
        raw_profile_importance_vectors.append(raw_importance)
        profile_importance_vectors.append(normalized_importance)
        profiling_rows.append({
            "profile_index": profile_idx,
            "candidate_index": profile_indices[profile_idx - 1],
            "indexA": idx_a,
            "indexB": idx_b,
            "label": label,
            "raw_importance": raw_importance,
            "normalized_importance": normalized_importance,
            "normalized_sum": sum(normalized_importance.values()),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        })
        print(f"    Raw importance: {raw_importance}")
        print(f"    Normalized importance sum: {sum(normalized_importance.values()):.4f}")

    all_scores = [
        int(score)
        for importance in raw_profile_importance_vectors
        for score in importance.values()
    ]
    score_counts = collections.Counter(all_scores)
    print("\n[Adaptive Profiling] Importance score distribution")
    print(f"  Raw counts: {dict(sorted(score_counts.items()))}")
    print(
        "  0/3 mass: "
        f"{score_counts.get(0, 0) + score_counts.get(3, 0)}/{len(all_scores)} "
        f"({((score_counts.get(0, 0) + score_counts.get(3, 0)) / len(all_scores)) if all_scores else 0.0:.2%})"
    )
    print("\n[Adaptive Profiling] Token usage")
    print(f"  Prompt tokens     : {profiling_token_usage['prompt_tokens']}")
    print(f"  Completion tokens : {profiling_token_usage['completion_tokens']}")
    print(f"  Total tokens      : {profiling_token_usage['total_tokens']}")

    selector = HybridAttributeSelector(
        attributes=attributes,
        top_k_attributes=top_k_attributes,
        min_k_attributes=min_k_attributes,
        max_k_attributes=max_k_attributes,
        cumulative_importance_threshold=cumulative_importance_threshold,
        required_attribute_roles=required_attribute_roles,
        attribute_roles=inferred_attribute_roles,
        top_k_retrieval=top_k_retrieval,
        fusion_strategy="weighted_average",
        predictor_weight=0.5,
        retrieval_weight=0.5,
    )
    selector.fit(np.asarray(profile_features), profile_importance_vectors)

    selected_attributes_by_pair = {}
    selection_sizes = []
    print("\n[Adaptive Transfer] Selecting attributes for all candidate pairs")
    for pair_idx, (idx_a, idx_b) in enumerate(candidate_pairs):
        feature_vector = pair_feature_vector(
            df_A,
            df_B,
            idx_a,
            idx_b,
            attributes,
            embedding_model=embedding_model,
        )
        selection = selector.select_attributes(feature_vector)
        selected_attributes_by_pair[(idx_a, idx_b)] = selection["selected_attributes"]
        selection_sizes.append(len(selection["selected_attributes"]))

    print(
        "Adaptive attributes per pair: "
        f"min={min(selection_sizes)}, "
        f"mean={np.mean(selection_sizes):.2f}, "
        f"max={max(selection_sizes)}"
    )
    print(f"Profiled candidate indices: {profile_indices}")

    profiling_summary = {
        "selection_policy": {
            "top_k_attributes": top_k_attributes,
            "min_k_attributes": min_k_attributes,
            "max_k_attributes": max_k_attributes,
            "cumulative_importance_threshold": cumulative_importance_threshold,
            "required_attribute_roles": list(required_attribute_roles or []),
            "attribute_roles": inferred_attribute_roles,
        },
        "profile_pairs": profile_pairs,
        "profile_indices": profile_indices,
        "score_counts": dict(sorted(score_counts.items())),
        "zero_three_mass": score_counts.get(0, 0) + score_counts.get(3, 0),
        "score_total": len(all_scores),
        "profiling_token_usage": profiling_token_usage,
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
        df_a = normalize_dataframe_records(df_a)
        df_b = normalize_dataframe_records(df_b)
    return df_a, df_b, gt_set


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_run_logs(dataset_name, result_df, summary):
    log_dir = LOG_ROOT / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = log_dir / "final_results.csv"
    summary_path = log_dir / "run_summary.json"

    result_df.to_csv(result_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print(f"\nSaved final pair results to {result_path}")
    print(f"Saved run summary to {summary_path}")


def run_pipeline(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    blocking = config["blocking"]
    adaptive = config["adaptive"]

    print(f"\n=== Running adaptive pipeline: {dataset_name} ===")
    print(f"Blocking hyperparameters: {blocking}")
    print(f"Adaptive hyperparameters: {adaptive}")
    print(f"Random seed: {RANDOM_SEED}")
    set_cache_dir(config["cache_dir"])
    print(f"Matcher cache dir: {config['cache_dir']}")

    df_a, df_b, gt_set = load_configured_dataset(config)

    model = SentenceTransformer(MODEL_NAME)
    np.random.seed(RANDOM_SEED)

    # Full-attribute blocking, computed once and reused downstream.
    table_a_vectors = embed_dataframe_sbert(df_a, model, batch_size=256)
    table_b_vectors = embed_dataframe_sbert(df_b, model, batch_size=256)

    print("TableA full-attribute vectors shape:", table_a_vectors.shape)
    print("TableB full-attribute vectors shape:", table_b_vectors.shape)

    planes_list = create_random_planes(
        num_tables=blocking["num_tables"],
        num_planes=blocking["num_planes"],
        dim=table_a_vectors.shape[1],
        seed=RANDOM_SEED,
    )
    candidate_pairs = query_lsh_fast(
        table_a_vectors,
        table_b_vectors,
        planes_list,
        num_flips=blocking["num_flips"],
        top_k=blocking["top_k"],
    )

    print("Number of full-attribute candidate pairs after top-k:", len(candidate_pairs))

    pair_similarity_scores = compute_candidate_pair_scores(
        candidate_pairs,
        table_a_vectors,
        table_b_vectors,
    )

    print("Blocking attributes used: full non-ID records")
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

    selected_attributes_by_pair = None
    profiling_summary = {}
    if ENABLE_ADAPTIVE_TRANSFER:
        adaptive_embedding_model = model if ADAPTIVE_USE_SEMANTIC_FEATURES else None
        selected_attributes_by_pair, _, profiling_summary = build_adaptive_attribute_map(
            df_a,
            df_b,
            candidate_pairs,
            pair_similarity_scores,
            gt_set,
            embedding_model=adaptive_embedding_model,
            profile_sample_size=adaptive["profile_sample_size"],
            top_k_retrieval=adaptive["top_k_retrieval"],
            top_k_attributes=adaptive.get("top_k_attributes"),
            min_k_attributes=adaptive.get("min_k_attributes", 2),
            max_k_attributes=adaptive.get("max_k_attributes", 5),
            cumulative_importance_threshold=adaptive.get("cumulative_importance_threshold", 0.8),
            required_attribute_roles=adaptive.get("required_attribute_roles"),
            attribute_roles=adaptive.get("attribute_roles"),
            seed=RANDOM_SEED,
        )

    result_df = infer_candidates_pairwise(
        df_a,
        df_b,
        candidate_pairs,
        selected_attributes_by_pair=selected_attributes_by_pair,
    )

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

    total_prompt_tokens = result_df["prompt_tokens"].sum()
    total_completion_tokens = result_df["completion_tokens"].sum()
    total_tokens = result_df["total_tokens"].sum()
    cache_hits = int(result_df["cache_hit"].sum()) if "cache_hit" in result_df else 0
    profiling_token_usage = profiling_summary.get("profiling_token_usage", {})
    profiling_prompt_tokens = int(profiling_token_usage.get("prompt_tokens", 0))
    profiling_completion_tokens = int(profiling_token_usage.get("completion_tokens", 0))
    profiling_total_tokens = int(profiling_token_usage.get("total_tokens", 0))
    pipeline_prompt_tokens = int(total_prompt_tokens) + profiling_prompt_tokens
    pipeline_completion_tokens = int(total_completion_tokens) + profiling_completion_tokens
    pipeline_total_tokens = int(total_tokens) + profiling_total_tokens

    print("\n--- Token Usage ---")
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
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Cache hits: {cache_hits}/{len(result_df)}")

    cand_set = set(zip(result_df["indexA"], result_df["indexB"]))
    in_cands = sum(1 for pair in gt_set if pair in cand_set)
    found_llm = tp

    print(f"\nTrue matches total         : {len(gt_set)}")
    print(f"True matches in candidates : {in_cands}  (blocking recall)")
    print(f"True matches found by LLM  : {found_llm}  (end-to-end recall)")

    summary = {
        "dataset": dataset_name,
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "config": config,
        "blocking": {
            "total_pairs": total_pairs,
            "candidate_pairs": candidate_pairs_count,
            "reduction_percentage": reduction_percentage,
            "gt_matches": len(gt_set),
            "found_matches": found,
            "pair_completeness": pc,
        },
        "adaptive": profiling_summary,
        "matching": {
            "num_result_pairs": len(result_df),
            "prompt_tokens": int(total_prompt_tokens),
            "completion_tokens": int(total_completion_tokens),
            "total_tokens": int(total_tokens),
            "cache_hits": cache_hits,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_matches_in_candidates": in_cands,
            "true_matches_found_by_llm": int(found_llm),
        },
        "token_usage": {
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
    }
    write_run_logs(dataset_name, result_df, summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive ER pipeline for one configured dataset.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="DBLP-ACM",
        help="Dataset configuration to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.dataset)


if __name__ == "__main__":
    main()
