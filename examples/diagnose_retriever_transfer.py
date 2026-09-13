#!/usr/bin/env python3
"""Offline diagnostics for the Step-1.5 transfer stage.

Replays the retriever/predictor half of `HybridAttributeSelector` against the
profiling data already stored in a completed run's `run_summary.json`. No LLM
calls and no embedding model are needed: the LLM importance vectors are read
back from the log, and the pair features are recomputed deterministically from
the normalized tables.

Two questions are answered:

1. Is cosine the right neighbour metric? Leave-one-out over the profiled pairs
   compares cosine against Euclidean, and against variants that append the
   blocking similarity as an explicit feature dimension.
2. Is the cumulative-importance selection policy reachable on this schema?
   Selection sizes are reported under the current policy and under scale-free
   alternatives, using the LLM's own importance vectors as the reference.

Run from the repository root, for example:

    python examples/diagnose_retriever_transfer.py \
        --run-summary logs/DBLP-ACM/runs/20260615_221942_step15_importance_based/run_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.attribute_selection.adaptive import (  # noqa: E402
    AttributeImportancePredictor,
    compute_attribute_features,
    compute_numeric_scales,
    flatten_pair_features,
    infer_attribute_roles,
)
from code.main import DATASET_CONFIGS, load_configured_dataset  # noqa: E402


METRICS = ("cosine", "euclidean", "cosine_simdim", "euclidean_simdim")


def neighbour_scores(query, matrix, metric):
    """Higher is more similar, for every metric."""
    if metric.startswith("cosine"):
        q_norm = np.linalg.norm(query)
        m_norm = np.linalg.norm(matrix, axis=1)
        denom = np.where(m_norm * q_norm > 0, m_norm * q_norm, 1.0)
        return matrix @ query / denom
    return -np.linalg.norm(matrix - query, axis=1)


def safe_normalize(vector):
    total = float(np.sum(vector))
    if total <= 0:
        return np.full(len(vector), 1.0 / len(vector))
    return vector / total


def retrieve_importance(query, feature_matrix, importance_matrix, metric, top_k):
    scores = neighbour_scores(query, feature_matrix, metric)
    k = min(int(top_k), len(scores))
    order = np.argsort(-scores)[:k]
    return safe_normalize(importance_matrix[order].mean(axis=0)), scores[order]


def spearman(a, b):
    """Rank correlation without a scipy dependency."""
    def ranks(x):
        order = np.argsort(x)
        out = np.empty(len(x), dtype=float)
        out[order] = np.arange(len(x), dtype=float)
        return out

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom else 0.0


def cumulative_policy(importance, min_k, max_k, threshold):
    order = np.argsort(-importance)
    selected, cumulative = [], 0.0
    for idx in order:
        if len(selected) >= max_k:
            break
        selected.append(int(idx))
        cumulative += float(importance[idx])
        if len(selected) >= min_k and cumulative >= threshold:
            break
    return selected


def ratio_policy(importance, min_k, max_k, lam):
    """Scale-free rule: keep attributes carrying at least lam x uniform mass."""
    n = len(importance)
    order = np.argsort(-importance)
    selected = [int(idx) for idx in order if importance[idx] >= lam / n]
    if len(selected) < min_k:
        selected = [int(idx) for idx in order[:min_k]]
    return selected[:max_k]


def gap_policy(importance, min_k, max_k):
    """Cut at the largest drop in the ranked importance profile."""
    order = np.argsort(-importance)
    ranked = importance[order]
    upper = min(max_k, len(ranked))
    if upper <= min_k:
        return [int(idx) for idx in order[:upper]]
    gaps = ranked[:upper - 1] - ranked[1:upper]
    cut = int(np.argmax(gaps[min_k - 1:])) + min_k
    return [int(idx) for idx in order[:cut]]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-summary", required=True, help="Path to a step15 run_summary.json.")
    parser.add_argument("--top-k-retrieval", type=int, default=None, help="Override the run's top_k_retrieval.")
    parser.add_argument("--ratio-lambda", type=float, default=1.5, help="Lambda for the ratio-to-uniform policy.")
    args = parser.parse_args()

    summary = json.loads(Path(args.run_summary).read_text())
    dataset = summary["dataset"]
    adaptive = summary["adaptive"]
    if not adaptive.get("profiling_rows"):
        raise SystemExit(f"{args.run_summary} has no profiling_rows (not a step15 run?)")

    policy = adaptive["selection_policy"]
    min_k = int(policy.get("min_k_attributes") or 1)
    max_k = int(policy.get("max_k_attributes") or 0)
    threshold = float(policy.get("cumulative_importance_threshold") or 0.0)
    top_k_retrieval = args.top_k_retrieval or int(summary["config"]["adaptive"].get("top_k_retrieval", 5))

    df_a, df_b, _ = load_configured_dataset(DATASET_CONFIGS[dataset])
    rows = adaptive["profiling_rows"]
    attributes = list(rows[0]["normalized_importance"].keys())
    max_k = max_k or len(attributes)
    numeric_scales = compute_numeric_scales(df_a, df_b, attributes)

    features, importances, similarities = [], [], []
    for row in rows:
        feature_dict = compute_attribute_features(
            df_a.iloc[int(row["indexA"])].to_dict(),
            df_b.iloc[int(row["indexB"])].to_dict(),
            attributes,
            embedding_model=None,
            numeric_scales=numeric_scales,
        )
        features.append(flatten_pair_features(feature_dict, attributes))
        importances.append([float(row["normalized_importance"][attr]) for attr in attributes])
        similarities.append(float(row["similarity_score"]))

    features = np.asarray(features, dtype=float)
    importances = np.asarray(importances, dtype=float)
    similarities = np.asarray(similarities, dtype=float)
    augmented = np.hstack([features, similarities.reshape(-1, 1)])
    n_pairs, n_attrs = importances.shape

    print(f"\n=== {dataset} | {adaptive['profile_sampler']} | {n_pairs} profiled pairs | "
          f"{n_attrs} attributes | {features.shape[1]}-dim features ===")
    print(f"Roles: {infer_attribute_roles(attributes)}")

    print("\n--- 1. Neighbour-metric leave-one-out ---")
    print(f"{'metric':<18}{'spearman':>10}{'MAE':>10}{'set-Jacc':>10}{'nbr spread':>12}{'nbr range':>22}")
    for metric in METRICS:
        matrix = augmented if metric.endswith("simdim") else features
        spears, maes, jaccs, spreads, lo, hi = [], [], [], [], [], []
        for i in range(n_pairs):
            mask = np.arange(n_pairs) != i
            predicted, scores = retrieve_importance(
                matrix[i], matrix[mask], importances[mask], metric, top_k_retrieval
            )
            truth = importances[i]
            spears.append(spearman(predicted, truth))
            maes.append(float(np.mean(np.abs(predicted - truth))))
            jaccs.append(jaccard(
                cumulative_policy(predicted, min_k, max_k, threshold),
                cumulative_policy(truth, min_k, max_k, threshold),
            ))
            all_scores = neighbour_scores(matrix[i], matrix[mask], metric)
            spreads.append(float(np.std(all_scores)))
            lo.append(float(all_scores.min()))
            hi.append(float(all_scores.max()))
        print(f"{metric:<18}{np.mean(spears):>10.4f}{np.mean(maes):>10.4f}{np.mean(jaccs):>10.4f}"
              f"{np.mean(spreads):>12.4f}   [{np.mean(lo):>8.4f}, {np.mean(hi):>8.4f}]")
    print("  spearman/MAE/set-Jacc: predicted vs. the LLM's own importance vector (higher is better, MAE lower).")
    print("  nbr spread/range: dispersion of the similarity scores the metric assigns; a narrow")
    print("  range means the top-k neighbourhood is barely distinguishable from the rest.")

    print("\n--- 2. Ridge predictor vs. retriever vs. hybrid (leave-one-out, best metric per row) ---")
    print(f"{'estimator':<24}{'spearman':>10}{'MAE':>10}{'set-Jacc':>10}")
    for name in ("predictor", "retriever(cosine)", "retriever(euclidean)", "hybrid(cosine)", "hybrid(euclidean)"):
        spears, maes, jaccs = [], [], []
        for i in range(n_pairs):
            mask = np.arange(n_pairs) != i
            truth = importances[i]
            ridge = AttributeImportancePredictor().fit(features[mask], importances[mask]).predict(features[i])[0]
            if name == "predictor":
                predicted = ridge
            else:
                metric = "euclidean" if "euclidean" in name else "cosine"
                retrieved, _ = retrieve_importance(features[i], features[mask], importances[mask], metric, top_k_retrieval)
                predicted = retrieved if name.startswith("retriever") else safe_normalize(0.5 * ridge + 0.5 * retrieved)
            spears.append(spearman(predicted, truth))
            maes.append(float(np.mean(np.abs(predicted - truth))))
            jaccs.append(jaccard(
                cumulative_policy(predicted, min_k, max_k, threshold),
                cumulative_policy(truth, min_k, max_k, threshold),
            ))
        print(f"{name:<24}{np.mean(spears):>10.4f}{np.mean(maes):>10.4f}{np.mean(jaccs):>10.4f}")

    print("\n--- 3. Selection-policy reachability (on the LLM's own importance vectors) ---")
    needed = [int(np.searchsorted(np.cumsum(np.sort(v)[::-1]), threshold) + 1) for v in importances]
    top_max_k = [float(np.cumsum(np.sort(v)[::-1])[min(max_k, n_attrs) - 1]) for v in importances]
    print(f"  attributes needed to reach cumulative {threshold}: mean={np.mean(needed):.2f} max={max(needed)}")
    print(f"  mass actually carried by the top-{max_k}: mean={np.mean(top_max_k):.4f} max={max(top_max_k):.4f}")
    if np.mean(top_max_k) < threshold:
        print(f"  >>> max_k={max_k} binds for essentially every pair: the threshold is unreachable on this schema.")

    print(f"\n{'policy':<34}{'mean k':>9}{'min':>6}{'max':>6}{'saturated at max_k':>22}")
    policies = {
        f"cumulative>={threshold} (current)": lambda v: cumulative_policy(v, min_k, max_k, threshold),
        f"ratio>={args.ratio_lambda}x uniform": lambda v: ratio_policy(v, min_k, max_k, args.ratio_lambda),
        "largest-gap cut": lambda v: gap_policy(v, min_k, max_k),
    }
    for label, fn in policies.items():
        sizes = np.array([len(fn(v)) for v in importances])
        print(f"{label:<34}{sizes.mean():>9.2f}{sizes.min():>6}{sizes.max():>6}"
              f"{(sizes == max_k).mean():>21.0%}")
    print("  These are upper bounds: they use the LLM's true importance, so any spread here is")
    print("  what a perfect transfer stage could express under that policy.")


if __name__ == "__main__":
    main()
