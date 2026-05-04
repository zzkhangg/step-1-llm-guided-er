"""Minimal Step 1.5 usage example.

Run after blocking has already produced candidate pairs. The existing global
tuple condensation / attribute filtering should remain before blocking when the
pipeline uses it.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.attribute_selection.adaptive import (
    HybridAttributeSelector,
    compute_attribute_features,
    condense_pair_by_attributes,
    flatten_pair_features,
    importance_vector_from_ranked,
    select_profile_pairs,
)


attributes = ["name", "address", "city", "phone"]

table_a = [
    {"name": "Pasta House", "address": "10 King St", "city": "Brisbane", "phone": "5551000"},
    {"name": "Coffee Bar", "address": "5 Queen St", "city": "Brisbane", "phone": "5552000"},
    {"name": "Sushi Place", "address": "8 Roma St", "city": "Brisbane", "phone": "5553000"},
]
table_b = [
    {"name": "The Pasta House", "address": "10 King Street", "city": "Brisbane", "phone": "5551000"},
    {"name": "Coffee Shop", "address": "500 Queen St", "city": "Brisbane", "phone": "5559999"},
    {"name": "Sushi Place", "address": "8 Roma Street", "city": "Brisbane", "phone": "5553001"},
]

# Candidate pairs and aggregate embedding similarities come from blocking.
candidate_pairs = [(0, 0), (1, 1), (2, 2)]
pair_similarity_scores = np.array([0.92, 0.51, 0.78])

profile_pairs, profile_indices = select_profile_pairs(
    candidate_pairs,
    pair_similarity_scores,
    sample_size=2,
)

profile_features = []
profile_importance_vectors = []

for idx_a, idx_b in profile_pairs:
    feature_dict = compute_attribute_features(table_a[idx_a], table_b[idx_b], attributes)
    profile_features.append(flatten_pair_features(feature_dict, attributes))

    # Mock LLM profiling output. In production, this comes from the profiling
    # prompt over selected candidate pairs.
    ranked_attributes = ["name", "phone", "address"]
    importance = importance_vector_from_ranked(attributes, ranked_attributes)
    profile_importance_vectors.append([importance[attr] for attr in attributes])

selector = HybridAttributeSelector(
    attributes,
    fusion_strategy="weighted_average",
    top_k_retrieval=2,
    min_k_attributes=2,
    max_k_attributes=4,
    cumulative_importance_threshold=0.8,
)
selector.fit(np.vstack(profile_features), np.asarray(profile_importance_vectors))

new_pair = candidate_pairs[1]
new_features = flatten_pair_features(
    compute_attribute_features(table_a[new_pair[0]], table_b[new_pair[1]], attributes),
    attributes,
)
selection = selector.select_attributes(new_features)

condensed_a, condensed_b = condense_pair_by_attributes(
    table_a[new_pair[0]],
    table_b[new_pair[1]],
    selection["selected_attributes"],
)

print("Profile pair indices:", profile_indices)
print("Selected attributes:", selection["selected_attributes"])
print("Record A for LLM:", condensed_a)
print("Record B for LLM:", condensed_b)
