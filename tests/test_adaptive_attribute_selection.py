import os
import tempfile
import unittest

import numpy as np

from code.attribute_selection.adaptive import (
    AttributeImportancePredictor,
    FEATURE_ORDER,
    HybridAttributeSelector,
    ProfileRetriever,
    compute_attribute_features,
    condense_pair_by_attributes,
    flatten_pair_features,
    infer_attribute_roles,
    importance_vector_from_grouped,
    importance_vector_from_ranked,
    select_profile_pairs,
)
from code.attribute_selection.llm_guided import parse_adaptive_attribute_importance
from code.llm_guided import normalize_importance_vector


class AdaptiveAttributeSelectionTests(unittest.TestCase):
    def test_stratified_sampling_biases_medium_and_keeps_coverage(self):
        candidate_pairs = [(i, i) for i in range(30)]
        scores = np.linspace(0.0, 1.0, 30)

        _, indices = select_profile_pairs(candidate_pairs, scores, sample_size=10)

        low_count = sum(scores[idx] <= np.quantile(scores, 1 / 3) for idx in indices)
        high_count = sum(scores[idx] >= np.quantile(scores, 2 / 3) for idx in indices)
        medium_count = len(indices) - low_count - high_count

        self.assertEqual(len(indices), 10)
        self.assertGreaterEqual(medium_count, low_count)
        self.assertGreaterEqual(medium_count, high_count)
        self.assertGreater(low_count, 0)
        self.assertGreater(high_count, 0)

    def test_stratified_sampling_handles_empty_input(self):
        pairs, indices = select_profile_pairs([], [], sample_size=5)
        self.assertEqual(pairs, [])
        self.assertEqual(indices, [])

    def test_feature_extraction_fixed_dimensions(self):
        attributes = ["name", "price"]
        features = compute_attribute_features(
            {"name": "Apple iPhone", "price": "100"},
            {"name": "Apple Phone", "price": "80"},
            attributes,
        )
        flat = flatten_pair_features(features, attributes)

        self.assertEqual(len(flat), len(attributes) * len(FEATURE_ORDER))
        self.assertEqual(list(features["name"].keys()), list(FEATURE_ORDER))

    def test_numeric_and_non_numeric_handling(self):
        attributes = ["name", "price"]
        features = compute_attribute_features(
            {"name": "abc", "price": "100"},
            {"name": "abc", "price": "90"},
            attributes,
            numeric_scales={"price": 20},
        )

        self.assertEqual(features["name"]["is_numeric"], 0)
        self.assertEqual(features["name"]["numeric_similarity"], 0)
        self.assertEqual(features["price"]["is_numeric"], 1)
        self.assertAlmostEqual(features["price"]["numeric_similarity"], 0.5)
        self.assertGreaterEqual(features["price"]["token_overlap"], 0)

    def test_ranked_importance_vector_is_normalized(self):
        vector = importance_vector_from_ranked(
            ["name", "address", "city", "phone"],
            ["name", "phone"],
        )

        self.assertAlmostEqual(vector["name"], 2 / 3, places=4)
        self.assertAlmostEqual(vector["phone"], 1 / 3, places=4)
        self.assertEqual(vector["address"], 0)
        self.assertAlmostEqual(sum(vector.values()), 1.0)

    def test_grouped_importance_vector_is_normalized(self):
        vector = importance_vector_from_grouped(
            ["name", "address", "city", "phone"],
            {"high": ["name", "address"], "medium": ["phone"], "low": ["city"]},
        )

        self.assertAlmostEqual(vector["name"], 3 / 9)
        self.assertAlmostEqual(vector["address"], 3 / 9)
        self.assertAlmostEqual(vector["phone"], 2 / 9)
        self.assertAlmostEqual(vector["city"], 1 / 9)
        self.assertAlmostEqual(sum(vector.values()), 1.0)

    def test_ridge_predictor_returns_normalized_non_negative_vectors(self):
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        predictor = AttributeImportancePredictor().fit(X, Y)
        pred = predictor.predict(np.array([[0.8, 0.1, 0.1]], dtype=float))

        self.assertEqual(pred.shape, (1, 3))
        self.assertTrue(np.all(pred >= 0))
        self.assertAlmostEqual(pred[0].sum(), 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "predictor.joblib")
            predictor.save(path)
            loaded = AttributeImportancePredictor.load(path)
            loaded_pred = loaded.predict(np.array([[0.8, 0.1, 0.1]], dtype=float))
            self.assertAlmostEqual(loaded_pred[0].sum(), 1.0)

    def test_retriever_returns_top_k_and_averaged_normalized_importance(self):
        features = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=float)
        importance = np.array([[1, 0], [0, 1], [0, 1]], dtype=float)

        retriever = ProfileRetriever().fit(features, importance)
        aggregate, indices, scores = retriever.predict_importance(np.array([1, 0]), top_k=2)

        self.assertEqual(len(indices), 2)
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(aggregate.sum(), 1.0)
        self.assertTrue(np.allclose(aggregate, np.array([0.5, 0.5])))

    def test_hybrid_selector_weighted_average_and_confidence_based(self):
        attributes = ["name", "phone", "city"]
        profile_features = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        profile_importance = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=float,
        )

        weighted = HybridAttributeSelector(
            attributes,
            fusion_strategy="weighted_average",
            top_k_retrieval=2,
            top_k_attributes=2,
        ).fit(profile_features, profile_importance)
        weighted_result = weighted.select_attributes(np.array([1, 0, 0], dtype=float))

        confidence = HybridAttributeSelector(
            attributes,
            fusion_strategy="confidence_based",
            top_k_retrieval=2,
            threshold=0.2,
        ).fit(profile_features, profile_importance)
        confidence_result = confidence.select_attributes(np.array([1, 0, 0], dtype=float))

        self.assertEqual(len(weighted_result["selected_attributes"]), 2)
        self.assertIn("name", weighted_result["selected_attributes"])
        self.assertIn("retrieved_indices", weighted_result["retrieval_debug"])
        self.assertGreaterEqual(len(confidence_result["selected_attributes"]), 1)
        self.assertAlmostEqual(sum(confidence_result["combined_importance"].values()), 1.0)

    def test_hybrid_selector_dynamic_budget_uses_importance_mass(self):
        attributes = ["name", "phone", "city", "address", "type"]
        selector = HybridAttributeSelector(
            attributes,
            min_k_attributes=2,
            max_k_attributes=5,
            cumulative_importance_threshold=0.8,
        )

        selector.estimate_importance = lambda _: (
            np.array([0.75, 0.15, 0.05, 0.03, 0.02]),
            np.array([0.75, 0.15, 0.05, 0.03, 0.02]),
            np.array([0.75, 0.15, 0.05, 0.03, 0.02]),
            {"retrieved_indices": [], "cosine_similarity_scores": []},
        )
        concentrated = selector.select_attributes(np.zeros(1))

        selector.estimate_importance = lambda _: (
            np.array([0.25, 0.22, 0.20, 0.18, 0.15]),
            np.array([0.25, 0.22, 0.20, 0.18, 0.15]),
            np.array([0.25, 0.22, 0.20, 0.18, 0.15]),
            {"retrieved_indices": [], "cosine_similarity_scores": []},
        )
        diffuse = selector.select_attributes(np.zeros(1))

        self.assertEqual(concentrated["selected_attributes"], ["name", "phone"])
        self.assertEqual(diffuse["selected_attributes"], ["name", "phone", "city", "address"])

    def test_hybrid_selector_fixed_top_k_remains_backward_compatible(self):
        attributes = ["name", "phone", "city", "address"]
        selector = HybridAttributeSelector(
            attributes,
            top_k_attributes=3,
            min_k_attributes=1,
            max_k_attributes=4,
            cumulative_importance_threshold=0.5,
        )
        selector.estimate_importance = lambda _: (
            np.array([0.7, 0.2, 0.08, 0.02]),
            np.array([0.7, 0.2, 0.08, 0.02]),
            np.array([0.7, 0.2, 0.08, 0.02]),
            {"retrieved_indices": [], "cosine_similarity_scores": []},
        )

        result = selector.select_attributes(np.zeros(1))

        self.assertEqual(result["selected_attributes"], ["name", "phone", "city"])

    def test_infer_attribute_roles_marks_identity_and_context_fields(self):
        roles = infer_attribute_roles(["title", "authors", "venue", "year", "misc"])

        self.assertEqual(roles["title"], "identity")
        self.assertEqual(roles["authors"], "identity")
        self.assertEqual(roles["venue"], "context")
        self.assertEqual(roles["year"], "context")
        self.assertEqual(roles["misc"], "other")

    def test_hybrid_selector_enforces_required_identity_role(self):
        attributes = ["year", "venue", "title", "authors"]
        selector = HybridAttributeSelector(
            attributes,
            min_k_attributes=2,
            max_k_attributes=3,
            cumulative_importance_threshold=0.8,
            required_attribute_roles=["identity"],
        )
        selector.estimate_importance = lambda _: (
            np.array([0.5, 0.35, 0.1, 0.05]),
            np.array([0.5, 0.35, 0.1, 0.05]),
            np.array([0.5, 0.35, 0.1, 0.05]),
            {"retrieved_indices": [], "cosine_similarity_scores": []},
        )

        result = selector.select_attributes(np.zeros(1))

        self.assertEqual(result["selected_attributes"], ["year", "venue", "title"])

    def test_hybrid_selector_replaces_context_when_required_role_at_max_budget(self):
        attributes = ["year", "venue", "category", "title"]
        selector = HybridAttributeSelector(
            attributes,
            top_k_attributes=3,
            max_k_attributes=3,
            required_attribute_roles=["identity"],
        )
        selector.estimate_importance = lambda _: (
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            {"retrieved_indices": [], "cosine_similarity_scores": []},
        )

        result = selector.select_attributes(np.zeros(1))

        self.assertEqual(result["selected_attributes"], ["year", "venue", "title"])

    def test_hybrid_selector_does_not_enforce_when_no_identity_available(self):
        attributes = ["year", "venue", "category"]
        selector = HybridAttributeSelector(
            attributes,
            top_k_attributes=2,
            max_k_attributes=2,
            required_attribute_roles=["identity"],
        )
        selector.estimate_importance = lambda _: (
            np.array([0.6, 0.3, 0.1]),
            np.array([0.6, 0.3, 0.1]),
            np.array([0.6, 0.3, 0.1]),
            {"retrieved_indices": [], "cosine_similarity_scores": []},
        )

        result = selector.select_attributes(np.zeros(1))

        self.assertEqual(result["selected_attributes"], ["year", "venue"])

    def test_hybrid_selector_required_roles_empty_keeps_old_behavior(self):
        attributes = ["year", "venue", "title"]
        selector = HybridAttributeSelector(
            attributes,
            top_k_attributes=2,
            required_attribute_roles=[],
        )
        selector.estimate_importance = lambda _: (
            np.array([0.5, 0.4, 0.1]),
            np.array([0.5, 0.4, 0.1]),
            np.array([0.5, 0.4, 0.1]),
            {"retrieved_indices": [], "cosine_similarity_scores": []},
        )

        result = selector.select_attributes(np.zeros(1))

        self.assertEqual(result["selected_attributes"], ["year", "venue"])

    def test_condense_pair_by_attributes_keeps_only_selected(self):
        record_a = {"name": "A", "address": "X", "phone": "1"}
        record_b = {"name": "B", "address": "Y", "phone": "2"}

        condensed_a, condensed_b = condense_pair_by_attributes(record_a, record_b, ["name", "phone"])

        self.assertEqual(condensed_a, {"name": "A", "phone": "1"})
        self.assertEqual(condensed_b, {"name": "B", "phone": "2"})

    def test_adaptive_profile_parser_reads_nested_integer_scores(self):
        content = """
        {
          "attribute_importance": {
            "name": 3,
            "address": 1,
            "phone": 0
          }
        }
        """

        scores = parse_adaptive_attribute_importance(content, ["name", "address", "phone", "city"])

        self.assertEqual(scores, {"name": 3, "address": 1, "phone": 0, "city": 0})

    def test_adaptive_profile_parser_clamps_and_handles_bad_json(self):
        content = """
        ```json
        {
          "attribute_importance": {
            "name": 4,
            "address": -2,
            "phone": "2"
          }
        }
        ```
        """

        scores = parse_adaptive_attribute_importance(content, ["name", "address", "phone"])
        bad_scores = parse_adaptive_attribute_importance("not json", ["name", "address"])

        self.assertEqual(scores, {"name": 3, "address": 0, "phone": 2})
        self.assertEqual(bad_scores, {"name": 0, "address": 0})

    def test_pipeline_importance_normalization_before_fit(self):
        raw, normalized = normalize_importance_vector(
            {"name": 3, "address": 1, "phone": 0},
            ["name", "address", "phone", "city"],
        )

        self.assertEqual(raw, {"name": 3.0, "address": 1.0, "phone": 0.0, "city": 0.0})
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertAlmostEqual(normalized["name"], 0.75)
        self.assertAlmostEqual(normalized["address"], 0.25)

    def test_pipeline_importance_normalization_fallback_for_all_zero(self):
        _, normalized = normalize_importance_vector(
            {"name": 0, "address": 0},
            ["name", "address"],
        )

        self.assertEqual(normalized, {"name": 0.5, "address": 0.5})


if __name__ == "__main__":
    unittest.main()
