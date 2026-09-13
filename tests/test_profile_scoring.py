"""Covers the 'decisive' Step-1.5 profiling mode.

The original profiling prompt asks for a 0-3 score per attribute, scored in isolation.
The decisive mode asks for the subset directly, so the parser must turn a list of names
into the same score-dict shape the ordinal parser produces -- otherwise the selector,
which is shared by both modes, cannot consume it.
"""

import unittest

from code.attribute_selection.llm_guided import (
    DECISIVE_PROFILE_PROMPT,
    DEFAULT_PROFILE_SCORING,
    PROFILE_SCORING_MODES,
    parse_adaptive_attribute_importance,
    parse_decisive_attribute_set,
    query_llm_profile_importance_with_usage,
)

ATTRS = ["name", "addr", "city", "phone", "type"]


class DefaultsTests(unittest.TestCase):
    def test_default_mode_is_unchanged(self):
        # Flipping this silently would make new runs incomparable to reported results.
        self.assertEqual(DEFAULT_PROFILE_SCORING, "ordinal")

    def test_advertised_modes(self):
        self.assertEqual(set(PROFILE_SCORING_MODES), {"ordinal", "decisive"})

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            query_llm_profile_importance_with_usage(0, 0, None, None, scoring="binary")

    def test_prompt_declares_every_placeholder(self):
        for token in ("{attributes}", "{record_a_json}", "{record_b_json}"):
            self.assertIn(token, DECISIVE_PROFILE_PROMPT)


class ParseTests(unittest.TestCase):
    def test_selected_attributes_score_one_and_the_rest_zero(self):
        scores = parse_decisive_attribute_set(
            '{"decisive_attributes": ["phone", "name"]}', ATTRS
        )
        self.assertEqual(scores, {"name": 1.0, "addr": 0.0, "city": 0.0, "phone": 1.0, "type": 0.0})

    def test_shape_matches_the_ordinal_parser(self):
        # Both feed the same selector, so the key set must agree exactly.
        ordinal = parse_adaptive_attribute_importance(
            '{"attribute_importance": {"name": 3, "phone": 2}}', ATTRS
        )
        decisive = parse_decisive_attribute_set('{"decisive_attributes": ["name"]}', ATTRS)
        self.assertEqual(sorted(ordinal), sorted(decisive))

    def test_json_fences_are_tolerated(self):
        scores = parse_decisive_attribute_set(
            '```json\n{"decisive_attributes": ["city"]}\n```', ATTRS
        )
        self.assertEqual(scores["city"], 1.0)

    def test_bare_list_is_accepted(self):
        # A plausible deviation from the requested envelope; cheaper to accept than to
        # discard a whole profiled pair over formatting.
        self.assertEqual(parse_decisive_attribute_set('["addr"]', ATTRS)["addr"], 1.0)

    def test_hallucinated_names_are_dropped(self):
        scores = parse_decisive_attribute_set(
            '{"decisive_attributes": ["phone", "zipcode"]}', ATTRS
        )
        self.assertEqual(sorted(scores), sorted(ATTRS))
        self.assertEqual(scores["phone"], 1.0)

    def test_unparseable_response_falls_back_to_all_zeros(self):
        # Normalizing an all-zero vector yields a uniform one, matching how the ordinal
        # parser degrades -- a bad response must not silently favour some attribute.
        scores = parse_decisive_attribute_set("sorry, I cannot help", ATTRS)
        self.assertEqual(set(scores.values()), {0.0})

    def test_empty_selection_falls_back_to_all_zeros(self):
        self.assertEqual(set(parse_decisive_attribute_set('{"decisive_attributes": []}', ATTRS).values()), {0.0})

    def test_wrong_payload_type_falls_back_to_all_zeros(self):
        self.assertEqual(set(parse_decisive_attribute_set('{"decisive_attributes": 3}', ATTRS).values()), {0.0})

    def test_duplicates_do_not_inflate_scores(self):
        scores = parse_decisive_attribute_set(
            '{"decisive_attributes": ["phone", "phone", "phone"]}', ATTRS
        )
        self.assertEqual(scores["phone"], 1.0)


class NormalizationTests(unittest.TestCase):
    """Decisive scores must survive the shared normalization step intact."""

    def test_binary_scores_normalize_to_a_uniform_subset(self):
        from code.main import normalize_importance_vector

        scores = parse_decisive_attribute_set(
            '{"decisive_attributes": ["name", "phone"]}', ATTRS
        )
        _, normalized = normalize_importance_vector(scores, ATTRS)
        self.assertAlmostEqual(normalized["name"], 0.5)
        self.assertAlmostEqual(normalized["phone"], 0.5)
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertEqual({normalized[a] for a in ("addr", "city", "type")}, {0.0})

    def test_all_zero_scores_normalize_to_uniform_over_everything(self):
        from code.main import normalize_importance_vector

        _, normalized = normalize_importance_vector({a: 0.0 for a in ATTRS}, ATTRS)
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertEqual(len(set(normalized.values())), 1)


if __name__ == "__main__":
    unittest.main()
