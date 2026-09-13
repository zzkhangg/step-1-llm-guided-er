"""Pins the encoding of absent values in the per-attribute similarity features.

Both feature implementations used to answer 1.0 when neither record populated an
attribute, so a column the two sources simply do not share looked like the strongest
possible agreement signal. On Amazon-Walmart that covers 2.7% of feature cells; on the
two datasets reported so far it covers none, which is why it went unnoticed. The
contract is that an absent value produces no evidence, not positive evidence.
"""

import unittest

from code.attribute_selection import adaptive
from code.attribute_selection import supervised


class AdaptiveMissingValueTests(unittest.TestCase):
    def test_both_missing_is_not_agreement(self):
        self.assertEqual(adaptive._token_overlap("", ""), 0.0)
        self.assertEqual(adaptive._edit_similarity("", ""), 0.0)
        self.assertEqual(adaptive._exact_match("", ""), 0.0)

    def test_one_missing_is_not_agreement(self):
        self.assertEqual(adaptive._token_overlap("sony", ""), 0.0)
        self.assertEqual(adaptive._edit_similarity("sony", ""), 0.0)
        self.assertEqual(adaptive._exact_match("sony", ""), 0.0)

    def test_values_without_word_characters_do_not_divide_by_zero(self):
        self.assertEqual(adaptive._token_overlap("---", "---"), 0.0)

    def test_present_values_are_unaffected(self):
        self.assertEqual(adaptive._token_overlap("sony tv", "sony tv"), 1.0)
        self.assertEqual(adaptive._edit_similarity("sony", "sony"), 1.0)
        self.assertEqual(adaptive._exact_match("Sony ", "sony"), 1.0)

    def test_feature_vector_reports_zeros_for_an_absent_attribute(self):
        features = adaptive.compute_attribute_features(
            {"title": "canon eos", "brand": ""},
            {"title": "canon eos", "brand": None},
            ["title", "brand"],
        )
        self.assertEqual(features["brand"]["token_overlap"], 0.0)
        self.assertEqual(features["brand"]["edit_similarity"], 0.0)
        self.assertEqual(features["brand"]["exact_match"], 0.0)
        self.assertEqual(features["title"]["token_overlap"], 1.0)


class SupervisedMissingValueTests(unittest.TestCase):
    def test_both_missing_is_not_agreement(self):
        self.assertEqual(supervised.token_overlap("", ""), 0.0)
        self.assertEqual(supervised.edit_similarity("", ""), 0.0)
        self.assertEqual(supervised.exact_match("", ""), 0.0)

    def test_whitespace_only_values_do_not_divide_by_zero(self):
        self.assertEqual(supervised.token_overlap("   ", "   "), 0.0)
        self.assertEqual(supervised.exact_match("   ", "   "), 0.0)

    def test_present_values_are_unaffected(self):
        self.assertEqual(supervised.token_overlap("sony tv", "sony tv"), 1.0)
        self.assertEqual(supervised.edit_similarity("sony", "sony"), 1.0)
        self.assertEqual(supervised.exact_match("Sony ", "sony"), 1.0)

    def test_field_similarities_zero_an_absent_column(self):
        features = supervised.compute_field_similarities(
            {"title": "canon eos", "brand": ""},
            {"title": "canon eos", "brand": ""},
            ["title", "brand"],
        )
        self.assertEqual(features["brand_token_overlap"], 0.0)
        self.assertEqual(features["brand_edit_sim"], 0.0)
        self.assertEqual(features["brand_exact"], 0.0)


if __name__ == "__main__":
    unittest.main()
