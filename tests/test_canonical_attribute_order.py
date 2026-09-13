"""Guards the canonical prompt field order.

Both selection stages rank attributes by importance and previously passed that
ranking straight through to the matcher prompt. Because the ranking moves with the
profile sample, reseeding permuted the prompt as well as changing the attribute
subset -- and on DBLP-ACM permuting a fixed subset flipped 5.4% of the matcher's
answers, against 0.1% for an identical prompt. These tests pin the field order to
the table's column order so that attribute choice is the only thing a reseed varies.
"""

import unittest

import numpy as np
import pandas as pd

from code.attribute_selection.adaptive import HybridAttributeSelector
from code.main import apply_global_tuple_strategy


ATTRS = ["title", "authors", "venue", "year"]


def canonical(attributes, order):
    position = {attr: idx for idx, attr in enumerate(order)}
    return sorted(attributes, key=lambda attr: position[attr])


class Step1CanonicalOrderTests(unittest.TestCase):
    def setUp(self):
        self.df_a = pd.DataFrame([dict.fromkeys(ATTRS, "x")])
        self.df_b = pd.DataFrame([dict.fromkeys(ATTRS, "x")])

    def test_manual_selection_is_reordered_to_column_order(self):
        scrambled = ["year", "authors", "title"]
        out_a, out_b, summary = apply_global_tuple_strategy(
            self.df_a, self.df_b, "manual", set(), [],
            threshold=0.0, top_k=None, n_pos=1, n_neg=1,
            manual_attributes=scrambled,
        )

        expected = ["title", "authors", "year"]
        self.assertEqual(summary["selected_attributes"], expected)
        self.assertEqual(list(out_a.columns), expected)
        self.assertEqual(list(out_b.columns), expected)
        self.assertTrue(summary["canonical_attribute_order"])

    def test_reordering_preserves_the_selected_set(self):
        scrambled = ["venue", "title"]
        _, _, summary = apply_global_tuple_strategy(
            self.df_a, self.df_b, "manual", set(), [],
            threshold=0.0, top_k=None, n_pos=1, n_neg=1,
            manual_attributes=scrambled,
        )
        self.assertEqual(set(summary["selected_attributes"]), set(scrambled))


class Step15CanonicalOrderTests(unittest.TestCase):
    def test_canonicalization_collapses_permutations_onto_distinct_sets(self):
        rng = np.random.default_rng(0)
        selector = HybridAttributeSelector(
            attributes=ATTRS,
            min_k_attributes=2,
            max_k_attributes=4,
            cumulative_importance_threshold=0.8,
            top_k_retrieval=3,
            fusion_strategy="weighted_average",
            predictor_weight=0.5,
            retrieval_weight=0.5,
        )
        importance = rng.random((12, len(ATTRS)))
        selector.fit(rng.random((12, len(ATTRS) * 6)), importance / importance.sum(1, keepdims=True))

        raw, fixed, sets = set(), set(), set()
        for _ in range(200):
            selected = selector.select_attributes(rng.random(len(ATTRS) * 6))["selected_attributes"]
            raw.add(tuple(selected))
            fixed.add(tuple(canonical(selected, ATTRS)))
            sets.add(frozenset(selected))

        # The selector emits more orderings than there are subsets; canonicalizing
        # must leave exactly one ordering per subset and invent no new subsets.
        self.assertGreater(len(raw), len(sets))
        self.assertEqual(len(fixed), len(sets))
        self.assertEqual({frozenset(order) for order in fixed}, sets)

    def test_canonical_order_matches_table_column_order(self):
        for selected in (["year", "title"], ["venue", "authors", "title"], list(reversed(ATTRS))):
            with self.subTest(selected=selected):
                ordered = canonical(selected, ATTRS)
                self.assertEqual(ordered, [attr for attr in ATTRS if attr in selected])


if __name__ == "__main__":
    unittest.main()
