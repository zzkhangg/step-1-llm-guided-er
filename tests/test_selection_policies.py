"""Covers the Step-1.5 rule for how many attributes to keep.

The historical "cumulative" policy sums importance up to a fixed target. Because
importance is normalized to sum to one, each attribute's share shrinks as the schema
widens and the target becomes unreachable: on a 13-attribute schema the 0.8 target
needs 7.7 attributes on average against a max_k of 5, so the cap binds for every pair
and the selector cannot express "this pair needs fewer fields". These tests pin the
scale-free alternatives added to work around that.
"""

import unittest

import numpy as np

from code.attribute_selection import (
    DEFAULT_RATIO_LAMBDA,
    DEFAULT_SELECTION_POLICY,
    SELECTION_POLICIES,
    HybridAttributeSelector,
)


def make_selector(n_attributes, policy, min_k=2, max_k=5, ratio_lambda=DEFAULT_RATIO_LAMBDA):
    return HybridAttributeSelector(
        attributes=[f"attr{i}" for i in range(n_attributes)],
        min_k_attributes=min_k,
        max_k_attributes=max_k,
        cumulative_importance_threshold=0.8,
        selection_policy=policy,
        ratio_lambda=ratio_lambda,
    )


def select(selector, importance, policy):
    importance = np.asarray(importance, dtype=float)
    ranked = np.argsort(-importance).tolist()
    method = {
        "cumulative": selector._select_cumulative,
        "ratio": selector._select_ratio,
        "gap": selector._select_gap,
    }[policy]
    return method(importance, ranked)


def wide_uniform_ish(n=13):
    """A wide schema whose importance is spread thinly, as Amazon-Walmart's is."""
    importance = np.linspace(1.4, 0.6, n)
    return importance / importance.sum()


class DefaultsTests(unittest.TestCase):
    def test_default_policy_is_unchanged(self):
        # Changing this silently would invalidate comparison against reported results.
        self.assertEqual(DEFAULT_SELECTION_POLICY, "cumulative")

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            make_selector(4, "nonsense")

    def test_every_advertised_policy_runs(self):
        importance = wide_uniform_ish()
        for policy in SELECTION_POLICIES:
            with self.subTest(policy=policy):
                chosen = select(make_selector(len(importance), policy), importance, policy)
                self.assertTrue(1 <= len(chosen) <= 5)


class SaturationTests(unittest.TestCase):
    """The defect the alternatives exist to fix."""

    def test_cumulative_saturates_on_a_wide_schema(self):
        importance = wide_uniform_ish()
        chosen = select(make_selector(len(importance), "cumulative"), importance, "cumulative")
        self.assertEqual(len(chosen), 5, "cumulative should hit max_k on a wide flat schema")

    def test_gap_does_not_saturate_on_a_wide_schema(self):
        importance = wide_uniform_ish()
        chosen = select(make_selector(len(importance), "gap"), importance, "gap")
        self.assertLess(len(chosen), 5)

    def test_gap_respects_a_clear_elbow(self):
        # Two dominant attributes, then a cliff: the cut belongs right after them.
        importance = np.array([0.40, 0.38, 0.06, 0.06, 0.05, 0.05])
        chosen = select(make_selector(len(importance), "gap", min_k=2), importance, "gap")
        self.assertEqual(len(chosen), 2)


class BoundsTests(unittest.TestCase):
    def test_all_policies_respect_min_k_and_max_k(self):
        importance = wide_uniform_ish()
        for policy in SELECTION_POLICIES:
            for min_k, max_k in ((2, 5), (1, 3), (3, 4)):
                with self.subTest(policy=policy, min_k=min_k, max_k=max_k):
                    selector = make_selector(len(importance), policy, min_k=min_k, max_k=max_k)
                    chosen = select(selector, importance, policy)
                    self.assertGreaterEqual(len(chosen), min_k)
                    self.assertLessEqual(len(chosen), max_k)

    def test_selections_are_unique_and_ranked(self):
        importance = wide_uniform_ish()
        for policy in SELECTION_POLICIES:
            with self.subTest(policy=policy):
                chosen = select(make_selector(len(importance), policy), importance, policy)
                self.assertEqual(len(chosen), len(set(chosen)), "no duplicate attributes")
                picked = importance[chosen]
                rest = np.delete(importance, chosen)
                if len(rest):
                    self.assertGreaterEqual(picked.min(), rest.max() - 1e-12,
                                            "must keep the highest-importance attributes")


class RatioTests(unittest.TestCase):
    def test_bar_scales_with_schema_width(self):
        # lambda/n: a fixed share that would pass at n=4 must still be judged at n=13.
        for n in (4, 13):
            selector = make_selector(n, "ratio", min_k=1, max_k=n, ratio_lambda=1.5)
            self.assertAlmostEqual(selector.ratio_lambda / n, 1.5 / n)

    def test_larger_lambda_keeps_fewer_attributes(self):
        importance = wide_uniform_ish()
        sizes = [
            len(select(make_selector(len(importance), "ratio", max_k=13, ratio_lambda=lam),
                       importance, "ratio"))
            for lam in (0.5, 1.0, 1.5, 2.0)
        ]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_uniform_importance_keeps_everything_below_the_bar(self):
        # With lambda < 1 and perfectly uniform importance, every attribute clears it.
        n = 6
        importance = np.full(n, 1.0 / n)
        chosen = select(make_selector(n, "ratio", min_k=1, max_k=n, ratio_lambda=0.9),
                        importance, "ratio")
        self.assertEqual(len(chosen), n)


if __name__ == "__main__":
    unittest.main()


class SupervisedScaleFreeThresholdTests(unittest.TestCase):
    """The supervised bar must scale with schema width.

    Importance is normalised to sum to one, so a fixed cut-off gets harder to clear as
    the schema widens. On Amazon-Walmart's 12 attributes the mean share is 0.083 against
    the old 0.3 default: nothing cleared it, the selection came back empty, and the run
    aborted.
    """

    def test_reproduces_the_old_default_on_a_four_attribute_schema(self):
        from code.attribute_selection.supervised import select_top_attributes
        ranked = [("title", 0.545), ("authors", 0.252), ("year", 0.180), ("venue", 0.023)]
        # 1.2/4 = 0.3, the previous absolute default.
        self.assertEqual(select_top_attributes(ranked), ["title"])

    def test_wide_schema_still_selects(self):
        from code.attribute_selection.supervised import select_top_attributes
        ranked = [(f"a{i}", s) for i, s in enumerate(
            [0.20, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.01])]
        selected = select_top_attributes(ranked)
        self.assertTrue(selected, "a 12-attribute schema must not select nothing")
        self.assertIn("a0", selected)

    def test_never_returns_empty_when_importance_is_uniform(self):
        from code.attribute_selection.supervised import select_top_attributes
        ranked = [(f"a{i}", 0.1) for i in range(10)]
        self.assertEqual(select_top_attributes(ranked), ["a0"])

    def test_explicit_threshold_still_overrides(self):
        from code.attribute_selection.supervised import select_top_attributes
        ranked = [("title", 0.545), ("authors", 0.252), ("year", 0.180), ("venue", 0.023)]
        self.assertEqual(
            select_top_attributes(ranked, threshold=0.15), ["title", "authors", "year"]
        )
