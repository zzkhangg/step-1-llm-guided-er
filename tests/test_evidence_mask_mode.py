"""The mask must gate selection identically in every mode, and differ only in the prompt."""
import unittest

import numpy as np

from code.attribute_selection.adaptive import (
    HybridAttributeSelector,
    attribute_blankness,
    attribute_evidence_mask,
)


class _Selector(HybridAttributeSelector):
    """Ranks attributes by a fixed importance vector, bypassing profiling."""

    def __init__(self, attributes, importance, **kw):
        super().__init__(attributes=attributes, **kw)
        self._importance = np.asarray(importance, dtype=float)

    def estimate_importance(self, pair_features):
        v = self._importance.copy()
        return v, v.copy(), v.copy(), {}


class EvidenceMaskModeTests(unittest.TestCase):
    def setUp(self):
        self.attributes = ["modelno", "title", "brand"]
        # modelno is ranked first, which is what makes the mask matter here.
        self.sel = _Selector(self.attributes, [0.6, 0.3, 0.1],
                             min_k_attributes=1, max_k_attributes=3,
                             selection_policy="gap")
        self.features = np.zeros((len(self.attributes), 6))
        self.rec_a = {"modelno": "ax17", "title": "pentel ez2 pencil", "brand": "pentel"}
        self.rec_b = {"modelno": "", "title": "pentel ez 2 pencil", "brand": "pentel"}

    def test_mask_marks_the_one_sided_blank(self):
        mask = attribute_evidence_mask(self.rec_a, self.rec_b, self.attributes)
        self.assertEqual(mask, [False, True, True])

    def test_masked_attribute_is_never_selected(self):
        mask = attribute_evidence_mask(self.rec_a, self.rec_b, self.attributes)
        selected = self.sel.select_attributes(self.features, evidence_mask=mask)["selected_attributes"]
        self.assertNotIn("modelno", selected)
        self.assertIn("title", selected)

    def test_keep_mode_shows_the_masked_attribute_without_selecting_it(self):
        mask = attribute_evidence_mask(self.rec_a, self.rec_b, self.attributes)
        selected = self.sel.select_attributes(self.features, evidence_mask=mask)["selected_attributes"]
        unmasked = self.sel.select_attributes(self.features)["selected_attributes"]
        display = list(selected) + [a for a in unmasked if a not in selected]
        # The prompt regains the blank field; the selection that produced it does not.
        self.assertIn("modelno", display)
        self.assertIn("title", display)
        self.assertNotIn("modelno", selected)
        self.assertGreater(len(display), len(selected))

    def test_pair_with_no_blanks_is_identical_in_both_modes(self):
        rec_b = dict(self.rec_b, modelno="ax17")
        mask = attribute_evidence_mask(self.rec_a, rec_b, self.attributes)
        self.assertTrue(all(mask))
        selected = self.sel.select_attributes(self.features, evidence_mask=mask)["selected_attributes"]
        unmasked = self.sel.select_attributes(self.features)["selected_attributes"]
        self.assertEqual(sorted(selected), sorted(unmasked),
                         "with nothing masked the two modes must share a cache entry")


if __name__ == "__main__":
    unittest.main()


class BothBlankMaskModeTests(unittest.TestCase):
    """The three-way blank classification the `both_blank` display mode rests on."""

    ATTRS = ["title", "brand", "modelno"]

    def test_classifies_each_blank_case(self):
        a = {"title": "widget", "brand": "acme", "modelno": ""}
        b = {"title": "widget", "brand": "", "modelno": ""}
        self.assertEqual(
            attribute_blankness(a, b, self.ATTRS),
            ["both_present", "one_sided", "both_blank"],
        )

    def test_whitespace_only_counts_as_blank(self):
        a = {"title": "   ", "brand": "acme", "modelno": "x1"}
        b = {"title": "\t\n", "brand": "acme", "modelno": "x1"}
        self.assertEqual(
            attribute_blankness(a, b, self.ATTRS),
            ["both_blank", "both_present", "both_present"],
        )

    def test_missing_key_is_blank(self):
        self.assertEqual(attribute_blankness({}, {}, ["title"]), ["both_blank"])
        self.assertEqual(attribute_blankness({"title": "x"}, {}, ["title"]), ["one_sided"])

    def test_evidence_mask_agrees_with_blankness(self):
        a = {"title": "widget", "brand": "acme", "modelno": ""}
        b = {"title": "widget", "brand": "", "modelno": ""}
        self.assertEqual(
            attribute_evidence_mask(a, b, self.ATTRS),
            [k == "both_present" for k in attribute_blankness(a, b, self.ATTRS)],
        )
