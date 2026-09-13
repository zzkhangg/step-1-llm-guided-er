"""Pins the rule that an attribute absent on either side is not selected for that pair.

Every similarity feature computed on such an attribute is zero regardless of the other
record's value, so it can neither support nor contradict a match; selecting it spends
prompt tokens and a selection slot on nothing. Two paths could previously select one --
a rule filling its minimum-attribute quota, and role enforcement promoting an "identity"
attribute regardless of predicted importance. On Amazon-Walmart the second path promoted
an empty modelno, and 26 of the 135 matches the selector missed had been condensed to a
single attribute that one record left blank.
"""

import unittest

import numpy as np

from code.attribute_selection.adaptive import (
    HybridAttributeSelector,
    attribute_evidence_mask,
)


ATTRS = ["title", "brand", "modelno"]


class EvidenceMaskTests(unittest.TestCase):
    def test_mask_requires_both_sides_populated(self):
        a = {"title": "canon eos", "brand": "canon", "modelno": ""}
        b = {"title": "canon eos", "brand": "", "modelno": "xyz"}
        self.assertEqual(attribute_evidence_mask(a, b, ATTRS), [True, False, False])

    def test_mask_treats_whitespace_and_none_as_absent(self):
        a = {"title": "  ", "brand": None, "modelno": "x"}
        b = {"title": "canon", "brand": "canon", "modelno": "x"}
        self.assertEqual(attribute_evidence_mask(a, b, ATTRS), [False, False, True])


class SelectorEvidenceTests(unittest.TestCase):
    def selector(self, **kwargs):
        params = dict(
            attributes=ATTRS,
            min_k_attributes=1,
            max_k_attributes=3,
            selection_policy="gap",
            required_attribute_roles=["identity"],
            attribute_roles={"title": "identity", "brand": "context", "modelno": "identity"},
        )
        params.update(kwargs)
        sel = HybridAttributeSelector(**params)
        # Fit on profiles that make modelno look like the most important attribute.
        features = np.tile(np.linspace(0.1, 0.9, len(ATTRS) * 6), (4, 1))
        importance = np.tile(np.array([0.2, 0.1, 0.7]), (4, 1))
        sel.fit(features, importance)
        return sel

    def test_absent_attribute_is_not_selected(self):
        sel = self.selector()
        features = np.linspace(0.1, 0.9, len(ATTRS) * 6)
        chosen = sel.select_attributes(features, evidence_mask=[True, True, False])
        self.assertNotIn("modelno", chosen["selected_attributes"])
        self.assertTrue(chosen["selected_attributes"], "a subset must still be produced")

    def test_role_enforcement_cannot_promote_an_absent_attribute(self):
        # modelno carries the "identity" role and the highest predicted importance, so
        # both the ranking and role enforcement would otherwise pull it in.
        sel = self.selector()
        features = np.linspace(0.1, 0.9, len(ATTRS) * 6)
        chosen = sel.select_attributes(features, evidence_mask=[True, False, False])
        self.assertEqual(chosen["selected_attributes"], ["title"])

    def test_mask_is_ignored_when_nothing_carries_evidence(self):
        # Some subset must still be sent; falling back to the ranking beats sending none.
        sel = self.selector()
        features = np.linspace(0.1, 0.9, len(ATTRS) * 6)
        chosen = sel.select_attributes(features, evidence_mask=[False, False, False])
        self.assertTrue(chosen["selected_attributes"])

    def test_omitting_the_mask_keeps_previous_behaviour(self):
        sel = self.selector()
        features = np.linspace(0.1, 0.9, len(ATTRS) * 6)
        without = sel.select_attributes(features)["selected_attributes"]
        allowed = sel.select_attributes(features, evidence_mask=[True, True, True])["selected_attributes"]
        self.assertEqual(without, allowed)


if __name__ == "__main__":
    unittest.main()
