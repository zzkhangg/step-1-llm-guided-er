import unittest

from code.entity_split import entity_split


class EntitySplitTests(unittest.TestCase):
    def test_components_stay_together_and_partition_covers_every_record(self):
        gold = {(0, 0), (0, 1), (1, 1), (2, 2), (3, 3), (4, 4)}
        parts = entity_split(10, 12, gold)
        owners = {}
        for name, part in parts.items():
            for side in ("a", "b"):
                for index in part[side]:
                    self.assertNotIn((side, index), owners)
                    owners[side, index] = name
        self.assertEqual(len(owners), 22)
        for a, b in gold:
            self.assertEqual(owners["a", a], owners["b", b])
        self.assertEqual(parts, entity_split(10, 12, gold))

    def test_gold_input_order_does_not_change_split(self):
        gold = [(i, i) for i in range(20)]
        self.assertEqual(entity_split(25, 25, gold), entity_split(25, 25, gold[::-1]))
