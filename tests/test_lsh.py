import unittest
from itertools import product

import numpy as np

from code.lsh import query_lsh_fast


class LSHProbeTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[1.0, 1.0, 1.0]])
        self.b = np.array(list(product([-1.0, 1.0], repeat=3)))
        self.planes = [np.eye(3)]

    def test_each_radius_matches_hamming_distance_oracle(self):
        for radius in range(4):
            with self.subTest(radius=radius):
                pairs = query_lsh_fast(
                    self.a, self.b, self.planes, num_flips=radius, top_k=8
                )
                expected = {(0, j) for j, b in enumerate(self.b)
                            if np.count_nonzero(b < 0) <= radius}
                self.assertEqual(set(pairs), expected)
                self.assertEqual(len(pairs), len(expected))

    def test_default_matches_one_bit_radius(self):
        self.assertEqual(
            query_lsh_fast(self.a, self.b, self.planes, top_k=8),
            query_lsh_fast(self.a, self.b, self.planes, num_flips=1, top_k=8),
        )

    def test_top_k_ranks_by_similarity_and_deduplicates_tables(self):
        pairs = query_lsh_fast(
            self.a, self.b, self.planes * 2, num_flips=3, top_k=1
        )
        self.assertEqual(pairs, [(0, 7)])

    def test_empty_exact_bucket(self):
        self.assertEqual(query_lsh_fast(
            self.a, self.b[:1], self.planes, num_flips=0
        ), [])

    def test_invalid_radius(self):
        for radius in (-1, 4, 1.5, True, "1"):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                query_lsh_fast(self.a, self.b, self.planes, num_flips=radius)

    def test_numpy_integer_radius(self):
        self.assertEqual(len(query_lsh_fast(
            self.a, self.b, self.planes, num_flips=np.int64(0)
        )), 1)
