from collections import defaultdict
from itertools import combinations
from numbers import Integral
import numpy as np


def create_random_planes(num_tables: int, num_planes: int, dim: int, seed: int = 42):
    rng = np.random.RandomState(seed)   # ← fixed seed, reproducible planes
    planes = []
    for _ in range(num_tables):
        planes.append(rng.randn(num_planes, dim))
    return planes


def query_lsh_fast(tableA_vectors, tableB_vectors, planes_list, num_flips=1, top_k=5):
    """
    Fast LSH blocking using integer hash keys + dict lookup (no matrix comparison).

    num_flips is the maximum Hamming distance to probe, including the exact
    bucket. Zero probes only the exact bucket; one preserves the original
    behaviour. Probe count per table is sum(comb(num_planes, r), r=0..num_flips).
    """
    candidate_pairs = []
    num_planes = planes_list[0].shape[0]
    if isinstance(num_flips, bool) or not isinstance(num_flips, Integral):
        raise ValueError("num_flips must be an integer")
    if not 0 <= num_flips <= num_planes:
        raise ValueError("num_flips must be between 0 and num_planes")
    if any(planes.shape[0] != num_planes for planes in planes_list):
        raise ValueError("All LSH tables must have the same number of planes")

    # Reuse XOR masks for every query and table; include all smaller radii.
    probe_masks = [
        sum(1 << bit for bit in flipped_bits)
        for radius in range(num_flips + 1)
        for flipped_bits in combinations(range(num_planes), radius)
    ]

    # ── pre-hash all of table B into dicts: {hash_int -> [idx, ...]} ──
    print("Pre-hashing Table B...")
    tableB_tables = []
    for planes in planes_list:
        projections = np.dot(tableB_vectors, planes.T)      # (nB, num_planes)
        bits_matrix = (projections > 0).astype(np.uint8)    # (nB, num_planes)
        buckets = defaultdict(list)
        # vectorized: pack each row of bits into an integer
        powers = (1 << np.arange(num_planes, dtype=np.int64))
        hash_ints = bits_matrix @ powers                     # (nB,) — one int per record
        for idx, h in enumerate(hash_ints):
            buckets[int(h)].append(idx)
        tableB_tables.append((buckets, powers))
    print("Pre-hashing done.")

    # ── query each record in A ──
    for i, vecA in enumerate(tableA_vectors):
        candidates = set()

        for t, planes in enumerate(planes_list):
            buckets, powers = tableB_tables[t]

            proj  = np.dot(planes, vecA)
            bitsA = (proj > 0).astype(np.uint8)
            h     = int(bitsA @ powers)

            for mask in probe_masks:
                candidates.update(buckets.get(h ^ mask, ()))

        if not candidates:
            continue

        candidate_list = list(candidates)
        sims       = np.dot(tableB_vectors[candidate_list], vecA)
        top_idxs   = np.argsort(-sims)[:top_k]
        for idx in top_idxs:
            candidate_pairs.append((i, candidate_list[idx]))

        if i % 500 == 0:
            print(f"  Processed {i}/{len(tableA_vectors)} | "
                  f"candidates so far: {len(candidate_pairs)}")

    return candidate_pairs
