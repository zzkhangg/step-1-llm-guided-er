from collections import defaultdict
import numpy as np


# LSH parameters
num_tables = 15
num_planes = 6


def create_random_planes(num_tables: int, num_planes: int, dim: int):
    planes = []
    for _ in range(num_tables):
        planes.append(np.random.randn(num_planes, dim))
    return planes


def _bits_to_int(bits: np.ndarray) -> int:
    """Convert binary array to integer for fast dict lookup."""
    return int(''.join(bits.astype(str)), 2)


def query_lsh_fast(tableA_vectors, tableB_vectors, planes_list, num_flips=1, top_k=5):
    """
    Fast LSH blocking using integer hash keys + dict lookup (no matrix comparison).
    """
    candidate_pairs = []
    num_planes = planes_list[0].shape[0]

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

            # exact match — O(1) dict lookup
            if h in buckets:
                candidates.update(buckets[h])

            # multi-probe — flip each bit
            for flip_idx in range(num_planes):
                flipped    = bitsA.copy()
                flipped[flip_idx] ^= 1
                h_flipped  = int(flipped @ powers)
                if h_flipped in buckets:
                    candidates.update(buckets[h_flipped])

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