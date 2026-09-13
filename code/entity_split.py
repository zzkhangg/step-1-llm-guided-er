"""Deterministic, gold-connected entity partitions for development experiments."""

import numpy as np


def entity_split(n_a, n_b, gold, seed=42):
    parent = list(range(n_a + n_b))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in sorted(gold):
        parent[root(n_a + b)] = root(a)
    groups = {}
    for node in range(n_a + n_b):
        groups.setdefault(root(node), []).append(node)
    # Stratify matched components and unmatched records to retain gold coverage.
    strata = [[], [], []]
    for group in groups.values():
        has_a = any(x < n_a for x in group)
        has_b = any(x >= n_a for x in group)
        strata[0 if has_a and has_b else 1 if has_a else 2].append(group)
    rng = np.random.RandomState(seed)
    result = {name: {"a": [], "b": []} for name in ("train", "validation", "test")}
    for groups in strata:
        order = rng.permutation(len(groups))
        cuts = (int(len(groups) * .6), int(len(groups) * .8))
        for rank, idx in enumerate(order):
            name = "train" if rank < cuts[0] else "validation" if rank < cuts[1] else "test"
            for node in groups[idx]:
                result[name]["a" if node < n_a else "b"].append(
                    node if node < n_a else node - n_a)
    for part in result.values():
        part["a"].sort()
        part["b"].sort()
    return result
