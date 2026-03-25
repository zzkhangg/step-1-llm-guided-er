import json
import pandas as pd
import numpy as np
from pathlib import Path


# ── 1. Generate and save negative pairs from LSH blocking ──

def generate_negative_pairs(candidate_pairs, gt_set, save_path: str = None):
    """
    Extract negative pairs from LSH candidates — pairs that
    survived blocking but are NOT true matches.
    These are hard negatives — similar records that don't match.

    Parameters
    ----------
    candidate_pairs : list of (idxA, idxB) from LSH blocking
    gt_set          : set of (idxA, idxB) true matches
    save_path       : optional path to save negatives as CSV
    """
    negative_pairs = [
        (a, b) for a, b in candidate_pairs
        if (a, b) not in gt_set
    ]

    print(f"Total candidates  : {len(candidate_pairs)}")
    print(f"True matches      : {len(gt_set)}")
    print(f"Hard negatives    : {len(negative_pairs)}")

    df_negatives = pd.DataFrame(negative_pairs, columns=['idxA', 'idxB'])
    df_negatives['label'] = 0

    if save_path:
        df_negatives.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")

    return df_negatives


def load_negative_pairs(save_path: str):
    """Load previously saved negative pairs."""
    df = pd.read_csv(save_path)
    print(f"Loaded {len(df)} negative pairs from {save_path}")
    return df


# ── 2. Build labeled dataset from gt_set + negatives ──

def build_labeled_pairs(gt_set, df_negatives, n_pos=None, n_neg=None, seed=42):
    """
    Combine positive pairs from gt_set with negative pairs.
    Optionally sample n_pos positives and n_neg negatives for balance.

    Parameters
    ----------
    gt_set      : set of (idxA, idxB) true matches
    df_negatives: DataFrame with columns idxA, idxB, label
    n_pos       : number of positives to sample (None = use all)
    n_neg       : number of negatives to sample (None = use all)
    """
    rng = np.random.RandomState(seed)

    # positive pairs
    pos_pairs = [(a, b, 1) for a, b in gt_set]
    if n_pos and n_pos < len(pos_pairs):
        idx = rng.choice(len(pos_pairs), n_pos, replace=False)
        pos_pairs = [pos_pairs[i] for i in idx]

    # negative pairs
    neg_pairs = [(int(row['idxA']), int(row['idxB']), 0)
                 for _, row in df_negatives.iterrows()]
    if n_neg and n_neg < len(neg_pairs):
        idx = rng.choice(len(neg_pairs), n_neg, replace=False)
        neg_pairs = [neg_pairs[i] for i in idx]

    labeled_pairs = pos_pairs + neg_pairs
    rng.shuffle(labeled_pairs)

    print(f"Positives : {len(pos_pairs)}")
    print(f"Negatives : {len(neg_pairs)}")
    print(f"Total     : {len(labeled_pairs)}")

    return labeled_pairs

def normalize_id(val):
    """Normalize ID to string for consistent comparison."""
    return str(val).strip()


def build_id_maps(df_A, df_B, col_A: str, col_B: str):
    """
    Build ID → positional index maps for both tables.

    Parameters
    ----------
    df_A, df_B : pd.DataFrame
    col_A      : id column name in df_A
    col_B      : id column name in df_B
    """
    idA_to_pos = {normalize_id(row_id): pos for pos, row_id in enumerate(df_A[col_A])}
    idB_to_pos = {normalize_id(row_id): pos for pos, row_id in enumerate(df_B[col_B])}
    return idA_to_pos, idB_to_pos


def build_gt_set(df_gt, idA_to_pos, idB_to_pos,
                 col_id1: str, col_id2: str):
    """
    Build ground truth set of (posA, posB) index pairs.

    Parameters
    ----------
    df_gt       : ground truth DataFrame
    idA_to_pos  : dict from build_id_maps
    idB_to_pos  : dict from build_id_maps
    col_id1     : id column for table A in gt
    col_id2     : id column for table B in gt
    """
    gt_set = set()
    for _, row in df_gt.iterrows():
        a = normalize_id(row[col_id1])
        b = normalize_id(row[col_id2])
        if a in idA_to_pos and b in idB_to_pos:
            gt_set.add((idA_to_pos[a], idB_to_pos[b]))
    return gt_set
