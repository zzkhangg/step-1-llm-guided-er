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
