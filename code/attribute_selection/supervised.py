import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from difflib import SequenceMatcher

from ..lsh import query_lsh_fast


# ── 1. Field-level similarity features ──

# A value missing from both records carries no evidence either way, so all three features
# score it 0.0 rather than treating "both blank" as agreement. Ranking attributes by
# classifier weight is otherwise biased towards sparsely populated columns, whose blanks
# co-occur with the label far more often than their values do.
def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity between token sets."""
    set_a = set(str(a).lower().split())
    set_b = set(str(b).lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def edit_similarity(a: str, b: str) -> float:
    """Normalized edit distance similarity."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def exact_match(a: str, b: str) -> float:
    """1 if values are identical and present, 0 otherwise."""
    if not str(a).strip() or not str(b).strip():
        return 0.0
    return float(str(a).lower().strip() == str(b).lower().strip())


def compute_field_similarities(rowA: dict, rowB: dict, cols: list) -> dict:
    """
    Compute per-field similarity features between two records.
    Returns a flat feature dict with 3 features per field.
    """
    features = {}
    for col in cols:
        a = str(rowA.get(col, '')) if not pd.isna(rowA.get(col, '')) else ''
        b = str(rowB.get(col, '')) if not pd.isna(rowB.get(col, '')) else ''
        features[f'{col}_token_overlap'] = token_overlap(a, b)
        features[f'{col}_edit_sim']      = edit_similarity(a, b)
        features[f'{col}_exact']         = exact_match(a, b)
    return features


# ── 2. Build feature matrix from labeled pairs ──

def build_feature_matrix(df_A, df_B, labeled_pairs, cols):
    """
    Build feature matrix X and label vector y from labeled pairs.

    Parameters
    ----------
    df_A, df_B     : DataFrames
    labeled_pairs  : list of (idxA, idxB, label) where label is 1/0
    cols           : attribute columns to compute similarity on
    """
    rows = []
    labels = []
    for idx_a, idx_b, label in labeled_pairs:
        rowA = df_A.iloc[idx_a].to_dict()
        rowB = df_B.iloc[idx_b].to_dict()
        features = compute_field_similarities(rowA, rowB, cols)
        rows.append(features)
        labels.append(label)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


# ── 3. Train classifier and extract feature importance ──

def train_attribute_selector(df_A, df_B, labeled_pairs, cols):
    """
    Train logistic regression on field-level similarity features.
    Returns selected attributes ranked by discriminative power.

    Parameters
    ----------
    labeled_pairs : list of (idxA, idxB, label)
    cols          : all candidate attribute columns
    """
    print(f"Training on {len(labeled_pairs)} labeled pairs...")
    print(f"Positive matches : {sum(1 for _, _, l in labeled_pairs if l == 1)}")
    print(f"Negative matches : {sum(1 for _, _, l in labeled_pairs if l == 0)}")

    if not cols:
        raise ValueError("Supervised tuple strategy requires at least one common attribute column.")
    if not labeled_pairs:
        raise ValueError("Supervised tuple strategy requires labeled positive and negative pairs.")

    labels = {label for _, _, label in labeled_pairs}
    if labels != {0, 1}:
        raise ValueError(
            "Supervised tuple strategy requires both positive and negative pairs. "
            f"Observed labels: {sorted(labels)}"
        )

    X, y = build_feature_matrix(df_A, df_B, labeled_pairs, cols)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train logistic regression
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_scaled, y)

    # ── analyze weights per field ──
    feature_names  = X.columns.tolist()
    weights        = np.abs(clf.coef_[0])

    # aggregate importance per attribute (max across 3 similarity metrics)
    attr_importance = {}
    for col in cols:
        col_features = [f for f in feature_names if f.startswith(col + '_')]
        col_indices  = [feature_names.index(f) for f in col_features]
        attr_importance[col] = weights[col_indices].max()

    total_importance = sum(attr_importance.values())
    if total_importance > 0:
        attr_importance = {
            attr: score / total_importance
            for attr, score in attr_importance.items()
        }

    # rank attributes by importance
    ranked = sorted(attr_importance.items(), key=lambda x: -x[1])

    print("\n[Attribute Importance Ranking]")
    for attr, score in ranked:
        print(f"  {attr:<20} : {score:.4f}")

    # evaluate classifier
    y_pred = clf.predict(X_scaled)
    print("\n[Classifier Performance on Training Data]")
    print(classification_report(y, y_pred))

    return clf, scaler, ranked, X.columns.tolist()


# ── 4. Select top attributes based on importance ──

# The importance scores are normalised to sum to one, so an attribute's mean share is
# 1/m and a fixed cut-off is not scale-free: the wider the schema, the smaller every
# share, until no attribute clears the bar and the strategy selects nothing. That is not
# hypothetical -- on Amazon-Walmart's 12 attributes the mean share is 0.083 against the
# 0.3 default, the selection came back empty, and the run aborted.
#
# The bar is therefore expressed as a multiple of the uniform share. The multiple is 1.2
# because that reproduces the previous behaviour exactly on a four-attribute schema
# (1.2/4 = 0.3, the old default), so the DBLP-ACM results this value was never chosen
# against are unchanged. It was not picked to make any particular dataset work.
UNIFORM_SHARE_MULTIPLE = 1.2


def select_top_attributes(ranked, threshold=None, top_k=None,
                          uniform_share_multiple=UNIFORM_SHARE_MULTIPLE):
    """
    Keep attributes carrying at least ``uniform_share_multiple`` times the uniform share.

    Parameters
    ----------
    ranked    : list of (attr, score) from train_attribute_selector, scores summing to 1
    threshold : absolute cut-off, retained for callers that need the old behaviour;
                when given it overrides the relative bar
    top_k     : maximum number of ranked attributes to retain
    uniform_share_multiple : bar as a multiple of 1/len(ranked)
    """
    if not ranked:
        return []

    if threshold is not None:
        bar = float(threshold)
        bar_text = f"score >= {bar:g}"
    else:
        bar = float(uniform_share_multiple) / len(ranked)
        bar_text = f"score >= {uniform_share_multiple:g}/{len(ranked)} = {bar:.4f}"

    selected = [attr for attr, score in ranked if score >= bar]
    if not selected:
        # Every attribute can sit below the bar when importance is near-uniform, which
        # says the attributes are indistinguishable, not that none of them is usable.
        # Falling back to the top-ranked one keeps the run alive and is still a
        # defensible global subset; returning nothing is not.
        selected = [ranked[0][0]]
        print(f"\nNo attribute cleared {bar_text}; falling back to the top-ranked one.")
    if top_k is not None:
        selected = selected[: int(top_k)]
    top_k_text = "all" if top_k is None else str(top_k)
    print(f"\nSelected attributes ({bar_text}, top_k={top_k_text}): {selected}")
    return selected


def supervised_selection(
    df_A,
    df_B,
    df_train,          # gold pairs
    id_col_A,
    id_col_B,
    idA_to_pos,
    idB_to_pos,
    tableA_vectors,    # vectors used for LSH
    tableB_vectors,
    planes_list,
    blocking_top_k=5,
    neg_ratio=1.0,     # negatives : positives
    threshold=None,
    save_model=False,
    model_path="attr_selector_clf.pkl",
    scaler_path="attr_selector_scaler.pkl"
):
    """
    Full supervised attribute selection using LSH-based hard negatives.
    """

    # ── 1. Build positive pairs from ground truth ──
    positives = []
    perfect_mapping_set = set()

    for _, row in df_train.iterrows():

        a = str(row[id_col_A]).strip()
        b = str(row[id_col_B]).strip()

        if a in idA_to_pos and b in idB_to_pos:
            idxA = idA_to_pos[a]
            idxB = idB_to_pos[b]
            positives.append((idxA, idxB, 1))
            perfect_mapping_set.add((idxA, idxB))

    print(f"Positive pairs: {len(positives)}")

    # ── 2. Run LSH blocking to get candidate pairs ──
    print("\nRunning LSH blocking...")
    candidate_pairs = query_lsh_fast(
        tableA_vectors,
        tableB_vectors,
        planes_list,
        top_k=blocking_top_k
    )

    print(f"Total candidate pairs from LSH: {len(candidate_pairs)}")

    # ── 3. Generate HARD negatives ──
    negatives = []
    for idxA, idxB in candidate_pairs:
        if (idxA, idxB) not in perfect_mapping_set:
            negatives.append((idxA, idxB, 0))

    print(f"Hard negatives before sampling: {len(negatives)}")

    # ── 4. Balance dataset (optional) ──
    n_pos = len(positives)
    n_neg_keep = int(n_pos * neg_ratio)

    if len(negatives) > n_neg_keep:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(negatives), size=n_neg_keep, replace=False)
        negatives = [negatives[i] for i in indices]

    print(f"Negatives used: {len(negatives)}")

    # ── 5. Combine labeled pairs ──
    labeled_pairs = positives + negatives

    print(f"Total training pairs: {len(labeled_pairs)}")

    # ── 6. Select common columns ──
    cols = [c for c in df_A.columns if c in df_B.columns]

    # ── 7. Train attribute selector ──
    clf, scaler, ranked, feature_names = train_attribute_selector(
        df_A,
        df_B,
        labeled_pairs,
        cols
    )

    # ── 8. Select top attributes ──
    selected_attrs = select_top_attributes(ranked, threshold=threshold)

    # ── 9. Save model if needed ──
    if save_model:
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    # ── 10. Return reduced tables + ranking ──
    return (
        df_A[selected_attrs].copy(),
        df_B[selected_attrs].copy(),
        ranked
    )
