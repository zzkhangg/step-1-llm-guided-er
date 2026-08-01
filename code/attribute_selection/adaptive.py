"""Pair-adaptive attribute selection for post-blocking ER candidates.

This module implements the Step 1.5 profiling flow:
candidate-pair sampling, attribute-level similarity features, conversion of
LLM profiling output to importance vectors, and hybrid predictor/retriever
selection before LLM pairwise matching.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import math
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_ORDER = (
    "token_overlap",
    "edit_similarity",
    "exact_match",
    "semantic_similarity",
    "numeric_similarity",
    "is_numeric",
)

OTHER_ATTRIBUTE_HINTS = (
    "image_url",
    "imageurl",
    "source_url",
    "original_id",
    "source_id",
    "record_id",
    "row_id",
    "url",
    "link",
    "image",
)

IDENTITY_ATTRIBUTE_HINTS = (
    "title",
    "name",
    "modelno",
    "model",
    "isbn",
    "doi",
    "sku",
    "upc",
    "email",
    "phone",
)

STRONG_SUPPORT_ATTRIBUTE_HINTS = (
    "brand",
    "authors",
    "author",
)

CONTEXT_ATTRIBUTE_HINTS = (
    "year",
    "venue",
    "city",
    "state",
    "country",
    "type",
    "category",
    "class",
)

RELATIVE_NUMERIC_ATTRIBUTE_HINTS = (
    "price",
    "cost",
    "amount",
    "msrp",
)


def _clean_value(value) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _edit_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _exact_match(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    return float(a.lower().strip() == b.lower().strip())


def _parse_number(value: str):
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        match = re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", value)
        if match:
            return float(match.group(0))
    return None


def _uses_relative_numeric_scale(attribute: str) -> bool:
    attr_text = str(attribute).lower()
    return any(hint in attr_text for hint in RELATIVE_NUMERIC_ATTRIBUTE_HINTS)


def compute_numeric_scales(table_a, table_b, attributes, method="iqr") -> Dict[str, float]:
    """Compute dataset-level scales for numeric attributes that need them."""
    scales = {}
    for attr in attributes:
        if _uses_relative_numeric_scale(attr):
            continue

        values = []
        for table in (table_a, table_b):
            if attr not in table:
                continue
            for value in table[attr]:
                number = _parse_number(_clean_value(value))
                if number is not None:
                    values.append(number)

        if not values:
            continue

        arr = np.asarray(values, dtype=float)
        if method == "iqr":
            q1, q3 = np.percentile(arr, [25, 75])
            scale = float(q3 - q1)
        elif method == "std":
            scale = float(np.std(arr))
        elif method == "range":
            scale = float(np.max(arr) - np.min(arr))
        else:
            raise ValueError("method must be one of: iqr, std, range")

        if scale <= 0:
            scale = float(max(np.max(np.abs(arr)), 1.0))
        scales[attr] = scale
    return scales


def _safe_normalize(vector: np.ndarray, fallback_size: int | None = None) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total > 0:
        return arr / total
    size = fallback_size if fallback_size is not None else arr.size
    if size == 0:
        return arr
    return np.full(size, 1.0 / size, dtype=float)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        return _safe_normalize(arr)
    out = np.zeros_like(arr, dtype=float)
    for idx, row in enumerate(arr):
        out[idx] = _safe_normalize(row, fallback_size=arr.shape[1])
    return out


def _as_2d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def infer_attribute_roles(attributes: Sequence[str]) -> Dict[str, str]:
    """Infer coarse evidence roles from attribute names."""
    roles = {}
    for attr in attributes:
        attr_text = str(attr).lower()
        if any(hint in attr_text for hint in OTHER_ATTRIBUTE_HINTS):
            roles[attr] = "other"
        elif any(hint in attr_text for hint in IDENTITY_ATTRIBUTE_HINTS):
            roles[attr] = "identity"
        elif any(hint in attr_text for hint in STRONG_SUPPORT_ATTRIBUTE_HINTS):
            roles[attr] = "strong_support"
        elif any(hint in attr_text for hint in CONTEXT_ATTRIBUTE_HINTS):
            roles[attr] = "context"
        else:
            roles[attr] = "other"
    return roles


def _similarity_scores(candidate_pairs, pair_similarity_scores) -> np.ndarray:
    if isinstance(pair_similarity_scores, Mapping):
        return np.asarray(
            [pair_similarity_scores.get(pair, pair_similarity_scores.get(tuple(pair), 0.0)) for pair in candidate_pairs],
            dtype=float,
        )
    scores = np.asarray(pair_similarity_scores, dtype=float)
    if scores.shape[0] != len(candidate_pairs):
        raise ValueError("pair_similarity_scores must align with candidate_pairs")
    return scores


def _stratified_quotas(target, medium_ratio=0.6, low_ratio=0.2, high_ratio=0.2):
    ratios = {"medium": medium_ratio, "low": low_ratio, "high": high_ratio}
    quotas = {name: int(round(target * ratio)) for name, ratio in ratios.items()}
    quotas["medium"] += target - sum(quotas.values())
    return quotas


def _similarity_strata(scores):
    q_low, q_high = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "low": np.where(scores <= q_low)[0].tolist(),
        "medium": np.where((scores > q_low) & (scores < q_high))[0].tolist(),
        "high": np.where(scores >= q_high)[0].tolist(),
    }


def select_random_profile_pairs(candidate_pairs, sample_size, seed=42):
    """Uniform random baseline for choosing post-blocking profile pairs."""
    pairs = list(candidate_pairs)
    if sample_size <= 0 or not pairs:
        return [], []

    target = min(int(sample_size), len(pairs))
    rng = np.random.RandomState(seed)
    selected = rng.choice(len(pairs), size=target, replace=False).tolist()
    return [pairs[idx] for idx in selected], selected


def compute_importance_proxy_scores(candidate_pairs, pair_similarity_scores, pair_features) -> np.ndarray:
    """Score pairs by how informative they are likely to be before LLM profiling.

    The score is label-free. It favors ambiguous blocked pairs whose attributes
    provide mixed evidence, because those pairs are more useful for learning
    which attributes should carry importance during transfer.
    """
    pairs = list(candidate_pairs)
    if not pairs:
        return np.asarray([], dtype=float)

    features = _pair_feature_matrix(pairs, pair_features)
    scores = np.clip(_similarity_scores(pairs, pair_similarity_scores), 0.0, 1.0)

    feature_count = len(FEATURE_ORDER)
    if features.shape[1] >= feature_count and features.shape[1] % feature_count == 0:
        grouped = features.reshape(features.shape[0], -1, feature_count)
        # Use the agreement signals, excluding the is_numeric indicator.
        attribute_evidence = np.max(grouped[:, :, : feature_count - 1], axis=2)
    else:
        attribute_evidence = np.clip(features, 0.0, 1.0)

    evidence_mean = attribute_evidence.mean(axis=1)
    evidence_contrast = np.clip(attribute_evidence.std(axis=1) / 0.5, 0.0, 1.0)
    evidence_mixedness = 1.0 - np.clip(np.abs(evidence_mean - 0.5) / 0.5, 0.0, 1.0)

    if len(scores) <= 1:
        similarity_uncertainty = np.ones_like(scores, dtype=float)
    else:
        center = float(np.median(scores))
        spread = float(np.max(np.abs(scores - center)))
        if spread <= 1e-12:
            similarity_uncertainty = np.ones_like(scores, dtype=float)
        else:
            similarity_uncertainty = 1.0 - np.clip(np.abs(scores - center) / spread, 0.0, 1.0)

    proxy_scores = (
        0.50 * evidence_contrast
        + 0.30 * similarity_uncertainty
        + 0.20 * evidence_mixedness
    )
    return np.clip(proxy_scores, 0.0, 1.0)


def select_importance_based_profile_pairs(
    candidate_pairs,
    pair_similarity_scores,
    pair_features,
    sample_size,
    seed=42,
):
    """Select profile pairs with high label-free expected importance signal."""
    pairs = list(candidate_pairs)
    if sample_size <= 0 or not pairs:
        return [], []

    target = min(int(sample_size), len(pairs))
    proxy_scores = compute_importance_proxy_scores(pairs, pair_similarity_scores, pair_features)
    rng = np.random.RandomState(seed)
    tie_breaker = rng.random(len(pairs))
    selected = np.lexsort((tie_breaker, -proxy_scores))[:target].tolist()
    return [pairs[idx] for idx in selected], selected


def select_profile_pairs(
    candidate_pairs,
    pair_similarity_scores,
    sample_size,
    medium_ratio=0.6,
    low_ratio=0.2,
    high_ratio=0.2,
    seed=42,
):
    """Stratify candidate pairs by similarity quantiles and sample for profiling.

    Returns
    -------
    tuple
        ``(selected_pairs, selected_indices)`` where selected indices refer to
        the original ``candidate_pairs`` sequence.
    """
    pairs = list(candidate_pairs)
    if sample_size <= 0 or not pairs:
        return [], []

    scores = np.clip(_similarity_scores(pairs, pair_similarity_scores), 0.0, 1.0)
    target = min(int(sample_size), len(pairs))
    strata = _similarity_strata(scores)

    rng = np.random.RandomState(seed)
    quotas = _stratified_quotas(
        target,
        medium_ratio=medium_ratio,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
    )

    selected = []
    for name in ("medium", "low", "high"):
        available = [idx for idx in strata[name] if idx not in selected]
        if not available:
            continue
        take = min(quotas[name], len(available))
        if take > 0:
            selected.extend(rng.choice(available, size=take, replace=False).tolist())

    for name in ("medium", "low", "high"):
        if len(selected) >= target:
            break
        available = [idx for idx in strata[name] if idx not in selected]
        if not available:
            continue
        take = min(target - len(selected), len(available))
        selected.extend(rng.choice(available, size=take, replace=False).tolist())

    selected = selected[:target]
    return [pairs[idx] for idx in selected], selected


def _pair_feature_matrix(candidate_pairs, pair_features) -> np.ndarray:
    features = np.asarray(pair_features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.shape[0] != len(candidate_pairs):
        raise ValueError("pair_features must align with candidate_pairs")
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)


def _select_diverse_indices(pair_features, candidate_indices, sample_size, rng):
    available_indices = list(dict.fromkeys(int(idx) for idx in candidate_indices))
    target = min(int(sample_size), len(available_indices))
    if target <= 0:
        return []
    if target == len(available_indices):
        return available_indices

    local_features = pair_features[available_indices]
    center = local_features.mean(axis=0)
    center_distances = np.linalg.norm(local_features - center, axis=1)
    first_candidates = np.flatnonzero(np.isclose(center_distances, center_distances.max()))
    first_position = int(rng.choice(first_candidates))

    selected_positions = [first_position]
    selected_position_set = {first_position}
    min_distances = np.linalg.norm(local_features - local_features[first_position], axis=1)
    min_distances[first_position] = -np.inf

    while len(selected_positions) < target:
        best_distance = np.max(min_distances)
        best_positions = np.flatnonzero(np.isclose(min_distances, best_distance))
        next_position = int(rng.choice(best_positions))
        selected_positions.append(next_position)
        selected_position_set.add(next_position)

        distances = np.linalg.norm(local_features - local_features[next_position], axis=1)
        min_distances = np.minimum(min_distances, distances)
        for position in selected_position_set:
            min_distances[position] = -np.inf

    return [available_indices[position] for position in selected_positions]


def select_diverse_profile_pairs(candidate_pairs, pair_features, sample_size, seed=42):
    """Select profile pairs with farthest-first coverage in pair-feature space."""
    pairs = list(candidate_pairs)
    if sample_size <= 0 or not pairs:
        return [], []

    features = _pair_feature_matrix(pairs, pair_features)
    rng = np.random.RandomState(seed)
    selected = _select_diverse_indices(features, range(len(pairs)), sample_size, rng)
    return [pairs[idx] for idx in selected], selected


def select_stratified_diverse_profile_pairs(
    candidate_pairs,
    pair_similarity_scores,
    pair_features,
    sample_size,
    medium_ratio=0.6,
    low_ratio=0.2,
    high_ratio=0.2,
    seed=42,
):
    """Stratify by blocking similarity, then diversify within each stratum.

    This is label-free: similarity comes from blocking embeddings and diversity
    comes from pair-level attribute agreement features.
    """
    pairs = list(candidate_pairs)
    if sample_size <= 0 or not pairs:
        return [], []

    features = _pair_feature_matrix(pairs, pair_features)
    scores = np.clip(_similarity_scores(pairs, pair_similarity_scores), 0.0, 1.0)
    target = min(int(sample_size), len(pairs))
    strata = _similarity_strata(scores)
    quotas = _stratified_quotas(
        target,
        medium_ratio=medium_ratio,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
    )
    rng = np.random.RandomState(seed)

    selected = []
    for name in ("medium", "low", "high"):
        available = [idx for idx in strata[name] if idx not in selected]
        take = min(quotas[name], len(available))
        if take > 0:
            selected.extend(_select_diverse_indices(features, available, take, rng))

    for name in ("medium", "low", "high"):
        if len(selected) >= target:
            break
        available = [idx for idx in strata[name] if idx not in selected]
        take = min(target - len(selected), len(available))
        if take > 0:
            selected.extend(_select_diverse_indices(features, available, take, rng))

    selected = selected[:target]
    return [pairs[idx] for idx in selected], selected


def _semantic_similarity(a: str, b: str, embedding_model=None) -> float:
    if embedding_model is None:
        return 0.0
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    embeddings = embedding_model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    sim = float(np.dot(embeddings[0], embeddings[1]))
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def compute_attribute_features(
    record_a,
    record_b,
    attributes,
    embedding_model=None,
    numeric_scales=None,
):
    """Compute fixed per-attribute similarity features for one candidate pair."""
    numeric_scales = numeric_scales or {}
    features = {}

    for attr in attributes:
        a = _clean_value(record_a.get(attr, ""))
        b = _clean_value(record_b.get(attr, ""))
        num_a = _parse_number(a)
        num_b = _parse_number(b)
        is_numeric = float(num_a is not None and num_b is not None)

        numeric_similarity = 0.0
        if is_numeric:
            scale = numeric_scales.get(attr)
            if scale is None or float(scale) <= 0:
                scale = max(abs(num_a), abs(num_b), 1.0)
            numeric_similarity = 1.0 - min(abs(num_a - num_b) / float(scale), 1.0)

        features[attr] = {
            "token_overlap": float(np.clip(_token_overlap(a, b), 0.0, 1.0)),
            "edit_similarity": float(np.clip(_edit_similarity(a, b), 0.0, 1.0)),
            "exact_match": _exact_match(a, b),
            "semantic_similarity": _semantic_similarity(a, b, embedding_model),
            "numeric_similarity": float(np.clip(numeric_similarity, 0.0, 1.0)),
            "is_numeric": is_numeric,
        }

    return features


def flatten_pair_features(feature_dict, attributes):
    """Flatten feature dict using the required attribute-major order."""
    values = []
    for attr in attributes:
        attr_features = feature_dict.get(attr, {})
        for feature_name in FEATURE_ORDER:
            values.append(float(attr_features.get(feature_name, 0.0)))
    return np.asarray(values, dtype=float)


def importance_vector_from_ranked(attributes, ranked_attributes):
    """Convert ranked influential attributes into a normalized importance dict."""
    attrs = list(attributes)
    ranked = [attr for attr in ranked_attributes if attr in attrs]
    weights = {attr: 0.0 for attr in attrs}
    n = len(ranked)
    for pos, attr in enumerate(ranked):
        weights[attr] = float(n - pos)
    total = sum(weights.values())
    if total > 0:
        weights = {attr: value / total for attr, value in weights.items()}
    return weights


def importance_vector_from_grouped(attributes, grouped_attributes):
    """Convert grouped high/medium/low attributes into a normalized vector."""
    attrs = list(attributes)
    group_weights = {"high": 3.0, "medium": 2.0, "low": 1.0}
    weights = {attr: 0.0 for attr in attrs}
    for group, group_attrs in grouped_attributes.items():
        weight = group_weights.get(str(group).lower(), 0.0)
        for attr in group_attrs:
            if attr in weights:
                weights[attr] = weight
    total = sum(weights.values())
    if total > 0:
        weights = {attr: value / total for attr, value in weights.items()}
    return weights


class AttributeImportancePredictor:
    """Lightweight Ridge predictor for pair-specific attribute importance."""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.n_attributes_ = None

    def fit(self, X, Y):
        X_arr = _as_2d(X)
        Y_arr = _normalize_rows(_as_2d(Y))
        self.n_attributes_ = Y_arr.shape[1]
        self.model.fit(X_arr, Y_arr)
        return self

    def predict(self, X):
        if self.n_attributes_ is None:
            raise ValueError("Predictor must be fit before predict")
        pred = self.model.predict(_as_2d(X))
        return _normalize_rows(pred)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class ProfileRetriever:
    """Retrieve similar profiled pairs and average their importance vectors."""

    def __init__(self):
        self.profile_feature_matrix = None
        self.profile_importance_matrix = None

    def fit(self, profile_feature_matrix, profile_importance_matrix):
        self.profile_feature_matrix = _as_2d(profile_feature_matrix)
        self.profile_importance_matrix = _normalize_rows(_as_2d(profile_importance_matrix))
        if self.profile_feature_matrix.shape[0] != self.profile_importance_matrix.shape[0]:
            raise ValueError("Feature and importance matrices must have the same number of rows")
        return self

    def retrieve(self, query_feature_vector, top_k=5):
        if self.profile_feature_matrix is None:
            raise ValueError("Retriever must be fit before retrieve")
        query = np.asarray(query_feature_vector, dtype=float).reshape(1, -1)
        sims = cosine_similarity(query, self.profile_feature_matrix)[0]
        k = min(int(top_k), len(sims))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        order = np.argsort(-sims)[:k]
        return order, sims[order]

    def predict_importance(self, query_feature_vector, top_k=5):
        indices, scores = self.retrieve(query_feature_vector, top_k=top_k)
        if len(indices) == 0:
            n_attrs = self.profile_importance_matrix.shape[1]
            aggregate = np.full(n_attrs, 1.0 / n_attrs, dtype=float)
        else:
            aggregate = _safe_normalize(self.profile_importance_matrix[indices].mean(axis=0))
        return aggregate, indices.tolist(), scores.tolist()


class HybridAttributeSelector:
    """Combine Ridge prediction and profile retrieval for adaptive selection."""

    def __init__(
        self,
        attributes,
        predictor_weight=0.5,
        retrieval_weight=0.5,
        fusion_strategy="weighted_average",
        top_k_retrieval=5,
        top_k_attributes=None,
        min_k_attributes=1,
        max_k_attributes=None,
        cumulative_importance_threshold=None,
        required_attribute_roles=None,
        attribute_roles=None,
        threshold=None,
    ):
        self.attributes = list(attributes)
        self.predictor_weight = predictor_weight
        self.retrieval_weight = retrieval_weight
        self.fusion_strategy = fusion_strategy
        self.top_k_retrieval = top_k_retrieval
        self.top_k_attributes = top_k_attributes
        self.min_k_attributes = min_k_attributes
        self.max_k_attributes = max_k_attributes
        self.cumulative_importance_threshold = cumulative_importance_threshold
        self.required_attribute_roles = list(required_attribute_roles or [])
        inferred_roles = infer_attribute_roles(self.attributes)
        if attribute_roles:
            inferred_roles.update({attr: role for attr, role in attribute_roles.items() if attr in inferred_roles})
        self.attribute_roles = inferred_roles
        self.threshold = threshold
        self.predictor = AttributeImportancePredictor()
        self.retriever = ProfileRetriever()

    def fit(self, profile_features, profile_importance_vectors):
        Y = self._importance_matrix(profile_importance_vectors)
        self.predictor.fit(profile_features, Y)
        self.retriever.fit(profile_features, Y)
        return self

    def _importance_matrix(self, vectors):
        if isinstance(vectors, np.ndarray):
            return _normalize_rows(vectors)
        rows = []
        for vector in vectors:
            if isinstance(vector, Mapping):
                rows.append([float(vector.get(attr, 0.0)) for attr in self.attributes])
            else:
                rows.append(vector)
        return _normalize_rows(np.asarray(rows, dtype=float))

    def _to_dict(self, vector):
        return {attr: float(vector[idx]) for idx, attr in enumerate(self.attributes)}

    def _entropy_confidence(self, vector):
        probs = _safe_normalize(vector, fallback_size=len(self.attributes))
        if len(probs) <= 1:
            return 1.0
        entropy = -float(np.sum([p * math.log(p) for p in probs if p > 0]))
        max_entropy = math.log(len(probs))
        return float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

    def _enforce_required_roles(self, selected_indices, ranked_indices):
        if not self.required_attribute_roles:
            return selected_indices

        selected = list(dict.fromkeys(int(idx) for idx in selected_indices))
        selected_attrs = {self.attributes[idx] for idx in selected}
        max_k = len(self.attributes)
        if self.max_k_attributes is not None:
            max_k = max(1, min(int(self.max_k_attributes), len(self.attributes)))

        for role in self.required_attribute_roles:
            role_attrs = {attr for attr, attr_role in self.attribute_roles.items() if attr_role == role}
            if not role_attrs or selected_attrs & role_attrs:
                continue

            replacement_idx = next(
                (idx for idx in ranked_indices if self.attributes[idx] in role_attrs),
                None,
            )
            if replacement_idx is None:
                continue

            if len(selected) < max_k:
                selected.append(int(replacement_idx))
                selected_attrs.add(self.attributes[replacement_idx])
                continue

            replace_pos = next(
                (
                    pos
                    for pos in range(len(selected) - 1, -1, -1)
                    if self.attribute_roles.get(self.attributes[selected[pos]]) != role
                ),
                None,
            )
            if replace_pos is not None:
                removed_attr = self.attributes[selected[replace_pos]]
                selected_attrs.discard(removed_attr)
                selected[replace_pos] = int(replacement_idx)
                selected_attrs.add(self.attributes[replacement_idx])

        ranked_position = {idx: pos for pos, idx in enumerate(ranked_indices)}
        selected.sort(key=lambda idx: ranked_position.get(idx, len(ranked_position)))
        return selected

    def estimate_importance(self, pair_features):
        predictor_importance = self.predictor.predict(pair_features)[0]
        retrieval_importance, retrieved_indices, retrieval_scores = self.retriever.predict_importance(
            pair_features,
            top_k=self.top_k_retrieval,
        )

        if self.fusion_strategy == "weighted_average":
            combined = (
                self.predictor_weight * predictor_importance
                + self.retrieval_weight * retrieval_importance
            )
        elif self.fusion_strategy == "confidence_based":
            predictor_conf = self._entropy_confidence(predictor_importance)
            retrieval_conf = self._entropy_confidence(retrieval_importance)
            total_conf = predictor_conf + retrieval_conf
            if total_conf == 0:
                predictor_alpha = retrieval_alpha = 0.5
            else:
                predictor_alpha = predictor_conf / total_conf
                retrieval_alpha = retrieval_conf / total_conf
            combined = predictor_alpha * predictor_importance + retrieval_alpha * retrieval_importance
        else:
            raise ValueError("fusion_strategy must be 'weighted_average' or 'confidence_based'")

        combined = _safe_normalize(combined, fallback_size=len(self.attributes))
        return (
            combined,
            predictor_importance,
            retrieval_importance,
            {
                "retrieved_indices": retrieved_indices,
                "cosine_similarity_scores": retrieval_scores,
            },
        )

    def select_attributes(self, pair_features):
        combined, predictor_importance, retrieval_importance, retrieval_debug = self.estimate_importance(pair_features)
        ranked_indices = np.argsort(-combined).tolist()

        if self.top_k_attributes is not None:
            selected_indices = ranked_indices[: int(self.top_k_attributes)]
        elif self.cumulative_importance_threshold is not None:
            max_k = len(ranked_indices)
            if self.max_k_attributes is not None:
                max_k = max(1, min(int(self.max_k_attributes), len(ranked_indices)))
            min_k = max(1, min(int(self.min_k_attributes), max_k))
            target = float(self.cumulative_importance_threshold)

            selected_indices = []
            cumulative = 0.0
            for idx in ranked_indices:
                if len(selected_indices) >= max_k:
                    break
                selected_indices.append(idx)
                cumulative += float(combined[idx])
                if len(selected_indices) >= min_k and cumulative >= target:
                    break
        elif self.threshold is not None:
            selected_indices = [idx for idx in ranked_indices if combined[idx] >= self.threshold]
        else:
            selected_indices = ranked_indices

        if not selected_indices:
            selected_indices = [int(np.argmax(combined))]
        selected_indices = self._enforce_required_roles(selected_indices, ranked_indices)

        return {
            "selected_attributes": [self.attributes[idx] for idx in selected_indices],
            "combined_importance": self._to_dict(combined),
            "predictor_importance": self._to_dict(predictor_importance),
            "retrieval_importance": self._to_dict(retrieval_importance),
            "retrieval_debug": retrieval_debug,
        }


def condense_pair_by_attributes(record_a, record_b, selected_attributes):
    """Return two records containing only the selected attributes."""
    attrs = list(selected_attributes)
    return (
        {attr: record_a.get(attr, "") for attr in attrs},
        {attr: record_b.get(attr, "") for attr in attrs},
    )
