"""Run three independent adaptive/full matcher draws on the frozen DBLP-ACM test split."""
import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from code.entity_split import entity_split
from code.embeddings import embed_dataframe_sbert
from code.lsh import create_random_planes, query_lsh_fast
from code.main import DATASET_CONFIGS, DEFAULT_MODEL_NAME, load_configured_dataset
from code.main import compute_candidate_pair_scores, pair_feature_vector
from code.attribute_selection.adaptive import attribute_evidence_mask, compute_numeric_scales
from code.matcher import infer_candidates_pairwise, set_cache_enabled
from code.main import check_api_errors

ROOT = Path("logs/DBLP-ACM/train_validation_2026-09-09")


class SemanticLookup:
    def __init__(self, model, frames):
        texts = sorted({str(v).strip().lower() for frame in frames for v in frame.to_numpy().ravel()})
        self.values = dict(zip(texts, model.encode(texts, batch_size=128, convert_to_numpy=True,
                                                   normalize_embeddings=True, show_progress_bar=True)))

    def encode(self, texts, **kwargs):
        return np.asarray([self.values[str(t).strip().lower()] for t in texts])


def main():
    cfg = DATASET_CONFIGS["DBLP-ACM"]
    a, b, gold = load_configured_dataset(cfg)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    split = protocol["split"]
    ia, ib = split["test"]["a"], split["test"]["b"]
    test_a, test_b = a.iloc[ia].reset_index(drop=True), b.iloc[ib].reset_index(drop=True)
    model = SentenceTransformer(DEFAULT_MODEL_NAME, local_files_only=True)
    va = embed_dataframe_sbert(test_a, model, model_name=DEFAULT_MODEL_NAME)
    vb = embed_dataframe_sbert(test_b, model, model_name=DEFAULT_MODEL_NAME)
    chosen = json.loads((ROOT / "selected_blocking.json").read_text())
    planes = create_random_planes(chosen["num_tables"], chosen["num_planes"], va.shape[1], seed=42)
    candidate_pairs = query_lsh_fast(va, vb, planes, num_flips=chosen["num_flips"], top_k=chosen["top_k"])
    (ROOT / "test_candidates.json").write_text(json.dumps(candidate_pairs) + "\n")
    selector = joblib.load(ROOT / "adaptive.joblib")["selector"]
    scales = joblib.load(ROOT / "adaptive.joblib")["numeric_scales"]
    semantic = SemanticLookup(model, [test_a, test_b])
    attrs = {}
    for i, j in candidate_pairs:
        features = pair_feature_vector(test_a, test_b, i, j, selector.attributes, semantic, scales)
        mask = attribute_evidence_mask(test_a.iloc[i].to_dict(), test_b.iloc[j].to_dict(), selector.attributes)
        selected = selector.select_attributes(features, evidence_mask=mask)["selected_attributes"]
        attrs[i, j] = [x for x in selector.attributes if x in selected]
    set_cache_enabled(False)
    for draw in (1, 2, 3):
        for arm in ("adaptive", "full"):
            out = ROOT / f"test_{arm}_draw{draw}.csv"
            if out.exists():
                continue
            frame = infer_candidates_pairwise(
                test_a, test_b, candidate_pairs, max_workers=16,
                selected_attributes_by_pair=attrs if arm == "adaptive" else None,
                checkpoint_dir=str(ROOT / f"test_{arm}_draw{draw}_checkpoint"),
            )
            check_api_errors(frame)
            frame.to_csv(out, index=False)
            print(f"completed {arm} draw {draw}: {len(frame)} pairs", flush=True)
    (ROOT / "test_repeats_complete.json").write_text(json.dumps({
        "test_evaluated": True, "draws_per_arm": 3, "candidate_pairs": len(candidate_pairs),
        "configuration": chosen,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
