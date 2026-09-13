"""Development-only entity split, blocking validation, and train-only profiling.

Run as python -m examples.run_dblp_train_validation. Test is never evaluated.
"""
import hashlib
import io
import json
import time
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from code.entity_split import entity_split
from code.embeddings import embed_dataframe_sbert
from code.lsh import create_random_planes, query_lsh_fast
from code.main import (
    DATASET_CONFIGS, DEFAULT_MODEL_NAME, load_configured_dataset,
    build_adaptive_attribute_map, compute_candidate_pair_scores,
    compute_numeric_scales, pair_feature_vector, attribute_evidence_mask,
    set_profile_cache_dir, set_cache_enabled, infer_candidates_pairwise,
    check_api_errors, json_default,
)
from code.attribute_selection.adaptive import _clean_value
from code.llm_client import get_llm_model

OUT = Path("logs/DBLP-ACM/train_validation_2026-09-09")


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=json_default) + "\n")
    tmp.replace(path)


class BatchedSemanticEncoder:
    def __init__(self, model, frames):
        texts = sorted({_clean_value(v) for frame in frames
                        for v in frame.to_numpy().ravel()})
        values = model.encode(texts, batch_size=128, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=True)
        self.values = dict(zip(texts, values))

    def encode(self, texts, **kwargs):
        return np.asarray([self.values[t] for t in texts])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = DATASET_CONFIGS["DBLP-ACM"]
    a, b, gold = load_configured_dataset(config)
    split = entity_split(len(a), len(b), gold)
    grid = list(product([5, 15], [6, 8], [0, 1, 2], [5, 10, 20]))
    protocol = {
        "split": split, "seed": 42, "ratio": [0.6, 0.2, 0.2],
        "search_universe": "within-partition tables; smaller than original benchmark",
        "status": "development dataset previously examined; not pristine held-out test",
        "embedding_model": DEFAULT_MODEL_NAME, "llm_model": get_llm_model(),
        "grid_columns": ["num_tables", "num_planes", "num_flips", "top_k"],
        "grid": grid, "recall_target": .95,
        "selection": "recall>=.95: fewest candidates, highest recall, lexicographic config; otherwise highest recall then fewest candidates",
        "adaptive": "50 train profiles, diversity, decisive, gap, min=1, max=5, identity required, drop mask; no adaptive hyperparameter sweep",
        "source_sha256": {name: hashlib.sha256((Path(config["base_path"]) / config[name]).read_bytes()).hexdigest()
                          for name in ["table_a", "table_b", "ground_truth"]},
    }
    if (OUT / "protocol.json").exists():
        old = json.loads((OUT / "protocol.json").read_text())
        if old != json.loads(json.dumps(protocol)):
            raise RuntimeError("Existing experiment protocol differs")
    else:
        save("protocol.json", protocol)
    parts = {}
    for name in ("train", "validation"):
        ia, ib = split[name]["a"], split[name]["b"]
        ma, mb = {v: i for i, v in enumerate(ia)}, {v: i for i, v in enumerate(ib)}
        g = {(ma[x], mb[y]) for x, y in gold if x in ma and y in mb}
        parts[name] = (a.iloc[ia].reset_index(drop=True), b.iloc[ib].reset_index(drop=True), g)
        print(name, "A/B/gold:", len(ia), len(ib), len(g), flush=True)
    model = SentenceTransformer(DEFAULT_MODEL_NAME, local_files_only=True)
    vectors = {}
    for name, (aa, bb, _) in parts.items():
        vectors[name] = tuple(embed_dataframe_sbert(f, model, model_name=DEFAULT_MODEL_NAME) for f in (aa, bb))

    def block(name, cfg):
        va, vb = vectors[name]
        planes = create_random_planes(cfg[0], cfg[1], va.shape[1], seed=42)
        with redirect_stdout(io.StringIO()):
            return query_lsh_fast(va, vb, planes, num_flips=cfg[2], top_k=cfg[3])

    rows = []
    # Generate at the largest k once, then retain the ranked prefix per query.
    for tables, planes, flips in product([5, 15], [6, 8], [0, 1, 2]):
        start = time.monotonic()
        pairs = block("validation", (tables, planes, flips, 20))
        elapsed = time.monotonic() - start
        for k in [5, 10, 20]:
            counts, selected = {}, []
            for pair in pairs:
                counts[pair[0]] = counts.get(pair[0], 0) + 1
                if counts[pair[0]] <= k:
                    selected.append(pair)
            found = len(set(selected) & parts["validation"][2])
            row = dict(zip(protocol["grid_columns"], [tables, planes, flips, k]))
            row.update(candidates=len(selected), found=found, gold=len(parts["validation"][2]),
                       recall=found / len(parts["validation"][2]), shared_top20_seconds=elapsed)
            rows.append(row)
        save("blocking_grid.json", rows)
        print("Blocking", tables, planes, flips, "done", flush=True)
    eligible = [r for r in rows if r["recall"] >= .95]
    def key(r):
        tail = tuple(r[c] for c in protocol["grid_columns"])
        return ((r["candidates"], -r["recall"]) if eligible else (-r["recall"], r["candidates"])) + tail
    best = min(eligible or rows, key=key)
    save("selected_blocking.json", {**best, "target_met": bool(eligible)})
    print("Selected blocking:", best, flush=True)
    cfg = tuple(best[c] for c in protocol["grid_columns"])
    candidates = {name: block(name, cfg) for name in parts}
    for name, pairs in candidates.items():
        save(name + "_candidates.json", pairs)
    train_a, train_b, _ = parts["train"]
    val_a, val_b, val_gold = parts["validation"]
    semantic = BatchedSemanticEncoder(model, [train_a, train_b, val_a, val_b])
    model_path = OUT / "adaptive.joblib"
    if model_path.exists():
        bundle = joblib.load(model_path)
        selector, scales = bundle["selector"], bundle["numeric_scales"]
    else:
        set_profile_cache_dir(str(OUT / "profile_cache"))
        _, selector, summary = build_adaptive_attribute_map(
            train_a, train_b, candidates["train"],
            compute_candidate_pair_scores(candidates["train"], *vectors["train"]),
            embedding_model=semantic, profile_sample_size=50,
            profile_sampler="diversity", profile_scoring="decisive",
            profile_max_workers=4, selection_policy="gap", min_k_attributes=1,
            max_k_attributes=5, required_attribute_roles=["identity"], seed=42,
        )
        save("train_profiling.json", summary)
        if summary["profiling_error_count"]:
            raise RuntimeError("Profiling errors; refusing to fit validation on fallback labels")
        scales = compute_numeric_scales(train_a, train_b, selector.attributes)
        joblib.dump({"selector": selector, "numeric_scales": scales}, model_path)
    chosen = {}
    for i, j in candidates["validation"]:
        features = pair_feature_vector(val_a, val_b, i, j, selector.attributes, semantic, scales)
        mask = attribute_evidence_mask(val_a.iloc[i].to_dict(), val_b.iloc[j].to_dict(), selector.attributes)
        selected = selector.select_attributes(features, evidence_mask=mask)["selected_attributes"]
        chosen[i, j] = [attr for attr in selector.attributes if attr in selected]
    set_cache_enabled(False)
    results = {}
    for arm, attrs in [("adaptive", chosen), ("full", None)]:
        frame = infer_candidates_pairwise(val_a, val_b, candidates["validation"],
            max_workers=16, selected_attributes_by_pair=attrs,
            checkpoint_dir=str(OUT / (arm + "_checkpoint")))
        frame.to_csv(OUT / (arm + "_validation.csv"), index=False)
        check_api_errors(frame)
        positive = {(int(r.indexA), int(r.indexB)) for r in frame.itertuples() if r.answer == "Yes"}
        tp, fp = len(positive & val_gold), len(positive - val_gold)
        fn = len(val_gold) - tp
        results[arm] = {"tp": tp, "fp": fp, "fn_end_to_end": fn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / len(val_gold), "f1": 2 * tp / (2 * tp + fp + fn),
            "matching_tokens": int(frame.total_tokens.sum()), "candidates": len(frame)}
        save("validation_results.json", results)
        print(arm, results[arm], flush=True)
    save("complete.json", {"test_evaluated": False, "validation_draws_per_arm": 1})


if __name__ == "__main__":
    main()
