"""Audit completed development runs and write comparable validation metrics."""
import json
from pathlib import Path

import pandas as pd


def main():
    base = Path("logs/DBLP-ACM/train_validation_2026-09-09")
    protocol = json.loads((base / "protocol.json").read_text())
    split = protocol["split"]
    for side in ("a", "b"):
        values = [i for part in split.values() for i in part[side]]
        assert len(values) == len(set(values))
        assert sorted(values) == list(range(len(values)))
    raw_gold = pd.read_csv("datasets/DBLP-ACM/gold.csv")
    aa = pd.read_csv("datasets/DBLP-ACM/DBLP.csv", encoding="latin1")
    bb = pd.read_csv("datasets/DBLP-ACM/ACM.csv", encoding="latin1")
    amap, bmap = dict(zip(aa.id, aa.index)), dict(zip(bb.id, bb.index))
    gold = {(amap[r.idDBLP], bmap[r.idACM]) for r in raw_gold.itertuples()}
    owners = {side: {i: name for name, part in split.items() for i in part[side]}
              for side in ("a", "b")}
    assert all(owners["a"][a] == owners["b"][b] for a, b in gold)
    # Only validation gold is used to score predictions.
    va, vb = split["validation"]["a"], split["validation"]["b"]
    ai, bi = {v: i for i, v in enumerate(va)}, {v: i for i, v in enumerate(vb)}
    val_gold = {(ai[a], bi[b]) for a, b in gold if a in ai and b in bi}
    candidates = {tuple(p) for p in json.loads((base / "validation_candidates.json").read_text())}
    profile = json.loads((base / "train_profiling.json").read_text())
    assert profile["profiling_error_count"] == 0
    train_candidates = {tuple(p) for p in json.loads((base / "train_candidates.json").read_text())}
    assert len(profile["profile_pairs"]) == 50
    assert all(tuple(p) in train_candidates for p in profile["profile_pairs"])
    output = {"test_evaluated": False, "draws_per_arm": 1, "arms": {}}
    for arm in ("adaptive", "full"):
        df = pd.read_csv(base / (arm + "_validation.csv"))
        pairs = list(zip(df.indexA, df.indexB))
        assert len(pairs) == len(set(pairs)) == len(candidates)
        assert set(pairs) == candidates
        assert df.answer.isin(["Yes", "No"]).all()
        assert not df.cache_hit.any()
        pred = {pair for pair, answer in zip(pairs, df.answer) if answer == "Yes"}
        tp, fp = len(pred & val_gold), len(pred - val_gold)
        fn = len(val_gold) - tp
        matching = int(df.total_tokens.sum())
        profiling = profile["profiling_token_usage"]["total_tokens"] if arm == "adaptive" else 0
        output["arms"][arm] = dict(tp=tp, fp=fp, fn=fn,
            precision=tp / len(pred) if pred else 0, recall=tp / len(val_gold),
            f1=2 * tp / (2 * tp + fp + fn), candidates=len(df),
            mean_attributes=float(df.selected_attribute_count.mean()),
            matching_tokens=matching, profiling_tokens=profiling,
            pipeline_tokens=matching + profiling)
    adaptive, full = output["arms"]["adaptive"], output["arms"]["full"]
    output["f1_difference"] = adaptive["f1"] - full["f1"]
    output["token_saving_fraction"] = 1 - adaptive["pipeline_tokens"] / full["pipeline_tokens"]
    (base / "validated_results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
