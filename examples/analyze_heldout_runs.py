#!/usr/bin/env python3
"""Validate and summarize DBLP-Scholar repetitions without any LLM calls."""

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.main import DATASET_CONFIGS, load_configured_dataset


DRAW1 = {
    "adaptive_draw1": "20260904_143805_step15_diversity_decisive_gap_mink1",
    "full_draw1": "20260904_151442_full",
}


def metrics(answers, gold):
    predicted = {pair for pair, answer in answers.items() if answer == "Yes"}
    candidates = set(answers)
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    candidate_gold = len(candidates & gold)
    precision = tp / len(predicted) if predicted else 0.0
    candidate_recall = tp / candidate_gold if candidate_gold else 0.0
    recall = tp / len(gold) if gold else 0.0
    return {
        "tp": tp, "fp": fp, "fn": candidate_gold - tp,
        "precision": precision,
        "candidate_recall": candidate_recall,
        "candidate_f1": 2 * tp / (len(predicted) + candidate_gold) if predicted or candidate_gold else 0.0,
        "end_to_end_recall": recall,
        "end_to_end_f1": 2 * tp / (len(predicted) + len(gold)) if predicted or gold else 0.0,
    }


def read_draw(path, gold):
    summary = json.loads(path.read_text())
    matching = summary["matching"]
    assert matching["api_errors"] == 0 and not matching["metrics_invalid"], path
    assert not summary["runtime_controls"]["matcher_cache_enabled"], path
    adaptive = summary.get("adaptive", {})
    if adaptive:
        assert adaptive["profiling_error_count"] == 0, path
        assert adaptive["sampling"]["label_usage"] == "ground_truth_not_used_for_sampling", path
        assert adaptive["selection_size_mean"] < summary["attribute_selection"]["attribute_count"], path
    frame = pd.read_csv(path.parent / "final_results.csv", keep_default_na=False)
    assert len(frame) == matching["num_result_pairs"] == 13080, path
    assert not frame.duplicated(["indexA", "indexB"]).any(), path
    assert frame.answer.isin(["Yes", "No"]).all(), path
    assert (frame.llm_error == "").all(), path
    assert not frame.cache_hit.astype(bool).any(), path
    answers = {(int(row.indexA), int(row.indexB)): row.answer for row in frame.itertuples()}
    observed = metrics(answers, gold)
    for key, value in observed.items():
        assert abs(value - matching[key]) < 1e-10, (path, key, value, matching[key])
    attrs = collections.Counter(attr for raw in frame.selected_attributes for attr in json.loads(raw))
    row = {
        "run_summary": str(path.relative_to(REPO_ROOT)),
        **observed,
        "blocking_recall": summary["blocking"]["pair_completeness"],
        "pipeline_tokens": summary["token_usage"]["pipeline_total_tokens"],
        "failed_attempt_tokens": matching.get("reported_failed_attempt_tokens"),
        "failed_attempts": matching.get("failed_attempts"),
        "unreported_usage_attempts": matching.get("unreported_usage_attempts"),
        "checkpoint_hits": matching.get("checkpoint_hits", 0),
        "mean_attributes": float(frame.selected_attribute_count.mean()),
        "attribute_counts": dict(attrs),
        "requests_started_at": matching.get("requests_started_at"),
        "requests_finished_at": matching.get("requests_finished_at"),
        "providers": matching["providers"],
    }
    return row, answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="logs/heldout/validated_results.json")
    args = parser.parse_args()
    _, _, gold = load_configured_dataset(DATASET_CONFIGS["DBLP-Scholar"])
    root = REPO_ROOT / "logs/DBLP-Scholar/runs"
    paths = {name: root / run / "run_summary.json" for name, run in DRAW1.items()}
    for path in sorted(root.glob("*/run_summary.json")):
        summary = json.loads(path.read_text())
        checkpoint = summary["runtime_controls"].get("matcher_checkpoint_dir")
        if checkpoint and Path(checkpoint).name in {
            "adaptive_draw2", "adaptive_draw3", "full_draw2", "full_draw3",
        }:
            paths[Path(checkpoint).name] = path

    rows, answers = {}, {}
    for name, path in paths.items():
        if path.exists():
            rows[name], answers[name] = read_draw(path, gold)
    reference = set(next(iter(answers.values())))
    assert all(set(a) == reference for a in answers.values()), "Candidate sets differ across draws"

    expected = {f"{arm}_draw{i}" for arm in ("adaptive", "full") for i in (1, 2, 3)}
    missing = sorted(expected - rows.keys())
    windows = {}
    for window, draws in (("original_2026-09-04", (1,)), ("amended_draws_2_3", (2, 3))):
        arms = {}
        for arm in ("adaptive", "full"):
            selected = [rows[f"{arm}_draw{i}"] for i in draws if f"{arm}_draw{i}" in rows]
            if selected:
                arms[arm] = {
                    "n": len(selected),
                    **{key: {"mean": statistics.mean(r[key] for r in selected),
                             "min": min(r[key] for r in selected),
                             "max": max(r[key] for r in selected)}
                       for key in ("precision", "candidate_recall", "candidate_f1",
                                   "end_to_end_recall", "end_to_end_f1", "pipeline_tokens")},
                }
        windows[window] = arms

    output = {"missing_draws": missing, "draws": rows, "windows": windows}
    destination = REPO_ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print("\nDraw                 Precision  Cand. R  Cand. F1  E2E R    E2E F1   Tokens")
    for name, row in sorted(rows.items()):
        print(f"{name:<20} {row['precision']:.4f}     {row['candidate_recall']:.4f}   "
              f"{row['candidate_f1']:.4f}    {row['end_to_end_recall']:.4f}   "
              f"{row['end_to_end_f1']:.4f}   {row['pipeline_tokens']:,}")
    print(f"\nMissing draws: {', '.join(missing) or 'none'}")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
