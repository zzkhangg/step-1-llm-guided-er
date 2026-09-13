"""Analysis for the two 2026-08-31 Amazon jobs.

Job 1 (mask): compares --evidence-mask-mode drop against keep. The comparison is only
meaningful on the pairs whose prompt actually differs between the two modes; on every other
pair the two runs build the same prompt and share a cache entry, so they agree by
construction.

The split is on the `selected_attributes` column, NOT on `prompt_hash`: that column is a
hash of the prompt *template* only (code/matcher.py:54), which is constant across every pair
and both runs, so splitting on it puts all 88,296 pairs in the "identical" bucket and finds
nothing. What actually varies per pair is the attribute list, and in `keep` mode that list
carries the blank attributes the mask barred from ranking but left visible -- so a differing
attribute list is exactly a differing prompt, and exactly a differing matcher cache key
(which is a hash of the condensed records, code/matcher.py:72).

Job 2 (repeats): three same-day draws of `full` against three of the adaptive
configuration on one fixed 20,000-pair slice. Reports each draw, the mean/sd/range per arm,
and a pooled McNemar over the draws paired in order.

Usage:  python -m examples.analyze_mask_and_repeats
"""
import csv
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.main import DATASET_CONFIGS, load_configured_dataset  # noqa: E402

RUNS = "logs/Amazon-Walmart/runs"


def newest(pred):
    hits = sorted(d for d in os.listdir(RUNS) if pred(d))
    return os.path.join(RUNS, hits[-1]) if hits else None


def read_run(path):
    """-> {(a,b): (answer_is_yes, prompt_hash, errored)}"""
    out = {}
    with open(os.path.join(path, "final_results.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["indexA"]), int(row["indexB"]))
            out[key] = (
                row["answer"].strip().lower().startswith("y"),
                row["selected_attributes"],
                bool(row.get("llm_error", "").strip()),
            )
    return out


def prf(pred_yes, gt):
    tp = sum(1 for k, y in pred_yes.items() if y and k in gt)
    fp = sum(1 for k, y in pred_yes.items() if y and k not in gt)
    fn = len(gt) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def mcnemar(b, c):
    """Exact two-sided binomial test on the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_gt(subset_keys=None):
    _, _, gt = load_configured_dataset(DATASET_CONFIGS["Amazon-Walmart"])
    if subset_keys is not None:
        gt = {g for g in gt if g in subset_keys}
    return gt


def job1():
    drop = newest(lambda d: d.startswith("20260831") and d.endswith("gap_mink1"))
    keep = newest(lambda d: d.startswith("20260831") and "maskkeep" in d)
    print("=" * 78)
    print("JOB 1 -- evidence mask: drop vs keep, same day")
    print(f"  drop run: {drop}")
    print(f"  keep run: {keep}")
    if not (drop and keep):
        print("  !! one of the runs is missing, skipping")
        return
    D, K = read_run(drop), read_run(keep)
    shared = [k for k in D if k in K and not D[k][2] and not K[k][2]]
    gt = load_gt(set(shared))

    same = [k for k in shared if D[k][1] == K[k][1]]
    diff = [k for k in shared if D[k][1] != K[k][1]]
    print(f"\n  comparable pairs      : {len(shared)}")
    print(f"    identical prompt    : {len(same)}  (mask cannot touch these)")
    print(f"    differing prompt    : {len(diff)}  <- the mask's actual scope")

    flips_same = sum(1 for k in same if D[k][0] != K[k][0])
    print(f"    answer flips on identical prompts: {flips_same}  (expect 0 -- shared cache)")
    flips_diff = sum(1 for k in diff if D[k][0] != K[k][0])
    print(f"    answer flips on differing prompts: {flips_diff}")

    for name, run in (("drop", D), ("keep", K)):
        p, r, f, tp, fp, fn = prf({k: run[k][0] for k in shared}, gt)
        print(f"\n  {name:4s}  P={p:.4f}  R={r:.4f}  F1={f:.4f}   TP={tp} FP={fp} FN={fn}")

    b = c = 0  # b: drop right & keep wrong ; c: keep right & drop wrong
    for k in diff:
        dg, kg = (D[k][0] == (k in gt)), (K[k][0] == (k in gt))
        if dg and not kg:
            b += 1
        elif kg and not dg:
            c += 1
    print(f"\n  McNemar on the {len(diff)} differing pairs")
    print(f"    drop right / keep wrong : {b}")
    print(f"    keep right / drop wrong : {c}")
    print(f"    p = {mcnemar(b, c):.3g}")
    print("    (both runs same day, shared matcher cache -> no drift confound)")

    ch = Counter()
    for k in diff:
        ch[(D[k][0], K[k][0])] += 1
    print(f"    answer changes: No->Yes {ch[(False, True)]}, Yes->No {ch[(True, False)]}")


def job2():
    print("\n" + "=" * 78)
    print("JOB 2 -- Amazon 20,000-pair slice, 3 same-day draws per arm")
    # Runs are identified by reading the summaries rather than by guessing directory names.
    draws = {"full": [], "adaptive": []}
    for d in sorted(os.listdir(RUNS)):
        path = os.path.join(RUNS, d)
        sp = os.path.join(path, "run_summary.json")
        if not (d.startswith("20260831") or d.startswith("20260901")) or not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        sub = s.get("blocking", {}).get("candidate_subset") or s.get("candidate_subset")
        if not sub:
            continue
        arm = "adaptive" if "step15" in d else "full"
        draws[arm].append((d, s["matching"]))

    gt_cache = {}
    for arm, items in draws.items():
        print(f"\n  {arm} ({len(items)} draws)")
        f1s = []
        for d, m in items:
            print(f"    {d:58s} P={m['end_to_end_precision']:.4f} "
                  f"R={m['end_to_end_recall']:.4f} F1={m['end_to_end_f1']:.4f}")
            f1s.append(m["end_to_end_f1"])
        if len(f1s) >= 2:
            mu = sum(f1s) / len(f1s)
            sd = (sum((x - mu) ** 2 for x in f1s) / (len(f1s) - 1)) ** 0.5
            print(f"    mean F1 = {mu:.4f}   sd = {sd:.4f}   range = [{min(f1s):.4f}, {max(f1s):.4f}]")

    fa = [x for _, x in draws["full"]]
    ad = [x for _, x in draws["adaptive"]]
    if fa and ad:
        lo_f, hi_f = min(x["end_to_end_f1"] for x in fa), max(x["end_to_end_f1"] for x in fa)
        lo_a, hi_a = min(x["end_to_end_f1"] for x in ad), max(x["end_to_end_f1"] for x in ad)
        overlap = not (hi_f < lo_a or hi_a < lo_f)
        print(f"\n  full range     [{lo_f:.4f}, {hi_f:.4f}]")
        print(f"  adaptive range [{lo_a:.4f}, {hi_a:.4f}]")
        print(f"  ranges overlap : {overlap}"
              + ("  -> the gap is NOT separable from run-to-run noise" if overlap
                 else "  -> separated; the gap survives the noise"))

    # Paired McNemar, draw i of full against draw i of adaptive.
    pairs = list(zip([d for d, _ in draws["full"]], [d for d, _ in draws["adaptive"]]))
    if pairs:
        print("\n  paired McNemar per draw (full vs adaptive, correctness)")
        for df_, da_ in pairs:
            F, A = read_run(os.path.join(RUNS, df_)), read_run(os.path.join(RUNS, da_))
            shared = [k for k in F if k in A and not F[k][2] and not A[k][2]]
            key = tuple(sorted(shared)[:1])
            if key not in gt_cache:
                gt_cache[key] = load_gt(set(shared))
            gt = gt_cache[key]
            b = sum(1 for k in shared if (F[k][0] == (k in gt)) and not (A[k][0] == (k in gt)))
            c = sum(1 for k in shared if (A[k][0] == (k in gt)) and not (F[k][0] == (k in gt)))
            print(f"    {df_[:24]} vs {da_[:34]}: full-only-right={b} "
                  f"adaptive-only-right={c} p={mcnemar(b, c):.3g}")


if __name__ == "__main__":
    job1()
    job2()
