#!/usr/bin/env python3
"""Run every Step-1.5 profile sampler across several seeds and report variance.

Answers the question the single-seed runs cannot: are the differences between
samplers larger than the run-to-run noise of a single sampler?

Blocking is held fixed (its seed is independent of ``--seed``), so every run in
the sweep sees an identical candidate-pair set and the only thing that varies is
which pairs get profiled.

Run from the repository root, for example:

    python examples/sweep_sampler_seeds.py --dataset Fodors-Zagat \
        --seeds 42,43,44,45,46 --exclude-attributes class
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLERS = ("random", "similarity_stratified", "diversity", "stratified_diversity", "importance_based")


def latest_run_dir(dataset, log_mode):
    runs = sorted((REPO_ROOT / "logs" / dataset / "runs").glob(f"*_{log_mode}"))
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="Fodors-Zagat")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seeds.")
    parser.add_argument("--samplers", default=",".join(SAMPLERS), help="Comma-separated samplers.")
    parser.add_argument("--exclude-attributes", default="", help="Passed through to code.main.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    results = {sampler: {} for sampler in samplers}
    failures = []
    aborted = False

    for sampler in samplers:
        for seed in seeds:
            cmd = [
                sys.executable, "-m", "code.main",
                "--dataset", args.dataset,
                "--mode", "step15",
                "--profile-sampler", sampler,
                "--seed", str(seed),
                "--disable-matcher-cache",
            ]
            if args.exclude_attributes:
                cmd += ["--exclude-attributes", args.exclude_attributes]

            print(f"\n=== {sampler} seed={seed} ===", flush=True)
            if args.dry_run:
                print(" ".join(cmd))
                continue

            proc = subprocess.run(cmd, cwd=REPO_ROOT)
            if proc.returncode != 0:
                print(f"  FAILED (exit {proc.returncode}), skipping", flush=True)
                continue

            log_mode = f"step15_{sampler}" if seed == 42 else f"step15_{sampler}_seed{seed}"
            run_dir = latest_run_dir(args.dataset, log_mode)
            if run_dir is None:
                print("  no run directory found, skipping", flush=True)
                continue
            summary = json.loads((run_dir / "run_summary.json").read_text())
            matching = summary["matching"]

            # A failed LLM call is scored as a non-match and the run still writes a
            # complete summary, so an exhausted API budget produces numerically valid
            # but meaningless metrics (F1 = 0.0 for a fully failed run). Never let those
            # into the aggregate: a nonzero error count disqualifies the run.
            api_errors = int(matching.get("api_errors", 0) or 0)
            if api_errors:
                total = int(matching.get("num_result_pairs", 0) or 0)
                print(f"  DISCARDED: {api_errors}/{total} LLM calls failed "
                      f"({api_errors / total:.1%}); metrics are not meaningful", flush=True)
                failures.append((sampler, seed, api_errors, total, run_dir.name))
                if api_errors == total:
                    print("  All calls failed -- the API budget is likely exhausted. "
                          "Aborting the sweep rather than burning the remaining runs.", flush=True)
                    aborted = True
                    break
                continue

            results[sampler][seed] = {
                "f1": matching["end_to_end_f1"],
                "precision": matching["end_to_end_precision"],
                "recall": matching["end_to_end_recall"],
                "tp": matching["tp"],
                "tokens": summary["token_usage"]["pipeline_total_tokens"],
                "run_dir": run_dir.name,
            }
        if aborted:
            break

    if args.dry_run:
        return

    # Merge into any existing sweep file rather than overwriting it: a sweep is
    # routinely resumed one sampler at a time, and a plain write would silently
    # discard the runs from earlier invocations.
    out_path = REPO_ROOT / "logs" / args.dataset / "sampler_seed_sweep.json"
    merged = {}
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            print(f"\nWarning: {out_path} is not valid JSON; starting a fresh file.")
            merged = {}
    for sampler, per_seed in results.items():
        merged.setdefault(sampler, {}).update({str(seed): row for seed, row in per_seed.items()})
    merged = {sampler: dict(sorted(rows.items(), key=lambda kv: int(kv[0])))
              for sampler, rows in merged.items()}
    out_path.write_text(json.dumps(merged, indent=2))

    if failures:
        print(f"\n\n=== {len(failures)} run(s) DISCARDED for LLM API errors ===")
        for sampler, seed, errors, total, name in failures:
            print(f"  {sampler:<24} seed={seed:<4} {errors}/{total} failed   {name}")
        print("  These contribute no metrics. Re-run them once the API budget is restored.")
    if aborted:
        print("\n=== SWEEP ABORTED: every LLM call in a run failed. ===")
        print("  Remaining sampler/seed combinations were not attempted.")

    # Report over the merged file, so a sweep resumed one sampler at a time still
    # shows every run accumulated so far rather than just this invocation's.
    all_seeds = sorted({int(s) for rows in merged.values() for s in rows})
    print(f"\n\n=== {args.dataset}: end-to-end F1 across seeds {all_seeds} ===")
    print(f"{'sampler':<24}{'n':>4}{'mean':>9}{'stdev':>9}{'min':>9}{'max':>9}{'range':>9}   per-seed")
    all_means = {}
    for sampler, rows in merged.items():
        f1s = [row["f1"] for row in rows.values()]
        if not f1s:
            continue
        sd = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        all_means[sampler] = statistics.mean(f1s)
        per_seed = " ".join(f"{rows[str(s)]['f1']:.4f}" if str(s) in rows else "  --  " for s in all_seeds)
        print(f"{sampler:<24}{len(f1s):>4}{statistics.mean(f1s):>9.4f}{sd:>9.4f}{min(f1s):>9.4f}"
              f"{max(f1s):>9.4f}{max(f1s) - min(f1s):>9.4f}   {per_seed}")

    complete = [s for s in all_means if len(merged[s]) > 1]
    if len(all_means) > 1 and complete:
        spread = max(all_means.values()) - min(all_means.values())
        worst_within = max(
            max(r["f1"] for r in merged[s].values()) - min(r["f1"] for r in merged[s].values())
            for s in complete
        )
        print(f"\nBetween-sampler spread of means : {spread:.4f}")
        print(f"Largest within-sampler range    : {worst_within:.4f}")
        verdict = "larger than" if spread > worst_within else "NOT larger than"
        print(f"=> Sampler differences are {verdict} single-sampler seed noise.")
        if len(all_means) < len(SAMPLERS) or any(len(merged[s]) < len(all_seeds) for s in all_means):
            print("   (Read with care: the sweep is incomplete -- see the per-sampler n above.)")

    print("\nNote: only 'random' and 'similarity_stratified' actually resample when the seed")
    print("changes. 'diversity' is near-deterministic and 'stratified_diversity' and")
    print("'importance_based' are fully deterministic (the seed only breaks ties), so any")
    print("F1 variation in those rows measures LLM response non-determinism, not sampling")
    print("variance -- which makes them a useful noise floor for reading the other two.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
