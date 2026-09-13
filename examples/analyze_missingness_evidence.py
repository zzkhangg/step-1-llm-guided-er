"""Is a one-sided blank attribute evidence of a non-match?

The evidence mask currently zeroes an attribute when it is blank on *either* side, which
collapses two situations that are not the same thing:

  both_present  -- A and B both carry a value; the attribute can agree or contradict
  one_sided     -- A carries a value, B does not (or vice versa)
  both_blank    -- neither side carries a value; there is nothing to say

The supervisor's question is whether `one_sided` deserves to reach the matcher rather than
be masked away with `both_blank`. That is an empirical question about the data, answerable
with the gold labels and no LLM calls: if a one-sided blank were evidence of a non-match, the
match rate among one_sided pairs would sit well below the match rate among both_present
pairs. If instead absence is just incomplete cataloguing -- a property of the source table
rather than of the entity -- the two rates will be close, and showing the gap to the matcher
adds noise rather than signal.

Reported per attribute, over the post-blocking candidate pairs of an existing run.

Usage:  python -m examples.analyze_missingness_evidence [run_dir]
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from code.main import DATASET_CONFIGS, load_configured_dataset  # noqa: E402

DEFAULT_RUN = "logs/Amazon-Walmart/runs/20260831_202242_step15_diversity_decisive_gap_mink1"


def blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip()


def main(run_dir=DEFAULT_RUN, dataset="Amazon-Walmart"):
    df_a, df_b, gt = load_configured_dataset(DATASET_CONFIGS[dataset])
    pairs = []
    with open(os.path.join(run_dir, "final_results.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            pairs.append((int(row["indexA"]), int(row["indexB"])))

    attrs = [c for c in df_a.columns if c in df_b.columns]
    n_match = sum(1 for p in pairs if p in gt)
    print(f"dataset {dataset}   candidate pairs {len(pairs)}   of which gold matches {n_match} "
          f"({n_match / len(pairs):.2%})\n")

    hdr = (f"{'attribute':18s} {'both_present':>22s} {'one_sided':>22s} {'both_blank':>22s}")
    print(hdr)
    print(f"{'':18s} {'n / match-rate':>22s} {'n / match-rate':>22s} {'n / match-rate':>22s}")
    print("-" * len(hdr))

    for attr in attrs:
        va, vb = df_a[attr].values, df_b[attr].values
        buckets = {"both_present": [0, 0], "one_sided": [0, 0], "both_blank": [0, 0]}
        for ia, ib in pairs:
            ba, bb = blank(va[ia]), blank(vb[ib])
            key = "both_blank" if (ba and bb) else ("both_present" if not (ba or bb) else "one_sided")
            buckets[key][0] += 1
            if (ia, ib) in gt:
                buckets[key][1] += 1
        cells = []
        for key in ("both_present", "one_sided", "both_blank"):
            n, m = buckets[key]
            cells.append(f"{n:7d} / {m / n:7.3%}" if n else f"{0:7d} /       -")
        print(f"{attr:18s} " + " ".join(f"{c:>22s}" for c in cells))

    print("\nRead the middle column against the left one. A one_sided match rate close to the")
    print("both_present rate means absence carries no signal about matching, and masking is")
    print("the right call. A clearly lower rate means absence is evidence and should be shown.")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
