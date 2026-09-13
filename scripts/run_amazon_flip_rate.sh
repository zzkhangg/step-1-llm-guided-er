#!/usr/bin/env bash
# How much do the matcher's answers move between two runs of one configuration?
#
# On DBLP-ACM two fresh runs of no-selection disagreed on 632 of 13,080 pairs (4.8%) and on
# 0.056 F1. Several Amazon-Walmart differences are smaller than that, so whether those
# comparisons mean anything depends on whether the same instability is present here -- and
# that is not safe to assume in either direction. Amazon-Walmart has seven times the pairs,
# which averages out random flips, but the DBLP-ACM movement was a systematic shift in how
# willingly the model answered Yes, and that does not average out.
#
# Measuring the disagreement rate needs two runs, not six: it is the quantity underneath the
# F1 spread, and it is what decides whether full repetition is worth 12 more hours.
#
# Both runs take the same fixed slice of candidate pairs, so the only thing differing
# between them is the matcher. Each repeat writes to its own cache directory rather than
# using --disable-matcher-cache: an empty directory forces real calls just the same, but the
# answers are kept, so a run that dies at 99% resumes instead of re-billing 20,000 calls.
# Sharing one directory would make the second run replay the first and report a
# disagreement rate of zero.
#
# --allow-api-errors is deliberate here. The guard exists because a failed call is scored as
# a non-match, which corrupts precision and recall; but this experiment does not use those,
# it compares the two runs pair by pair. The analysis drops any pair that errored in either
# run, so a handful of failures costs a little sample size and biases nothing. The previous
# attempt lost 20,000 paid calls to 7 failures for exactly this reason.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/fliprate

for i in 1 2; do
    name="full_rep${i}"
    [ -s "logs/fliprate/${name}.done" ] && { echo "=== ${name} :: already done"; continue; }
    echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
    if "${PYTHON}" -u -m code.main --dataset Amazon-Walmart --exclude-attributes original_id \
           --mode full --matcher-cache-dir "cache/fliprate/${name}" --allow-api-errors \
           --max-candidate-pairs 20000 --candidate-sample-seed 7 \
           > "logs/fliprate/${name}.log" 2>&1; then
        date > "logs/fliprate/${name}.done"
        echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
    else
        echo "=== ${name} :: FAILED"; tail -5 "logs/fliprate/${name}.log"
    fi
done
echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
