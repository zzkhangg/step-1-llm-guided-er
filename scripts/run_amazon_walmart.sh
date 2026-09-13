#!/usr/bin/env bash
# Amazon-Walmart: the four configurations compared in the Week 3 report.
#
# Run from the repo root with the conda env active:
#   conda activate comp3710
#   bash scripts/run_amazon_walmart.sh
#
# The configurations run one at a time on purpose. Running them concurrently shares
# the OpenRouter rate limit, so each one slows the others down without finishing the
# set any sooner.
#
# Ordered so the headline comparison (decisive vs. no selection) completes first: if
# the run has to be cut short, the first two configurations are the ones the report
# needs. The matcher response cache stays enabled, so pairs where two configurations
# select the same attributes share one answer and cannot contribute a spurious
# difference -- the same paired design used on DBLP-ACM.
#
# original_id is excluded: it is the row number of each table (1..22074 and 1..2554),
# so it carries no information about either record, and its two integers would
# otherwise be shown to the matcher and fed to the transfer model as a numeric feature.

set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve the interpreter explicitly: a detached shell (nohup, cron) does not source the
# profile that puts the conda env on PATH, so a bare "python" is not found there.
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
if [ ! -x "${PYTHON}" ]; then
    echo "Interpreter not found: ${PYTHON} -- set PYTHON=/path/to/python" >&2
    exit 1
fi

mkdir -p logs/amazon_console

common=(--dataset Amazon-Walmart --exclude-attributes original_id)

run() {
    local name="$1"; shift
    local out="logs/amazon_console/${name}.log"
    echo "=== ${name} :: started $(date '+%H:%M:%S') ==="
    if "${PYTHON}" -m code.main "${common[@]}" "$@" >"${out}" 2>&1; then
        echo "=== ${name} :: done $(date '+%H:%M:%S') -> ${out}"
    else
        echo "=== ${name} :: FAILED, see ${out}" >&2
        tail -20 "${out}" >&2
        return 1
    fi
}

# Step 1.5 uses --selection-policy gap, not the pipeline default. The default
# "cumulative" rule is not scale-free: importance is normalized to sum to one, so on a
# 12-attribute schema the 0.8 target needs more attributes than max_k allows and the cap
# binds for every pair. That is measured, not predicted -- the earlier Amazon-Walmart run
# recorded selection_size min = mean = max = 5, i.e. a fixed subset dressed up as adaptive
# selection. "gap" cuts at the largest drop in the ranked importance profile, has no
# threshold to tune, and on DBLP-ACM matches the best hand-tuned cumulative target.
POLICY=(--selection-policy gap --min-k-attributes 1)

# Ordered by what the report needs most if the set has to be cut short.
run full \
    --mode full

run step15_decisive \
    --mode step15 --profile-sampler diversity --profile-scoring decisive "${POLICY[@]}"

run step1_supervised \
    --mode step1 --tuple-strategy supervised --step1-no-reblock \
    --tuple-n-pos 50 --tuple-n-neg 50

run step15_ordinal \
    --mode step15 --profile-sampler diversity --profile-scoring ordinal "${POLICY[@]}"

echo
echo "All four configurations finished. Summaries:"
ls -dt logs/Amazon-Walmart/runs/* | head -4
