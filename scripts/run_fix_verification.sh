#!/usr/bin/env bash
# 1) DBLP-ACM with the evidence-mask fix -- DBLP is the selection dataset, so changing
#    the method and re-measuring there is legitimate. Mostly cache hits.
# 2) Amazon-Walmart supervised, which previously aborted because a fixed 0.3 cut-off is
#    unreachable on a 12-attribute schema.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/fixver

run() {
    name="$1"; shift
    echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
    if "${PYTHON}" -u -m code.main "$@" > "logs/fixver/${name}.log" 2>&1; then
        echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
        grep -E "Selected global attributes|End-to-end F1|LLM API errors|Saved run summary" \
            "logs/fixver/${name}.log" | sed 's/^/    /'
    else
        echo "=== ${name} :: FAILED $(date '+%F %H:%M:%S')"
        tail -15 "logs/fixver/${name}.log"
    fi
}

run dblp_decisive_gap_fixed \
    --dataset DBLP-ACM --mode step15 --profile-sampler diversity \
    --profile-scoring decisive --selection-policy gap --min-k-attributes 1

run amazon_supervised_fixed \
    --dataset Amazon-Walmart --exclude-attributes original_id \
    --mode step1 --tuple-strategy supervised --step1-no-reblock \
    --tuple-n-pos 50 --tuple-n-neg 50

echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
