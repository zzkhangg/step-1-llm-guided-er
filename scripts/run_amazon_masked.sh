#!/usr/bin/env bash
# Re-runs the two Step-1.5 configurations on Amazon-Walmart with the evidence-mask fix
# in place. The earlier runs of these two predate that fix, so the published Amazon table
# currently mixes two versions of the selector; these runs make it consistent.
#
# The fix was found by inspecting Amazon-Walmart failures, so these numbers are no longer
# a clean out-of-sample estimate and must be reported as such.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/masked

common=(--dataset Amazon-Walmart --exclude-attributes original_id
        --profile-sampler diversity --selection-policy gap --min-k-attributes 1)

run() {
    name="$1"; shift
    echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
    if "${PYTHON}" -u -m code.main "${common[@]}" "$@" > "logs/masked/${name}.log" 2>&1; then
        echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
        grep -E "End-to-end F1|LLM API errors|Adaptive attributes per pair|Saved run summary" \
            "logs/masked/${name}.log" | sed 's/^/    /'
    else
        echo "=== ${name} :: FAILED $(date '+%F %H:%M:%S')"
        tail -15 "logs/masked/${name}.log"
    fi
}

run decisive_masked --mode step15 --profile-scoring decisive
run ordinal_masked  --mode step15 --profile-scoring ordinal

echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
