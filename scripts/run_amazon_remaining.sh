#!/usr/bin/env bash
# Resumes the two Amazon-Walmart configurations left unfinished when the earlier
# run was killed. full and step15_decisive are already complete; their results are
# in logs/Amazon-Walmart/runs/.
#
# Launched with setsid so it detaches from the launching session's process group:
# nohup alone only ignores SIGHUP, and the previous attempt died when its parent
# session was torn down mid-run.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/amazon_console

common=(--dataset Amazon-Walmart --exclude-attributes original_id)
POLICY=(--selection-policy gap --min-k-attributes 1)

run() {
    name="$1"; shift
    out="logs/amazon_console/${name}.log"
    echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
    if "${PYTHON}" -u -m code.main "${common[@]}" "$@" >"${out}" 2>&1; then
        echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
        grep -E "End-to-end F1|LLM API errors|Saved run summary" "${out}" | sed 's/^/    /'
    else
        echo "=== ${name} :: FAILED $(date '+%F %H:%M:%S'), see ${out}"
        tail -20 "${out}"
    fi
}

run step1_supervised \
    --mode step1 --tuple-strategy supervised --step1-no-reblock \
    --tuple-n-pos 50 --tuple-n-neg 50

run step15_ordinal \
    --mode step15 --profile-sampler diversity --profile-scoring ordinal "${POLICY[@]}"

echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
