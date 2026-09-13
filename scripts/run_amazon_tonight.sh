#!/usr/bin/env bash
# Two Amazon-Walmart configurations, run back to back overnight.
#
# 1. decisive + gap + min_k=2  -- targets the one place condensation loses. With the
#    evidence-mask fix in place, decisive/gap/min_k=1 ties full on per-pair accuracy
#    (McNemar p=0.96) but trails on F1 because its errors fall on the positive class.
#    Splitting the errors by selection size shows why: at k=1 it is wrong 868 times
#    against full's 840, while at k=2 it is wrong 175 against full's 211. Forcing a
#    floor of two attributes removes the only bucket where it loses.
#
# 2. cumulative with the cap removed -- supplies the missing middle of the frontier.
#    Amazon currently has two points, 12 attributes and 1.34; the 0.8 target needs
#    about 7.7 of 12, so uncapping asks whether accuracy degrades gradually or falls
#    off a cliff between them.
#
# They run one at a time: concurrent runs share the OpenRouter rate limit, so running
# both at once finishes no sooner and slows each one down.
#
# NOTE ON DURABILITY: matcher answers are written to cache/Amazon-Walmart as they are
# produced, so a crash costs wall-clock time but never re-bills a pair already answered.
# Re-running this script after an interruption resumes cheaply.

set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/tonight

common=(--dataset Amazon-Walmart --exclude-attributes original_id
        --mode step15 --profile-sampler diversity --profile-scoring decisive)

run() {
    name="$1"; shift
    echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
    if "${PYTHON}" -u -m code.main "${common[@]}" "$@" > "logs/tonight/${name}.log" 2>&1; then
        echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
        grep -E "End-to-end F1|LLM API errors|Adaptive attributes per pair|Saved run summary" \
            "logs/tonight/${name}.log" | sed 's/^/    /'
    else
        echo "=== ${name} :: FAILED $(date '+%F %H:%M:%S')"
        tail -15 "logs/tonight/${name}.log"
    fi
}

run decisive_gap_mink2 --selection-policy gap --min-k-attributes 2
run decisive_cum_uncapped --selection-policy cumulative --max-k-attributes 0 --min-k-attributes 1

echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
