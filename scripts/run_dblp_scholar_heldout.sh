#!/usr/bin/env bash
# Held-out evaluation on DBLP-Scholar.
#
# The configuration below is FROZEN by docs/heldout_preregistration_2026-09-04.md.
# Do not edit any flag in this file. If a flag must change, amend the Deviations log
# in that document first, with the date and the reason, and report the run as
# exploratory rather than held-out.
#
# Recovery changes are recorded in the 2026-09-07 Deviations entry. Each draw has
# its own pair-indexed checkpoint; cross-draw response caching stays disabled.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/heldout
exec 9>logs/heldout/driver.lock
flock -n 9 || { echo "Another held-out driver is running"; exit 1; }

"${PYTHON}" -c 'from code.llm_client import get_llm_model, provider_order, reasoning_enabled
assert get_llm_model() == "deepseek/deepseek-v4-flash-0731", "Frozen matcher model changed"
assert not provider_order(), "Frozen protocol uses unpinned routing"
assert not reasoning_enabled(), "Frozen protocol disables reasoning"' || exit 1

adaptive=(--mode step15 --profile-sampler diversity --profile-scoring decisive
          --selection-policy gap --min-k-attributes 1)
baseline=(--mode full)

for i in 1 2 3; do
    for cfg in adaptive full; do
        name="${cfg}_draw${i}"
        [ -f "logs/heldout/${name}.done" ] && { echo "=== ${name} :: already done, skipping"; continue; }
        if [ "$cfg" = "adaptive" ]; then args=("${adaptive[@]}"); else args=("${baseline[@]}"); fi
        echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
        checkpoint="cache/heldout/checkpoints/${name}"
        echo "=== restart $(date -Iseconds); checkpoint=${checkpoint} ===" >> "logs/heldout/${name}.log"
        if "${PYTHON}" -u -m code.main --dataset DBLP-Scholar --disable-matcher-cache \
            "${args[@]}" --matcher-checkpoint-dir "${checkpoint}" >> "logs/heldout/${name}.log" 2>&1; then
            date -Iseconds > "logs/heldout/${name}.done"
            echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
            tail -25 "logs/heldout/${name}.log"
        else
            echo "=== ${name} :: FAILED; completed pairs remain in ${checkpoint} ==="
            tail -20 "logs/heldout/${name}.log"
            exit 1
        fi
    done
done
echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
