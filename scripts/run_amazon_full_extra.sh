#!/usr/bin/env bash
# Two more `full` draws on the 20,000-pair slice, run on the SAME DAY as the adaptive draws.
#
# Why this exists: logs/fliprate/full_rep{1,2} were run on 2026-08-27. On 2026-08-31 the
# matcher answers Yes far more often than it did then -- measured on 69,087 byte-identical
# prompts, Yes went 1,589 -> 3,250, with No->Yes flips outnumbering Yes->No 2,269 to 608.
# That is a systematic shift, not sampling noise, so an 27/08 `full` draw cannot be compared
# against a 31/08 adaptive draw. Reusing rep1/rep2 as two of the three `full` draws would
# have rebuilt the exact confound this whole exercise is meant to remove.
#
# rep1/rep2 stay valid for what they were built for -- they were run within one day of each
# other, so the 5.20% flip rate they measure is sound. They are just not usable as arms of a
# cross-configuration comparison against today's runs.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/amzsub cache/amzsub

for i in 4 5; do
    name="full_rep${i}"
    [ -s "logs/amzsub/${name}.done" ] && { echo "=== ${name} :: already done"; continue; }
    echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
    if "${PYTHON}" -u -m code.main --dataset Amazon-Walmart --exclude-attributes original_id \
           --mode full --max-candidate-pairs 20000 --candidate-sample-seed 7 \
           --allow-api-errors --matcher-cache-dir "cache/amzsub/${name}" \
           > "logs/amzsub/${name}.log" 2>&1; then
        date > "logs/amzsub/${name}.done"
        echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
        grep -E "End-to-end (precision|recall|F1)|LLM API errors" "logs/amzsub/${name}.log" | sed 's/^/    /'
    else
        echo "=== ${name} :: FAILED"; tail -15 "logs/amzsub/${name}.log"
    fi
done
echo "=== EXTRA FULL DRAWS DONE $(date '+%F %H:%M:%S') ==="
