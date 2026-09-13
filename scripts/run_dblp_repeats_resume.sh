#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/repeats
for i in 2 3; do
    for cfg in adaptive full; do
        name="${cfg}_draw${i}"
        [ -s "logs/repeats/${name}.done" ] && { echo "=== ${name} :: already done, skipping"; continue; }
        echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
        if [ "$cfg" = "adaptive" ]; then
            args=(--mode step15 --profile-sampler diversity --profile-scoring decisive
                  --selection-policy gap --min-k-attributes 1)
        else
            args=(--mode full)
        fi
        if "${PYTHON}" -u -m code.main --dataset DBLP-ACM --disable-matcher-cache \
               "${args[@]}" > "logs/repeats/${name}.log" 2>&1; then
            date > "logs/repeats/${name}.done"
            echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
        else
            echo "=== ${name} :: FAILED"; tail -5 "logs/repeats/${name}.log"
        fi
    done
done
echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
