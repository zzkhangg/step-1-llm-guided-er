#!/usr/bin/env bash
# Three independent draws each of the two headline DBLP-ACM configurations.
#
# The matcher is not deterministic across runs even at temperature 0: OpenRouter spreads
# one model slug over providers that differ in quantization, so two fresh runs of the same
# configuration disagreed on 632 of 13,080 pairs and on 0.056 F1. Every number reported so
# far is a single draw, which is not enough to separate a real effect from that spread.
#
# --disable-matcher-cache is what makes the repetition real: with the cache on, a repeat
# run replays the stored answers and reproduces the first draw exactly, measuring nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/repeats

for i in 1 2 3; do
    for cfg in adaptive full; do
        name="${cfg}_draw${i}"
        echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
        if [ "$cfg" = "adaptive" ]; then
            args=(--mode step15 --profile-sampler diversity --profile-scoring decisive
                  --selection-policy gap --min-k-attributes 1)
        else
            args=(--mode full)
        fi
        "${PYTHON}" -u -m code.main --dataset DBLP-ACM --disable-matcher-cache \
            "${args[@]}" > "logs/repeats/${name}.log" 2>&1 \
            && echo "=== ${name} :: done $(date '+%F %H:%M:%S')" \
            || { echo "=== ${name} :: FAILED"; tail -5 "logs/repeats/${name}.log"; }
    done
done
echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
