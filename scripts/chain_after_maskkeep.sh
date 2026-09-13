#!/usr/bin/env bash
# Wait for the in-flight evidence-mask run to finish, then measure the flip rate.
# Sequential on purpose: both are API-bound and share one rate limit, so overlapping them
# delays the first answer without advancing the second.
cd "$(dirname "$0")/.."
while pgrep -f "code\.main --dataset Amazon-Walmart .* --evidence-mask-mode keep" >/dev/null; do
    sleep 60
done
echo "=== maskkeep finished, starting flip-rate $(date '+%F %H:%M:%S') ==="
exec bash scripts/run_amazon_flip_rate.sh
