#!/usr/bin/env bash
# Two jobs, chained, both resumable via .done markers.
#
# JOB 1 -- evidence mask: keep vs drop, back to back on ONE day.
#   The earlier comparison ran the two modes a week apart, across a period in which the
#   matcher drifted toward answering Yes more often -- the same direction as the effect that
#   was attributed to showing blank columns. That result is retracted. Running both modes
#   inside one window removes the drift confound; nothing else about the setup changes.
#
#   Both modes share ONE matcher cache directory on purpose. 64,857 of the 88,296 pairs have
#   no blank attribute, so the two modes build a byte-identical prompt for them; replaying
#   the first run's answer there is correct by construction and removes matcher noise from
#   the 73% of pairs the mask cannot touch. Only the 23,439 pairs whose prompt actually
#   differs get fresh calls in the second run. That is also what makes job 1 affordable:
#   ~112k calls instead of ~177k.
#
#   Analysis is pair-by-pair (McNemar) on those 23,439 pairs, not a difference of F1 -- the
#   F1 gap is far below the 5.2% run-to-run noise floor measured on this dataset.
#
# JOB 2 -- Amazon repeats on a fixed 20,000-pair slice.
#   Two identical `full` runs over this same slice disagreed on 5.20% of pairs and 0.078 F1,
#   while the full-vs-adaptive gap is 0.023 -- three times smaller than the noise. Every
#   Amazon number currently rests on a single draw and cannot support a comparison. Three
#   draws per arm turn the chapter from point estimates into ranges.
#
#   logs/fliprate/full_rep{1,2} are already draws 1 and 2 of the `full` arm on this exact
#   slice and seed, so only full_rep3 is needed here; the adaptive arm needs all three.
#   Each repeat writes its own cache directory: an empty directory forces real calls just as
#   --disable-matcher-cache would, but the answers are kept, so a run that dies at 99%
#   resumes instead of re-billing 20,000 calls.
#
#   --allow-api-errors is deliberate in job 2 for the same reason as the flip-rate script:
#   the analysis is paired and drops any pair that errored in either run. Job 1 does NOT use
#   it -- its runs double as the same-day reference numbers for the report.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$HOME/miniconda3/envs/comp3710/bin/python}"
mkdir -p logs/masktest logs/amzsub cache/masktest cache/amzsub

run() {
    local dir="$1" name="$2"; shift 2
    if [ -s "logs/${dir}/${name}.done" ]; then echo "=== ${name} :: already done, skipping"; return; fi
    echo "=== ${name} :: started $(date '+%F %H:%M:%S') ==="
    if "${PYTHON}" -u -m code.main "$@" > "logs/${dir}/${name}.log" 2>&1; then
        date > "logs/${dir}/${name}.done"
        echo "=== ${name} :: done $(date '+%F %H:%M:%S')"
        grep -E "End-to-end F1|End-to-end precision|End-to-end recall|LLM API errors|Adaptive attributes per pair|Saved run summary" \
            "logs/${dir}/${name}.log" | sed 's/^/    /'
    else
        echo "=== ${name} :: FAILED $(date '+%F %H:%M:%S')"; tail -15 "logs/${dir}/${name}.log"
    fi
}

base=(--dataset Amazon-Walmart --exclude-attributes original_id)
adaptive=(--mode step15 --profile-sampler diversity --profile-scoring decisive
          --selection-policy gap --min-k-attributes 1)

echo "########## JOB 1: evidence mask keep vs drop ##########"
run masktest mask_drop "${base[@]}" "${adaptive[@]}" \
    --evidence-mask-mode drop --matcher-cache-dir cache/masktest/shared
run masktest mask_keep "${base[@]}" "${adaptive[@]}" \
    --evidence-mask-mode keep --matcher-cache-dir cache/masktest/shared

echo "########## JOB 2: Amazon 20k-slice repeats ##########"
slice=(--max-candidate-pairs 20000 --candidate-sample-seed 7 --allow-api-errors)

run amzsub full_rep3 "${base[@]}" --mode full "${slice[@]}" \
    --matcher-cache-dir cache/amzsub/full_rep3
for i in 1 2 3; do
    run amzsub "adaptive_rep${i}" "${base[@]}" "${adaptive[@]}" "${slice[@]}" \
        --matcher-cache-dir "cache/amzsub/adaptive_rep${i}"
done

echo "########## ALL DONE $(date '+%F %H:%M:%S') ##########"
