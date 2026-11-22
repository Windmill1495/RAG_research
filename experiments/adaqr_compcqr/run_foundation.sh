#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python=${ADAQR_PYTHON:-python3}
output="$PWD/outputs/train_half_v1"
lock="$output/.fusion.lock"
sft=${ADAQR_SFT_ADAPTER:-"$PWD/inputs/sft"}
export ADAQR_GPU_MIN_FREE_MIB=${ADAQR_GPU_MIN_FREE_MIB:-18500}
export OPENAI_API_KEY=${OPENAI_API_KEY:-local-lucene-only-placeholder}
if [[ -n "${JAVA_HOME:-}" ]]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

usage() {
    cat <<'HELP'
bash run_foundation.sh prepare
bash run_foundation.sh status
bash run_foundation.sh components
bash run_foundation.sh baseline
bash run_foundation.sh pairs
bash run_foundation.sh dpo [physical_gpu]
bash run_foundation.sh evaluate [single|compcqr] [--dry-run]
HELP
}

pick_gpus() {
    "$python" -B -c '
import sys
from parallel_adaqr_half import gpu_map
indices = [int(value) for value in sys.argv[1].split(",")]
devices = gpu_map(indices)
print(",".join(devices[index] for index in indices))
' "$1"
}

score_gpus=${ADAQR_SCORE_GPUS:-0,1}
case "${1:-help}" in
    prepare|status|baseline)
        exec "$python" -u -B parallel_adaqr_half.py "$1"
        ;;
    components)
        exec "$python" -u -B parallel_components.py run
        ;;
    pairs)
        test -s "$output/rewards_baseline/manifest.json"
        CUDA_VISIBLE_DEVICES="" exec flock -n "$lock" "$python" -u -B train_fusion.py pairs \
            --rewards "$output/rewards_baseline" --alpha 0.5 --delta 0.1 \
            --output "$output/preferences_baseline.jsonl"
        ;;
    dpo)
        test -s "$output/preferences_baseline.jsonl"
        CUDA_VISIBLE_DEVICES=$(pick_gpus "${2:-${score_gpus%%,*}}") || exit 1
        export CUDA_VISIBLE_DEVICES
        exec flock -n "$lock" "$python" -u -B train_fusion.py dpo \
            --preferences "$output/preferences_baseline.jsonl" --sft-adapter "$sft" \
            --seed 42 --epochs 1 --max-length 2048 --output "$output/dpo_baseline"
        ;;
    evaluate)
        mode=${2:-single}
        case "$mode" in
            single) gpus=${score_gpus%%,*} ;;
            compcqr) gpus=$score_gpus ;;
            *) usage >&2; exit 2 ;;
        esac
        : "${ADAQR_EVAL_PROJECT_ROOT:?Set the original signed RAG project root}"
        : "${ADAQR_BASELINE_RUN:?Set the completed fixed 2,000-turn baseline directory}"
        : "${ADAQR_TEST_SUBSET:?Set the ordered test_2000.json path}"
        : "${ADAQR_OFFICIAL_TEST:?Set the original qrecc-test.json path}"
        report="$output/eval_$mode"
        extras=()
        adapter="$output/dpo_baseline"
        if [[ "${3:-}" == --dry-run ]]; then
            extras=(--dry-run)
            adapter=$sft
        else
            test -s "$adapter/adaqr_metadata.json"
            CUDA_VISIBLE_DEVICES=$(pick_gpus "$gpus") || exit 1
            export CUDA_VISIBLE_DEVICES
            if [[ -f "$report/manifest.json" ]]; then extras=(--resume); fi
        fi
        mkdir -p "$output"
        exec flock -n "$lock" "$python" -u -B evaluate_fusion.py \
            --project-root "$ADAQR_EVAL_PROJECT_ROOT" --adapter "$adapter" \
            --baseline-run "$ADAQR_BASELINE_RUN" --subset-file "$ADAQR_TEST_SUBSET" \
            --official-test "$ADAQR_OFFICIAL_TEST" --inference "$mode" \
            --aux-model "${ADAQR_LLAMA_DIR:-"$PWD/models/Llama-3.1-8B-Instruct"}" \
            --aux-seed 42 --output-dir "$report" "${extras[@]}"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
