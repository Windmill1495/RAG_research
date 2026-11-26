#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python=${ADAQR_PYTHON:-/data1/zixuan/adaqr_compcqr_env/bin/python}
foundation_project=${ADAQR_FOUNDATION_PROJECT:-/data1/zixuan/RAG_Research}
foundation_experiment="$foundation_project/experiments/adaqr_compcqr"
foundation_run=${ADAQR_FOUNDATION_RUN:-"$foundation_experiment/outputs/train_half_v1"}
output=${ADAQR_EVIDENCE_OUTPUT:-"$PWD/outputs/server16_evidence"}
verifier=${ADAQR_VERIFIER_DIR:-/data1/zixuan/models/DeBERTa-v3-base-mnli-fever-anli}
sft=${ADAQR_SFT_ADAPTER:-"$foundation_experiment/outputs/train4000_v1/sft"}
lock="$foundation_experiment/outputs/train4000_v1/.fusion.lock"
gpu_pool=${ADAQR_EVIDENCE_GPUS:-2,3,4,5}
export ADAQR_GPU_MIN_FREE_MIB=${ADAQR_GPU_MIN_FREE_MIB:-18000}
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=${OPENAI_API_KEY:-local-lucene-only-placeholder}

reward_file="$output/rewards_evidence/checkpoint.jsonl"
preferences="$output/preferences_evidence.jsonl"
adapter="$output/dpo_evidence"
evidence_args=(
    --foundation-run "$foundation_run" --output "$output" --verifier "$verifier"
    --weight "${ADAQR_EVIDENCE_WEIGHT:-1.0}" --epsilon "${ADAQR_EVIDENCE_EPSILON:-0.1}"
    --batch-size "${ADAQR_EVIDENCE_BATCH_SIZE:-16}"
)

usage() {
    cat <<'HELP'
bash run_server16_evidence.sh check
bash run_server16_evidence.sh smoke [limit]
bash run_server16_evidence.sh rewards
bash run_server16_evidence.sh pairs
bash run_server16_evidence.sh dpo
bash run_server16_evidence.sh evaluate [single|compcqr] [--dry-run]
bash run_server16_evidence.sh status
HELP
}

pick_gpus() {
    "$python" -B -c '
import sys
from parallel_adaqr_half import gpu_map
pool = tuple(int(value) for value in sys.argv[1].split(","))
count = int(sys.argv[2])
if min(pool) < 0 or len(set(pool)) != len(pool):
    raise ValueError("ADAQR_EVIDENCE_GPUS must contain distinct physical GPU IDs")
if count > len(pool) or (count == 4 and len(pool) != 4):
    raise ValueError(f"Select {count} physical GPUs in ADAQR_EVIDENCE_GPUS")
indices = pool[:count] if count else pool
devices = gpu_map(indices)
print(",".join(devices[index] for index in indices))
' "$gpu_pool" "$1"
}

acquire_gpu_lock() {
    exec 9>> "$lock"
    if ! flock -n 9; then
        printf 'Another foundation or Evidence GPU job holds %s\n' "$lock" >&2
        exit 1
    fi
}

run_evidence() {
    local stage=$1
    shift
    "$python" -u -B server16_evidence.py "$stage" "${evidence_args[@]}" "$@"
}

stop_workers() {
    local pid
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

case "${1:-help}" in
    check|status)
        CUDA_VISIBLE_DEVICES="" run_evidence "$1"
        if [[ $1 == status ]]; then
            if [[ -s "$preferences" ]]; then
                printf 'Evidence preferences: %s pairs\n' "$(wc -l < "$preferences")"
            fi
            if [[ -s "$adapter/adaqr_metadata.json" && -s "$adapter/adapter_model.safetensors" ]]; then
                printf 'Evidence DPO adapter: %s\n' "$adapter"
            fi
        fi
        ;;
    smoke)
        CUDA_VISIBLE_DEVICES="" run_evidence smoke --device cpu --limit "${2:-3}"
        ;;
    rewards)
        acquire_gpu_lock
        devices=$(pick_gpus 0)
        IFS=, read -r -a gpu_uuids <<< "$devices"
        pids=()
        trap stop_workers EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        for rank in "${!gpu_uuids[@]}"; do
            CUDA_VISIBLE_DEVICES="${gpu_uuids[rank]}" \
                "$python" -u -B server16_evidence.py score "${evidence_args[@]}" \
                --workers "${#gpu_uuids[@]}" --rank "$rank" --device cuda:0 &
            pids+=("$!")
        done
        for rank in "${!pids[@]}"; do
            wait "${pids[rank]}"
            unset 'pids[rank]'
        done
        trap - EXIT INT TERM
        CUDA_VISIBLE_DEVICES="" run_evidence merge --workers "${#gpu_uuids[@]}"
        ;;
    pairs)
        test -s "$reward_file"
        temp=$(mktemp -d "$output/.preferences-XXXXXX")
        trap 'rm -f -- "$temp/preferences.jsonl"; rmdir -- "$temp"' EXIT
        CUDA_VISIBLE_DEVICES="" "$python" -u -B train_fusion.py pairs \
            --rewards "$output/rewards_evidence" --alpha 0.5 --delta 0.1 \
            --output "$temp/preferences.jsonl"
        if [[ -e "$preferences" ]]; then
            cmp -- "$temp/preferences.jsonl" "$preferences"
        else
            ln -- "$temp/preferences.jsonl" "$preferences"
        fi
        printf 'Evidence preferences: %s pairs\n' "$(wc -l < "$preferences")"
        ;;
    dpo)
        test -s "$preferences"
        test -s "$sft/adapter_config.json"
        test -s "$sft/adapter_model.safetensors"
        if [[ -s "$adapter/adaqr_metadata.json" && -s "$adapter/adapter_model.safetensors" ]]; then
            printf 'Completed Evidence DPO adapter: %s\n' "$adapter"
            exit 0
        fi
        acquire_gpu_lock
        CUDA_VISIBLE_DEVICES=$(pick_gpus 4)
        export CUDA_VISIBLE_DEVICES
        launch=("$python" -u -B -m torch.distributed.run --standalone --nproc-per-node=4 server16_dpo_ddp.py)
        arguments=(--preferences "$preferences" --sft-adapter "$sft" --seed 42 --max-length 2048 --output "$adapter")
        "${launch[@]}" reference "${arguments[@]}"
        CUDA_VISIBLE_DEVICES=$(pick_gpus 4)
        exec "${launch[@]}" train "${arguments[@]}"
        ;;
    evaluate)
        mode=${2:-single}
        case "$mode" in
            single) gpu_count=1 ;;
            compcqr) gpu_count=2 ;;
            *) usage >&2; exit 2 ;;
        esac
        report="$output/eval_$mode"
        extra=()
        case "${3:-}" in
            --dry-run)
                [[ -s "$adapter/adaqr_metadata.json" ]] || adapter=$sft
                export CUDA_VISIBLE_DEVICES=""
                extra=(--dry-run)
                ;;
            '')
                test -s "$adapter/adaqr_metadata.json"
                acquire_gpu_lock
                CUDA_VISIBLE_DEVICES=$(pick_gpus "$gpu_count")
                export CUDA_VISIBLE_DEVICES
                if [[ -f "$report/manifest.json" ]]; then extra=(--resume); fi
                ;;
            *) usage >&2; exit 2 ;;
        esac
        exec "$python" -u -B evaluate_fusion.py \
            --project-root "${ADAQR_EVAL_PROJECT_ROOT:-$foundation_project}" --adapter "$adapter" \
            --baseline-run "${ADAQR_BASELINE_RUN:-$foundation_project/outputs/bm25_gpu4_experiments/main-test2000}" \
            --subset-file "${ADAQR_TEST_SUBSET:-/data1/zixuan/qrecc_full/subsets/test_2000.json}" \
            --official-test "${ADAQR_OFFICIAL_TEST:-/data1/zixuan/qrecc_full/raw/qrecc-test.json}" \
            --inference "$mode" --aux-model "${ADAQR_LLAMA_DIR:-/data1/zixuan/models/Llama-3.1-8B-Instruct}" \
            --aux-seed 42 --compcqr-selection-top-k 5 --output-dir "$report" "${extra[@]}"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
