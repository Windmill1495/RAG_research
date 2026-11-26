#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
project=$(realpath ../..)
python=/data1/zixuan/adaqr_compcqr_env/bin/python
output="$PWD/outputs/train_half_v1"
lock="$PWD/outputs/train4000_v1/.fusion.lock"
sft="$PWD/outputs/train4000_v1/sft"
test_data=/data1/zixuan/qrecc_full/subsets/test_2000.json
official_test=/data1/zixuan/qrecc_full/raw/qrecc-test.json
comparison="$project/outputs/bm25_gpu4_experiments/main-test2000"

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export OPENAI_API_KEY=sk-local-bm25-unused

usage() {
    cat <<'HELP'
bash run_adaqr_half.sh prepare-and-archive
bash run_adaqr_half.sh components
bash run_adaqr_half.sh baseline
bash run_adaqr_half.sh baseline-gpu2345
bash run_adaqr_half.sh fetch-nli
bash run_adaqr_half.sh evidence
bash run_adaqr_half.sh pairs baseline|evidence
bash run_adaqr_half.sh dpo baseline|evidence [4|5]
bash run_adaqr_half.sh evaluate baseline|evidence [single|compcqr]
bash run_adaqr_half.sh status
HELP
}

stage=${1:-help}
case "$stage" in
    baseline-gpu2345)
        ADAQR_GPU_MIN_FREE_MIB="${ADAQR_GPU_MIN_FREE_MIB:-18500}" \
            exec "$python" -u -B parallel_adaqr_components_gpu2345.py baseline
        ;;
    prepare|prepare-and-archive|components|baseline|evidence|fetch-nli|status)
        shift
        exec "$python" -u -B parallel_adaqr_half.py "$stage" "$@"
        ;;
    pairs|dpo|evaluate)
        arm=${2:-}
        if [[ "$arm" != baseline && "$arm" != evidence ]]; then
            usage >&2
            exit 2
        fi
        ;;
    help|-h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

preferences="$output/preferences_${arm}.jsonl"
adapter="$output/dpo_${arm}"
pick_gpus() {
    "$python" -B -c '
import sys
from parallel_adaqr_half import gpu_map
indices = list(map(int, sys.argv[1:]))
values = gpu_map(indices)
print(",".join(values[index] for index in indices))
' "$@"
}

case "$stage" in
    pairs)
        CUDA_VISIBLE_DEVICES="" exec flock -n "$lock" "$python" -u -B train_fusion.py pairs \
            --rewards "$output/rewards_$arm" --alpha 0.5 --delta 0.1 --output "$preferences"
        ;;
    dpo)
        gpu=${3:-4}
        if [[ "$gpu" != 4 && "$gpu" != 5 ]]; then
            printf 'DPO GPU must be 4 or 5\n' >&2
            exit 2
        fi
        test -s "$preferences"
        CUDA_VISIBLE_DEVICES=$(pick_gpus "$gpu") || exit 1
        export CUDA_VISIBLE_DEVICES
        exec flock -n "$lock" "$python" -u -B train_fusion.py dpo \
            --preferences "$preferences" --sft-adapter "$sft" --seed 42 --epochs 1 \
            --max-length 2048 --output "$adapter"
        ;;
    evaluate)
        mode=${3:-single}
        if [[ "$mode" == compcqr ]]; then
            CUDA_VISIBLE_DEVICES=$(pick_gpus 4 5) || exit 1
        elif [[ "$mode" == single ]]; then
            CUDA_VISIBLE_DEVICES=$(pick_gpus 4) || exit 1
        else
            usage >&2
            exit 2
        fi
        export CUDA_VISIBLE_DEVICES
        test -s "$adapter/adaqr_metadata.json"
        report="$output/eval_${arm}_${mode}"
        resume=()
        if [[ -f "$report/manifest.json" ]]; then
            resume=(--resume)
        fi
        exec flock -n "$lock" "$python" -u -B evaluate_fusion.py \
            --project-root "$project" --adapter "$adapter" --baseline-run "$comparison" \
            --subset-file "$test_data" --official-test "$official_test" --inference "$mode" \
            --aux-model /data1/zixuan/models/Llama-3.1-8B-Instruct --aux-seed 42 \
            --output-dir "$report" "${resume[@]}"
        ;;
esac
