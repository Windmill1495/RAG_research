#!/usr/bin/env bash
set -euo pipefail

root=/data1/zixuan/RAG_Research/experiments/adaqr_compcqr
python=/data1/zixuan/adaqr_compcqr_env/bin/python
model=/data1/zixuan/models/Mistral-7B-v0.1
cd "$root"

run=outputs/train4000_v1
mkdir -p "$run"
exec 9>"$run/.sft.lock"
flock -n 9
test ! -e "$run/sft"
printf '%s  %s\n' \
  '99aa030d0cf00e2760cd1c5fa9519bd095dc7007908ccab3cd0480b57be6e254' \
  'assets/adaqr_sft_train4000_nonfirst.jsonl' | sha256sum -c -

while [[ ! -s "$model/.weights-ready" ]] || \
  ! CUDA_VISIBLE_DEVICES= "$python" -B -c 'import torch, transformers, peft, accelerate, bitsandbytes' >/dev/null 2>&1; do
  if [[ ! -s "$model/.weights-ready" ]] && ! tmux has-session -t =adaqr-paper-mistral 2>/dev/null; then
    printf '%s\n' 'The model download stopped before verification. See fetch-mistral.log.'
    exit 1
  fi
  printf '%s Waiting for verified Mistral weights and the SFT environment\n' "$(date -Is)"
  sleep 60
done

while true; do
  gpu=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | \
    awk -F, '$2 + 0 >= 23000 && !found {print $1; found=1}')
  if [[ -n $gpu ]]; then
    break
  fi
  printf '%s Waiting for an idle GPU\n' "$(date -Is)"
  sleep 60
done

export CUDA_VISIBLE_DEVICES="$gpu"
printf '%s Starting train_4000 SFT on %s\n' "$(date -Is)" "$gpu"
exec "$python" -u -B train_fusion.py sft --seed-source user_qrecc \
  --seed-data assets/adaqr_sft_train4000_nonfirst.jsonl \
  --base-model "$model" --output "$run/sft"
