#!/usr/bin/env bash
set -euo pipefail

cd /data1/zixuan/RAG_Research/experiments/adaqr_compcqr
export CUDA_VISIBLE_DEVICES=
python=/data1/zixuan/adaqr_compcqr_env/bin/python

while ! grep -q '56,003 of 56,003 shards' prepare-paper-index.log; do
  if ! tmux has-session -t =adaqr-paper-corpus 2>/dev/null; then
    printf '%s\n' 'The official archive preparation stopped. See prepare-paper-index.log.'
    exit 1
  fi
  printf '%s Waiting for official QReCC archive extraction\n' "$(date -Is)"
  sleep 60
done

while ! "$python" -m pip show pyserini >/dev/null 2>&1; do
  printf '%s Waiting for the Pyserini environment\n' "$(date -Is)"
  sleep 60
done

exec ionice -c 3 nice -n 10 "$python" -u -B prepare_paper_index.py \
  --archive /data1/zixuan/qrecc_full/incoming/QReCC_full/passages.zip \
  --work /data1/zixuan/qrecc_full/lucene-papers \
  --index-python "$python"
