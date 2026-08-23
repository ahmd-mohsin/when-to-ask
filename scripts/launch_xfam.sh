#!/usr/bin/env bash
# T7 cross-family replication (decisions/028 Amendment A item 9).
#
# R2's collection protocol VERBATIM on the SAME 60 tasks with the SAME frozen
# class artifact -- only the model family changes. Because the artifact is
# per-task, this needs no new registry work and no LLM budget.
#
# Usage:
#   scripts/launch_xfam.sh smoke                 # 3 tasks x 2 seeds, GATE
#   scripts/launch_xfam.sh full 12               # 60 tasks x 12 seeds
#   scripts/launch_xfam.sh full 24               # extend to R2 parity
#   MODEL=google/gemma-3-27b-it scripts/launch_xfam.sh smoke   # fallback
#
# Interrupt-safe: existing <run>.json files are skipped, so `full 24` after
# `full 12` ADDS seeds 12-23 rather than redoing work.
#
# --classes keeps the sealed pool (swe_60+) untouched by construction.
# max_steps=50 matches R2 exactly; trajectories under a different cap are NOT
# comparable, which is the whole point of this run.
set -u
cd /home/ubuntu/when-to-ask

MODE="${1:-smoke}"
NRUNS="${2:-12}"
# decisions/028 Amendment C: the originally pre-registered 3.2-24B ships no HF
# chat_template, so the collector cannot build a turn for it. 2501 is the
# text-only predecessor -- same family, same size class, same 40 x 5120 depth
# and width -- and ships its own template, so nothing about the prompt format
# is chosen by us.
MODEL="${MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
SLUG=$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')

export PATH=/opt/dlami/nvme/wta-venv/bin:$PATH
export HF_HOME=/opt/dlami/nvme/hf
export TMPDIR=/opt/dlami/nvme/tmp
PY=/opt/dlami/nvme/wta-venv/bin/python

if [ "$MODE" = "smoke" ]; then
  OUT="data/xfam_${SLUG}_smoke"; NTASKS=3; NRUNS=2
else
  OUT="data/xfam_${SLUG}"; NTASKS=60
fi
LOGDIR="/opt/dlami/nvme/logs/xfam_${SLUG}_${MODE}"
mkdir -p "$LOGDIR" "$TMPDIR" /opt/dlami/nvme/wta-scratch

echo "=== T7 $MODE launch $(date -Is) ==="
echo "model : $MODEL"
echo "out   : $OUT   (tasks=$NTASKS seeds=$NRUNS)"

hf buckets --help >/dev/null 2>&1 || {
  echo "FATAL: 'hf buckets' unavailable (need huggingface-hub 1.x on PATH)"; exit 1; }

SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$SHARDS" -ge 1 ] || { echo "FATAL: no GPUs detected"; exit 1; }
echo "detected $SHARDS GPU(s) -> --num-shards $SHARDS"

for i in $(seq 0 $((SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" scripts/collect_v2.py \
    --model-id "$MODEL" \
    --classes data/interpretation_classes.json \
    --n-tasks "$NTASKS" --n-runs "$NRUNS" \
    --max-steps 50 \
    --shard "$i" --num-shards "$SHARDS" \
    --out "$OUT" \
    --scratch-dir /opt/dlami/nvme/wta-scratch \
    > "$LOGDIR/shard$i.log" 2>&1 &
  echo "shard $i -> pid $! (gpu $i, log $LOGDIR/shard$i.log)"
done

wait
echo "=== T7 $MODE all shards exited $(date -Is) ==="
echo "next: python scripts/xfam_smoke_gate.py --a0 $OUT"
