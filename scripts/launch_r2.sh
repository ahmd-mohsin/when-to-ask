#!/usr/bin/env bash
# R2 main collection - 60 tasks x 24 seeds, task-sharded one process per GPU.
#
# Preflight baked in (see AWS_RUNBOOK 2d): `hf` on PATH (else every task whose
# image is not already cached is SILENTLY skipped), image store + TMPDIR on the
# NVMe. Shard count is detected from the box, never hardcoded - R2's 60 tasks
# shard cleanly across any card count.
#
# max_steps=50 (not the 40 default): at 40, 37% of runs hit the cap and were
# truncated, which would hide the later-trajectory drift this dataset exists to
# measure. Trajectories collected under a DIFFERENT cap are not comparable --
# if you change it, start a fresh --out dir rather than resuming into an old one.
#
# Interrupt-safe: existing <run>.json files are skipped, so re-running resumes,
# including across a change in card count.
set -u
cd /home/ubuntu/when-to-ask

export PATH=/opt/dlami/nvme/wta-venv/bin:$PATH
export HF_HOME=/opt/dlami/nvme/hf
export TMPDIR=/opt/dlami/nvme/tmp
PY=/opt/dlami/nvme/wta-venv/bin/python
LOGDIR=/opt/dlami/nvme/logs/r2
mkdir -p "$LOGDIR" "$TMPDIR" /opt/dlami/nvme/wta-scratch

echo "=== R2 launch $(date -Is) ==="

# hf is how every task image is fetched; missing hf degrades silently, so fail loud.
hf buckets --help >/dev/null 2>&1 || {
  echo "FATAL: 'hf buckets' unavailable (need huggingface-hub 1.x on PATH)"; exit 1; }
echo "hf: $(command -v hf)"

SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$SHARDS" -ge 1 ] || { echo "FATAL: no GPUs detected"; exit 1; }
echo "detected $SHARDS GPU(s) -> --num-shards $SHARDS"

for i in $(seq 0 $((SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" scripts/collect_v2.py \
    --model-id Qwen/Qwen3-32B \
    --classes data/interpretation_classes.json \
    --n-tasks 60 --n-runs 24 \
    --max-steps 50 \
    --shard "$i" --num-shards "$SHARDS" \
    --out data/a0_v3_32b \
    --scratch-dir /opt/dlami/nvme/wta-scratch \
    > "$LOGDIR/shard$i.log" 2>&1 &
  echo "shard $i -> pid $! (gpu $i, log $LOGDIR/shard$i.log)"
done

wait
echo "=== R2 all shards exited $(date -Is) ==="
