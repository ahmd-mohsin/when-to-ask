#!/usr/bin/env bash
# 028 Amendment H -- the full_info arm.
#
# Same 3 python tasks, same Qwen3-32B, same protocol as R2. The ONLY change is
# the instruction file: full_info/instruction.md resolves every blocker in
# prose, where baseline/instruction.md leaves them ambiguous. Everything else
# (max_steps=50, the 0.7/0.9/1.1/1.3 ladder cycled by seed, the nudge,
# exec_timeout, the top_k clamp) is untouched, so the arms are comparable.
#
# Tasks are restricted to swe_0,swe_10,swe_11 -- the three the sweap_json
# parser can score. swe_1 and swe_12 are JS and the parser mis-reads jest
# output (see results/test_outcome_vector.json), so collecting them would burn
# GPU on runs nothing can score.
#
# Preflight baked in, both learned the hard way:
#   - `hf` on PATH, else every task whose image is not cached is SILENTLY
#     skipped and the run "succeeds" on nothing.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- R2 lost 55 runs to
#     OOM in the temp-1.3 arm without it.
#
# Interrupt-safe: existing <run>.json files are skipped, so re-running resumes.
# Writes to its OWN --out dir; it must never be merged into data/a0_v3_32b,
# which is the baseline arm and the universe every published number uses.
set -u
cd /home/ubuntu/when-to-ask

export PATH=/opt/dlami/nvme/wta-venv/bin:$PATH
export HF_HOME=/opt/dlami/nvme/hf
export TMPDIR=/opt/dlami/nvme/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/dlami/nvme/wta-venv/bin/python
OUT=/ssd3/wta_data/a0_full_info_32b        # EBS, not the ephemeral NVMe
LOGDIR=/opt/dlami/nvme/logs/full_info
mkdir -p "$LOGDIR" "$TMPDIR" /opt/dlami/nvme/wta-scratch "$OUT"

echo "=== full_info arm launch $(date -Is) ==="

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
    --mode full_info \
    --only-tasks swe_0,swe_10,swe_11 \
    --n-tasks 5 --n-runs 24 \
    --max-steps 50 \
    --shard "$i" --num-shards "$SHARDS" \
    --out "$OUT" \
    --scratch-dir /opt/dlami/nvme/wta-scratch \
    > "$LOGDIR/shard$i.log" 2>&1 &
  echo "shard $i -> pid $! (gpu $i, log $LOGDIR/shard$i.log)"
done

wait
echo "=== full_info arm all shards exited $(date -Is) ==="
