#!/usr/bin/env bash
# R2 watchdog: adopt the running shards, restart any that die, stop cleanly when
# the collection is finished.
#
# Safe to start while R2 is already running - it never touches a live shard, it
# only fills gaps. Restarting is cheap because collect_v2 resumes: existing
# <run>.json files are skipped, so a restart costs at most the in-flight run.
#
#   tmux new-session -d -s r2wd 'bash scripts/watchdog_r2.sh'
#   touch /opt/dlami/nvme/logs/r2/STOP    # ask the watchdog to stand down
set -u
cd /home/ubuntu/when-to-ask

export PATH=/opt/dlami/nvme/wta-venv/bin:$PATH
export HF_HOME=/opt/dlami/nvme/hf
export TMPDIR=/opt/dlami/nvme/tmp
PY=/opt/dlami/nvme/wta-venv/bin/python
LOGDIR=/opt/dlami/nvme/logs/r2
WDLOG="$LOGDIR/watchdog.log"
STOP="$LOGDIR/STOP"
SHARDS=4
INTERVAL=60          # seconds between checks
FAST_EXIT=180        # a restart that dies sooner than this did no real work
MAX_RESTARTS=20      # per shard, then give up and keep reporting

mkdir -p "$LOGDIR"
log() { echo "$(date -Is) $*" | tee -a "$WDLOG"; }

# Anchored to the interpreter path: an unanchored pattern also matches any shell
# whose command line merely CONTAINS the pattern text (including this script's
# own callers), which would make a dead shard look alive and defeat the watchdog.
alive() { pgrep -f -- "^$PY scripts/collect_v2\.py .* --shard $1 --num-shards $SHARDS " >/dev/null; }

start_shard() {
  local i=$1
  CUDA_VISIBLE_DEVICES=$i "$PY" scripts/collect_v2.py \
    --model-id Qwen/Qwen3-32B \
    --classes data/interpretation_classes.json \
    --n-tasks 60 --n-runs 24 \
    --max-steps 50 \
    --shard "$i" --num-shards "$SHARDS" \
    --out data/a0_v3_32b \
    --scratch-dir /opt/dlami/nvme/wta-scratch \
    >> "$LOGDIR/shard$i.log" 2>&1 &
  echo $!
}

runs_done() { cat data/a0_v3_32b/events.s*.jsonl 2>/dev/null | grep -c '"run_done"'; }

declare -a RESTARTS FAST DONE_
for i in $(seq 0 $((SHARDS-1))); do RESTARTS[$i]=0; FAST[$i]=0; DONE_[$i]=0; done

log "watchdog up (interval ${INTERVAL}s, $SHARDS shards); adopting whatever is running"
for i in $(seq 0 $((SHARDS-1))); do
  alive "$i" && log "  shard $i: adopted (already running)" || log "  shard $i: NOT running, will start"
done

while :; do
  if [ -f "$STOP" ]; then log "STOP file present - standing down (shards left alone)"; exit 0; fi

  finished=0
  for i in $(seq 0 $((SHARDS-1))); do
    if [ "${DONE_[$i]}" -eq 1 ]; then finished=$((finished+1)); continue; fi
    if alive "$i"; then continue; fi

    # shard is gone: either it finished its slice, or it crashed
    before=$(runs_done)
    if [ "${RESTARTS[$i]}" -ge "$MAX_RESTARTS" ]; then
      log "shard $i: down, but hit MAX_RESTARTS=$MAX_RESTARTS - not restarting"
      DONE_[$i]=1; continue
    fi
    RESTARTS[$i]=$(( ${RESTARTS[$i]} + 1 ))
    log "shard $i: DOWN -> restart #${RESTARTS[$i]} (runs_done=$before)"
    t0=$(date +%s)
    pid=$(start_shard "$i")
    sleep "$FAST_EXIT"
    if kill -0 "$pid" 2>/dev/null; then
      FAST[$i]=0
      log "shard $i: restart #${RESTARTS[$i]} healthy (pid $pid)"
    else
      dt=$(( $(date +%s) - t0 ))
      FAST[$i]=$(( ${FAST[$i]} + 1 ))
      log "shard $i: restart exited after ${dt}s (fast-exit ${FAST[$i]}/2)"
      if [ "${FAST[$i]}" -ge 2 ]; then
        DONE_[$i]=1
        log "shard $i: two fast exits and no work left -> treating slice as COMPLETE"
      fi
    fi
  done

  if [ "$finished" -eq "$SHARDS" ]; then
    log "all $SHARDS shards complete - watchdog exiting. total runs_done=$(runs_done)"
    exit 0
  fi

  # heartbeat: progress, stalls, memory pressure
  nd=$(runs_done)
  last=$(cat data/a0_v3_32b/events.s*.jsonl 2>/dev/null | grep '"run_done"' \
         | tail -1 | sed 's/.*"ts": "//;s/".*//')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')
  oom=$(grep -ch "memory allocation failed with OOM" "$LOGDIR"/shard*.log 2>/dev/null | paste -sd, -)
  log "heartbeat: runs_done=$nd/1440 last=$last mem=[$mem] oom_warns=[$oom]"

  if [ -n "$last" ]; then
    age=$(( $(date +%s) - $(date -d "$last" +%s) ))
    [ "$age" -gt 10800 ] && log "WARNING: no run_done in $((age/60)) min across all shards"
  fi

  sleep "$INTERVAL"
done
