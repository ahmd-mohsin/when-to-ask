# AWS runbook — what to run when the GPU instance is ready

Order matters; each step gates the next (decisions/011). Paste each step's
output back to Claude before moving on — several steps are stop-and-review
points by design.

## 0. Setup (once)

```bash
# instance: 1 GPU with >= 24 GB (g5.xlarge / g6e.xlarge class), Ubuntu DLAMI
git clone <this repo> && cd <repo>
pip install -e .[gpu,dev]
python -m pytest -q          # 90 tests, all CPU — should pass as on the laptop
```

## 1. Hook proof (Phase-0 go/no-go, ~5 min)

```bash
python scripts/prove_hook.py
```

Downloads Qwen2.5-Coder-7B-Instruct (~15 GB), runs one under-specified task
N=2, prints PASS/FAIL per spec-A0 check. **If FAIL, stop and report.**

## 2. A0 collection v1.6 (grounded ADR 012 + multi-layer ADR 014) —
##    the full-batch procedure; this is the LAST GPU step

```bash
# (a) extract pre-patch source context from the task docker images (leak
#     analysis in the script header; falls back gracefully, recorded in the
#     manifest). If data/task_context/ is empty (ephemeral NVMe wiped on a
#     stop/start), re-run this; --scratch-dir keeps multi-GB archives off root.
python scripts/extract_task_context.py --n-tasks 20 --scratch-dir /opt/dlami/nvme/wta-scratch

# (b) collect capturing 4 LAYERS in one forward pass (--layers, ADR 014) so
#     the layer sweep is laptop-only forever after. Same seeds/prompts as
#     before -> directly comparable; only the saved activations change.
python scripts/collect_a0.py --n-tasks 20 --layers 0.4,0.5,0.6,0.7
```

Writes `data/a0/<task>/<run>.{npz,json,txt}` (npz `h` now **(R, 4, 3584)**) +
per-task `prompt.txt` + `collection_manifest.json` (versions, GPU, grounding
mode, `reader.layer_indices`, per-run timing/finish-reason) + `events.jsonl`.
Verify: npz shape (R, 4, 3584) float16, finite, nonzero variance; 20/20
`mode=docker`; distinct-signature count (< 30% fires decisions/008 — tell
Claude). Then tarball `data/a0/` + `data/task_context/`, print sha256, report.
The box can be stopped after — all sweeps/gates run on the laptop.

## 2b. v2 collection — REAL agent trajectories (decisions/017) — CURRENT step

```bash
git pull && python -m pytest -q          # expect 108 green
python scripts/collect_v2.py --n-tasks 20 --scratch-dir /opt/dlami/nvme/wta-scratch
```

Runs 8 seeded agent trajectories per task inside each task's docker container
(4-layer capture + cadence/cue/value reads baked in). Budget ~4-8 GPU-hours
(15 turns × ~1k tokens × 160 runs). Watch the manifest per run: `steps`,
`finished` (reached TASK_DONE), `reads_by_trigger` (expect nonzero `value`),
`actions`. Broken images are skipped and logged, not fatal. Tarball
`data/a0_v2/` back to the laptop when done. For the 32B pass afterwards:
same command + `--model-id Qwen/Qwen2.5-Coder-32B-Instruct` on a g5.12xlarge.

## 2c. SCALE collection — 60 train tasks at Qwen3-32B (decisions/018+019) —
##     CURRENT step. Instance: **g7e.2xlarge** (1x RTX PRO 6000 Blackwell,
##     96 GB VRAM, ~$3.36/hr on-demand us-east-1; verified 2026-07-19).
##     Single-GPU fit for 32B bf16 (~65 GB + KV), ~3x faster than the sharded
##     4xA10G alternative -> est. $60-90 total. Fallback if no capacity:
##     g5.12xlarge (4x A10G, spot). NOT g6e.xlarge (48 GB < 65 GB).
##     At boot on g7e: (1) use a CURRENT DLAMI (Blackwell needs cu128+;
##     torch cu130 as on the 14B box is fine); (2) `df -h` to find the local
##     NVMe mount — it may not be /opt/dlami/nvme on this family; point
##     --scratch-dir at it.

```bash
git pull && python -m pytest -q          # expect 118 green

# (a) SMOKE FIRST (~1-2 h, <$10) — Qwen3-32B has never run in this harness.
#     3 tasks x 2 runs; verifies: no <think> blocks in segments (thinking
#     pinned OFF by default, decisions/019), one-bash-block protocol
#     compliance, memory fits, and gives a measured per-run time for the
#     full-run cost estimate. PASTE THE MANIFEST + one .txt back to Claude
#     BEFORE launching the full run.
python scripts/collect_v2.py \
    --model-id Qwen/Qwen3-32B \
    --n-tasks 3 --n-runs 2 \
    --classes data/interpretation_classes.json \
    --out data/a0_v2_32b_smoke \
    --scratch-dir /opt/dlami/nvme/wta-scratch

# (b) FULL RUN (only after smoke review) — 60 train tasks x 8 seeds.
python scripts/collect_v2.py \
    --model-id Qwen/Qwen3-32B \
    --n-tasks 60 \
    --classes data/interpretation_classes.json \
    --out data/a0_v2_32b \
    --scratch-dir /opt/dlami/nvme/wta-scratch
```

Notes that matter:
- `--classes` is REQUIRED: restricts collection to the 60 artifact (train)
  tasks. Without it, sorted-dir order reaches swe_60+ — the SEALED TEST
  POOL — before swe_7/8/9. Never collect swe_60+.
- Thinking mode is OFF by default and recorded in the manifest (do NOT pass
  --enable-thinking). If smoke segments still contain <think> text, STOP and
  report — that's a template regression, not something to work around.
- Rough full-run budget: 480 runs; at 14B a run averaged ~2.5 min, sharded
  32B is ~2-4x slower → expect ~40-80 box-hours (~$100-200 spot). The smoke
  run replaces this guess with a measured number — trust the measurement.
- Broken images are skipped and logged (they still count toward --n-tasks;
  report "SKIPPED" manifest entries).
- Watch per-run: steps, finished, reads_by_trigger (nonzero value), actions.
- When done: tarball data/a0_v2_32b/ back to the laptop, print sha256.
  Laptop then reruns audit_labels + sweep + run_full_gates --kfold 5
  (step 3) against the 32B data.

## 3. Offline training + sweeps + gates (laptop, CPU — no AWS)

```bash
python scripts/audit_labels.py                       # human-readable label audit
python scripts/sweep.py --layers 0,1,2,3             # rank layers (gate1+gate5) + eps/window
python scripts/run_full_gates.py --layer <best> --eps-settle <best> --window <best> --kfold 5
```

`sweep.py` prints the layer table (best by gate-5 lean-separation) and the
eps×window table (best A3 settle rate), then the exact `run_full_gates.py`
command for the trustworthy k-fold gate numbers. That gate run is the owner
STOP point (decisions/011/013/014).

train_offline prints: label coverage + class balance, A1 held-out AUROC, the
GRL-treadmill check, A3 settle rate + benign-spread reference — each number's
audit trail lands in models/ (labels_debug.jsonl, a2_history.jsonl). Read the
audit file; if labels look wrong, the fix is the class artifact / lexicons,
never the downstream.

## 4. A4 gates — STOP POINT

```bash
python scripts/run_gates.py --data data/a4_heldout.npz --a2 models/a2.pt
```

Prints the seven gate numbers unfiltered. **This is the science gate: the
owner reviews before ANY Part B result is trusted (decisions/011). Red gates
are findings, not bugs — nothing gets tuned to pass.**

## 5. Part B eval (only after sign-off)

Online trigger + Ask-F1 + matched-N baselines on HiL-Bench per `specs/eval.md`.

## Cost notes (modest budget, decisions/004)

- Steps 1-2 are the only GPU-bound steps until eval; a g5.xlarge spot
  instance covers them. A0 at 20 tasks × N=8 × ~768 tokens ≈ a few GPU-hours.
- Everything else (training A2 on logged activations, calibration, gates) is
  CPU and can run on the laptop overnight at real-data scale.
