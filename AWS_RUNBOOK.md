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

## 2. A0 collection v1.5 (grounded — ADR 012) — DONE for the 3-task sample;
##    this is the full-batch procedure

```bash
# (a) extract pre-patch source context from the task docker images
#     (leak analysis in the script header; falls back gracefully, always
#     recorded in the manifest)
python scripts/extract_task_context.py --n-tasks 20

# (b) collect: grounded prompts, 1536-token generations, full diagnostics
python scripts/collect_a0.py --n-tasks 20
```

Writes `data/a0/<task>/<run>.{npz,json,txt}` + per-task `prompt.txt` +
`collection_manifest.json` (versions, GPU, grounding mode per task, per-run
timing/finish-reason) + `events.jsonl`. Check the closing summary: how many
tasks were actually grounded (`mode=docker`), and the distinct-signature
count (< 30% diverse fires decisions/008's escalation — tell Claude).

Bring `data/a0/` (tar.gz) back to the laptop.

## 3. Offline training + calibration + label audit (laptop, CPU)

```bash
python scripts/train_offline.py          # labels -> d -> A2 -> A3 (+ diagnostics)
python scripts/audit_labels.py           # human-readable audit of label decisions
```

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
