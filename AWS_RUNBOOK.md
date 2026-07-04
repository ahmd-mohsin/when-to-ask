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

## 2. A0 sample collection (small first)

```bash
python scripts/collect_a0.py --n-tasks 3          # sample: ~10 min, sanity
python scripts/collect_a0.py --n-tasks 20         # first real batch
```

Writes `data/a0/<task>/<run>.{npz,json,txt}` + `diversity_report.json`.
Check the diversity line: if < 30% of tasks show >= 2 distinct answers,
escalation rule fires (decisions/008) — raise temperature / enable persona,
tell Claude.

**Bring the 3-task sample back to the laptop.** The composite-label builder
(file + region + sub-goal + error signature → decision identity; registry →
interpretation classes, decisions/005) is deliberately written against real
log content, not guessed — this is the one remaining implementation piece,
and it unblocks step 3.

## 3. Offline training + calibration (laptop or AWS, CPU is fine)

Label the A0 logs, then: build `d` (A1), train A2, calibrate tau + benign
spread (A3). Script lands with the label builder; the underlying functions
are all built and fixture-validated (`scripts/run_pipeline_smoke.py`).

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
