# 014 — Multi-layer capture, parallel sweeps, k-fold gates

---
status: agreed (owner 2026-07-09: "do all the fixes")
date: 2026-07-09
---

**Context.** The first gate run (decisions/013) left gate 5 (lean separation)
unresolved, with two confounds: we had only captured mid-layer 14 (so a layer
sweep needed a fresh AWS run), and gates 1/5/7 were measured on 2 held-out
seeds → only 2–3 decisions → too noisy to trust.

**Decisions.**

1. **Multi-layer capture — capture-all on disk, select-one-at-load.**
   `hf_reader` captures a configurable set of layers in the SAME forward pass
   (default {0.4,0.5,0.6,0.7}×depth). On disk the read matrix becomes
   `(R, L, H)`; `RunLog.layers` records the resolved indices;
   `load_run_log(..., layer=)` slices to a single `(R, H)` so EVERY downstream
   stage (labeling, A1, A2, A3, gates) is unchanged — it still receives 1-D
   per-read `h`. `build_labels(layer=)` forwards the selection. This makes the
   layer sweep a pure-CPU laptop operation with **zero AWS re-runs** after one
   final multi-layer collection. Legacy single-layer `(R, H)` data still loads
   (the `layer` arg is ignored). ~4× storage (~180 MB total) — trivial.

2. **eps/window sweep is parallel, not serial.** These act on the
   already-trained lean vectors, so `run_full_gates.py` / `sweep.py` loop
   eps×window through A3 + gate 7 on one trained A2 — no retrain. Fixes the
   7%-settle-rate finding (dry-run: 7% → 20% at eps=0.8, window=3).

3. **k-fold gates (`--kfold N`) — the power fix.** Cross-fit over non-OOD
   `(task, seed)` groups (never split a run across folds; `kfold_group_indices`
   in `a4_gates.py`), retrain A2 per fold, measure gates 1/5/7 on the held
   fold, POOL. Raises measured decisions from 2–3 to 13–16 and reports
   mean ± std. OOD tasks stay separate for gate 6. The layer sweep uses fast
   single-split to RANK layers; k-fold is run once on the winning layer for the
   trustworthy number.

4. **"More class coverage" is DEFERRED** — one variable at a time. Run the
   layer sweep first; only if gate 5 stays weak across all layers do we expand
   signatures/data. (What it would buy: 25% class-label coverage starves the
   lean head; more labeled reads/class give gate 5 more to separate.)

**Consequences.** RunLog schema is back-compatible (existing v1.5 data still
loads). `run_full_gates.py` is refactored into importable pipeline functions
(`prepare_split`, `fit_a1/a2/a3`, `gates_1to6`, `compute_gate7`, `kfold_gates`)
that `sweep.py` reuses — one implementation. Workflow now: one final AWS
multi-layer collection, then `sweep.py` (layer + eps) and
`run_full_gates.py --kfold 5 --layer <best>` all on the laptop → owner review.
