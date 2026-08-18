# Runbook: 026 label rebuild + gate rerun (GPU box)

Owner-facing, copy-paste order. Context: decisions/026 (coordinate repair) —
the R2 labels were corrupted; this rebuilds them and reruns the gates with the
corrected statistics. Laptop-side verification is complete (026 Amendment A);
nothing below runs on the laptop except step 0.

## 0. Laptop, before the box: archive-protect old results

`gate5_permutation_test.py` defaults its output into `results/`, and
`gate2_text_control.py` writes `results/gate2_text_control*.json`. The R2-era
`results/` must not be clobbered:

```bash
mv results results_r2_frozen
```

(The R2 artifacts are also inside `wta_r2_results_20260811.tar.zst` — verified
sha256 — so this is belt-and-braces.)

## 1. Box prep (only if the VM was stopped since last use)

HANDOFF_R2_GATES.md §7, in order: verify backup checksum → extract → verify
against tar listing → repoint the `data/a0_v3_32b` symlink → rebuild venv →
fix docker/containerd roots (stop `docker docker.socket containerd`, mkdir
both roots, start containerd THEN docker, `docker run hello-world`) → then GPU
work. `models/` must live on a volume with room (labels.npz ~6.5 GB).

```bash
git pull   # box tracks main; the 026 commits are pushed to both branches
python -m pytest -q   # expect 232 green before spending GPU-box time
```

## 2. Rebuild labels (fixed coordinates) — ~2h

```bash
python scripts/generate_labels.py --a0 /ssd/wta_data/a0_v3_32b --out models/v3_32b_fixed --tokenizer auto
```

Sanity gates before continuing (all from the printed coverage/log +
labels.npz meta):

- reads == 787,281 over 1415 runs (the 1385-run frozen build was 683,285)
- `txt_join_mismatch == 0` summed over tasks (any nonzero = sidecar/txt
  disagreement — STOP and inspect that run)
- `segment_clamped == 0`; `token_clamped` ≈ 1-2% diffuse with no task
  dominating (laptop sample: 1.3%, overshoot ≤2 tokens — benign drift)
- record the fork census line (forked blockers) next to the frozen run's 62

`models/v3_32b/` (the corrupted-label run) is FROZEN — never write into it.

## 3. Gates — pre-registered layer 3 first

```bash
python scripts/run_full_gates.py --a0 /ssd/wta_data/a0_v3_32b --classes data/interpretation_classes.json --out models/v3_32b_fixed --labels-npz models/v3_32b_fixed/labels.npz --kfold 5 --layer 3
```

Notes: `--labels-npz` skips the in-script label rebuild (026); the kfold path
now pools gates 1/2/3/5/6/7 (2/3/6 were unmeasured at 32B before) and dumps
per-fold lean embeddings to `models/v3_32b_fixed/lhe_folds/`.

CAVEAT: `--labels-npz` must point at a labels.npz built with the SAME --layer
convention as the kfold you are running. `generate_labels --layer None` stores
the mid layer = slot 3, so step 2's npz serves the layer-3 run ONLY. For the
other 7 layers (step 6) run per layer:

```bash
python scripts/generate_labels.py --a0 /ssd/wta_data/a0_v3_32b --out models/v3_32b_fixed_L0 --tokenizer auto --layer 0
python scripts/run_full_gates.py --a0 /ssd/wta_data/a0_v3_32b --out models/v3_32b_fixed_L0 --labels-npz models/v3_32b_fixed_L0/labels.npz --kfold 5 --layer 0
```

(or omit --labels-npz and let run_full_gates build labels itself per layer.)

## 4. Single-split layer 3 (saves a1/a2/a3 artifacts) + l_he permutation

```bash
python scripts/run_full_gates.py --a0 /ssd/wta_data/a0_v3_32b --out models/v3_32b_fixed --labels-npz models/v3_32b_fixed/labels.npz --layer 3
python scripts/gate5_lhe_permutation.py --folds models/v3_32b_fixed/lhe_folds --out results/gate5_lhe_permutation.json
```

The l_he permutation is HANDOFF §1b's "missing step": the run-level
permutation in the learned L space where the old 0.894 lived. Its headline is
the GLOBAL Stouffer p, not a per-cell count.

## 5. Controls on the fixed labels

```bash
python scripts/gate5_permutation_test.py --labels models/v3_32b_fixed/labels.npz --out results/gate5_permutation_test_fixed.json
python scripts/gate2_text_control.py --labels models/v3_32b_fixed/labels.npz --debug models/v3_32b_fixed/labels_debug.jsonl --a0 /ssd/wta_data/a0_v3_32b
python scripts/gate2_text_control.py --labels models/v3_32b_fixed/labels.npz --debug models/v3_32b_fixed/labels_debug.jsonl --a0 /ssd/wta_data/a0_v3_32b --causal
python scripts/gate2_text_control.py --labels models/v3_32b_fixed/labels.npz --debug models/v3_32b_fixed/labels_debug.jsonl --a0 /ssd/wta_data/a0_v3_32b --causal --mask-anchors
```

The permutation script now reports: per-decision EXACT p where the assignment
space is small, `testable` flags with min-achievable p, and the global
Stouffer test — read the global p as the headline, and "k of m testable"
instead of "k of 35". gate2_text_control now slices the same raw text the
labeler scored (its old numbers were asymmetrically pro-text on CRLF runs).

## 6. Remaining 7 layers (optional until layer 3 is read)

Per-layer as in step 3's caveat block. Hours each; run after the owner has
seen layer 3.

## 7. STOP

decisions/011: gate numbers go to the owner unmodified. Record everything as
ADR 026 Amendment B (numbers + which of §5's corrected statistics produced
them). Reading guidance from Amendment A(3): the repair's effect on gates has
UNKNOWN sign — neither a flip nor a confirmation is the "expected" outcome.
The sealed pool stays untouched either way (decisions/019 gating unchanged).
