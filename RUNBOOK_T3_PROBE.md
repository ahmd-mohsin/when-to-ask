# Runbook: T3 probe-robustness appendix (GPU box, CPU-only work)

Owner-facing, copy-paste order. Context: decisions/028 T3 — the last gate-2
escape. Full-dim (5120) linear probe + a 2-layer MLP (hidden 512, the one
pre-declared nonlinear family) on the FIXED labels, gate-2 task, same s6,s7
split as the text control. Reference to beat: causal + anchors-masked TEXT
baseline **0.730**. Whatever the numbers, they go in the paper as-run.

No GPU needed — it is sklearn on streamed activations. It needs the box only
because `models/v3_32b_fixed/labels.npz` (~6.5 GB h) lives there. Budget:
~2-4h wall, peak RAM ~4-6 GB (h is streamed row-selected, never fully loaded).

## 1. Box prep

```bash
git pull   # box tracks main; 028 execution commits are pushed to both branches
python -m pytest -q   # expect 236 green before running anything
```

## 2. Run (one command)

```bash
python scripts/gate2_probe_robustness.py --labels models/v3_32b_fixed/labels.npz --out results/gate2_probe_robustness.json
```

The script prints three accuracies as it goes:

- `acc_256d_linear` — consistency check; should land ≈ **0.2745** (026 B's
  recorded number on the fixed labels). If it is far off, STOP and paste the
  log — the split or labels differ and the run must not be interpreted.
- `acc_fulldim_linear` — the "projection crippled it" escape (old labels
  scored 0.5037; fixed-label value unknown until this run).
- `acc_fulldim_mlp` — the "linear probes are too weak" escape. Config frozen
  in the script (hidden 512, relu, adam, early stopping, random_state 0).

## 3. Paste back

Paste `results/gate2_probe_robustness.json` (or commit it from the box —
fresh file, overwrites nothing). Interpretation is pre-committed in 028 T3:
if the MLP closes the gap to 0.730 the internals story gets a caveat; if not,
the negative is escape-proof. Either way it ships in the appendix.
