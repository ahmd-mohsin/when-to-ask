# R2 gate run — results, data handoff, and what's next

Run 2026-08-14 → 08-16 on the Blackwell box. This is the decisions/011 STOP
point: the A4 gate numbers are below, unmodified. Nothing was tuned to make a
gate pass; red gates are reported as findings.

## 1. Headline: gate 5 fails at every layer (a powered negative)

k-fold 5, frozen 1385-run snapshot, eps 0.6 / window 3:

    L  frac | g5_ratio  +-sd   g5_sil | g1_acc chance | g7_medK  fpos | g5dec
    0  0.20 |   0.832  0.461   0.094  |  0.735  0.482 |   19.9   0.95 |  32
    1  0.30 |   0.890  0.366   0.094  |  0.768  0.482 |   20.8   0.95 |  32
    2  0.41 |   0.840  0.294   0.089  |  0.782  0.482 |   20.4   0.95 |  32
    3  0.50 |   0.894  0.279   0.091  |  0.778  0.482 |   20.0   0.95 |  32   <- PRE-REGISTERED
    4  0.59 |   0.869  0.362   0.088  |  0.747  0.482 |   20.2   0.95 |  32
    5  0.70 |   0.764  0.313   0.042  |  0.729  0.482 |   20.8   0.95 |  32
    6  0.80 |   0.746  0.322   0.063  |  0.756  0.482 |   20.7   0.95 |  32
    7  0.84 |   0.625  0.192   0.051  |  0.747  0.482 |   19.6   0.95 |  32

The between/within ratio never reaches 1.0 across the full depth span, and the
pre-registered mid-layer is the BEST of the eight — so per decisions/015's
precedent, "wrong layer" is ruled out. Corroborated three ways:

- CPU vs GPU at layer 3 differ by 0.029 on g5_ratio — an order of magnitude
  below the fold-to-fold sd (0.44). The conclusion does not depend on device.
- The completed 1415-run collection tightens layer 3 to **0.805 +- 0.148 on 38
  decisions** (vs 0.894 +- 0.279 on 32). More data, half the variance, same
  verdict.
- 7B (decisions/015) was 0.726 +- 0.303. Scaling 7B -> 32B did not rescue it.

Pre-registered CPU layer-3 report, verbatim:

    g5_ratio   [0.6491, 1.7621, 0.5357, 0.9848, 0.6842]   mean 0.9232
    g5_sil     [-0.0452, 0.2580, 0.0200, 0.1269, 0.0624]  mean 0.0844
    g1_acc     [0.8442, 0.8750, 0.6416, 0.8099, 0.7086]   mean 0.7759
    g1_chance  [0.5, 0.5, 0.4120, 0.5, 0.5]               mean 0.4824
    g7_medK    [22, 26, 14, 26, 17]                       mean 21.0
    g7_fracpos [1.0, 1.0, 0.75, 1.0, 1.0]                 mean 0.95
    folds 5

## 2. The single-split run is uninformative — and why

Only 1 of 7 gates produced a number on the `--held-seeds s6,s7` split:

    gate1_topic_leakage:     n_decisions=0        INSUFFICIENT
    gate2_decision_recovery: 0.41018507318253294  (chance 0.010101010101010102)
    gate3_fork_collocation:  n_same=0, n_diff=3967  INSUFFICIENT, no theta
    gate5_lean_separation:   n_decisions=0        INSUFFICIENT
    gate6_ood_transfer:      SKIPPED (no theta from gate 3)
    gate7_lead_time:         INSUFFICIENT forked decisions

**The holdout contains zero forked decisions.** It is 2 of 24 seeds — 79 runs
over 50 tasks, ~1.6 runs per task — and a fork needs two runs of the SAME task
committing to DIFFERENT classes. Most tasks contribute one run, and one run
cannot disagree with itself. The `s6,s7` default was sized for an 8-seed design
(s0-s5 train / s6-s7 eval = 25% holdout); R2 collected 24 seeds, so the holdout
silently shrank to 8.3%. decisions/021 §7 pre-registered k-fold for exactly
this reason.

Note gate 6 (cross-task transfer, 4 OOD tasks) is downstream of gate 3's theta,
so a seed-split failure silently removed the cross-task evidence too. Worth
addressing: the strongest generalisation gate depends on the weakest split.

Gate 2 is strongly positive: **0.410 vs 0.0101 chance (~40x)**. T recovers WHICH
decision is in play; nothing yet says it encodes WHICH WAY it resolved.

## 3. Diagnostics (raw activations, no A2)

decisions/015's decisive diagnostic, micro-averaged as in 015, layer 3:

    structural  +0.097 over chance   (7B reference: +0.015)
    value       +0.126 over chance

Near/far (decisions/016 method) refutes literal-copying for the value arm:

                  ALL reads      NEAR (<=12 tok)   FAR (>=24 tok)
    value      +0.126            -0.233            +0.135
    structural +0.097            -0.110            +0.101

Separability lives >=24 tokens from any signature mention, not at the emission
moment — it is durably represented, not a copied token in context.

**Caveat on averaging:** macro (mean over decisions) and micro (pooled over
reads) disagree sharply here — macro puts structural at +0.004, micro at +0.097.
decisions/015's reference number is micro; use micro for comparison.

## 4. Collection state

    1415 / 1440 runs   (was 1385; 30 recovered)

The 55 originally-missing runs were 100% temp-1.3 (temperature = seed % 4 ->
0.7/0.9/1.1/1.3), i.e. one temperature arm, biased toward the longest
trajectories. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovered 30:

    temp 0.7   360/360      temp 1.1   360/360
    temp 0.9   360/360      temp 1.3   335/360  (was 305/360)

The remaining 25 failed 2-8 times each on clear cards — genuine capacity, not
fragmentation (waste fell 4.41 GiB -> 0.42 GiB). Every remaining lever (batch,
context, captured layers, max-steps) would change the measurement, so they are
left missing. The temp-1.3 arm is still ~7% short, biased toward long runs.

## 5. Data handoff

Bundles on the box (persistent `/ssd2`):

| file | size | contents |
|---|---|---|
| `wta_r2_handoff.tar.gz` | 6.2 GB | the 30 recovered runs + all results |
| `wta_v3_models_slim.tar.gz` | 22 MB | results only (no regenerable `labels.npz`) |

Each run ships 4 files; `h` is `(reads, 8, 5120) float16`, so **all 8 captured
layers travel with the data** and any layer can be re-derived locally.

Merge the recovered runs into an existing 1385-run copy:

    cp -rn recovered_runs/swe_* <your>/a0_v3_32b/
    cp recovered_runs/events.s*.jsonl recovered_runs/collection_manifest.s*.json <your>/a0_v3_32b/
    ls <your>/a0_v3_32b/*/*.npz | wc -l    # expect 1415

**Trained models:** only layer 3 single-split (`a2.pt`, `a1_direction.npy`,
`a3_calibration.npz`). `kfold_gates()` calls `fit_a2()` with no output path, so
the k-fold path trains 5 models per layer and saves none. Per-layer trained
models need a single-split re-run per layer (~4 h each).

## 6. What's next: labelling, not collection

**All 214 decisions already have >=4 runs.** The loss is at labelling:

    commitment (run,decision) pairs: 4931
      labeled  : 1574 (31.9%)
      UNLABELED: 3357 (68.1%)   -- 3272 of them "no signature hits"

Where labelling succeeds, 43% of decisions turn out to be forks. Labelling the
rest plausibly takes 62 forked decisions past 100 — the only lever that answers
the power criticism (32 pooled decisions, 3-9 per fold, 4% class coverage).
More seeds would not help; more tasks would burn the sealed pool.

See `decisions/025-judge-labelling-required.md` — it pre-registers the
labelling-teacher change BEFORE any new number exists, requires
`scripts/validate_judge.py` against the frozen `judge_validation_pairs.json`
first, and pre-commits that judge-labelled gates are a separate arm that never
replaces §1.

## 7. Reproducing on a fresh box

`/opt/dlami/nvme` is AWS instance store — **wiped on every VM stop**. Gone with
it: the venv, the ~62 GB HF cache, and Docker's `data-root` + containerd `root`.
That last one is the non-obvious failure: both daemons come up "active" against
an empty volume and every task container dies with
`apply layer error ... metadata.db: no such file or directory`, which reads like
an image problem and is not. Fix: stop `docker docker.socket containerd`,
`mkdir -p` both roots, start containerd **then** docker, verify with
`docker run hello-world`. Do this before any collection run.

Order that works: verify backup checksum -> extract -> verify extraction against
the tar listing -> repoint symlinks -> rebuild venv -> fix docker/containerd ->
launch GPU work.

Two more traps: `data/a0_v3_32b` is a symlink (dangles after a wipe), and
`labels.npz` stores the full `h` matrix (~6.5 GB), so `models/` needs to live on
a volume with room — not the 29 GB root disk.

## 8. Sealed pool

Untouched. decisions/019 gates the sealed test on gate 5 passing; it did not
pass, so no sealed-test execution is authorised.
