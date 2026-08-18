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

## 1b. ADDENDUM 2026-08-18: gate 5 was compared to the wrong reference

The §1 table reports between/within against the gate's own note, "want ratio
>> 1". **1.0 is not this statistic's no-signal value.** For n points per class
it is

    E[ratio | random labels] = sqrt(2 / (n - 1))

because the numerator is a distance between CENTROIDS (shrinking as 1/sqrt(n))
while the denominator is a distance between POINTS (which does not shrink).
Verified to three decimals against the real `gate5_lean_separation` code by
Monte Carlo, dimension-independent (d = 64/512/4096) and class-count
independent (`scripts/gate5_noise_floor.py`):

    n/class     2      3      4      5      8     12     20
    closed  1.414  1.000  0.816  0.707  0.535  0.426  0.324
    MC      1.418  1.002  0.819  0.706  0.535  0.427  0.324

`a4_gates.py:225` admits a decision at `m.sum() >= 4` with >= 2 classes — as
few as **2 reads per class, where pure noise scores 1.414 and clears the
stated bar**. The reported mean pools decisions of different n, so it is not
comparable to any single constant.

**The honest null is a RUN-LEVEL permutation**, not that closed form: reads
are not independent (dozens come from one generation), so the independent unit
is the run. `scripts/gate5_permutation_test.py` holds activations fixed,
shuffles which RUN committed to which class, and recomputes. On the frozen
1385-run `models/v3_32b/labels.npz`, raw layer-sliced activations projected to
128 dims (JL; observed and null go through the same projection):

    gate5-eligible decisions            35
    observed ratio            mean   0.392
    RUN-level null            mean   0.396     <- the honest reference
    read-level null           mean   0.258     <- too permissive
    observed / run-null              0.991x
    decisions with p_run < 0.05       0 / 35
    median runs per decision            6

Note the run-level null is HIGHER than the read-level null: shuffling within
runs destroys the within-run clumping that inflates centroid separation, so
read-level permutation (and the iid closed form) both understate the null and
flatter the result. Any permutation test here must be run-level.

**What this changes.** The negative gets stronger and much more defensible.
"0.894 < 1.0, so it fails" was never the right test; "the statistic is
indistinguishable from a run-level label permutation, on every one of 35
decisions" is. It is also the standard device for this claim (Hewitt & Liang
control tasks), and it is independent of every open labelling question.

**What this does NOT yet establish.** The numbers above are on RAW activations
(decisions/015's decisive diagnostic), not on A2's learned L space where §1's
0.894 lives. Running the same run-level permutation inside `run_full_gates`
on `l_he` is the missing step, and it is the one that would let §1's headline
be restated in permutation terms. Until then, do not write "0.894 is at its
noise floor" — that is not what was measured.

Consistency check, SUPERSEDED same day -- see §2b: gate 2 (which decision is
in play) looked strongly positive at 0.410 vs 0.0101 chance, suggesting "topic
is encoded, lean is not". A lexical control run after this section was written
shows a causal bag-of-words baseline reaching 0.704 on the same task, so gate 2
does not establish that topic is encoded *in the activations* either. Read §1b
and §2b together: neither gate survives a correct baseline.

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

## 2b. ADDENDUM 2026-08-18: gate 2 does not survive a lexical control

Gate 2 (decision recovery from the A2 topic vector T) is the strongest number
in this file: 0.4102 vs 0.0101 chance, ~40x. But activations are COMPUTED FROM
the surrounding text, and different decisions live in different files with
different identifiers -- so "40x chance" needs a baseline that is not chance.

`scripts/gate2_text_control.py` fits the same task (which of 109 held-out
decisions does this read belong to) on the same reads and the same s6,s7 split,
from the raw trace text instead. 40,000 train / 9,002 test reads:

    features                                        accuracy
    chance                                            0.0092
    activations, raw h -> 256d  (JL)                  0.2516
    activations, raw h -> 1024d (JL)                  0.3604
    gate 2 as reported (A2 topic space T)             0.4102
    activations, raw h FULL 5120d (no projection)     0.5037
    text, TF-IDF, causal prefix-only 800 chars        0.7040
    text, TF-IDF, +-400 chars, 961 anchors MASKED     0.7103
    text, TF-IDF, +-400 chars                         0.7365

**A bag-of-words classifier beats the gate.** Text at 0.704-0.737 vs gate 2's
0.4102 (1.7x) and vs the best activation probe at 0.5037 (1.4x). Whatever
gate 2 measures, plain lexical content of the transcript carries more of it.

Three objections, each tested and each closed:

- *"The control is circular -- the decision label IS argmax of anchor hits in
  that window (labeling.py:315), so TF-IDF is just relearning the labeler."*
  Masking all 961 anchor strings removes 7.8% of characters and costs 2.6
  points (0.7365 -> 0.7103). The signal is broad topical vocabulary, not the
  labeler's keywords.
- *"Text has lookahead the model never had."* The +-window includes 400 chars
  written AFTER the read. Restricting to a strictly causal 800-char prefix --
  only what existed when the activation was produced -- costs 3 points
  (0.7365 -> 0.7040). It still beats every activation probe. **This is the
  result that matters for a lead-time claim: activations do not identify the
  live decision earlier than the already-written text does.**
- *"The random projection crippled the activations."* Partly true and worth
  reporting: 256d 0.2516 -> 1024d 0.3604 -> full 5120d 0.5037. At full
  dimension the gap narrows from 2.0x to 1.4x, but does not close.

Two incidental findings worth carrying: **A2's learned topic space T (0.4102)
is WORSE than raw full-dimension activations (0.5037)**, so the disentangling
step is losing topic information rather than isolating it; and the activation
probe here is linear (multinomial logistic) -- a nonlinear probe was not tried
and is the one remaining way the gap could close.

**Status of the three headline gates after this session:**

    gate 5  at its run-level permutation null (0/35 decisions p<0.05)  -- 1b
    A1      held-out AUROC 0.599 (was 0.765 at 14B)  -- run_full_gates_single.log:69
    gate 2  beaten 1.4x by a causal bag-of-words baseline              -- here

The internal-state claim is not supported by any of the three. That is the
finding; it is now controlled rather than asserted.

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
