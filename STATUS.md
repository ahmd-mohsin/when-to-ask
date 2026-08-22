# STATUS — what every code name means, and what is done vs left

**Living document. Last updated: 2026-08-22.**
Update rule at the bottom: this file is edited in the SAME commit as any
experiment that lands. If it disagrees with reality, reality wins and this
file is stale — fix it.

---

## Read this first (30 seconds)

We are executing [decisions/028](decisions/028-iclr-experimental-program.md),
the frozen experimental program for the ICLR paper
([paper/OUTLINE.md](paper/OUTLINE.md)). The paper is a MEASUREMENT paper:
"where does an agent's need-to-ask signal live?" Answer so far: not in
single-run internals, not in surface behavior, not in generic embeddings.
The mechanism (a working ask-trigger) was **dropped** at a pre-registered
gate in 027 and is future work. Nothing in this cycle claims a mechanism.

---

## The decoder ring — four different numbering schemes

They collide. This is the single biggest source of confusion in the repo.

### 1. `T#` and `F#` — the 028 paper deliverables (**T = Table, F = Figure**)

Six deliverables numbered **1–6 in ONE sequence**; the letter says whether
the output is a table or a figure in the paper. This is why the numbers look
gappy when you read only the T's or only the F's — they interleave.

| ID | Kind | What it is |
|---|---|---|
| **T1** | Table | THE centerpiece: representation × fork-signal matrix (rows R1–R6b below) |
| **F2** | Figure | Rollout-budget ablation — "how many runs buy the signal" |
| **T3** | Table | Probe-robustness appendix — closes the last gate-2 escape |
| **F4** | Figure | Lead-time and fork anatomy — onset, completion, ask window |
| **T5** | Table | Cross-scale replication — 7B / 14B generality rows |
| **T6** | Table | Ground-truth error bounds — how wrong can the labels be |

### 2. `R#` **inside T1** — the ROWS of that table

Each row asks: can THIS representation tell that two runs resolved the same
blocker differently? Scored as same-vs-different separation AUROC on the
frozen commitment-pair pool.

| Row | Representation |
|---|---|
| R1 | Raw activations, 8 layers (32B) |
| R2 | Learned L-space (the A2 autoencoder space) |
| R3 | Surface behavior hashes (hashed char-3grams) |
| R4 | MiniLM embedding of the mutating turn |
| R5 | Strong embedder — BAAI/bge-large-en-v1.5 |
| R6a | Single-run LLM introspection (the Ask-or-Assume-style detector) |
| R6b | Ensemble LLM comparison (judge sees TWO runs, same/different) |

### 3. `R#` as a **COLLECTION RUN** — totally different thing ([decisions/021](decisions/021-real-run-design.md))

GPU data-collection phases. **This is the collision:** collection-R2 is the
dataset; T1-row-R2 is a probe. They share no meaning.

| Phase | What | Status |
|---|---|---|
| R0 | Config check (confirmed the top_k=20 clamp) | done |
| R1 | Diversity pilot, 6 tasks × 12 seeds | done, passed |
| R2 | **Main collection: 60 tasks × 24 seeds → 1,415 runs** | done — this is all our data |
| R3 | OOD (harbor_sql) + sealed swe_60+ pool | **NEVER RUN** — see "known gaps" |
| R4 | Analysis (CPU) | done — this is what 026/027/028 are |
| R5 | Phase-4 live eval | **NEVER RUN** — died with the mechanism |

### 4. `A#` / `B` — pipeline stages and specs (`specs/`)

| ID | What |
|---|---|
| A0 | Data collection (traces + activations) |
| A1 | Ambiguity direction |
| A2 | Disentangling autoencoder → the T (topic) and L (lean) spaces |
| A3 | Commitment / calibration (tau, l_scale, theta) |
| A4 | The gates (see below) |
| B | Online ask-trigger spec (CUSUM) |
| B2 | Behavioral-divergence variant of B (the 027 pivot) |

### 5. `gate#` — the seven A4 gates

`gate1` topic leakage · `gate2` decision recovery · `gate3` fork collocation ·
`gate4` conflation · `gate5` lean separation · `gate6` OOD transfer ·
`gate7` lead time.

### 6. Data and model directories

| Path | Model | Runs |
|---|---|---|
| `data/a0` | Qwen2.5-Coder-7B | 160 (no ActionEvents recorded) |
| `data/a0_v2` | Qwen2.5-Coder-14B | 160 |
| `data/a0_v3_32b` | **Qwen3-32B — the main collection** | 1,415 |
| `models/v3_32b_fixed*` | Fixed-labeler labels for the above (026 repair) | — |
| `models/t5_*_fixed` | Labels regenerated for the 7B/14B scale rows | — |

---

## Status board — the 028 program

| ID | Deliverable | Status | Result / where |
|---|---|---|---|
| **F2** | Rollout-budget ablation | ✅ **DONE** | Forked-task fraction .096→.667 at N=2→24, still rising. Testable decisions 0 / 0 / 0.8 / 3.6 / 8 / 13. `results/rollout_budget_ablation.json` |
| **F4** | Lead-time / fork anatomy | ✅ **DONE** | 63 forks, 40/60 tasks; 89% commitments datable; onset turn 4 → completion turn 11 = **ask window median 4 turns**. `results/lead_time_fork_anatomy.json` |
| **T5** | Cross-scale replication | ✅ **DONE** | 14B: hashed .622 / MiniLM .666 (small pool). 7B: census only — zero ActionEvents recorded. `results/t5_cross_scale.json` |
| **T1·R1** | Raw activations | ✅ done (026) | At permutation null; global Stouffer p=.339 |
| **T1·R2** | Learned L-space | ✅ done (026) | Null; n_testable=0 |
| **T1·R3** | Surface hashes | ✅ done (027) | AUROC .555 — clustered CI **[0.492, 0.631], contains chance** |
| **T1·R4** | MiniLM | ✅ done (027) | AUROC .580 — clustered CI **[0.535, 0.628], excludes chance** but far below the 0.75 bar |
| **T1·R5** | bge-large | ✅ **DONE** | AUROC **0.573** — no better than MiniLM's .580 despite 15× the parameters. v1/v2 reproduced exactly (.555/.580). `results/t1_r5_strong_embedder.json` |
| **T1·R6a** | Single-run LLM introspection | ⏳ **STAGED, not run** | 238 items + 12 payload chunks ready. Waiting on Fable budget |
| **T1·R6b** | Ensemble LLM comparison | ⏳ **STAGED, not run** | 200 pairs + 10 payload chunks ready. **The load-bearing experiment** |
| **T3** | Probe robustness | ⏳ **OWNER — GPU box** | [RUNBOOK_T3_PROBE.md](RUNBOOK_T3_PROBE.md). Needs `models/v3_32b_fixed/labels.npz` |
| **T6** | Ground-truth error bounds | ⏳ **not started** | Marked optional in 028; **recommend upgrading to mandatory** (see gaps) |
| — | AUROC clustered CIs (028 Am.A item 7) | 🔄 partial | 32B hashed done. Remaining reps after R5. `scripts/t1_auroc_ci.py` |
| — | Trace-blind vs informed registry split (028 Am.A item 8) | ✅ **DONE** (hashed + MiniLM) | hashed .540 informed vs .564 blind; MiniLM .600 informed vs .558 blind — direction FLIPS between reps and CIs overlap → no systematic registry-leak effect. **On the trace-blind 40, MiniLM's CI [.499,.616] touches chance** → on the cleanest subset nothing generic beats chance. **But census differs sharply: 85% vs 57.5% forked.** `results/blind_vs_informed_split*.json` |
| — | Train-fold vs eval-fold check | ✅ **DONE** | Stage-1 detection train F1 **0.517** vs eval **0.507** — fails equally on data it was tuned on → signal absence, not overfitting |

Legend: ✅ done · 🔄 in progress · ⏳ not started

---

## Known gaps a reviewer will attack

1. **Label validity.** Ground truth is a lexicon anchor-matcher; its judge
   validation scored **0.765 against a pre-registered 0.90 bar and was
   stopped**. The "noise biases toward null" defense does NOT apply (the
   noise is instance-dependent and fork-erasing). **Fix: T6 + a ~50-item
   owner hand-label.** Highest leverage remaining item.
2. **Underpowered-vs-negative.** F2 itself shows 0 testable decisions at
   N≤4. Defense = lead with the global Stouffer test and the relative
   text-vs-activation comparison, not per-decision counts.
3. **"You didn't try hard enough."** Only R6b can answer this, by showing a
   ceiling: if an LLM CAN separate them, the generic failures are
   informative rather than lack of effort.
4. **No OOD — the missing collection-R3.** One model family, one benchmark,
   one scaffold. **The sealed pool was never *collected*, not merely never
   tested** — there is no data for swe_60+. Partially mitigated by the
   trace-blind/informed split (028 Am.A item 8) and by T5's cross-scale
   rows. `scripts/scaffold_robustness.py` cheaply covers the scaffold half.
   SQL/sealed OOD needs new class artifacts → out of scope this cycle;
   state as a limitation.
5. **Prevalence saturation.** Always-ask baselines hit F1 0.800 because 2/3
   of tasks fork. Report AUPRC or move to decision-level attribution.
6. **Census is registry-quality-dependent (NEW, 2026-08-22).** Trace-blind
   registries find forks in 57.5% of tasks vs 85% for trace-informed ones.
   Since the known labeler failure mode is UNDER-detection (prose anchors
   never uttered by code-first traces), the 2/3 headline is best presented
   as a **lower bound**, with this decomposition shown rather than hidden.
7. **Do not over-claim "zero signal" (NEW, 2026-08-22).** With honest
   clustered CIs the rows differ: hashed .555 [.492,.631] **contains**
   chance, MiniLM .580 [.535,.628] **excludes** it. The defensible claim is
   *detectable but useless* — a whisper of signal nowhere near the 0.75 a
   detector needs — not "no signal at all." Say it that way before a
   reviewer says it for us.

## What defends the negatives (say these out loud in the paper)

- **Train ≈ eval.** Stage-1 detection scores F1 0.517 on the folds it was
  tuned on vs 0.507 held out. The failure is not generalization — the
  signal cannot be fit even with the answers in view.
- **Nothing is fitted in T1 rows R3–R5.** Those AUROCs use a fixed
  representation and Euclidean distance; there is no capacity to overfit,
  so no train/test split applies.
- **Registry construction is not the cause.** The informed-vs-blind gap
  flips sign between representations (hashed .540/.564, MiniLM .600/.558)
  with overlapping CIs — noise, not a leak. Stronger still: restricted to
  the 40 tasks whose registry was authored **before any trace existed**,
  MiniLM's CI [.499, .616] includes chance, so on the cleanest subset no
  generic representation is distinguishable from chance.
- **Adaptive analysis inflates false POSITIVES.** The long 011→028 chain
  spent its researcher degrees of freedom trying to find signal and failed,
  which makes the null more credible, not less.
- **Encoder capacity is not the bottleneck.** bge-large (335M, 1024-d)
  scores .573 vs MiniLM (22M, 384-d) at .580 — 15× the parameters buys
  nothing. "Use a better embedder" is answered empirically, not by assertion.

## Sealed / untouched resources

- **Sealed test pool** (swe_60+, 40 tasks): still sealed, never spent.
- **Phase-4 eval harness**: built, never run.
- **Judge-labelling arm** (025): STOPPED at its validation gate.

---

## Process rules

- Every experiment ships a script in `scripts/`, output in fresh
  `results/` files, never overwriting prior artifacts.
- Numbers are reported **as-run**. No rerun with variations without
  amending [decisions/028](decisions/028-iclr-experimental-program.md) FIRST.
- Test suite must stay green (currently **245**).
- Commit each completed experiment with its results; push to BOTH
  `master:master` and `master:main`. Use Git Bash for git (PowerShell git
  stalls on this machine).

## ⚠️ Update rule for this file

**Whenever an experiment is implemented, run, or its status changes, update
the status board in the SAME commit.** Specifically: move the row's status
marker, paste the headline number and the results path, and if it closes or
opens a reviewer gap, edit "Known gaps" too. A stale STATUS.md is worse than
none — it is the file people trust to know where the project stands.
