# STATUS — what every code name means, and what is done vs left

**Living document. Last updated: 2026-08-25.**
Update rule at the bottom: this file is edited in the SAME commit as any
experiment that lands. If it disagrees with reality, reality wins and this
file is stale — fix it.

---

## Read this first (30 seconds)

We are executing [decisions/028](decisions/028-iclr-experimental-program.md),
the frozen experimental program for the ICLR paper
([paper/OUTLINE.md](paper/OUTLINE.md)). The paper is a MEASUREMENT paper:
"where does an agent's need-to-ask signal live?" Answer so far: not in
single-run internals, not in surface behavior, not in generic embeddings,
and — as of 2026-08-25 — not in full-context LLM judgment either, whether
it reads one run (R6a) or compares two (R6b).
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
| **T1·R5** | bge-large | ✅ **DONE** | AUROC **0.573**, clustered CI [.525, .627] — no better than MiniLM's .580 despite 15× the parameters. v1/v2 reproduced exactly. `results/t1_r5_strong_embedder.json` |
| **T1·R6a** | Single-run LLM introspection | ✅ **DONE** | 238/238 judged, 60/60 tasks. Task-detection **F1 0.709** (P .718/R .700) — beats the fitted detectors (.507/.582) but **below always-ask .800**; run-level acc .580 vs a .664 always-ask base rate. `results/r6_llm_cells.json` |
| **T1·R6b** | Ensemble LLM comparison | ✅ **DONE — the hoped-for ceiling did NOT appear** | 200/200 pairs, 46 tasks. AUROC **0.5786**, clustered CI **[0.481, 0.670] contains chance** — indistinguishable from MiniLM's .580. Judge said "different" 129/200 on a 100/100 pool (diff 73% / same 44% correct). `results/r6_llm_cells.json`, `results/r6b_auroc_ci.json` (028 Am.E) |
| **T3** | Probe robustness | ✅ **DONE** | Neither escape works. full-dim linear **.541**, full-dim MLP **.507** — both far below the causal+anchors-masked TEXT baseline **.730**; the MLP is *worse* than the linear probe. 256-d consistency check reproduced **.2745** exactly. `results/gate2_probe_robustness.json` |
| **T7** | Cross-family replication (2nd model, same 60 tasks) | 🔄 **FULL COLLECTION RUNNING** | 60 tasks x **24 seeds** on `Mistral-Small-24B-Instruct-2501`. Restarted 2026-08-23 17:17Z on the **028 Amendment D** fix (a `grep -r -` dumped 723 MB and wedged a shard >1h at 97% CPU with its GPU idle; `error_signature` now fingerprints the truncated observation the model actually saw — trajectories bit-identical, window untouched). 55 pre-fix runs KEPT. Measured rate ~9-12 runs/h -> **~5-6 days**; ~272 GB projected on `/ssd3` (490 GB free). Amendment D **confirmed on the real failure**: `swe_1-s10`, which previously hung >1h, now completes normally. Relaunched 2026-08-23 17:50Z with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (allocator only) after 3 of 4 cards were observed at 3.2-3.7 GiB free — R2's documented OOM band. |
| **T6** | Ground-truth error bounds | ⏳ **not started** | Marked optional in 028; **recommend upgrading to mandatory** (see gaps) |
| — | AUROC clustered CIs (028 Am.A item 7) | ✅ **DONE** for every 32B separation AUROC | R3/R4/R5 via `scripts/t1_auroc_ci.py`; R6b via `scripts/r6b_auroc_ci.py` (same estimator and constants, guard-checked to reproduce the cell). T5/T7 rows get theirs when those land |
| — | Trace-blind vs informed registry split (028 Am.A item 8) | ✅ **DONE** (all 3 reps) | Direction FLIPS across reps with overlapping CIs → no systematic registry-leak effect. Full 3×2 table above. **Census differs sharply: 85% vs 57.5% forked** → the 2/3 headline is a lower bound. `results/blind_vs_informed_split*.json` |
| — | Train-fold vs eval-fold check | ✅ **DONE** | Stage-1 detection train F1 **0.517** vs eval **0.507** — fails equally on data it was tuned on → signal absence, not overfitting |

Legend: ✅ done · 🔄 in progress · ⏳ not started

---

## The T1 separation rows, with honest CIs

Task-clustered bootstrap, 2000 draws, seed 0 (`scripts/t1_auroc_ci.py`;
R6b via `scripts/r6b_auroc_ci.py`, same estimator and constants).
The pooled point estimates are the pre-registered as-run numbers.

| Row | Params | Pooled AUROC | Clustered 95% CI | vs chance |
|---|---|---|---|---|
| R3 hashed char-3grams | — | .555 | [.492, .631] | contains .50 |
| R4 MiniLM-L6 | 22M | .580 | [.535, .628] | excludes |
| R5 bge-large | 335M | .573 | [.525, .627] | excludes |
| **R6b LLM ensemble comparison** | frontier | **.579** | **[.481, .670]** | **contains .50** |

R6b is scored on 200 stratified pairs (46 tasks), not the full 8,567-pair
pool, so its interval is the widest — that width is sample size, not
instability. The point estimate lands **on top of MiniLM's .580**: the row
that was supposed to establish a ceiling instead joined the band.

Split by registry provenance (`scripts/blind_vs_informed_split.py`):

| Row | trace-informed (20 tasks) | trace-blind (40 tasks) |
|---|---|---|
| hashed | .540 [.448, .656] | .564 [.471, .685] |
| MiniLM | .600 [.527, .679] | .558 [.499, .616] |
| bge | .546 [.483, .627] | .599 [.534, .673] |

**How to say this.** Every generic representation lands in a narrow
**.54–.60** band. Some intervals exclude chance, some do not, and which
ones do is not stable across representations or subgroups. The band is flat
across a 15× parameter increase and across registry provenance. So the
claim is *detectable but useless, and flat* — **NOT** "no signal at all,"
and **NOT** "nothing beats chance on the clean subset" (bge does, at
.599 [.534, .673]). Over-claiming either way is the easiest way to lose a
reviewer who reruns the bootstrap.

## T7 on the box (2026-08-23) — blockers found, and how they resolved

T7 was handed over as "launch-ready". It was not. None of this was visible
from the laptop; all of it was found before spending GPU time. Recorded here
because the failure modes recur on every VM stop.

**The root cause was one event: the ephemeral NVMe was wiped.** That destroys
`/opt/dlami/nvme/wta-venv` — which every launcher and runbook hard-codes — and
the docker/containerd roots with it. The venv has been rebuilt (py3.12,
torch 2.9.1+cu128, transformers 5.15.1, 4 GPUs visible), but **treat it as
ephemeral: it will not survive the next stop.**

**1. ✅ RESOLVED — the pre-registered models could not be loaded.**
`hf_reader` used `AutoModelForCausalLM` and read `config.num_hidden_layers`
directly. `Mistral-Small-3.2-24B` is `Mistral3ForConditionalGeneration`;
`mistral3` is absent from the CausalLM mapping, its depth/width live in
`config.text_config`, and its decoder blocks sit at
`model.model.language_model.layers`. Fixed by **028 Amendment B** (loader +
nested config + hook capture, flat Llama/Qwen path provably unchanged and
still passing its bit-compatibility test). Verified on real weights:
`loaded: Mistral3ForConditionalGeneration, n_layers=40 hidden=5120,
layers=[8,12,16,20,24,28,32,34]` (`results/xfam_loader_smoke.json`).

**2. ✅ RESOLVED — loading it was necessary but not sufficient.**
`Mistral-Small-3.2-24B` **ships no HF `chat_template`**, so `generate_segment`
could not build a single agent turn. Writing one would define what the model
sees — the protocol tuning Amendment A item 9 forbids. Fixed by **028
Amendment C**: swap to `mistralai/Mistral-Small-24B-Instruct-2501`, the
text-only predecessor — same family, same size class, **same 40 x 5120**, and
it ships its own template, so the prompt format is the vendor's exactly as R2
used Qwen's. Only the model id moves.

**3. ✅ RESOLVED — the hil-bench task images were gone.** `docker images`
listed ~100 `hilbench-swe:*` rows at **0B** with `docker run` failing
`blob not found` — phantom metadata, not images. The per-task Dockerfiles are
`FROM hilbench-swe:<attempt_id>`, i.e. they wrap prebuilt bases and cannot be
rebuilt from source.

⚠️ **Correction to an earlier entry in this file:** this was recorded as
needing an HF token because `ScaleAI/hil-bench-swe-images` returned 401.
**That was wrong.** The 401 came from querying the *model/dataset* API for
something that is an HF **bucket**; `hf buckets ls` / `cp` read it
**anonymously**, which is the path `warmup_images.sh` already uses. **No HF
token is required for T7.** The real fault was the wiped docker/containerd
roots: recreate both, start containerd *then* docker, or `docker load` dies
with `metadata.db: no such file or directory`. Restoration is scripted in
`scripts/restore_hilbench_images.py` (60 tasks, 174 GB, sealed pool excluded
by construction — the task list is derived exactly as `collect_v2` derives
it).

**4. ⚠️ NOTED, as-run — Mistral tokenizer regex.** transformers warns that
both Mistral tokenizers "will lead to incorrect tokenization" without
`fix_mistral_regex=True`. Reads are indexed by token position, so the flag is
now passed unconditionally (Amendment C.2); Qwen3 is byte-identical with and
without it, so R2 is untouched.

⚠️ **Correction (2026-08-27, Amendment F).** This entry previously claimed the
flag was "a no-op that silences a warned-about defect rather than a
demonstrated one", based on 13 synthetic probe strings. **That was wrong.** On
REAL agent transcripts the flag changes tokenization in **23 of 40** Mistral
traces (0 of 40 Qwen), diverging from as early as token 92. It also exposed a
genuine bug: the labeler loaded its tokenizer WITHOUT the flag while the
collector used it, drifting the token→char map that `token_idx` indexes.
Fixed in Amendment F; pinned by
`harness/contract/test_tokenizer_agreement.py`. Lesson logged: synthetic
probes are not evidence about real traces.

## Known gaps a reviewer will attack

1. **Label validity.** Ground truth is a lexicon anchor-matcher; its judge
   validation scored **0.765 against a pre-registered 0.90 bar and was
   stopped**. The "noise biases toward null" defense does NOT apply (the
   noise is instance-dependent and fork-erasing). **Fix: T6 + a ~50-item
   owner hand-label.** Highest leverage remaining item — and **R6b raised
   its priority (2026-08-25)**: an LLM judge reading the same excerpts the
   labeler used agrees with the labels at only **0.585**, which is equally
   consistent with a weak signal and with noisy ground truth. That cell
   cannot separate the two; the hand-label can.

   Two diagnostics (NOT pre-registered cells) now bracket this gap:
   - **Label learnability (2026-08-27, VM session):** a FITTED TF-IDF+LR
     cannot recover the lexicon label from the window text (+0.001 over
     majority) or the whole transcript (+0.002); underpowered at ~12
     runs/cell, but consistent across window/full/fitted/unfitted/LLM.
     `results/diag_label_learnability.json`.
   - **Fable-vs-lexicon relabelling (2026-08-29, laptop):** on the 80
     intersection items where the 025 validation produced accepted Fable
     judgments (31 cells, 79% agreement with lexicon), swapping the answer
     key from lexicon to Fable does NOT surface signal — hashed .558
     [.37,.74], MiniLM .578 [.28,.81], the same band; the lexicon-keyed
     .744 on this sliver rests on 6 diff pairs and its CI contains chance.
     Severely underpowered and sampled from lexicon-labelable items only.
     `results/diag_fable_label_signal.json`,
     `scripts/diagnose_fable_label_signal.py`. The full-pool version
     requires executing the prepared-but-never-run production judge pass
     (`models/v3_32b_judge/`, 306 work files) — which the 025 STOP gates and
     only a new 028 amendment can authorize.
2. **Underpowered-vs-negative.** F2 itself shows 0 testable decisions at
   N≤4. Defense = lead with the global Stouffer test and the relative
   text-vs-activation comparison, not per-decision counts.
3. **"You didn't try hard enough." STILL OPEN — and R6b did not close it
   (2026-08-25).** R6b was the designated answer: show a ceiling, and the
   generic failures become informative rather than lack of effort. It ran at
   full coverage and produced **no ceiling** (.579 [.481, .670]). So the
   negative now extends from generic representations to full-context LLM
   judgment *with* cross-run comparison — a stronger claim in one direction
   — but we still cannot separate "the signal is not there" from "nothing we
   tried recovers it." **Report both halves; do not let the extra coverage
   read as a proof of absence.** See 028 Amendment E.
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
   reviewer says it for us. **R6b is in the same position (2026-08-25):**
   .579 [.481, .670] contains chance while R4's does not, so report the
   interval and let the band speak — do not call R6b "chance" and do not
   call it "signal."

## What defends the negatives (say these out loud in the paper)

- **Train ≈ eval.** Stage-1 detection scores F1 0.517 on the folds it was
  tuned on vs 0.507 held out. The failure is not generalization — the
  signal cannot be fit even with the answers in view.
- **Nothing is fitted in T1 rows R3–R5 or R6b.** R3–R5 use a fixed
  representation and Euclidean distance; R6b is a zero-shot judgment against
  a frozen prompt. Nothing is trained, so there is no capacity to overfit and
  no train/test split applies.
- **Registry construction is not the cause.** The informed-vs-blind gap
  flips sign across representations (blind higher for hashed and bge,
  informed higher for MiniLM) with heavily overlapping CIs — noise, not a
  leak. See the full 3×2 table below.
- **Adaptive analysis inflates false POSITIVES.** The long 011→028 chain
  spent its researcher degrees of freedom trying to find signal and failed,
  which makes the null more credible, not less.
- **Encoder capacity is not the bottleneck.** bge-large (335M, 1024-d)
  scores .573 vs MiniLM (22M, 384-d) at .580 — 15× the parameters buys
  nothing. "Use a better embedder" is answered empirically, not by assertion.
- **Neither is judgment capacity, or single-run framing (NEW, 2026-08-25).**
  A frontier LLM shown BOTH runs' excerpts and asked the question directly
  scores .579 [.481, .670] — the same band. "Just ask a good model" and "your
  representations were too dumb" are now answered empirically too. The
  registry-blindness audit backs this: 70 judge Reads, every one a payload
  chunk, zero stray reads (028 Am.E.3).

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
- Test suite must stay green (currently **254 passed, 2 skipped**).
- Commit each completed experiment with its results; push to BOTH
  `master:master` and `master:main`. Use Git Bash for git (PowerShell git
  stalls on this machine).

## ⚠️ Update rule for this file

**Whenever an experiment is implemented, run, or its status changes, update
the status board in the SAME commit.** Specifically: move the row's status
marker, paste the headline number and the results path, and if it closes or
opens a reviewer gap, edit "Known gaps" too. A stale STATUS.md is worse than
none — it is the file people trust to know where the project stands.
