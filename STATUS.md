# STATUS — what every code name means, and what is done vs left

**Living document. Last updated: 2026-09-03.**
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

⚠️ **READ THIS BEFORE QUOTING ANY T1 NUMBER (2026-08-30).** The
pair-separation statistic that produces rows R3/R4/R5 has now been run
against targets that MUST be recoverable, and it does not recover them:
it scores **.788 / .849** on "are these two runs even on different
repositories," **.685** on "did they edit different files," and
**.589 / .474** on a *planted* fork whose answer is known by construction.
So .555/.580 was never comparable to the 0.75 bar — it must be read
against this instrument's own ~.85 ceiling, and it sits near the
machinery's noise floor. **R3/R4/R5 do not currently license "the signal
is not there."** R1/R2 use permutation tests, different machinery, and are
untouched. See `results/diag_positive_control.json` and gap 8.
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
| **Judge arm** (028 Am.G) | Fable relabels the 3,361 lexicon-unlabelable commitments | ✅ **DONE — 99.0% of pool, frozen** | **3,326/3,361 judged** over 9 usage-limit cycles (5 work files = 35 items unreached). Gate: 2,411 abstained (72.5%), **907 accepted**, 8 rejected (1 bad_class, 7 unlocatable evidence). **Coverage result (instrument-independent, stands): 907 labels where the lexicon had none, 126 new multi-run cells, 42.1% forked ≈ the lexicon's own 43%** — an independent labeler finds forks at the lexicon's rate. **Separation numbers (judge_only hashed .523 [.438,.616] / MiniLM .550 [.473,.636], 642 diff pairs, 54 tasks; union .575 / .569) must be read under the ⚠️ positive control** — same instrument, so the same ~.85 ceiling and planted-fork failure apply; they do NOT license a signal claim. What they DO add: a fully independent labeler on a DISJOINT item set moves the statistic by <.03, corroborating that the .54–.60 band is a property of the instrument, not the labels. `data/judge_labels_v3_32b.jsonl`, `results/judge_arm_phase1_full.json` |
| — | AUROC clustered CIs (028 Am.A item 7) | ✅ **DONE** for every 32B separation AUROC | R3/R4/R5 via `scripts/t1_auroc_ci.py`; R6b via `scripts/r6b_auroc_ci.py` (same estimator and constants, guard-checked to reproduce the cell). T5/T7 rows get theirs when those land |
| — | Trace-blind vs informed registry split (028 Am.A item 8) | ✅ **DONE** (all 3 reps) | Direction FLIPS across reps with overlapping CIs → no systematic registry-leak effect. Full 3×2 table above. **Census differs sharply: 85% vs 57.5% forked** → the 2/3 headline is a lower bound. `results/blind_vs_informed_split*.json` |
| — | Train-fold vs eval-fold check | ✅ **DONE** | Stage-1 detection train F1 **0.517** vs eval **0.507** — fails equally on data it was tuned on → signal absence, not overfitting |
| — | **Per-blocker / span re-anchor of `r_vec`** (diagnostic, NOT a pre-registered cell) | ✅ **DONE 2026-09-01 — aliasing is real, and is NOT the cause** | Guard reproduces the published rows exactly (.5545 / .5800). Aliasing is structural: one action resolves 2+ blockers **50.6%** of the time (max 5), so action-level anchoring cannot separate them. A sub-action `span_anchor` (±300 raw chars round that blocker's signature match, via `_norm_map`) cuts aliasing **0.499 → 0.064** and moves 75% of anchors — and AUROC **falls**: hashed .5545 → .5455 → **.5319**, MiniLM .5800 → .5695 → **.5482**. Defuses the 44%-aliasing attack on the published cell. ⚠️ **R4's chance-exclusion does not survive**: span CI [.497, .601] contains .50. `results/diag_per_blocker_anchor.json` |
| — | **Positive control for the T1 pair-separation machinery** (diagnostic, NOT a pre-registered cell) | ✅ **DONE 2026-08-30 — the instrument does not resolve** | Same `auroc` + Euclidean-on-`r_vec` statistic as R3/R4/R5, run against must-be-recoverable targets. **A** same-vs-diff TASK (different repos entirely): hashed **.788**, MiniLM **.849** — should be ~1.0. **B** same-vs-diff FILE SET within task: **.685 / .684**. **C** planted 2×2 fork (`return None` vs `raise ValueError` × 4 shell idioms): nuisance distance is ~2× signal (ratio **.60 / .49**); AUROC with idiom varying **.589 / .474** — MiniLM below chance on a fork known by construction. **D** real label stratified by lexicon decision margin: hashed .484/.637/.581, MiniLM .556/.601/.640 — not monotonic, so label confidence does not buy separation. Incidental: `_is_mutating` misses Python-mediated writes. `results/diag_positive_control.json`, `scripts/diagnose_positive_control.py` |
| — | **Replay-and-diff PILOT** (gates the 60-task box run; NOT a pre-registered cell) | ✅ **DONE 2026-09-03 — Q1 GATE PASSED, Q2 ceiling NOT met** | 5 tasks (swe_0/1/10/11/12), 115 runs, 3,385 execs replayed in docker, 0 failed, 22 min, **no GPU**. **Q1 replay fidelity 0.9936** (median 1.00, p25 1.00, 99.1% of runs ≥0.9) vs a 0.80 gate — replay reproduces the recorded exit codes, so the diffs describe runs that really happened. **Q2 arm A (cross-task, must be ~1.0) = 0.581**, so the pre-set 0.95 ceiling is NOT met and the pilot's own verdict is NO-GO. But that 0.581 is metric contamination, not representation failure — see the row below. Arms B .445 / C .533 rest on 3 tasks / 4 cells and are too thin to read. ⚠️ **Both script defaults are wrong on the box**: `models/v3_32b_fixed_debug/` does not exist (use `models/v3_32b_fixed/`, identity-checked 787281/154766/33422 + chosen=1,595) and `data/a0_v3_32b` symlinks to the **1,385**-run snapshot while the fixed labels are the **1,415**-run universe (use `/ssd/wta_data/a0_v3_32b`). Inputs are now recorded in the artifact. `results/replay_diff_pilot.json`, `scripts/replay_diff_pilot.py` |
| — | **Why the replay pilot's arm A landed at .581** (diagnostic, NOT a pre-registered cell) | ✅ **DONE 2026-09-03 — the no-op runs run the number** | **26 of 115 diffs are empty**, and `jaccard_distance(∅,∅)=0.0` scores two runs that changed *nothing* as an EXACT match — which is why **4.4% of cross-task pairs** matched exactly, impossible for different repos otherwise. Excluding empty diffs: arm A **0.581 → 0.817**, cross-task exact matches **.0437 → .0000**, cross-task mean distance **.997** (properly disjoint). The empties are REAL, not a replay defect: 12/26 ran no write command; the other 14 wrote only `mkdir -p && touch`, i.e. empty files, which carry no ±content lines (swe_12-s2: 15 writes, fidelity 1.00, empty diff). Also verified: `cd` does not persist across actions, but `collect_v2.py` drove the ORIGINAL collection through the same `DockerTaskEnv`, so the recorded run had no `cd` persistence either — replay is faithful, and exit-code fidelity is simply blind here. **Does NOT convert the pilot to a GO** (.817 < .95); it relocates the limit: same-task pairs are themselves far apart (mean **.880**) — two runs on the same repo routinely touch disjoint line sets. `results/diag_replay_empty_diffs.json`, `scripts/diagnose_replay_empty_diffs.py` |
| — | **What limits the replay pilot's arm A** (diagnostic, NOT a pre-registered cell) | ✅ **DONE 2026-09-03 — a mixture, and the pilot cannot resolve it** | Same-task pairs sit at mean distance **.880**, and that is the cap. **Not the metric**: jaccard/containment/dice agree within .006 (.8170/.8220/.8170), so the median 2.9× (p90 22×) size asymmetry is irrelevant. **Not noise**: **no line is common to all runs of any task** (in_ALL=0 for all five); the shared core is boilerplate (`+try:`,`+else:`) plus deletions of the same original lines, and 37–83% of a task's distinct lines are seen once — the unique remainder is real solution content. **It is a mixture**: **47 of 89** non-empty runs carry **no deletion lines at all** (they only add new content, never modify existing code). Split: modifying **armA .967** (n=42) vs additive-only **.821** (n=47). Channel is not the story — on those same 42, del-only .880 and add-only .937 are both *below* the full .967. ⚠️ **Does NOT make the pilot a GO**: task-clustered CI95 **[.782,.995]** vs **[.662,.971]** overlap across nearly their whole range, and the subgroup is chosen **post hoc on a property of the outcome**. Hypothesis to pre-register, not a result. **Next step is more TASKS, not runs** (clusters set the width): ~20 tasks ≈ 68 GB / ~90 min vs the 174 GB full run. `results/diag_replay_dispersion.json`, `scripts/diagnose_replay_dispersion.py` |
| — | **Canonical accuracy** — first ABSOLUTE (per-run) read-out, NOT a pre-registered cell | ✅ **DONE 2026-09-03 — no signal, and the lexicon version is confounded** | Every T1 cell asks a *relational* question (did A and B decide the same?). This asks a *per-run* one: did the run land on the resolution the registry marks canonical? Free re-scoring of the existing 1,595 commitments — no collection, no GPU. **214/214 blockers carry exactly one `canonical: true` class, always class index 0** (the artifact's `_provenance` says so outright). Scored **1,595 commitments / 165 blockers / 58 tasks**. **Accuracy .3505, task-clustered CI95 [.2698, .4310], against a uniform-random baseline of .3799 — lift −.0295, CI contains chance.** Flat across temperature (.347/.348/.356/.352); forked blockers .280 vs unforked .401. ⚠️ **CONFOUNDED, so .3505 must NOT be read as "runs are right 35% of the time"**: the labeler picks whichever class carries the most signatures **52.5%** of the time vs a **38.0%** chance rate (lift +.146), and canonical/index-0 carries *fewer* signatures than index 1 (3.91 vs 4.15) while index 1 is selected most (.478 vs .351). Per-blocker accuracy is **bimodal — 55% of blockers at exactly 0.0, 25% at exactly 1.0** — i.e. the label is largely a property of which signatures exist for that blocker, not of what the run decided. **Verdict: the absolute framing is not refuted, but the lexicon-based version of it is dead**; a consequence-grounded target (`test_patch` + `test_cmd` + `log_parser`, which every task ships) has none of this confound and is the next step. `results/canonical_accuracy.json`, `scripts/canonical_accuracy.py` |
| — | **Test-outcome vector** — consequence-grounded fork labels, NOT a pre-registered cell | ✅ **DONE 2026-09-03 — instrument validated; the label barely varies, and 0/67 runs solve anything** | Replaces the lexicon entirely: replay the run, restore test files from HEAD, apply the task's own `test_patch`, run its `test_cmd` runner, read the **FAIL_TO_PASS** pass/fail vector. Owes nothing to lexicon, anchors or judge. **3 python tasks / 67 runs / 0 failed.** ✅ **Instrument validated by two controls**: ground-truth patch → **all F2P PASS on all 3 tasks**; zero-action baseline → all-fail; and **9/9 runs that changed nothing reproduced the baseline exactly** (the no-op control — replay is deterministic and the tests are not flaky). **Findings: 0 of 67 runs fully solved any task**; the vector **varies in only 1 of 3 tasks** (swe_0: 4 distinct vectors, `000000`×12 / `001000`×4 / `100000`×4 / `101000`×1 — swe_10 and swe_11 are constant all-zero across 22 and 24 runs). **48% of runs (32/67) produce non-importable code** (pytest collection error, e.g. swe_10-s10 leaves an `IndentationError` in `linux.py`), so the outcome is dominated by *did the run break the build*, not *which interpretation did it choose*. ⚠️ **JS tasks excluded** (swe_1, swe_12 = 48 runs): the shipped `sweap_json` parser mis-parses jest — 0/4 F2P names matched on a pristine container and every test reported PASSED despite jest failure glyphs. Deviations recorded: selected F2P test files instead of the full suite (shipped `test_cmd` runs everything), and test files restored from HEAD so a run that edited tests cannot fake a pass. `results/test_outcome_vector.json`, `scripts/test_outcome_vector.py` |
| — | **Does the fork census survive a working-code filter?** (diagnostic screen, NOT a pre-registered cell) | ✅ **DONE 2026-09-03 — not on this sample** | Recomputes the fork census counting only runs that changed the repo **and** produced importable code, using `test_outcome_vector`'s per-run status. Forked blockers **2/11 → 0/10**; forked tasks **1/3 → 0/3**; **26/67** runs survive the filter. On these 3 tasks every fork is carried by runs that broke the build or wrote nothing. ⚠️ **11 blockers and 2 forks — this CANNOT restate the 40/60 census and is not evidence it is wrong.** It is a screen, and the reason to extend the test-outcome vector to more tasks (cheap, no GPU) before the census carries a causal claim. `results/diag_forks_on_importable.json`, `scripts/diagnose_forks_on_importable.py` |
| — | **`full_info` arm — is the bottleneck information or competence?** (028 **Amendment H**, pre-registered) | 🔄 **RUNNING 2026-09-03** | New collection: same 3 python tasks, same Qwen3-32B, same protocol, **only the instruction file changes** — `full_info/instruction.md` resolves every blocker in prose (verified on swe_0: appends `## BLOCKER DETAILS` whose resolutions match the canonical classes verbatim). `collect_v2.py` gains `--mode {baseline,full_info}`. Frozen baseline to beat: import-failure **32/67 = 47.8%**, solve rate **0/67**, swe_0 partial-pass **9/21**, distinct vectors **4/1/1**. Pre-registered reading: solve≈0 and import-failure≈48% ⇒ **competence is the bottleneck** and the paper must say Qwen3-32B does not meet its own hook's premise; swe_0 off zero ⇒ **information is first-order** and the census gains causal support. Scored by the SAME validated `test_outcome_vector` pipeline. `decisions/028` Am.H |

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

⚠️ **REVISED 2026-08-30 — the band must now be read against the
instrument's own ceiling, not against 0.75.** The positive control
(`results/diag_positive_control.json`) shows this statistic reaches only
**.788/.849** when asked whether two runs are on *different repositories*,
**.685** on *different files*, and **.589/.474** on a *planted* fork. Its
usable dynamic range is therefore ~0.29, and the interpretation label sits
about 0.055 into it. The honest sentence is now: **"on an instrument whose
empirical ceiling is .85 for cross-repository discrimination,
interpretation separation is .58; the gap between planted-fork performance
with idiom held constant versus varying shows the statistic is dominated by
shell-idiom variance, so we cannot yet distinguish absent signal from an
unresolving instrument."** Do NOT write "the signal is not in surface
behavior or embeddings" from these rows until the instrument is fixed or
the ceiling is reported alongside.

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
     `scripts/diagnose_fable_label_signal.py`.
   - **The production judge arm, FROZEN at 99.0% (2026-09-01, 028 Am.G/G.1)
     — this gap is now substantially CLOSED as an explanation of the null.**
     3,326/3,361 judged; 907 accepted labels on a DISJOINT item set from the
     lexicon's. Coverage is real and instrument-independent: 126 new
     multi-run cells at **42.1% forked**, against the lexicon's own 43%.
     Separation is not: judge_only hashed .523 [.438,.616] / MiniLM .550
     [.473,.636]; union .575 / .569. **A fully independent labeler moves the
     statistic by &lt;.03.** Combined with the span re-anchor (gap 8) also
     failing to move it, the remaining live explanation of the .54–.60 band
     is the instrument, not the labels. Note the agreement read-out trap
     documented in `results/diag_agreement_metric_check.json`: pooled kappa
     0.762 is inflated because chance agreement is computed over the union
     of disjoint per-blocker class inventories; macro-over-blockers kappa is
     **0.000**, and only 3 of 92 blockers support a computable kappa. Never
     quote a bare pooled kappa.
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

8. **The instrument never had a positive control — and now that it has one,
   it fails (NEW, 2026-08-30). Highest-priority item in the repo.**
   `scripts/diagnose_positive_control.py` runs the *identical* statistic
   behind R3/R4/R5 against targets that must be recoverable:

   | target | hashed | MiniLM |
   |---|---|---|
   | same-vs-diff TASK (different repos) | .788 | .849 |
   | same-vs-diff FILE SET (within task) | .685 | .684 |
   | planted fork, idiom varying | .589 | **.474** |
   | *the real interpretation label* | .555 | .580 |

   A planted fork (`return None` vs `raise ValueError`) is recovered when
   idiom is held constant and lost when it varies; nuisance distance is
   ~2× signal distance (ratio .60 / .49). **Consequence:** R3/R4/R5 do not
   license a claim about interpretation signal in either direction, and any
   paper text asserting "not in surface behavior / not in embeddings" from
   those rows must be rewritten or must report the ceiling next to it.
   This subsumes part of gap 3: the "you didn't try hard enough" escape is
   now answered — the trying was fine, the ruler was not.

   **(a) RE-ANCHORING IS NOT THE FIX — settled 2026-09-01**
   (`scripts/diagnose_per_blocker_anchor.py`,
   `results/diag_per_blocker_anchor.json`). The aliasing is real but
   *structural*: one action resolves 2+ blockers **50.6%** of the time (up to
   5; median 2), because the agent rewrites the whole file in a single
   heredoc. So no action-level anchor can separate blockers, and a
   sub-action `span_anchor` (±300 raw chars around that blocker's own
   signature match, mapped back through `_norm_map`) was run instead:

   | anchor | hashed | MiniLM | aliasing |
   |---|---|---|---|
   | run (published, guard) | .5545 [.491,.632] | .5800 [.536,.626] | .499 |
   | blocker | .5455 [.484,.623] | .5695 [.524,.615] | .508 |
   | **span** | **.5319 [.477,.600]** | **.5482 [.497,.601]** | **.064** |

   De-aliasing works (0.499 → 0.064, 75% of anchors move) and AUROC
   **decreases monotonically**. So the 44% aliasing does NOT explain the
   band — which also defuses it as a reviewer attack on the published cell.
   ⚠️ **But note: R4's "excludes chance" does not survive de-aliasing** —
   MiniLM's span CI [.497, .601] contains .50. Any claim resting on R4
   being the one row above chance is anchor-dependent and must say so.

   **(b) NEXT — replay-and-diff. Pilot written 2026-09-01, awaiting the box.**
   The only remaining construction that could yield a working instrument AND
   independent ground truth: two runs decided the same thing iff their final
   normalized `git diff` matches — removes idiom, anchor, lexicon and judge
   by construction. Full run is 60 tasks / 174 GB / ~38.5k execs, so it is
   gated behind `scripts/replay_diff_pilot.py` (5 tasks, 115 runs, **3,385
   execs**, resumable, sealed pool excluded by construction via
   `eligible_tasks`). The pilot answers two questions in order:
   **Q1 replay fidelity** — does replaying `action_text` in `segment_idx`
   order reproduce the recorded `exit N` codes? Below the 0.80 gate the
   diffs describe runs that never happened and the approach is void
   regardless of how its separation looks. **Q2** — the same three positive
   control arms (cross-task / file-set / interpretation), scored by
   1 − Jaccard over normalized changed-line sets so the AUROC is directly
   comparable to the published .555/.580. GO requires fidelity ≥ .80 **and**
   cross-task AUROC ≥ .95. If replay-and-diff also lands at ~.58 *with* a
   clean ceiling, that is the escape-proof negative the paper lacks; if it
   cannot clear the ceiling either, no instrument in this repo resolves and
   the paper says exactly that.
   Incidental defect found in passing: `wta.labeling._is_mutating` matches
   only `("sed -i", ">", ">>", "tee ", "patch ", "git apply", "perl -i")`,
   so a Python-mediated write (`python -c "...open(f,'w').write(s)"`)
   mutates the repo invisibly — it can neither anchor a commitment nor
   contribute to `r_vec`.

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
- Test suite must stay green (currently **257 passed, 2 skipped**).
- Commit each completed experiment with its results; push to BOTH
  `master:master` and `master:main`. Use Git Bash for git (PowerShell git
  stalls on this machine).

## ⚠️ Update rule for this file

**Whenever an experiment is implemented, run, or its status changes, update
the status board in the SAME commit.** Specifically: move the row's status
marker, paste the headline number and the results path, and if it closes or
opens a reviewer gap, edit "Known gaps" too. A stale STATUS.md is worse than
none — it is the file people trust to know where the project stands.
