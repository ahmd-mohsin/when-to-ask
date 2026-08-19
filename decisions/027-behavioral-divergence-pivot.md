# 027: Part B pivots to BEHAVIORAL divergence — the ask-signal lives in the ensemble

Date: 2026-08-19. Status: ACCEPTED (owner approved the pivot in-session,
2026-08-19: "I want the best paper — go ahead make it happen", after the
novelty sweep). Follows 026 Amendment B (the repaired gates: lean not readable
from single-run activations in raw or learned space) and the 16-agent novelty
sweep of 2026-08-19 (full output archived in session task w7n0rxmf0; key
findings restated in §2 so this entry is self-contained).

**Pre-registration timing.** This entry freezes the pivot's detector
definition, baselines, evaluation protocol, and go/no-go criteria BEFORE any
detection number is computed on R2 data. Same discipline as 025/026: the
comparison that decides the paper must not be shaped by its own results.

## 1. The decision

Part B (the online ask trigger) is rebuilt on **registry-free behavioral
divergence across N parallel trajectories** instead of activation features
(A1/A2/A3 probes). The trigger MACHINERY is unchanged — the pre-registered
CUSUM-on-bucket-spread design of specs/B-online-trigger.md (fire on spread of
r, never s; h in read/observe units; offline-calibrated reference) — only the
FEATURES feeding it change: topic and resolution vectors derived from each
run's observable behavior (files touched, edit regions, mutating-command
content, error signatures, subgoal prose), never from hidden states and never
from the task's blocker registry.

This supersedes the scope rule "Part B must not be built until A4 gates pass":
those gates gated the ACTIVATION detector; 026 Amendment B closed that
question (negative, controlled). The activation work is not discarded — it
becomes the controlled half of the paper's central claim (§4).

## 2. Why (novelty grounds, from the 2026-08-19 sweep)

No published work triggers mid-task human clarification from cross-trajectory
divergence in a multi-step agent — but the window is narrow (~2-4 months):

- arXiv 2602.11619 (Snowflake) has the detection signal (10-rollout divergence
  of SWE agents) but only abstains post-hoc; no fork localization, no asking.
- arXiv 2603.26233 ("Ask or Assume?") has the setting and behavior (mid-task
  asking on underspecified SWE-bench, answer injection) but a single-run
  prompted-introspection trigger and no ask-quality metric.
- arXiv 2605.05802 uses mid-trajectory cross-rollout divergence to terminate
  RL rollouts; 2604.14624 (CLARITI) and 2511.02208 (PPP) are trained single-run
  ask policies; 2502.13069 (Ambig-SWE, ICLR26) shows prompting cannot detect
  underspecification — the failure this mechanism fixes; 2608.11552 argues
  trajectory consistency beats single-run UQ with UNCONTROLLED evidence.

Consequence (sweep synthesis, adopted here): "mid-task asking helps" is no
longer a publishable claim alone. The paper's decisive table is
**divergence-triggered asking vs single-run introspection at matched ask
budgets**, plus the controlled activation negative. If the divergence trigger
does not beat introspection, the mechanism paper is not written (§6).

## 3. The detector (frozen before any number)

Registry-free features per run, per turn, computed only from what the agent
observably did (all fields exist in R2 logs; population rates verified:
action_text 99.9%, error_signature 96.4%, files 65.7%, subgoal 53.9%,
region 5.7%):

- **Commitment event**: a MUTATING action (`labeling._is_mutating`, frozen
  lexicon). Non-mutating turns never vote — the behavioral analog of "the
  teacher, never the trigger".
- **Resolution vector r**: L2-normalized hashed character-3-gram TF vector
  (dim 512, seed 0) over the normalized mutating command text + region spans
  + post-execution error signature. Runs that write different resolutions
  produce distant r's; identical edits collide.
- **Topic vector**: L2-normalized hashed TF vector (dim 256, seed 1) over
  files touched + subgoal terms, accumulated over the run's recent turns
  (window: last 3 turns). Buckets runs working the same decision without any
  registry knowledge.
- **Vote weight**: 1.0 at each mutating action (latest vote per run wins —
  online.py semantics); no retraction in v1.
- **Trigger**: `wta.online.AskTrigger` verbatim (spread of r within topic
  bucket, CUSUM, loop channel off in offline replay). One observe per turn
  per run in lockstep replay (round-robin by step index), matching the eval
  orchestrator's rhythm.

Calibration (offline teacher, mirrors A3's tau/l_scale discipline): theta,
reference, and h are set on TRAIN folds only — reference = 0.9 quantile of
within-bucket spread among runs whose committed resolution is the SAME by
ground-truth labels (the registry-labeled analog of benign_spread_reference);
theta/h from a small pre-declared grid (theta ∈ {0.3, 0.5, 0.7}, h ∈ {3, 6,
12}) maximizing detection F1 on train folds ONLY. The deployed detector
consumes no labels; only its scalar thresholds are teacher-calibrated. Split:
5-fold by task, seed 0, same kfold_group_indices machinery as the gates.

## 4. The paper (one paper, two halves)

Claim: **the need-to-ask signal in coding agents lives in the behavioral
ensemble, not in single-run internals or single-run introspection.**
Half A (done, 026): controlled negative — activations fail causal text
baselines under run-level permutation with testability floors. Half B (this
entry): the ensemble signal detects interpretation forks with lead time
before behavioral commitment and, fired as targeted questions, beats
single-run triggers at matched ask budgets on HiL-Bench. Fallbacks
pre-declared: if §6's stage-2 comparison loses, the paper is Half A + the
fork census + stage-1 detection analysis (the sweep's rank-1
"measurement-science" framing) and the mechanism is future work.

## 5. Evaluation protocol (two stages, cheapest-first)

**Stage 1 — offline detection head-to-head on R2 data (no GPU, no new runs).**
Replay the 1415 runs per task in lockstep; the detector fires or not.
Ground truth from the FIXED labeler (026): a task instance "needs ask" iff
≥1 decision has ≥2 runs committed to different classes (forked); commitment
positions from labeled commitment rows. Metrics, eval folds only:
task-level detection precision/recall/F1; decision-level attribution
(does the firing bucket contain the forked runs — report, not gated);
**lead time** = fire step vs gate7-style action-divergence step (median,
in turns). Offline baselines, budget-matched by construction: random-ask
(Bernoulli matched to the detector's fire rate, 20 seeds), always-ask-at-
step-k (k = detector's median fire step), ask-at-first-mutating-action.
**Introspection arm (stage-1b, LLM):** an Ask-or-Assume-style single-run
introspector — same model family reading one trajectory's prefix at the
same observation points, binary ask/no-ask — run via Fable subagents on a
budget-feasible task sample (all forked tasks + equal-size non-forked
sample), chunked + resumable, prompts registry-blind. Compared at matched
fire rate.
**Stage 2 — live interactive eval (GPU box), only if stage 1 GO.** The
Phase-4 harness as pre-registered in 022 with two added arms:
`divergence_ask` (this detector) and `introspection_ask` (the 2603.26233
analog). Train-pool tasks first; the sealed pool is spent ONCE, on the final
headline table, only after the owner reviews stage-2 train-pool numbers
(019's sealed-pool discipline transfers intact).

## 6. Go/no-go criteria (frozen now)

- **Stage-1 GO** (proceed to introspection arm + stage 2): detection F1 on
  eval folds exceeds the best offline baseline by ≥ 0.10 absolute AND median
  lead time ≥ 2 turns before the action-divergence step on detected forks.
- **Stage-1b GO** (mechanism paper viable): detection F1 ≥ introspection F1
  at matched fire rate (parity suffices — the mechanism also brings fork
  localization and lead time that introspection lacks; superiority
  strengthens but parity does not kill).
- **NO-GO** at either stage → the pre-declared fallback framing of §4, no
  further mechanism spend. Numbers reported either way; nothing tuned after
  the eval folds are touched (train-fold calibration only).

## 7. Relation to frozen artifacts

interpretation_classes.json, fork_type_annotations.json, the 026 label
pipeline, and all gate results stay frozen. The detector must never read the
class artifact at inference; contract tests enforce the leak rule (the
TaskContext never carries the registry — eval/policies.py:36 discipline
extends to the offline replay). The judge-labelling arm (025) remains
STOPPED and is unaffected. The sealed pool remains sealed until §5 stage 2's
final table.

## Amendment A (2026-08-19, same day — stage-1 NO-GO as-run; owner decision: both-sequenced)

Written AFTER the stage-1 numbers and BEFORE any v2-feature number exists.
Nothing in §1-§7 is edited; §6's stage-1 criterion stands and was NOT met.

**(1) Stage-1 result, as-run** (results/offline_headtohead_stage1.json,
1415 runs, 5 folds, frozen grid): detector F1 0.507 vs random-budget-matched
0.582 and trivial always-ask-style baselines 0.800 — NO-GO (bar was 0.90).
Context findings: 40/60 tasks contain a fork at 24 rollouts (the task-level
metric is prevalence-saturated — itself a census finding); on true-positive
fires, attribution 13/18 (the firing bucket contained the forked runs) and
median lead +3 turns. A driver bug (boolean-mask fold indexing) was fixed
after the freeze; it is in the DRIVER, not the frozen detector, and preceded
any successful run.

**(2) Diagnosis (measured, not asserted).** The v1 hashed char-3gram
r-vectors carry almost no resolution signal: same-class commitment pairs sit
at mean distance 0.783 vs different-class 0.834 (unit geometry: orthogonal =
1.414), separation AUROC 0.555. The benign reference calibrated to ~1.13 —
near saturation — so the CUSUM had nothing to work with. The trigger
machinery itself is validated (contract tests: planted forks fire, twins
quiet). This is spec B2 caveat #1 realized in full.

**(3) Owner decision (in-session): BOTH, SEQUENCED.** (a) The fallback
science paper (§4's pre-declared framing, now enriched: the fork signal is
absent from single-run activations (026) AND from surface-form behavioral
hashes (this entry) — it lives in semantically-interpreted ensemble
behavior) starts drafting immediately; it is the paper's first half
regardless. (b) ONE pre-declared feature iteration runs in parallel, gated:

- **v2 features**: r_vec = a local sentence-embedding (CPU,
  sentence-transformers/all-MiniLM-L6-v2 via transformers, mean-pooled,
  l2-normalized, deterministic) of the mutating turn's
  `subgoal + action_text + error_signature`. ONLY r_vec changes; topic
  vectors, trigger, grid, folds, metric, and every §6 criterion stay frozen.
- **HARD GATE, frozen now**: v2 same-vs-diff class separation AUROC ≥ 0.75
  on the same all-task commitment-pair pool that scored v1 at 0.555 (a
  feature-validity measure independent of detection thresholds; eval-fold
  DETECTION numbers remain untouched until the gate passes). Gate fails →
  automatic fallback, no further feature iterations, mechanism section
  dropped from the paper.
- Gate passes → stage-1 reruns once with v2 under the §6 criteria unchanged,
  reported alongside v1 whichever way it lands.
