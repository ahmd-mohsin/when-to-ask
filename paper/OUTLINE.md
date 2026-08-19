# Paper outline — "Where does an agent's need-to-ask live?"

Status: skeleton, 2026-08-19. RESOLVED same day (027 Amendment B): the
embedding gate FAILED (v2 AUROC 0.580 vs bar 0.75) → the fallback IS the
paper; §7 is dropped to future work and the claim sharpens to the
THREE-negative arc (internals → surface hashes → semantic embeddings).
Every number cited here already exists in the repo (source noted per item). Venue targets per the
novelty sweep: ICLR/NeurIPS/ACL analysis track (primary), NeurIPS D&B or
FSE/ASE (census-led variant). Submission window: 2-4 months (sweep verdict —
Snowflake 2602.11619 and the Ask-or-Assume group are each one increment away).

## Working titles

- Where Does an Agent's Need-to-Ask Live? Interpretation Forks, Internal
  States, and the Behavioral Ensemble
- Asking Requires an Ensemble: A Controlled Study of Clarification Signals
  in Coding Agents

## Abstract skeleton

Agents on underspecified tasks should ask; when they don't, N parallel
rollouts silently resolve the same ambiguity in different ways
("interpretation forks"). We ask where the need-to-ask signal actually
lives. (i) CENSUS: at 24 rollouts on HiL-Bench SWE tasks, 2/3 of tasks fork;
forks are behaviorally committed (mutating actions), datable, and typed.
(ii) INTERNALS, controlled negative: linear probes and learned disentangled
spaces on single-run activations do not beat causal bag-of-words controls
under run-level permutation tests with testability floors — at 7B, 14B, 32B.
(iii) SURFACE BEHAVIOR: hashed representations of what runs literally did
also carry no class signal (AUROC 0.555). [(iv) MECHANISM, conditional:
semantic embedding of behavior does; a CUSUM trigger over ensemble
divergence detects forks with lead time and beats single-run introspection
at matched ask budgets.] Together: the signal is absent where the field has
been looking (single-run internals, single-run introspection) and present
only in the semantically-interpreted ensemble. + methodology contributions
(run-level permutation with exact tests; testability floors; label-coordinate
audit).

## 1. Introduction

Hook: Ambig-SWE (2502.13069) shows prompting cannot detect underspecification;
HiL-Bench shows frontier with-ask performance is dismal (<=9.4%). The
field's proposed detectors are single-run: introspection (2603.26233),
trained policies (2604.14624, 2511.02208), internal-state probes (agentic-UQ
wave). Our question: WHERE does the signal live? Contributions list (census /
controlled internals negative / surface-behavior negative / [mechanism] /
methodology toolkit + pre-registration chain).

## 2. Related work (map from the 2026-08-19 novelty sweep — full citations there)

- Agent clarification: HiL-Bench (2604.09408), Ask-or-Assume (2603.26233),
  CLARITI (2604.14624), PPP (2511.02208), SAGE-Agent/ClarifyBench
  (2511.08798), Ambig-SWE (2502.13069), SteerBench-Work (2608.12654).
- Divergence as signal: When Agents Disagree With Themselves (2602.11619 —
  closest prior: same signal, abstention not asking), Selective Rollout
  (2605.05802 — same primitive, RL termination), trajectory-consistency UQ
  (2608.11552 — uncontrolled version of our claim), ClarifyGPT (single-turn),
  self-consistency, semantic entropy.
- Probing methodology: Hewitt & Liang control tasks, Ravfogel; our run-level
  permutation + testability floors extend this to agent trajectories.

## 3. Setting and the fork census

- HiL-Bench SWE tasks + blocker registry; R2 collection: Qwen3-32B, 60 tasks
  x 24 seeds (temp ladder 0.7-1.3), 1415 runs, 683K reads, composite action
  events (files/region/subgoal/error-signature). [decisions/021, 023; HANDOFF]
- Interpretation forks: definition; commitment = first mutating action
  carrying a class signature; fork census: 63 forked blockers; 40/60 tasks
  fork at N=24 [models/v3_32b_fixed, 026 Amendment B; stage-1 JSON].
  Structural/value typing 161/53 [data/fork_type_annotations.json, 024].
- The prevalence finding: at N=24 the task-level "should we ask at all"
  question saturates (always-ask F1 0.8) — the real question is WHICH
  decision and WHEN [results/offline_headtohead_stage1.json].

## 4. Half A: single-run internals — a controlled negative

- The three-scale story: gate5 at its run-level permutation null at 7B/14B/
  32B; exact enumeration (2/35 at floor, ~10-13/35 testable); global Stouffer
  p=0.339 (raw), p=0.16 in the learned space with n_testable=0 — the
  testability-floor finding [026 Amendment B, results/*fixed.json,
  gate5_lhe_permutation.json].
- Causal lexical control: text beats the best activation probe in the hardest
  (causal + anchors-masked) variant, 0.730 vs 0.4102/0.2745 [gate2_text_
  control*.json]. Probe-family robustness appendix: full-dim + nonlinear
  probe rerun (TODO — the one remaining gate-2 escape; box, hours).
- The label-coordinate audit as a methods case study: three coordinate
  systems, 13.4% relabeling, debug-trail honesty property [026, AUDIT_R2].
- 14B LORO cautionary tale: majority baselines (0.729 vs 0.916) — why
  1/n-chance references mislead [026 §C / scale-story audit].

## 5. Half A': behavioral representations also fail — surface AND semantic

- v1 hashed behavioral vectors: same-class 0.783 vs diff-class 0.834,
  AUROC 0.555 — surface form of WHAT runs did carries ~no resolution signal.
- v2 SEMANTIC embeddings (MiniLM over subgoal+command+error): AUROC 0.580 —
  barely better; the pre-registered gate (0.75) failed and the mechanism was
  dropped as pre-committed [results/feature_signal_gate.json, 027 Am. A/B].
- Detection consequence: stage-1 F1 0.507 vs matched-random 0.582
  [results/offline_headtohead_stage1.json]. Positives reported honestly:
  attribution 13/18, lead +3 turns — the trigger machinery works when the
  features do; the features are the wall.
- The sharpened claim this sets up: the forks are REAL and datable (§3
  census), but recognizing "same resolution vs different" is not a generic
  representation problem — it requires decision-aware interpretation. A run
  cannot disagree with itself, and an ensemble cannot be compared without
  knowing what to compare.

## 6. Methodology toolkit (woven through, summarized once)

Run-level permutation with exact enumeration; testability floors as a design
diagnostic (n_testable=0 at ~6 runs/decision); majority baselines for
LORO-style diagnostics; coordinate-audit + reproducible-debug-trail property;
the pre-registration chain (011 -> 015 -> 021 -> 025 -> 026 -> 027 with
frozen go/no-go and as-run NO-GO reporting).

## 7. DROPPED per 027 Amendment B (gate FAIL) — recorded as future work

The divergence-ask mechanism moves to the discussion/future-work section:
the census proves the target exists; the three negatives bound what generic
detectors can see; closing the loop requires learned decision-aware fork
recognition (CLARITI-adjacent) — explicitly out of scope here. Stage-2, the
introspection head-to-head, and the sealed pool remain unspent.

## 8. Discussion

Where the signal lives and why single-run detectors are structurally blind
to interpretation forks (a run cannot disagree with itself); design guidance:
rollout budgets vs testability floors; prevalence-aware ask metrics;
limitations (one model family at scale; lexicon-labeled ground truth with
audited noise; offline lockstep vs live asynchrony).

## 9. Reproducibility statement

The decisions/ chain with dates; all-numbers-as-run including the NO-GOs;
code + census release plan (needs owner call on what ships).

## Asset -> section map (nothing here requires new compute except as marked)

| Asset | Section |
|---|---|
| models/v3_32b_fixed (box) + census JSONs | 3 |
| 026 Amendment B gate reports + permutation JSONs | 4 |
| gate2 text controls (3 variants) | 4 |
| full-dim + nonlinear probe rerun | 4 (TODO, box) |
| feature_signal_gate.json + stage-1 JSON | 5 (+7 if pass) |
| introspection arm (Fable subagents, chunked) | 7 (conditional) |
| Phase-4 eval harness | 7 (conditional, box) |
| novelty sweep output (w7n0rxmf0) | 2 |
