# 029: Counterfactual Information Sensitivity — a labeler-free, per-turn ground truth for need-to-ask, and the structural-layer probe against it

Date: 2026-09-04. Status: AGREED (owner approved the design in-session,
2026-09-04, after an adversarial pass; three owner decisions recorded in §3).
Follows 028 Amendment H.1 (the `full_info` arm: information is first-order on
swe_0, competence caps swe_10/swe_11) and the 2026-09-01→09-04 diagnostic
chain that killed every existing label (positive control, per-blocker
anchor, replay-and-diff, canonical accuracy, test-outcome vector).

**Pre-registration timing.** Nothing below has been run. This entry freezes
the target's definition, the injection protocol, every control, the probe
estimators and splits, the null distributions, the gates and their
consequences, and the interpretation, BEFORE any CIS number exists. Outcomes
land as Amendment 029.1 appended below; §1–§8 are never edited.

**Boundary.** This is MEASUREMENT of where the need-to-ask signal lives — not
a trigger, not an ask policy, not a lead-time mechanism. It revisits the
lead-time question 027 Amendment A retired, with a **new ground truth**; the
target is derived from teacher-forced log-probabilities under an injected
context, not from `r_vec`, `AskTrigger` or the CUSUM, so it is not the
feature iteration 027 forbade. The sealed pool (swe_60+) is untouched. Every
number is reported beside its baseline, its ceiling and its null. A strong
Stage-2 result is reported as *decodability with lead-time anatomy*; any
trigger built on it is a separate pre-registration.

## 1. Why

The paper's hypothesis is that the need-to-ask signal lives in the agent's
internal state before it commits. The structural-layer test of it
(T1·R1/R2) sits at permutation null (026 B: Stouffer p=.339; layer 3 only —
`decisions/026:344`). That null is **uninterpretable**, because every label
the probe could be scored against has since been shown dead:

| label | why dead | artifact |
|---|---|---|
| lexicon same/different | the pair-separation statistic fails its positive control: .788/.849 on "different repositories", .589/.474 on a planted fork | `results/diag_positive_control.json` |
| lexicon canonical-correctness | index-0 confound: the labeler picks the class with more signatures 52.5% vs 38.0% chance | `results/canonical_accuracy.json` |
| Fable judge | same .52–.58 band, independent instrument | `results/judge_arm_phase1_full.json` |
| replay-and-diff exact match | 0/777 same-task non-empty pairs identical; ~12% line overlap | `results/diag_replay_empty_diffs.json` |
| test-outcome vector at baseline | constant zero on 2/3 tasks; 0/67 runs solve | `results/test_outcome_vector.json` |

Every one labels a run by **what it did**. On a model that solves nothing,
that has no variance. 028 Am.H.1 showed the ambiguity *is* first-order on
swe_0 (0/21 → 8/23, p=.003) and that swe_0 had **zero lexicon forks** —
every run chose the same wrong class on 3 of 4 blockers. Divergence-based
labels are structurally blind to the case where asking helps most.

Activations exist for 1,415 runs × 8 layers × every turn. What is missing is
a per-run, per-turn truth that (a) needs no success, (b) needs no labeler,
(c) sees consensus-wrong-defaults, (d) is turn-indexed.

## 2. Definition

For a recorded run, turn *k* (segment index), blocker *b* of its task:

```
CIS_bash(k, b) = Σ_{t ∈ block_k} [ log p(t | ctx_k + R_b, t_<)  −  log p(t | ctx_k, t_<) ]
```

- `block_k` = the tokens of the **last ```bash block** of the recorded
  segment k (`agent_loop.parse_action`'s block; the behavioural commitment).
  Segments with no block score the whole segment and are flagged
  `no_block`.
- `ctx_k` = the exact message list the model saw before segment k, rebuilt by
  driving `wta.agent_loop.run_agent` over the recorded segments with the
  recorded commands re-executed in the task image (`wta.cis_context`).
- `R_b` = the registry resolution of blocker b,
  `<task>/shared/ask-human-data/blocker_registry.json[…].resolution`, used
  **verbatim** (`.strip()` only). It is the string the benchmark's ask_human
  server returns to an agent that asks, and the string `full_info` renders
  (byte-exact, 60/60 tasks; `wta.cis_registry` asserts both at load).
- Injection: appended to the trailing user turn of `ctx_k` with the
  stepper's merge rule, wrapped as
  `"The following clarification is provided by the task author to help you complete this task:\n\n" + R_b`.
  One wrapper for every variant (own / foreign / rival) so it cancels in
  every contrast.
- Both terms are teacher-forced forward passes over the same rebuilt
  context (`wta.cis_scorer`, branched KV cache from G_k = history through
  assistant_{k−1}). No sampling.

Secondary quantities, reported not gated: per-token most-negative shift
inside the block (length-robust); whole-segment sum; the baseline
surprisal `S_k = Σ log p(t | ctx_k)` over the block.

**Probe target.** Differenced against matched foreign resolutions, so that
"any injected normative text shifts improbable actions" is removed before
anything is probed:

```
T(k) = max_b [ −CIS_bash(k, b) ]  −  mean_{b' ∈ foreign(k)} [ −CIS_bash(k, b') ],   task-centred
```

`max` over blockers because one action resolves 2+ blockers 50.6% of the
time (`results/diag_per_blocker_anchor.json`). Raw `max_b[−CIS]` is a
secondary target; the per-blocker vector is a Stage-1 object only.

**Foreign controls** (`cis_registry.foreign_controls`): 2 per (run, k), seed
0, shared across the turn's own blockers; from other tasks; matched on
blocker `type`, backtick presence, word length ±25%; no identifier of the
control appears in the target task's instruction; source task shares ≤2
identifiers with the target's instruction (same-repo exclusion —
`metadata.json` has no repo field). Fallback relaxations are deterministic
and logged.

## 3. Owner decisions (2026-09-04)

1. **Rival-resolution sign control (G0.11):** Claude drafts one rival
   resolution per non-canonical class for the 11 pilot blockers (23
   entries, `data/cis_rival_resolutions_pilot.json`); only
   `status: approved` entries are loaded; the fixture is committed with
   this entry, before any CIS number exists. **As-executed (2026-09-04,
   same day, before any number):** the owner delegated the line-by-line
   review to an independent reviewer agent launched from a clean context
   with only the task and the source files. Verdicts: 13 approved as
   drafted, 10 revised with the reviewer's replacement text (originals
   retained in the fixture), 0 rejected. Two pairs of rivals that were not
   mutually exclusive as drafted were fixed; four technical errors an agent
   could react to were removed; all 23 sit within the ±25% length window.
   The reviewer's caveats are recorded in the fixture's
   `_provenance.review` and bind the reading of G0.11: several rival
   classes are *families* (one text pins one member, attenuating the
   paired sign for runs labelled via another member), and the no-limit
   classes contradict an explicit instruction. The pilot must record the
   lexicon-committed class per unit; per-signature variants are a
   sequenced-after item if off-member units are common. The owner retains
   veto over any entry.
2. **Timing:** 029 opens now, in parallel with the pending 18-task
   test-outcome extension (no shared inputs).
3. **`R_b` policy:** verbatim. The defective ids (literal `\n` / `\"` /
   `\\`, the `.",` leak in `swe_38/ambiguous_expired_entry_refresh_failure_behavior`)
   are pinned as a list in `data/cis_registry_pins.json`, together with
   every resolution's sha256 and the id order per task.

## 4. Frozen protocol

Identical to the R2 collection in every respect the model can see, except
that contexts are rebuilt and the model is scored, not sampled:

- Model `Qwen/Qwen3-32B`, bf16, `device_map=auto`, tokenizer via
  `hf_reader._load_tokenizer` (Amendment F). `enable_thinking=False`.
- Context: `SYSTEM_PROMPT`; instruction = `baseline/instruction.md` +
  `INSTRUCTION_SUFFIX` + `"\n\n"` + `DELIBERATION_NUDGE` (R2 ran the nudge
  ON); observations `[exit N]\n{truncate_obs(out,1500,500)}\n\nNext step?`
  with **N = the recorded exit code** (`error_signature`) and the replayed
  text; the collector's no-bash-block nag; `TASK_DONE` stop.
- Prompt render: `apply_chat_template(messages_k, add_generation_prompt=False)`
  + `"<|im_start|>assistant\n<think>\n\n</think>\n\n"`, asserted equal to the
  collector's `add_generation_prompt=True` render every turn (the manual
  concatenation is wrong on the 102 runs with a leaked `</think>`).
- Segment ids: `tokenizer(segment, add_special_tokens=False)`. Block span
  via `labeling.token_char_positions` + `agent_loop._BASH_BLOCK`.
- Observation capture: in-container redirect to files, exit code from a
  trailing marker, 4 MB per-stream cap; `marker_approx` recorded above the
  cap (the `truncate_obs` marker embeds the full length).
- Features (baseline branch only): residual stream at the 8 captured layers
  `[13,19,26,32,38,45,51,54]`, at `P_k−1` (last header token; primary) and
  `P_bash−1` (token before the last "```bash"; secondary), float16.
- Context cap 65,536 prompt tokens; units above recorded
  `unscored_context_cap`.
- Universe guard: the full run aborts unless 1,415 runs / 41,538 segments
  are present; the pilot unless 67 / 1,845. Resumable per run. Fresh
  `results/cis_*.json` files; nothing prior overwritten.
- Data paths on the box: `/ssd/wta_data/a0_v3_32b`, `models/v3_32b_fixed/`,
  `/ssd3/wta_data/a0_full_info_32b`; work dir `/ssd3/wta-cis/<tag>/`.

## 5. Stage 0 — target validity gates (ALL must pass before Stage 2 is fit)

| gate | read-out | fixed today | bar | on failure |
|---|---|---|---|---|
| G0.1 | rebuilt prompt == collector render, token-for-token, swe_0-s0/s3/s7 (laptop, in the suite) | 115/115 turns | = 100% | bug; nothing runs |
| G0.2 | exit-code fidelity, pilot replay | .9936 (replay pilot) | ≥ 0.95 | replay unusable |
| G0.3 | double-replay agreement of CIS_bash: Spearman ρ / median \|Δ\| | — | ρ ≥ 0.95 | context too unstable to score |
| G0.4 | branched-cache vs from-scratch, 20 units: max \|Δ\| block sum / ρ | tiny-model test passes | ≤ 0.05 nats and ρ ≥ 0.999 | scorer bug |
| G0.5 | recorded NPZ read vs fresh state, layer 32, token_idx 7 of segment k: median cosine | — | ≥ 0.95 | rebuilt context ≠ what the model saw |
| G0.6 | **relevance:** own-vs-foreign AUROC of max_b\|CIS\| on mutating turns, paired task-clustered CI | — | ≥ 0.65, CI excludes 0.5 | instrument in the .54–.60 band → target invalid |
| G0.7 | **sign (free):** lexicon-labelled *non-canonical* commitments with CIS_bash < 0 at their commitment segment | H.1 predicts ≈ all on swe_0 | ≥ 75%, binomial p < .05 | CIS does not track truth |
| G0.8 | **redundancy (free):** median \|CIS\| on the `full_info` arm ÷ baseline arm, same 3 tasks | — | ≤ 0.5 | injection channel not doing what it claims |
| G0.9 | variance decomposition of CIS_bash: own-vs-foreign contrast share vs covariate share {task, k, log tokens, is_mutating, S_k} | — | contrast ≥ covariate | target is nuisance |
| G0.10 | \|Spearman(S_k, CIS_bash)\| on mutating turns | — | > 0.3 → raw CIS disqualified; only T(k) probed | (rule, not a stop) |
| G0.11 | **rival sign (pilot only):** for runs committed to rival class c, CIS under c's rival text > CIS under canonical text (paired) | — | ≥ 75%, binomial p < .05 | rival texts unusable; G0.7/G0.8 carry the sign question |

Commitment turns for G0.7/G0.11 come from `commit_rounds` converted through
`actions[task][run][k].segment_idx` (list position → segment; 16.6% diverge).

## 6. Stage 1 — the target's anatomy (a result on its own; no probe)

- CIS_bash vs (segment − commitment segment), per blocker `type` (missing
  parameters 85 / ambiguous requirements 61 / contradictory requirements 68)
  and per fork type.
- **Frozen lead-time baseline (segment units):** 1,424 datable (run, blocker)
  commitments; commitment segment median 12 (p25 6, p75 20); turns strictly
  before: 99.6% ≥1, 94.4% ≥3, 83.9% ≥5.
- **Pre-registered prediction:** HiL-Bench's blockers "surface only through
  progressive exploration"; CIS should be ≈0 in early exploration and rise
  before commitment. Flat, or moving only *at* commitment → no lead time
  exists to detect.
- The swe_0 consensus-wrong-default case as a worked example.

## 7. Stage 2 — the structural-layer probe

- Unit = (run, turn) on **mutating turns** (primary universe; all turns
  secondary). Features = the fresh pre-action residual, 8 layers × 2
  positions; primary = layer 32 at `P_k−1`. Target = `T(k)` binarised at the
  within-(task × universe) median (top-quartile reported) for
  `LogisticRegression(max_iter=2000)` scored by the tie-aware `auroc_from`;
  and raw, task-centred, for ridge with alpha ∈ {1e-2,…,1e4} chosen by
  inner group-5-fold on train tasks by Spearman. Train-only
  standardisation. The first regression in this repo.
- Splits: leave-task-out via `kfold_group_indices` (masks). The
  train-fold-vs-eval-fold check is reported (STATUS precedent .517/.507).
- Nulls: run-level permutation of the target within (task × k-bucket),
  keeping each run's sequence intact; Stouffer sum vs its own permutation
  distribution (`gate5_permutation_test.py:205-215`). With a continuous
  target every run-permutation is distinct (n_runs! arrangements), so the
  026 testability floor does not bind. **Temporal shuffle is not a null**
  (it destroys the k-profile Stage 1 measures) and is not used.
- Baselines from the same units and folds: covariate-only probes on
  {k, log block tokens, is_mutating, S_k}; nuisance probes from the same
  features for task identity (gate1: .81–.92 — the ceiling), k, segment
  length.
- **Bars (frozen):** AUROC ≥ 0.70 with task-clustered CI excluding 0.5
  **and** paired clustered-bootstrap margin over the covariate baseline
  ≥ 0.10 (paired on identical units and folds) **and** Stouffer p < 0.01.
  The 8-layer sweep's maximum is tested against the permutation null of the
  maximum. Below any bar → reported as a negative with identical rigour.
- Lead time: the earliest held-out turn at which the probe's score exceeds
  a threshold calibrated on the null of the same max-over-turns statistic
  (95th percentile under run-level permutation within task), reported
  against a firing-rate-matched random-turn baseline. Never an uncorrected
  "earliest turn above threshold".

## 8. Pre-registered interpretation (fixed before any number)

1. **Any G0 gate fails** → the counterfactual is not a usable label on this
   model; reported with the failing number; no probe is fit.
2. **G0 passes, Stage 1 flat or moving only at commitment** → there is no
   lead time to detect; the premise that a pre-commitment signal exists is
   falsified for this model.
3. **G0 and Stage 1 pass, Stage 2 fails** → the agent's behaviour is
   measurably sensitive to withheld information but its pre-action state
   does not encode that sensitivity — the clean negative R1/R2 never was,
   because the label is finally valid.
4. **All pass** → the need-to-ask signal is linearly decodable from
   pre-action internals with lead-time anatomy, against a labeler-free,
   intervention-defined truth. Reported as decodability, not as a trigger.

In every case CIS itself is a contribution: a per-turn, labeler-free ground
truth for need-to-ask on any benchmark that ships resolutions, at
forward-pass cost.

**What this cannot settle.** One model family, one benchmark, teacher-forced
(not sampled) actions, the prose-injection channel only; competence-limited
tasks contribute ≈0 by construction. **Sequenced after, not bundled:** the
rival-text scale-up beyond the pilot; the second family via T7 once resumed
(stalled at 885/1440); any trigger, as its own pre-registration.

**Stop rules.** Every cell lands as-run. No rerun with variations without
amending this entry first. Driver scripts in `scripts/`, outputs in fresh
`results/cis_*.json`, STATUS.md updated in the same commit.

## Pilot (the first thing that runs)

The 3 already-replayed python tasks (swe_0, swe_10, swe_11; 67 baseline runs)
plus the 68 `full_info` runs of the same tasks. Reads every G0 gate, the
variance decomposition, and tokens/second per unit for the full-run
estimate (≈0.45B tokens by the branched-cache count). Stage 2 does NOT run on
the pilot (3 tasks give no probe number).

## Amendment 029.1 (2026-09-05 — Stage 0 pilot as-run: NO-GO)

Run exactly as pre-registered on the box: 3 pilot tasks, 67 baseline runs
(1,845 units, 559 mutating), an independent replicate into a separate work
dir, and 68 `full_info` runs. Nothing tuned, nothing rerun with variations.
`results/cis_stage0_pilot.json`.

**§8 interpretation 1 fires: four G0 gates fail, so the counterfactual is not
a usable label on this model. No probe is fit. Stage 2 does not run.**

### 029.1.1 Every gate beside its frozen §5 bar

| gate | bar (§5, frozen) | as-run | verdict |
|---|---|---|---|
| G0.1 rebuilt prompt == render | = 100% | 17/17 contract tests green on the box | **pass** |
| G0.2 exit-code fidelity | ≥ 0.95 | mean **0.995**, frac≥0.9 = 0.985, n=67 | **pass** |
| G0.3 double-replay agreement | ρ ≥ 0.95 | ρ = **0.9994**, median \|Δ\| = 0.000, n=6,898 | **pass** |
| G0.4 branched-cache vs from-scratch | ≤ 0.05 nats **and** ρ ≥ 0.999 | max \|Δ\| block = **1.266 nats** (25× the bar), max tok 0.584, ρ = 0.9963 | **FAIL** |
| G0.5 recorded NPZ vs fresh state | median cos ≥ 0.95 | median **0.9999**, p10 = 0.9954, n=1,793 | **pass** |
| G0.6 relevance, own-vs-foreign AUROC | ≥ 0.65, CI excludes 0.5 | **0.855**, clustered CI **[0.826, 0.947]**, own \|CIS\| 5.28 vs foreign 1.35 | **pass** |
| G0.7 sign on non-canonical commitments | ≥ 75%, p < .05 | **95.2%** negative (99/104), p = 4.8e-24; canonical control 5.7% non-neg | **pass** |
| G0.8 redundancy, full_info ÷ baseline | ≤ 0.5 | ratio **0.578** (median \|CIS\| 3.054 vs 5.283) | **FAIL** |
| G0.9 variance, contrast vs covariate | contrast ≥ covariate | contrast increment **0.085** < covariates **0.109** | **FAIL** |
| G0.10 \|ρ(S_k, CIS)\| | > 0.3 disqualifies raw CIS | ρ = **−0.398** → **raw CIS disqualified**, only T(k) probeable | rule fired |
| G0.11 rival sign | ≥ 75%, p < .05 | **51.9%** (54/104), p = 0.384 — indistinguishable from a coin | **FAIL** |

### 029.1.2 What passed is not nothing

The instrument itself is sound where it was testable. Replay fidelity 0.995;
double-replay ρ 0.9994 with a median delta of exactly zero; the rebuilt
context matches the recorded activations at median cosine 0.9999. And the two
substantive validity gates that *did* pass are the interesting ones: CIS
separates a unit's **own** withheld resolution from a **foreign** one at
AUROC 0.855 (bar 0.65, CI well clear of chance), and on lexicon-labelled
**non-canonical** commitments it is negative 95.2% of the time against a 5.7%
canonical control, p = 4.8e-24. Sign and relevance both hold decisively.

### 029.1.3 Why it is still a NO-GO

Four independent failures, and they are not near-misses:

- **G0.4 is a scorer defect, not a threshold quibble.** 1.266 nats against a
  0.05 bar is 25× over. The branched-cache path does not reproduce
  from-scratch scoring at the block level even though rank order survives
  (ρ 0.996). Every downstream number is computed through that path.
- **G0.9 says the target is nuisance.** The own-vs-foreign contrast adds
  R² 0.085 on top of covariates {task, k, log tokens, is_mutating, S_k},
  which alone explain 0.109. The thing the gate exists to prevent — a target
  dominated by position and length rather than information — is what the
  decomposition found.
- **G0.11 removes the independent sign check.** Rival-text direction is
  51.9%, a coin flip. G0.7's 95.2% therefore rests on the lexicon, the
  labeler this whole program has been trying to escape, so the sign evidence
  is not labeler-free after all.
- **G0.8 at 0.578 vs a 0.5 bar** is the closest to its bar and the least
  alarming on its own, but it says the injection channel removes only ~42%
  of the signal it is supposed to remove.

G0.10 additionally disqualifies raw CIS (ρ = −0.398 with surprisal, bar 0.3),
so even a repaired instrument would only license probing T(k), not CIS itself.

### 029.1.4 Cost, measured

1,845 units at **3.04 s/unit** of scoring (5,605 s score + 339 s replay).
Projected full run **≈35 GPU-hours**. Recorded per §"Pilot" so the number
exists; it is not a recommendation to spend it.

### 029.1.5 What this does not say

It does not say CIS is a bad idea — G0.6 and G0.7 are genuine positive
evidence that the quantity tracks something real. It says **this
implementation, on this model, is not a usable label yet**: one gate is a
reproducibility defect that is likely fixable, one is a validity failure that
may not be, and the independent sign check did not survive. Per §8 and the
stop rules, nothing is refit and no variation is rerun without amending this
entry first.

## Amendment 029.2 (2026-09-05 — correction to 029.1's reading of G0.7, and the coherent diagnosis)

No new numbers. This amendment corrects an interpretation in 029.1 from the
same `results/cis_stage0_pilot.json`, states what the eleven gates say when
read together, and names what may be pre-registered next. Nothing runs
before that pre-registration (029.3) exists.

### 029.2.1 The correction: G0.7's control shows the sign does NOT track correctness

029.1.1 reports G0.7 as "95.2% negative (99/104); canonical control 5.7%
non-neg" and 029.1.2 calls it a decisive pass. The control number was read
backwards. `frac_canonical_nonneg = 0.057` means **2 of 35** commitments the
lexicon labels as *canonical* have CIS ≥ 0 — **33 of 35 (94.3%) are
negative**. Injecting the correct answer makes the correct action less likely
94% of the time, exactly as it does for the wrong action (95.2%). The sign of
CIS therefore does not discriminate canonical from non-canonical
commitments. **G0.7 passed its written criterion (≥ 75% of non-canonical
negative) and failed its intended meaning.** 029.1.2's sentence "sign and
relevance both hold decisively" is withdrawn as to sign; relevance stands.

G0.11 says the same thing from the other side: rival text vs canonical text,
paired on the run's own committed class, is 51.9% — a coin. Both sign checks
agree; there is no sign.

### 029.2.2 What the gates say together

`CIS_bash` is a large, generic **negative perturbation** — any injected
resolution lowers the likelihood of the exact recorded block — **scaled by
relevance**. Own resolutions move it ~4× more than length/type/code-matched
foreign ones (median |CIS| 5.28 vs 1.35; paired AUROC .855, clustered CI
[.826, .947]). That relevance signal is real, replicable (G0.3 ρ .9994), and
computed on a context that reproduces the model's recorded internal state to
cosine .9999 (G0.5). But correctness does not enter the sign, so `CIS_bash`
is a **relevance detector, not a correctness detector**, and "large-negative
= should have asked" (§2) is not licensed.

**Mechanism.** Teacher-forcing the *exact recorded tokens* is a surface
measure. Told the answer explicitly, the model would write even a
semantically identical block differently — names, comments, structure — so
the recorded token sequence becomes less likely whether or not the decision
was right. Relevance survives because it drives magnitude; correctness is
swamped. This is the same surface-variance-over-semantic-signal failure that
killed the six pairwise instruments (STATUS gap 8), reappearing in
likelihood space. G0.9 (contrast increment .085 against covariates .109 —
*comparable*, not "pure nuisance" as 029.1.3 put it; the gate failed as
written) and G0.10 (ρ(S_k, CIS) = −.40) are the same fact seen through two
other gates: length and surprisal of the block co-vary with how much any
perturbation can move it.

**G0.4 is numerical, not scientific.** 1.266 nats against a .05 bar with
rank order intact (ρ .996) and the branched path self-reproducible to a
median Δ of 0.000 (G0.3) is bf16 kernel non-determinism between a
cached-prefix forward and a full-sequence forward over 12k+ tokens. The bar
was set for fp32-style equality and was wrong for bf16 at 32B. The
from-scratch forward is not the better reference: the collector *generated*
with a KV cache, and G0.5 shows the branched path reproduces those recorded
states at .9999. G0.4 stands as failed as pre-registered; a corrected G0.4′
(compare `CIS_branch − CIS_scratch`, where the systematic offset cancels,
with a bf16-appropriate bar) may be pre-registered in 029.3 and applies only
forward.

### 029.2.3 Assets that survive

The context-rebuild pipeline (`wta.cis_context`), validated end-to-end
against the original activations (G0.5); replay reproducibility (G0.3); the
branched teacher-forced scorer (`wta.cis_scorer`); the relevance quantity
`T_rel(k) = max_b[−CIS_bash(k,b)] − mean_foreign[−CIS_bash]`; the registry
loader and the reviewed rival fixture. All reusable at forward-pass cost.

### 029.2.4 What 029.3 must pre-register before anything runs

The failure is in *what is scored* (exact tokens), not in the context, the
injection channel, or the model's sensitivity. The re-scoping to be frozen
in 029.3:

1. **A decision-level readout in place of exact-token likelihood.** At the
   pre-action position `P_k−1`, teacher-force one short templated
   *commitment statement per registry class* of each blocker (the same
   frame for every class; content from the class's own signatures — the
   reviewed rival fixture is the pilot prototype) and read the model's
   **implicit interpretation distribution** `p_k(c | b) ∝ exp log p(stmt_c |
   ctx_k)`. Derived per-turn labels: `P_k(canonical | b)` (lean toward the
   right answer) and the entropy of `p_k(· | b)` (uncertainty). The
   perturbation that swamped `CIS_bash` cancels: every option shares the
   template and the context; only the class content differs. Cost is ~30
   tokens per option on the cached prefix — roughly 1/50 of a CIS branch.
2. **Validity gates for the readout, all planted or free:** (a) the
   `full_info` arm (68 runs, already collected) must raise `P(canonical)`
   relative to baseline — the planted effect H.1 demonstrated behaviourally;
   (b) at lexicon-committed segments, the argmax of `p_k` must agree with
   the committed class above the lexicon's own validated agreement floor;
   (c) statements built from *foreign* tasks' classes must receive low
   probability (relevance, as G0.6); (d) G0.4′ as above.
3. **`T_rel(k)` retained as a second, already-validated target**, labelled
   as relevance, never as correctness.
4. **Stage 2 re-scoped** to three targets — relevance, lean-entropy, and
   `P(canonical)` — with §7's splits, nulls, covariate and nuisance
   baselines, paired-margin bar and null-calibrated lead time unchanged.
   The strongest result this can deliver, stated in advance: *the agent's
   pre-action internal state linearly encodes (i) that the withheld
   information is relevant now and (ii) its implicit lean among the
   registry's interpretations, with lead time before commitment.* That is
   the structural-layer question in two components. It is not a
   "should-ask" detector unless (ii) is shown to track the canonical class,
   which gate 2(a)–(b) decide.

Sections 1–8 and 029.1's numbers are untouched.
