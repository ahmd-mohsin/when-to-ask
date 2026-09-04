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
