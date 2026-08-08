# 022 — Phase 4 eval build: engine choices and pre-registered eval protocol

---
status: agreed (owner 2026-08-07: bridge=vLLM, judge=local vLLM, N deferred to
post-R1; protocol rules a–i below proposed by the build and ratified with the
approved build plan — owner may amend any of them BEFORE the sealed run)
date: 2026-08-07
---

Builds the deliverables listed in decisions/021 §8: the Phase-4 eval harness
(`specs/eval.md`), the live wiring of Part B, the server bridge (decisions/019
"bridge rows"), and the scaffold-robustness check. Building now is sanctioned by
decisions/011 (A4 gates are a *usage* gate); **running the sealed eval still
waits for the A4 gates on real data + owner sign-off.** Everything in this build
is CPU-validated (fakes, fixtures, the 14B artifact set); every GPU action lands
in AWS_RUNBOOK step 5.

## 1. Owner decisions (2026-08-07)

**Bridge engine = vLLM serve.** The no-activation arms (no-ask / full-info /
naive-ask) run inside hil-bench's own harness against a vLLM server hosting our
backbone (`hosting.type: self_hosted`), per decisions/019 tier-3. Same weights,
same pinned sampling params as the corresponding our-loop arms; the inference
engine differs from our in-process HF path — **disclosed**, not hidden. The
detector arm always runs in-process through `HFStreamReader` (activation reads
require layer hooks); it is never routed through vLLM. Rejected alternative: a
thin OpenAI-compatible wrapper around `HFStreamReader` (byte-identical
generation path, but no KV/prefix reuse — SWE-Agent resends the full history
every step, making bridge rows a multi-day 100+ GPU-hour item for zero expected
information gain over the disclosure).

**Judge hosting = local vLLM**, `casperhansen/llama-3.3-70b-instruct-awq` on
port **8808** — the vendored `judge_config.yaml` verbatim (model, temperature
0.05, OpenAI-compatible endpoint). One 96 GB card holds it (~40 GB AWQ). The
SAME endpoint serves both consumers: our loop's `ApiJudge`
(`JUDGE_BASE_URL=http://127.0.0.1:8808`) and their harness
(`--judge-config configs/hilbench/judge_config.yaml`). Comparability holds by
construction. Rejected: hosted API providers (none serve the AWQ quant in their
frozen config — a deviation on the metric-defining component).

**Eval N = config parameter, value deferred.** `configs/eval.yaml` ships
`n_runs: null`; `scripts/run_eval.py` refuses to start with null outside
`--smoke`. The value is pre-registered in a later decisions/ entry once the R1
diversity pilot shows how many runs a fork needs to surface. decisions/000 Q7's
N=4 remains the default candidate; smoke tests use N=4.

## 2. Pre-registered protocol rules (frozen BEFORE any eval task is run)

Per the brief's hazards (§5f) and 021 §7. None of these may change after the
sealed pool is unsealed.

- **(a) Committed-trajectory rule** (brief §5f-4, anti-luck-laundering): for
  every N-run arm, the pass's `resolved` is scored on ONE trajectory — the
  lowest-seed run that emitted the submit marker; if none finished, the
  lowest-seed run. Deterministic and fixed before outcomes are visible. A
  majority-vote patch may be logged as a secondary diagnostic, never headline.
- **(b) Ask dedup** (brief §5f-1, anti-question-spam): at most ONE answered ask
  per bucket per task for the detector arm (`inject_resolution` on answer;
  subsequent fires of the same bucket logged as `suppressed_refire`, counted in
  the interruption budget, not asked). B1/B2/B3 dedup on their own trigger key.
- **(c) Answer scope**: the judge's resolution is injected into **all N runs**
  of the task (method doc: "inject the human's answer into all runs, continue"),
  as a user turn before each run's next generation.
- **(d) Sampling**: single-trajectory arms (no-ask, full-info, naive-ask — in
  BOTH scaffolds) pin **temperature 1.0, top_p 1.0, top_k 0, min_p 0.0**,
  matching the vendored self-hosted Qwen reference config so bridge rows differ
  from our-loop rows in scaffold only. N-run arms (detector, B1, B2, B4) use
  the collection temperature ladder {0.7, 0.9, 1.1, 1.3} across the N runs —
  diversity is what the detector consumes, and every matched-N baseline shares
  the identical ladder, so compute AND diversity are matched.
- **(e) Nudge OFF**: the collector's DELIBERATION_NUDGE instructs "Do not ask
  for clarification; commit to your reading" — self-sabotage for any ask arm.
  `run_eval.py` asserts it is absent from every eval prompt.
- **(f) Question phrasing**: backbone-LLM-phrased from the divergent options
  (one low-temperature `generate_segment` call, recorded in the compute
  column), with a deterministic template fallback on bad generations —
  ClarifyGPT precedent for LLM-phrased clarifying questions; HiL-Bench's own
  ask arm is model-phrased. Registry text (descriptions, resolutions,
  example_questions) NEVER enters the phrasing prompt (leak rule; asserted in
  contract tests).
- **(g) Structural vs value forks** are separate pre-registered slices (021
  §7): the blocker→{structural|value} map is frozen in
  `data/fork_type_annotations.json`, committed and owner-signed BEFORE the
  sealed pool is unsealed. (Derivation from the interpretation-class artifact
  kinds, hand-audited — same workflow as decisions/005.)
- **(h) Bridge scope**: SWE only for now. SQL bridge rows (business-info
  server, sql configs) deferred until the sql OOD story is set (021 R3).
- **(i) B4 random arm** asks at uniformly random rounds with its per-task ask
  budget matched to the detector arm's measured asks/task (budget-matching is
  computed from the detector arm's logs of the SAME run set — B4 is therefore
  scored last; this uses only ask COUNTS, never gate numbers or outcomes).

## 2b. Addendum (2026-08-08, owner question): eval step budget = 100, matched

The owner flagged that cap-40 our-loop arms could not beat published scores if
runs never finish. Verified step budgets elsewhere: **HiL-Bench's own harness
defaults SWE-Agent to `max_steps=200`** (vendored agents.py:167 — the Table-1
protocol); **mini-swe-agent** SWE-bench config caps at **250** steps / $3
(cost binds first); **SWE-agent** (NeurIPS 2024) caps by cost (~$2–4,
auto-submit); **OpenHands** standard SWE-bench Verified config is
**`--max-iterations 100`**, and published ablations show 50→100 already
saturates (identical resolve rates). DECISION: the EVAL step budget is
**100 on both scaffolds** — `configs/eval.yaml max_steps: 100` and
`--max-steps 100` on the bridge invocation — so the scaffold delta cannot be
a step-budget delta; below-their-200 is disclosed. COLLECTION caps are
unchanged (R1 stays at the pre-registered 40; R1's stop_reason distribution
decides whether R2 raises it — a nuisance-parameter fix, not gate tuning).

## 3. Metric plumbing decisions

- **pass@3** semantics are the verbatim port of `run_hil_bench.py
  ::summarize_rows` (@352d14c) into `src/xtid/harness/passk.py` — including the
  `infra_error` filter, `trajectory_needs_rerun` exclusion, ordered-first-k
  "any resolved", and the `num_valid >= expected_passes` inclusion rule —
  applied IDENTICALLY to our-loop arms and bridge rows. Test execution
  (`resolved`) is always their code, invoked: `calculate_pass_at_1` over a
  materialized flat tasks dir + SWE-Agent-shaped preds.json.
- **Ask-F1 for bridge rows is recomputed by us** from the harness's
  `ask_human_logs.json` plus the true per-task registry blocker counts, through
  the already-ported `compute_hil_metrics`. Reason: upstream
  `batch_runner.py:1179` imports a nonexistent `compute_zero_hil_metrics`, so a
  zero-question ask pass silently contributes `n_blockers_present=0` to the
  recall denominator; and `summarize_rows` emits ask metrics only for the
  literal mode name `ask_human`. Same formula, correct denominators, one code
  path for both scaffolds.
- **third_party/ is never modified.** Bridge configuration enters through the
  harness's own CLI surface: `--config-mapping` and `--judge-config` take paths
  (ours live in `configs/hilbench/`). The litellm cost-limit path is benign for
  self-hosted models: an unknown model name auto-DISABLES the per-instance cost
  limit (verified in swe.py — "disable cost limit to avoid
  ModelConfigurationError"); run length is bounded by `--max-steps`.
- Bridge rows target the sealed pool via `python -m hil_bench.cli swe` over a
  materialized flat tasks dir (`scripts/materialize_hilbench_tasks.py`) —
  `run_hil_bench.py` re-downloads the HF dataset and can only take the first N
  datapoints, so it cannot select swe_60+.

## 4. What this unblocks / what stays gated

Built and CPU-green after this build: artifact loader → live detector runtime,
turn-stepper with ask/answer injection, question assembly, the 8-arm
orchestrator, pass@k port + scoring + headline-table assembly, bridge configs +
materializer, judge validation harness, scaffold-robustness analyzer, eval
smoke. Still gated on gates + sign-off: any GPU eval execution, the sealed run,
and every number in the headline table.
