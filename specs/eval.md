# Spec — Phase 4 evaluation

Runs only after the A4 gates pass on real data and the owner signs off
(decisions/011). Everything here is measurement; nothing feeds back into
training.

> EDIT (decisions/022, 2026-08-07): this spec is updated for the Phase-4 build.
> Every change from the 2026-07-03 version is marked with an explicit
> `EDIT (decisions/022)` block per specs/README.md (no silent absorption).

## Metrics

- **Ask-F1** — HiL-Bench's metric, used verbatim via the xtid port
  (`src/xtid/harness/ask_f1.py`, from `hil_bench/utils/compute_hil_metrics.py`;
  reimplementing it would break comparability).
- **Pass@k** — HiL-Bench harness outcome per task.

> EDIT (decisions/022): pass@k made precise — **pass@3** (3 passes per task per
> arm), semantics ported verbatim from `run_hil_bench.py::summarize_rows`
> (@352d14c) into `src/xtid/harness/passk.py`: passes filtered by the
> `infra_error` status and `trajectory_needs_rerun` markers, ordered-first-k
> "any resolved", attempts included only when `num_valid >= expected_passes`
> (an `--include-partial` escape hatch mirrors upstream). Per-pass `resolved`
> always comes from THEIR evaluator (`calculate_pass_at_1` over a flat tasks
> dir + preds.json) — invoked, never reimplemented. For N-run arms, each pass
> is scored on a **single committed trajectory** (pre-registered rule,
> decisions/022 §2a): the lowest-seed finished run, else the lowest-seed run —
> so N-parallelism cannot launder a lucky pass (brief §5f-4). Ask-F1 is scored
> on the asking decisions.

- **Regime-sliced recall** — fork / confident-convergent / clear slices; the
  fork-blocker slice is where the method must win.

> EDIT (decisions/022): the fork slice is reported **split into structural vs
> value forks** as separate pre-registered hypotheses (decisions/021 §7), from
> the frozen `data/fork_type_annotations.json` committed before unsealing.

- **Lead-time** — gate-7 machinery (`wta/a4_gates.py::gate7_lead_time`) on
  real trajectory logs with real action reads (proxy=False).
- **Interruption budget** — asks per task, and asks per fired-bucket
  (question assembly quality is qualitative, reported with examples).

> EDIT (decisions/022): interruption budget also counts `suppressed_refire`
> events (bucket re-fires deduped by the once-per-fork rule), and a **compute
> column** (turns, generated tokens, question-phrasing calls, N) is reported
> beside Ask-F1 per the brief §5e.

## Baselines — matched compute (same N, same backbone, same tasks)

| baseline | source | status |
|---|---|---|
| Vanilla `ask_human` prompting | HiL-Bench's ask_human arm | our-loop arm (`model_initiated`) + bridge row |
| Output-divergence at matched N (B1) | ClarifyGPT consistency check, xtid port `xtid/signals/output_divergence.py` | ported |
| Single-stream should-ask probe (B3) | OPENIA linear probe, xtid port `xtid/signals/probe.py` | ported |
| Verbalized-across-N (B2) | xtid port `xtid/signals/verbalized.py` | ported |
| Random/uniform routing (B4) | trivial, budget-matched to detector asks/task | new |
| EigenScore internal divergence (reference) | eigenscore port `xtid/signals/internal_divergence.py` | ported |

> EDIT (decisions/022): B2 and B4 added to the table (the brief §5d ladder
> already listed them); B0 vanilla-ask runs in OUR scaffold as the
> `model_initiated` arm (core claims share one scaffold, decisions/019
> tier-1), and ALSO as a bridge row in their harness (tier-3).

## Protocol

- harbor_swe = development + eval split by task; the sealed eval pool is
  swe_60+ (decisions/018 §4 seal rule: once run, numbers are final).

> EDIT (decisions/022): the 2026-07-03 line "harbor_sql held out entirely" is
> amended per decisions/021 R3 — ~20 harbor_sql tasks are collected as the
> gate-6 OOD family; sql remains out of the headline table and sql bridge rows
> are deferred (decisions/022 §2h).

- All baselines consume the SAME logged trajectories where applicable
  (matched compute is by construction, not by budget accounting).
- Thresholds (theta, tau, s_ref, CUSUM reference) come from the offline
  calibration artifacts; nothing is re-tuned on eval tasks.

> EDIT (decisions/022): protocol additions, all pre-registered in
> decisions/022 §1–2:
> - **Within-task bucketing**: one fresh `AskTrigger` per (task, pass)
>   (decisions/018 gate-4 finding — naive cross-task topic bucketing mixes
>   decisions).
> - **Ask dedup**: max one answered ask per bucket per task; answers injected
>   into all N runs as a user turn.
> - **Sampling**: single-trajectory arms pin temperature 1.0 / top_p 1.0 /
>   top_k 0 (the vendored self-hosted Qwen reference config) in both
>   scaffolds; N-run arms use the collection ladder {0.7, 0.9, 1.1, 1.3}
>   across the N runs, identical for detector and every matched-N baseline.
> - **Nudge OFF** for every eval arm (the collection nudge forbids asking).
> - **Judge**: local vLLM of `casperhansen/llama-3.3-70b-instruct-awq` on
>   :8808 (their frozen config verbatim), shared by our `ApiJudge` and their
>   harness; our port's agreement with their `ask_human_server` is validated
>   on `data/judge_validation_pairs.json` BEFORE any ask row is trusted
>   (brief §5a).
> - **Eval N**: config parameter (`configs/eval.yaml: n_runs`), value
>   pre-registered after the R1 pilot; `run_eval.py` refuses null outside
>   `--smoke`.

## Bridge rows

> EDIT (decisions/022): new section; the bridge has its own component spec at
> `specs/eval-bridge.md`. Summary: no-ask / full-info / naive-ask also run
> inside hil-bench's own harness (SWE-Agent scaffold) against a vLLM server
> hosting our backbone, so the scaffold delta is measured, not argued
> (decisions/019 tier-3). Ask-F1 for bridge rows is recomputed by us from
> `ask_human_logs.json` + true registry counts (upstream zero-question bug,
> decisions/022 §3). The detector arm never runs through vLLM.

## Status

> EDIT (decisions/022): the 2026-07-03 "deliberately deferred" status is
> obsolete — the composite-label builder it gated on landed (commit 2979e1b).
> The Phase-4 components are: `src/wta/eval/` (artifacts loader, turn-stepper,
> question assembly, full-info augmentation, patch extraction, policies,
> orchestrator, scoring), `src/xtid/harness/passk.py`, `scripts/run_eval.py`,
> `scripts/score_eval.py`, `scripts/run_eval_smoke.py`,
> `scripts/materialize_hilbench_tasks.py`, `scripts/validate_judge.py`,
> `scripts/scaffold_robustness.py`, `configs/eval.yaml`, `configs/hilbench/`.
> All CPU-validated (contract tests + `run_eval_smoke.py`); GPU execution
> lives in AWS_RUNBOOK step 5 and stays gated on the A4 stop point.
