# Spec — Phase 4 evaluation

Runs only after the A4 gates pass on real data and the owner signs off
(decisions/011). Everything here is measurement; nothing feeds back into
training.

## Metrics

- **Ask-F1** — HiL-Bench's metric, used verbatim via the xtid port
  (`src/xtid/harness/ask_f1.py`, from `hil_bench/utils/compute_hil_metrics.py`;
  reimplementing it would break comparability).
- **Pass@k** — HiL-Bench harness outcome per task.
- **Regime-sliced recall** — fork / confident-convergent / clear slices; the
  fork-blocker slice is where the method must win.
- **Lead-time** — gate-7 machinery (`wta/a4_gates.py::gate7_lead_time`) on
  real trajectory logs with real action reads (proxy=False).
- **Interruption budget** — asks per task, and asks per fired-bucket
  (question assembly quality is qualitative, reported with examples).

## Baselines — matched compute (same N, same backbone, same tasks)

| baseline | source | status |
|---|---|---|
| Vanilla `ask_human` prompting | HiL-Bench's ask_human arm | harness exists (hil-bench clone) |
| Output-divergence at matched N (B1) | ClarifyGPT consistency check, xtid port `xtid/signals/output_divergence.py` | ported |
| Single-stream should-ask probe | OPENIA linear probe, xtid port `xtid/signals/probe.py` | ported |
| EigenScore internal divergence (reference) | eigenscore port `xtid/signals/internal_divergence.py` | ported |

## Protocol

- harbor_swe = development + eval split by task; **harbor_sql held out
  entirely** (OOD, gate 6 and the transfer limitation).
- All baselines consume the SAME logged trajectories where applicable
  (matched compute is by construction, not by budget accounting).
- Thresholds (theta, tau, s_ref, CUSUM reference) come from the offline
  calibration artifacts; nothing is re-tuned on eval tasks.

## Status / deliberately deferred

The orchestration that drives HiL-Bench end-to-end with the online trigger
(N live runs inside the executor + judge + answer injection) is the one
remaining integration, deliberately deferred until the first real A0 sample
exists: the composite-label builder (file + region + sub-goal + error
signature from logged actions) must be written against real log content, not
guessed (AWS_RUNBOOK.md step 3). The pipeline downstream of labels is fully
built and fixture-validated (`scripts/run_pipeline_smoke.py`).
