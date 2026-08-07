# 021: The real run — full scale audit and experiment design

Date: 2026-07-22. Status: PROPOSED (owner has heavy GPUs; awaiting hardware
parameters). Supersedes the re-collection sketch in decisions/020.

Owner's charge: *"make sure nothing is left small in the sense that doesn't let
the genuine hypothesis shine through."* This document re-reads the method spec
(`when-to-ask-offline-online (1).md`), the process brief, and every spec in
`specs/` against what the code actually does, and lists every place where a
pilot-scale compromise could mask a real effect.

## 0. Were the pilots useless? No — but they were pilots

They bought (all still valid): the read-position policy and its bugs; hook-based
capture that makes 32B memory-safe; the v2 agent collector; action-based
commitment labeling; the value-read finding (decisions/016); the structural-vs-
value asymmetry (structural forks separate at raw-h 0.71-0.73 at 14B, value
forks ~chance at 7B/14B); the real lead-time number (median K=10.5 reads, 100%
positive); and the zero-leak artifact hygiene contract. **What they never bought
is a powered test of the make-or-break claim** — and this document explains why,
mechanically.

Nothing below is a tuning knob discovered by looking at gate outcomes. Every
item is either (a) a spec requirement never implemented, (b) a documented sweep
never run, or (c) a measurement resolution limit. Gate numbers stay unseen until
the run is done.

---

## 1. CRITICAL — sampling diversity was clamped (the likely root cause)

**`hf_reader.generate_segment` passes only `do_sample` and `temperature`. It
never sets `top_p` or `top_k`, so generation inherits the model's shipped
`generation_config.json`.** For Qwen3, that config sets **`top_k=20`** (and
`top_p` ~0.95/0.8). With the candidate set truncated to 20 tokens, raising
temperature from 0.7 to 1.0 buys very little real interpretation diversity —
the model re-samples surface wording, not decisions.

This is the most plausible single explanation for the finding that drove
decisions/020: 16 of 18 forks had a **single-run minority**. We were asking the
model to diverge while capping its ability to.

- **MUST VERIFY FIRST (5 minutes on the box):** dump the effective config —
  `AutoModelForCausalLM.from_pretrained(...).generation_config` — and record it
  in the manifest. Everything downstream is conditioned on this.
- **Fix:** pass sampling params explicitly and record them:
  `top_p=1.0, top_k=0` (disabled), `min_p=0.0`, temperature per the diversity
  ladder below. Never inherit silently again.

Related diversity gaps:

- **Temperature range 0.7–1.0, tied to `seed % 3`** (`TEMPS`). Too narrow, and
  temperature is confounded with seed. **Fix:** decouple; ladder
  {0.7, 0.9, 1.1, 1.3} crossed with seeds, recorded per run.
- **The persona/framing nudge specified in the method doc was never built.**
  Method doc A0 states plainly: *"Force the runs to actually disagree by varying
  seed and temperature (and optionally a light persona nudge)."* v1.5
  (`collect_a0.py`) had a deliberation nudge with a `--no-nudge` ablation; **the
  v2 collector dropped it entirely.** `collect_v2.py` appends only a grounding
  sentence. **Fix:** reinstate a recorded, ablatable nudge, plus an
  interpretation-NEUTRAL framing variation (e.g. "prefer the minimal change" vs
  "prefer the general solution") — it must not name any interpretation class, or
  it contaminates the label.
- **N=8 seeds.** Even with sampling fixed, 8 runs puts a 25%-probability
  interpretation in ~2 runs with high variance. **Fix: 24 seeds** on fork-bearing
  tasks (gate5 needs ≥2 runs per class per fork).

## 2. CRITICAL — trajectory-length censoring

Measured this session: **387/480 runs (81%) ran to the `max_steps=15` cap**;
per-blocker commit rate is 26% (finished) vs 16% (truncated); **half the forked
blockers have ZERO finished runs**. Late-in-trajectory decisions are censored,
not converged. `max_new_tokens=1024` also truncated at least one run
mid-bash-block (swe_5-s2).

- **Fix:** `max_steps` 15 → **40** (SWE-Agent-shaped budgets are 30–100);
  `max_new_tokens` 1024 → **2048**; keep `exec_timeout=120` (a few runs hit
  300–630s wall, dominated by generation not exec).
- Consider raising obs truncation (1500/500) — large-file observations may be
  starving the context that a late decision depends on.

## 3. CRITICAL — read density is structurally too sparse for agent loops

`StreamReadSelector` is constructed **fresh per turn**, and cadence fires at
positions K−1, 2K−1, … *within the turn*. **A turn shorter than 32 generated
tokens produces ZERO reads.** Consequences measured: median **8 reads/run**,
**26% of runs have zero reads**, 45% have <5. A decision the model makes in a
short turn is invisible to us no matter how good the labels are.

Spec A0 mandates a cadence sweep `{16, 32, 64}` — **never run on real data.**

- **Fix:** cadence **8** for agent loops (with 2048-token turns this is still
  bounded), and actually run the sweep. Reads are cheap under hook capture
  (KB/step); this is the highest-value-per-GPU-hour change on the list.
- Cue set is only 6 words and fired on 91/9401 reads (1%). Qwen3 *does* emit
  cues (Qwen2.5 emitted none) — expand the cue set (a spec-flagged Phase-2 sweep).
- Value reads: decisions/016 found value lean is transient within ~12 tokens of
  emission; cooldown=8 is plausible but unswept.

## 4. HIGH — the composite decision label was never implemented

The method doc's labeling note is explicit: *"Do not label a decision by the
file alone — that conflates two decisions that touch the same file. Use a
composite label: file + code region/span + stated sub-goal + error signature."*
`logging_schema.py:33` documents exactly those four fields. But
`agent_loop.py:107` records **only `{"files", "step"}`**.

**This predicts the gate-4 conflation regression observed at 14B**
(frac_collocated 0.257 → 0.785; same-task different-decision reads nearly as
close as same-decision). We shipped the failure mode the spec warned about.

- **Fix (build item):** record `region` (line span from sed/patch/heredoc
  targets), `subgoal` (the turn's THOUGHT text), `error_signature` (exit code +
  stderr head) per ActionEvent, and use them in the decision label.
- **Fallback the spec provides and we never built:** the **co-divergence** label
  source (runs whose leans diverge together at overlapping spans are evidence of
  the same decision). Build it if gate 4 stays red.

## 5. HIGH — task coverage leaves most of the benchmark unused

`third_party/hil-bench` has **105 swe + 105 sql** tasks. Our artifact covers
**60 swe and zero sql**.

- **Gate 6 (OOD transfer) has never had a real held-out family.** Its 0.28–0.30
  purity came from only 3 buckets forming — that is a no-data artifact, not a
  transfer result. decisions/018 planned ~20 sql OOD; never collected.
- The sealed test pool (swe_60+) still has no trajectories, so the eval split
  doesn't exist yet.
- **Fix:** derive class artifacts for sql OOD tasks and the sealed test pool
  (the artifact derivation workflow already exists + the deterministic auditor).

## 6. MEDIUM — layer coverage is narrow and unswept on real data

We capture 4 layers at fractions 0.4/0.5/0.6/0.7 (indices 26/32/38/45 of 64).
The only layer sweep ever run was at 7B, and it ranked within this same narrow
band. Early (0.2–0.3) and late (0.8–0.9) layers have never been looked at on
real agent data. Hook capture makes extra layers nearly free (KB/step).

- **Fix:** capture **8–10 layers** spanning 0.2–0.85; sweep at analysis time
  (select-at-load already supports this — zero extra GPU runs).

## 7. MEDIUM — statistical protocol

- Seed-holdout leaves few measurable decisions; with 24 seeds use **k-fold +
  leave-one-run-out** properly (machinery exists, `--kfold`).
- **Pre-register structural vs value forks as SEPARATE hypotheses.** Value forks
  were activation-invisible at both 7B and 14B; pooling them with structural
  forks dilutes the structural signal that is actually there (0.71–0.73 raw-h).
  This is the single most likely way a genuine effect gets buried in a null.

## 8. NOT BUILT — the deliverables the paper needs

- **Phase 4 eval harness** (`specs/eval.md`): Ask-F1 (verbatim port), Pass@k,
  regime-sliced recall, interruption budget, and the matched-N baselines
  (ClarifyGPT B1 output divergence, OPENIA probe, EigenScore) end-to-end through
  HiL-Bench's judge. **This is the headline table and it does not exist.**
- **Part B online trigger** — implemented, never run live (correctly gated on A4).
- **Capture-server bridge** (decisions/019) — serve the model behind an
  OpenAI-compatible endpoint so the no-ask / full-info / naive-ask arms run in
  hil-bench's own harness. Agreed in principle, not built.
- **Scaffold-robustness check** — fork signatures on SWE-Agent-shaped
  trajectories (our one-bash-block protocol is not the scaffold the frontier
  numbers come from; decisions/019 comparability).

---

## The run plan

**Ordering principle: verify the cheap causal fixes BEFORE spending the big
budget.** Items 1–3 are hypotheses about why forks were rare. If they are right,
fork rate jumps in a few GPU-hours. If they are wrong, a 10× collection would
just buy a bigger null — and we would want to know that for ~$50, not ~$5000.

### R0 — Config verification (minutes, ~0 GPU)
Dump the effective `generation_config`; record top_k/top_p/min_p in the
manifest. Confirms or refutes §1 immediately.

### R1 — Diversity pilot (~4–8 GPU-hours) — THE GO/NO-GO GATE
6 tasks known to carry structural forks (swe_36, swe_47, swe_50, swe_4, swe_12,
swe_30) × 12 seeds, with: explicit sampling params (top_k=0, top_p=1.0),
temperature ladder to 1.3, nudge reinstated, `max_steps=40`,
`max_new_tokens=2048`, `cadence=8`, 8–10 layers.

**Pre-registered success criterion (set BEFORE the run):** ≥3 of the 6 tasks
show ≥2 interpretation classes each committed by ≥2 distinct runs, and median
reads/run ≥ 25. Compare against the same 6 tasks at 8 seeds/cap-15 (we have that
baseline).

- **Met →** the clamp explanation holds; proceed to R2 at full scale.
- **Missed on diversity →** forks are genuinely rare for this backbone; the
  honest pivots are a *different/larger backbone*, or reframing to the
  structural-fork slice, or the negative result. Do NOT scale a null.
- **Missed on reads only →** drop cadence further and re-pilot (cheap).

### R2 — Main collection (scale set by hardware)
60 swe train tasks × 24 seeds × cap 40 ≈ **1440 runs**. At the pilot's measured
per-run cost (expect ~6–10 min with the longer cap), that is ~150–240 GPU-hours
on one card — trivially parallel across GPUs (tasks are independent; the
collector already resumes).

### R3 — OOD + sealed test (parallel with R2)
~20 harbor_sql tasks × 12 seeds (gate 6, first real transfer number) and the
sealed swe_60+ pool × 12 seeds (the eval split). Requires deriving class
artifacts for both first.

### R4 — Analysis (laptop/CPU, days not GPU)
Composite labels → layer sweep → cadence sweep → A1/A2/A3 → **k-fold gates,
structural and value pre-registered separately** → report. Red gate = stop and
report (standing rule).

### R5 — Phase 4 eval (GPU, after gates)
Build the capture-server bridge, run Ask-F1 / Pass@3 / baselines at matched N on
the sealed pool.

## Build items before R1 can launch

1. `hf_reader`: explicit sampling params + record effective generation_config.
2. `collect_v2`: `--top-p/--top-k/--min-p`, temperature ladder decoupled from
   seed, `--nudge` reinstated (+ ablation flag), defaults `max_steps=40`,
   `max_new_tokens=2048`, `cadence=8`, wider `--layers`.
3. `agent_loop`: composite observables (region, subgoal, error_signature).
4. Contract tests for each; suite stays green.

Items 1–2 are small and unblock the go/no-go pilot. Item 3 is the larger build
and is needed before R4, not before R1.
