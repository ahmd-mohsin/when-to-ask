# 016 — Value-triggered reads (validated), and the v2 agent-loop requirement

---
status: agreed (owner 2026-07-14); v2 collection design awaiting owner's next GPU session
date: 2026-07-14
---

## Value reads

**Finding (laptop, existing multi-layer data):** leave-one-run-out on RAW
layer-14 activations, split by distance to the nearest class-signature mention
in the trace: reads **within 12 tokens: acc 0.727 vs 0.500 chance** (55 reads,
6 decisions); reads ≥24 tokens away: chance (0.491). The value-fork lean IS in
the residual stream — **transiently, around the emission moment** — and the
32-token cadence straddled it. This revises decisions/015: the value-fork
negative was a *read-position* artifact, not absence of information.

**Consequence for the method:** a third read trigger, `value` (implemented in
`wta/reads.py`: fires when the generated delta emits a multi-digit literal,
`DEFAULT_VALUE_PATTERN`, cooldown 8 tokens; priority cue > value > cadence;
`collect_a0 --value-reads`). Still reads during generation — a cue-set
extension per decisions/006, never an action-boundary read.

**Honest limits:** 55 reads / 6 decisions is a pilot number; and the value
signal's lead-time is inherently short (~seconds — the state knows the value
only as it's about to write it), unlike structural forks' long-horizon signal
(median K≈7 reads). The lead-time asymmetry is itself a finding: structural
forks are foreseeable, value forks are only catchable-at-commit.

## The v2 agent-loop requirement (owner is right)

v1 traces are single-generation reasoning, chosen deliberately as the cheapest
falsifier of the representational claim. But HiL-Bench is an AGENT benchmark
and the method's claim is interrupting an agent before a wrong action. **The
paper's headline results must come from real tool-calling trajectories:**
multi-step runs through the hil-bench executor (docker), logging per-segment
reads (`segment_idx` in the RunLog schema exists for exactly this) and real
action events → real (not proxy) lead-time, and Ask-F1/Pass@k vs matched-N
baselines. Trace-level results are the pilot study that justified this spend.

**v2 collection = next GPU session:** agent loop (mini-swe-agent port) inside
the task containers, N=8, layers captured, cadence+cue+value reads, action
events logged at tool calls. Owner also plans a larger backbone: note
Qwen2.5-Coder-32B-Instruct is the natural step (bf16 ≈ 65 GB → g5.12xlarge
4×A10G with device_map=auto, already supported by hf_reader; a 70B coder-tuned
model doesn't exist in the Qwen2.5 line — Llama-3.3-70B is not code-tuned).

## What the paper needs (reporting-numbers view)

1. v2 agent trajectories: Ask-F1 + Pass@k vs vanilla-ask / output-divergence /
   single-stream probe at matched N (the benchmark's own metrics).
2. Pre-registered structural-vs-value boundary validation on fresh tasks.
3. Value-read rescue confirmed at collection time (--value-reads run).
4. Label-accuracy audit vs human (~100 commitments).
5. Scale point (7B vs 32B).
Gate 6 (OOD 0.28) and the lead-time asymmetry reported as measured limits.
