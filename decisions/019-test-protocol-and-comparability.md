# 019 — Test protocol: published numbers to beat, comparability rules, Qwen3-32B

---
status: agreed (owner 2026-07-18: "make the decision 019 and commit. also yes
i agree lets use qwen3-32B")
date: 2026-07-18
---

## 1. The published numbers to beat (verified against the HiL-Bench PDF,
##    extracted 2026-07-18 — not recalled from memory)

HiL-Bench paper (Scale AI, 2604.09408; PDF linked from
third_party/hil-bench/README.md), Table 1, **SWE domain, pass@3**:

| Model (closed, SWE-Agent scaffold) | Full info | w/ ask_human() | Recall | Precision | Ask-F1 |
|---|---|---|---|---|---|
| GPT-5.3-Codex  | 67.3% | 2.0% | 23.5% | 56.5% | 33.2% |
| GPT-5.4        | 67.3% | 1.3% | 27.0% | 52.3% | 35.6% |
| Gemini 3.1 Pro | 84.7% | 5.3% | 36.2% | 47.5% | 41.1% |
| Claude Opus 4.6| 69.1% | 9.4% | 34.6% | 26.3% | 29.9% |

Paper's own headline: 67–85% with full information vs ≤~10% when the model
must judge when to ask; **no frontier model exceeds Ask-F1 41 on SWE**. The
paper also LoRA-RLVR-finetunes **Qwen3-32B** (SkyRL; 120 train / 30 held-out
tasks per domain — nearly our split) and shows Ask-F1 + pass@3 improve in
lockstep (their Fig. 4; exact figure digits to be re-verified before quoting
in a draft — they are chart-embedded, not in a table).

**Publishability bars, in order:**
1. Our detector on an open 14B beating frontier *judgment* on the same
   benchmark: Ask-F1 > 41 and with-ask pass@3 meaningfully above ~10%.
2. Matching/beating their RLVR'd Qwen3-32B **without finetuning the LLM**
   (our probe is trained; the LLM is frozen).
3. The pre-registered internal bar (brief §5d): beat **B1 output-divergence
   at matched N on fork blockers, with earlier firing** — both sides produced
   by us; no published number exists.

## 2. Comparability rules (what makes our numbers citable side-by-side)

MATCH EXACTLY:
- **Tasks**: the public HiL-Bench dataset (already our substrate).
- **Judge**: their ask_human is a frozen **Llama-3.3-70B-Instruct** semantic
  judge (returns the blocker resolution or "irrelevant question"). Use their
  judge config VERBATIM — never a reimplementation.
- **Metric + protocol**: Ask-F1 via their scoring code; **pass@3** (3 seeds
  per task per arm).

CANNOT MATCH — DISCLOSE:
- Their frontier rows are closed models; internal-state methods require open
  weights. Frontier rows appear in our tables as *cited context only*,
  clearly marked different-model. Rule (pre-registered in the brief): every
  baseline is re-run on OUR backbone at matched N; reported numbers from
  other papers are never treated as head-to-head.
- Scaffold: they use SWE-Agent; our detector arm requires our instrumented
  loop (activation reads). Bridge option: their harness supports self_hosted
  models — run our backbone's no-ask / full-info / naive-ask rows inside
  THEIR harness as a scaffold-bridge row.
- Task subset: their Table 1 spans the released set; our sealed test is the
  unseen ~30 swe tasks (+~20 sql OOD). Subset disclosed.

Test arms (final run, all same backbone, matched N): B0 no-ask, full-info
ceiling, B1 output-divergence (load-bearing), our detector; secondary
B2 verbalized / B3 single-stream probe / B4 random per the brief's ladder.
ClarifyGPT paper numbers (MBPP/HumanEval, function-level) are method
citations only — different task regime, never number comparisons.

## 3. Backbone decision: the 32B pass uses **Qwen3-32B** (owner approved)

Rationale: it is the exact open model in the paper's RLVR experiment — the
only published same-model anchor available. This outweighs Qwen2.5-Coder-32B's
code specialization (trade-off noted; 14B primary remains Qwen2.5-Coder-14B,
already collected). Implementation notes:
- Qwen3 has hybrid thinking/non-thinking generation; the collection must PIN
  one mode (default: non-thinking for agent turns) and record it in the
  manifest; check what mode their RLVR setup used before the final run.
- Pipeline is layer-fraction based and reads model config at runtime — no
  code change expected beyond --model-id.

## 4. GPU: 32B does NOT fit the g6e.xlarge

Qwen3-32B at bf16 is ~65 GB of weights alone — over the L40S's 48 GB. Same
answer as decisions/016 gave for Qwen2.5-Coder-32B: use the already-planned
**g5.12xlarge (4× A10G 24 GB = 96 GB, device_map=auto, spot ~$2–2.5/hr)**.
Quantization stays REJECTED (it contaminates the measured activations,
decisions/016). The hook-based capture path (decisions/017 addendum) was
built precisely to make 32B memory-safe on that box. The 14B collections
stay on g6e.xlarge unchanged.

## Addendum (2026-07-18, same day): thinking mode + scaffold facts verified

**Thinking mode**: the paper NEVER specifies generation settings for its
Qwen3-32B RLVR runs (Appendix G covers only LoRA/SkyRL, the 120/30 split,
same SWE-Agent framework, and the reward scheme). However, the vendored
hil-bench repo's own self-hosted Qwen example configs
(configs/swe/ask_config_qwen3_30b_a3b_instruct_2507.yaml) use the
**Instruct (non-thinking) Qwen3 variant, temperature 1.0, via litellm
proxy**. DECISION: run Qwen3-32B in **non-thinking mode**, record the mode +
sampling params in the collection manifest, and state the paper's silence in
the eval section rather than claiming to match an unstated setting.

**Scaffold**: SWE-Agent is fully open source (Princeton, NeurIPS 2024) and is
vendored INSIDE our hil-bench clone (third_party/hil-bench/SWE-agent). We do
not use it for the detector arm for a structural reason, not availability:
SWE-Agent drives the model through an API client (litellm/proxy — see its
configs), so the model is a remote black box to the scaffold; activation
capture requires the model in-process with layer hooks. Our loop follows the
**mini-SWE-agent convention** (the SWE-Agent team's own published minimal
agent, also vendored) — a citable, simplified version of the same scaffold.

**Reviewer-acceptance plan for the scaffold difference** (three tiers):
1. Core claims (detector vs B0/B1/B2/B3 at matched N) share ONE scaffold —
   scaffold-independent, fully valid.
2. Paper Table 1 rows are cited context (different model AND scaffold,
   labeled as such).
3. BRIDGE ROWS kill the objection empirically: the arms that need no
   activations (no-ask, full-info, naive ask_human) are ALSO run in the
   paper's own harness with our backbone self-hosted (vLLM; the harness
   supports self_hosted/litellm hosting). If harness-baseline ≈ our-loop
   baseline, the scaffold delta is measured, not argued. Deep SWE-Agent
   integration (custom in-process model backend) stays a fallback only if
   demanded — weeks of engineering for what bridge rows already quantify.

## 5. Sequence (unchanged otherwise)

Train collection (runbook 2c, launching) → laptop gates → if gate 5 passes:
Part B + eval harness built and frozen (with within-task bucketing per the
gate-4 finding) → decisions/020 pre-registers the final test run → single
sealed test execution, numbers are final.
