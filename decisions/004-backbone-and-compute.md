# 004 — Backbone Qwen2.5-Coder-7B-Instruct; dev on CPU laptop, GPU runs on owner's AWS

---
status: agreed (compute) / assumed (backbone default not objected to)
date: 2026-07-02
---

**Context.** Owner: GPU work runs on **their AWS GPU instance**; development
happens on this laptop (no CUDA); budget is modest. The RunPod/Lambda plan
from the xtid pass is superseded.

**Decision.**
- Backbone default: `Qwen/Qwen2.5-Coder-7B-Instruct` (proposed in Q4,
  unobjected). bf16 ≈ 15 GB → fits a single A10G (g5.*) / L4 (g6.*) 24 GB
  instance with headroom; `load_in_4bit` path exists if needed.
- Everything must be **CPU-runnable end-to-end on synthetic fixtures**
  (no torch import required on the laptop path, following the xtid pattern).
  GPU-only code stays behind config (`backbone.kind: hf`).
- Handoff artifact: scripts the owner can run as-is on the AWS box
  (starting with the Phase-0 hook proof, `scripts/prove_hook.py`).
- Judge hosting (Llama-3.3-70B) decided later, at eval time — not needed for
  Phases 0–2 machinery.

**Consequences.** Modest budget → A0 collection scale starts at harbor_swe
only (100 tasks × N=8), harbor_sql held out for the OOD gate; revisit scale
after gate numbers exist.
