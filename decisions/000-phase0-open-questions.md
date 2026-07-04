# 000 — Phase-0 open questions (awaiting owner answers)

---
status: ANSWERED 2026-07-02 — see ADRs 001–010 for the recorded decisions
date: 2026-07-02
---

Each answered item becomes its own numbered ADR. "Proposed" = my default if you
just say "agreed"; blocking items are marked ⛔.

## Q1 ⛔ Relationship to the existing xtid codebase

The repo already contains the xtid de-risk pipeline (`src/xtid/`, claim C1, from
`pre-implementation brief.md`); its real GPU run never happened. The new brief
doesn't mention C1.

**Proposed:** treat the de-risk as superseded by this build; freeze `src/xtid/`
as an artifact; build the When-to-Ask method in a new package (`src/wta/`),
reusing the already-migrated pieces (HiL-Bench judge/Ask-F1/executor wrappers,
`HFWhiteBoxModel` hook, mini-swe-agent loop). Confirm — or tell me C1 still
gates this build.

## Q2 ⛔ ReDAct has no public code — brief-mandated stop

arXiv 2602.19396 resolves to **"Hiding in Plain Text: Detecting Concealed
Jailbreaks via Activation Disentanglement"** (Farzam et al., 2026-02-23) — the
same paper as your PDF `hiding in plain text jailbreak detection.pdf`; ReDAct is
its disentanglement module (goal-vs-framing ≈ our topic-vs-lean). Verified: no
Code link on arXiv, no code-availability statement in the full text, no
matching GitHub repo. Your brief says: stop and ask.

Also: **DEAL and REAL are the same paper.** arXiv 2506.08359 v1 = "DEAL:
Disentangling Transformer Head Activations for LLM Steering"; retitled v2 =
"REAL: Reading Out Transformer Activations…" (ICLR 2026 poster) — your PDF
`REAL Reading Out Transformer Activations for.pdf`. Official repo
`github.com/liam0949/REAL_ICLR`, MIT. Caveat: its mechanism is a **VQ-AE per
attention head** (behaviour-relevant vs irrelevant subspaces), not ReDAct's
two-headed MLP encoder — not a drop-in for our A2 shape.

Options:
- (a) **Reimplement ReDAct's architecture from the paper** (we have the PDF),
  cite as "following ReDAct" not "using ReDAct's code"; vendor REAL (MIT) as a
  reference implementation for the disentangling-training machinery. **Proposed.**
- (b) Email the ReDAct authors for code first; (a) in the meantime.
- (c) Pivot A2 to REAL's per-head VQ-AE mechanism (bigger method change).

## Q3 ⛔ Licence flags (before vendoring, per brief)

| Repo | Licence | Flag |
|---|---|---|
| nrimsky/CAA | MIT ✅ | — |
| andyzoujm/representation-engineering | MIT ✅ | — |
| liam0949/REAL_ICLR (DEAL/REAL) | MIT ✅ | — |
| farukakgul/LYNX | **none** | all-rights-reserved by default |
| saprmarks/geometry-of-truth | **none** | same |
| hil-bench (already cloned) | **no LICENSE file** | same |
| ClarifyGPT (already cloned) | **no LICENSE file** | same |

`third_party/` is git-ignored (only VERSIONS.md tracked), so local clones are
reference-use, not redistribution. **Proposed:** clone the MIT three now;
clone the unlicensed ones for local reference only, email authors for
permission/licence, and note the status in each PROVENANCE.md. For LYNX
specifically, the split-conformal *procedure* is textbook — if the authors
don't respond we implement the calibration from the paper and cite it as
"following LYNX". Confirm this handling.

## Q4 ⛔ Backbone model + hook mechanism

Prior xtid decision: `Qwen2.5-Coder-7B-Instruct`, rented GPU (RunPod/Lambda).
Doc candidates: Qwen3-Coder / GLM / DeepSeek / Llama-3.x.

Hooking is de-risked: `src/xtid/backbone/model.py::HFWhiteBoxModel` already
returns per-generated-token mid-layer hidden states inside the loop. It
currently **averages them per action** — the exact read-at-boundary pattern the
doc forbids — so the adaptation is: keep per-token vectors, subsample at
cadence/cues, never aggregate to the action boundary.

**Proposed:** keep Qwen2.5-Coder-7B-Instruct for A0–A4 (fits one 24–48 GB GPU,
known quantity); revisit scale-up only after gates pass.

## Q5 Read position (cadence/cue set)

**Proposed:** read every K=32 generated tokens AND at deliberation cues
("hmm", "wait", "let me", "actually", "should I", "alternatively" — token-level
match), logging token index per read; K ∈ {16, 32, 64} as a Phase-2 sweep.
LYNX's cue-token reading is precedent for the cue set.

## Q6 Layer + windows (Phase-2 sweep ranges — confirming ranges only)

**Proposed:** mid layer ∈ {0.4, 0.5, 0.6, 0.7}·depth; commitment window
w ∈ {3, 5, 8} reads; stability metric: max displacement of `r` over the last w
reads (fallback: mean pairwise distance).

## Q7 N and diversity

**Proposed:** A0 collection at **N=8** per task (seeds 0–7, temperature cycled
{0.7, 0.85, 1.0}, persona nudge OFF by default); online eval at **N=4** with
matched-compute baselines at the same N. Diversity report: per known blocker,
number of distinct interpretation classes hit + entropy of committed actions;
if <30% of fork-tasks show ≥2 interpretations, escalate (temperature up /
persona on) and report.

## Q8 ⛔ Registry → interpretation classes (finding: the class list doesn't exist yet)

The HiL-Bench clone has 200 per-task `blocker_registry.json` files (100 swe,
100 sql). Each blocker has `id`, `type`, `description`, ONE canonical
`resolution`, and `example_questions` — **not an enumerated set of competing
interpretation classes** as the doc's A2 supervision assumes.

Options:
- (a) **One-time LLM-assisted derivation**: for each blocker, derive 2–4
  interpretation classes from description + resolution + example_questions
  (each example question already implies the alternatives); freeze as
  `data/interpretation_classes.json`; hand-audit a random 20. This *becomes*
  the fixed registry artifact the doc refers to. **Proposed.**
- (b) Classes = {canonical resolution} ∪ observed committed-action clusters
  per blocker (no derivation step, but circular-ish with the fallback path).
- (c) You supply / manually author the class lists.

## Q9 ⛔ Compute budget + A0 scale

**Need from you:** budget (in $ or GPU-hours), provider account
(RunPod/Lambda — `scripts/provision_gpu.sh` exists), and judge hosting
(Llama-3.3-70B-AWQ on the same rented box vs an API provider).

**Proposed A0 scale:** harbor_swe only first (100 tasks × N=8); harbor_sql
held out entirely for the OOD gate (A4 check 6). Rough storage: ~2k reads/run
× 3584-dim fp16 ≈ 14 MB/run ≈ 11 GB total — fine on local disk.

## Q10 Off-registry scope

**Proposed:** implement the s-gated observable-action-cluster fallback, but
exercise it only in the OOD gate / limitation analysis — core A2 training uses
registry-derived classes only.

## Q11 Open-items defaults (non-blocking; noted as assumptions unless you object)

- Conformal calibration: one read per (run, decision) — sidesteps
  autocorrelation, per the doc.
- Per-bucket `r`-spread normalization: deferred to Phase-2 sweep.
- Oscillating runs: contribute low-weight votes via the soft commitment
  weights (never hard-excluded); swept in Phase 2.
