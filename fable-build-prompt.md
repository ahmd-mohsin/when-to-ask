# Build brief for Fable — "When-to-Ask" detector for HiL coding agents

## Read this first — what this is

This is **my research project** (targeting an ML venue; the contribution is a method, not a product). I am investigating when a coding agent facing an under-specified task should stop and ask the human, by comparing several parallel trajectories and detecting when they *confidently disagree* about how to resolve the same sub-decision — read from model activations, not from spoken output.

The attached design document (`when-to-ask-offline-online.md`) is the **canonical spec for the method**. It is the source of truth for *what* to build and *why* each piece exists. Where this brief and the doc disagree, the doc wins on method; this brief governs *process*.

Because this is research, two things are true and I need you to internalise them:

1. **The goal is to find out whether the method works, not to produce green checkmarks.** Some of the checks below (the A4 gates) are scientific hypotheses. If one fails, that is a *finding* I need to know about — stop and report it. Do **not** tune, leak evaluation data into training, or reshape a gate until it passes. Honest red is worth more to me than dishonest green.
2. **When something is unclear or under-specified, ask me — do not guess.** The doc has an "Open items" section and several design knobs on purpose. I would much rather answer five questions up front than discover you silently picked values. A list of the questions I already expect is below; add to it.

## How I want you to work: spec → harness → implement, phased

For every component, in this order:

1. **Spec.** Write a short, precise, testable specification: inputs, outputs, shapes/types, invariants, and the observable behaviour that would tell us it's correct. Keep it in `specs/`. One file per component (A0, A1, A2, A3, A4, B, eval).
2. **Harness.** Write the tests/checks that verify that spec *before* implementing it. Put them in `harness/`. Crucially, split the harness into two clearly separated kinds (see next section).
3. **Implement.** Satisfy the spec and pass the harness — but before writing anything from scratch, apply the migrate-first rule below. Only hand-write what is genuinely our own contribution; for everything a cited paper already implements, clone and migrate their code. Small, verifiable commits. If reality forces a spec change, change the spec file explicitly and tell me why.

Follow the doc's **phased build order** and treat it as a hard rule:

- **Do not implement the online trigger (Part B) until the A4 gates pass.** If topic invariance and fork-collocation don't hold, the trigger is meaningless and I don't want it built yet.
- Build **synthetic fixtures first** for every stage: construct toy activations with a *known* topic/lean/commitment structure so the whole pipeline (autoencoder, clustering, voting, CUSUM) can be validated end-to-end on data where we know the right answer, before any real model activations are involved. This de-risks the plumbing separately from the science.

## Migrate paper code — don't reimplement mechanisms we're citing

**Default to `git clone` + migrate the original authors' code. Only hand-write what is genuinely our own contribution.** For any mechanism that comes from a paper we cite, use *their* repository as the implementation, adapted minimally to our interfaces — do not rebuild it from the description.

Why this matters for the research, not just for speed: when the paper says "we use ReDAct's disentangling autoencoder" or "we calibrate the threshold with LYNX's split-conformal procedure," that claim must be *literally true* — the same code, the same mechanism — not a lookalike I re-derived. A faithful migration makes every such citation an honest, auditable statement. A from-scratch reimplementation quietly turns "we use X" into "we use something like X," which is both weaker and, if it diverges, wrong.

**How to migrate:**
- Resolve each repo from the paper's **arXiv "Code" link** (or Papers-with-Code). Do **not** invent a GitHub URL. If a paper has no linked repo, stop and tell me — I'll decide whether we reimplement or drop the dependency; don't silently reimplement.
- Keep migrated code **isolated and attributed**: a `third_party/<paper>/` directory per source, with a `PROVENANCE.md` recording the repo URL, commit hash, licence, and exactly what we changed to adapt it. Respect each licence (check compatibility before vendoring; flag anything restrictive to me).
- Adapt **minimally** — wrap their code behind our spec's interface rather than editing their internals where possible, so the mechanism stays recognisably theirs and upgrades stay easy.
- If migration needs a change that alters the mechanism's behaviour (not just plumbing), that's a research decision — flag it, don't just make it.

**Migrate vs build — the mapping (confirm repos in Phase 0):**

| Component | Source | Action |
|---|---|---|
| A1 ambiguity direction `d` (difference-in-means / CAA) | CAA (`nrimsky/CAA`), Geometry of Truth (`saprmarks/geometry-of-truth`) | **Migrate.** |
| A2 disentangling autoencoder (two-headed encoder, InfoNCE, orthogonality, gradient reversal, reconstruction) | ReDAct (backbone); DEAL (secondary) | **Migrate the architecture + training machinery.** |
| Activation read/hook utilities | Representation Engineering (`andyzoujm/representation-engineering`) | **Migrate/reuse.** |
| A3 threshold calibration (split conformal) | LYNX | **Migrate the calibration procedure.** |
| HiL-Bench harness, frozen judge, blocker registry, Ask-F1 | HiL-Bench | **Migrate/reuse verbatim** — reimplementing the benchmark would break comparability. |
| Output-divergence baseline | ClarifyGPT | **Migrate** so the baseline is faithful. |
| **A2 supervision + labelling** (topic = decision-identity from composite observables; `L` = interpretation class) | — | **Ours — implement.** (We wrap ReDAct's machinery with our targets.) |
| **A3 commitment definition** (`r` steady + `s` dropped) and stabilisation-point label | — | **Ours — implement.** (We feed LYNX's calibrator our score/label.) |
| **Part B: cross-trajectory matching, topic-keyed bucketing (leader + merge + hysteresis), mutable voting, dispersion + CUSUM trigger, loop channel, question assembly** | — | **Ours — implement from scratch.** This is the contribution; no paper does it. (CUSUM and clustering are textbook — use a standard library where clean — but the *system* is ours.) |

The rule of thumb: if a citation in the paper will point at it, migrate it. If it's the thing the paper is *about*, build it.

## Two kinds of checks in the harness — keep them separate

**(a) Contract / engineering tests** — ordinary software correctness. Make these pass.
Examples: tensor shapes and dtypes; the disentangling autoencoder's four loss terms are all wired in and non-zero; the decoder reconstructs a held-out `h` above a floor; leader-clustering assigns, merges on proximity, and holds a run via hysteresis exactly as specified; a vote is written on commitment and retracted on de-commitment; CUSUM fires on a synthetic *persistent* disagreement and does **not** fire on a synthetic transient blip; the loop channel adds signal for a run stuck on repeated environment states. On synthetic fixtures with known structure, these must pass.

**(b) Research validation gates (A4)** — hypotheses about the learned representation. **Do not "make these pass."** Run them on **held-out** data, report the number, and stop for my review. These are:

1. Topic-leakage: predict `r` from `T` alone → should be ~chance (report as ReDAct-style eta^2).
2. Decision-recovery: `T` predicts decision-identity well.
3. Fork-collocation: held-out same-decision/opposite-resolution runs have high topic cosine; also emit the same-vs-different cosine distributions so we can set `theta`.
4. Conflation: same-file/different-decision pairs do **not** collocate.
5. Lean-separation: within a bucket, different interpretations have well-separated `L` vectors.
6. OOD transfer: bucket purity on unseen task families (report as a limitation number).
7. Lead-time: on known forks, bucket disagreement rises K>0 steps before actions diverge.

If any gate fails, surface it with the number and your read on *why*, and wait. Never close the gap by training on the eval split or hand-selecting examples.

## Before you write any code — questions I expect you to ask me

Resolve these with me and record the answers in `decisions/` (short ADR-style entries). Do not assume defaults:

- **Backbone model**: which open-weight code agent model, and can we hook mid-layer residuals *inside the agent loop*? (Candidates in the doc: Qwen3-Coder / GLM / DeepSeek / Llama-3.x.)
- **HiL-Bench access**: do we have the harness, the frozen judge, and the blocker registry? The registry is my source of *interpretation classes* for supervising `L` — confirm we can read it.
- **Read position**: exact cadence or cue set for reading `h` across the reasoning span (the doc forbids reading only at the tool-call boundary — confirm the mechanism you'll use to sample earlier).
- **Layer + windows**: which mid layer; the commitment smoothing window; the stability metric. (These are Phase-2 sweeps — propose ranges, don't hardcode.)
- **N and diversity**: how many parallel trajectories, and how we induce divergence (seed/temperature/persona) — and how we *measure* the diversity we got.
- **Compute budget** and where training/inference run.
- **Off-registry decisions**: how we handle sub-decisions with no registry class (the doc's fallback is observable-action clusters gated by the ambiguity signal `s` — confirm scope).
- **Paper repos + licences**: for each source in the migrate-vs-build table, confirm the repo resolves from its arXiv page and report its licence. Flag any repo that's missing, or any licence that would restrict us from vendoring, before Phase 1.
- Anything in the doc's "Open items that stay empirical" you need pinned to proceed.

If a question blocks a phase, block and ask. If it doesn't, note the assumption in `decisions/` and continue.

## High-level flow

```mermaid
flowchart TD
    subgraph OFFLINE
        A0[A0: run agent N times, vary seed/temp\nlog h across reasoning span + observables]
        A1[A1: build ambiguity direction d\n= mean(should-ask) - mean(proceed)\n-> scalar signal s = dot h,d]
        LBL[label junctures from COMPOSITE observables\nfile+region+sub-goal+error; co-divergence fallback]
        A2[A2: disentangling autoencoder\nencoder -> T topic, L resolution-lean\ndecoder rebuilds h\nlosses: topic-sup + class-sup + gradient-reversal/orth + recon]
        A3[A3: commitment = r steady & s dropped\ncalibrate tau via split-conformal\non stabilization-point label]
        GATE{A4 gates on HELD-OUT data\nleakage / recovery / collocation /\nconflation / lean-sep / OOD / lead-time}
        A0 --> A1
        A0 --> LBL --> A2
        A1 -. anchors s-gate for off-registry .-> A2
        A2 --> A3 --> GATE
    end

    GATE -- fail: STOP, report, revisit labels/arch --> LBL
    GATE -- pass --> ONLINE

    subgraph ONLINE
        R[per read across reasoning span:\ns=dot h,d ; r=L h ; topic=T h ; commit?]
        C[cluster topic -> bucket\nleader + merge + hysteresis, theta from A4]
        V[if commit: mutable vote r into bucket\nelse retract]
        TRIG[per bucket >=2 votes:\ncommitment-weighted dispersion of r\n-> CUSUM  + lambda*loop channel]
        ASK[fire ask_human\noptions from divergent r/actions\ninject answer -> continue all runs]
        R --> C --> V --> TRIG --> ASK
    end

    ONLINE --> EVAL[Evaluation:\nAsk-F1 + Pass@k on HiL-Bench\nregime slices / lead-time /\nmatched-compute baselines]
```

## Deliverables per phase

- **Phase 0 — setup + questions.** Repo skeleton (`specs/ harness/ src/ decisions/ fixtures/ third_party/`), environment, `decisions/` answers to the questions above, synthetic-fixture generators with known structure. Resolve and clone the paper repos from the migrate-vs-build table, each into `third_party/<paper>/` with a `PROVENANCE.md` (URL, commit hash, licence, planned adaptations). Prove you can hook and log a mid-layer residual on one real task, N=2.
- **Phase 1 — ambiguity direction (A1).** Spec + harness + `d`; sanity checks on `s` separating held-out should-ask vs proceed states.
- **Phase 2 — encoder + gates (A2–A4).** Spec + harness + the disentangling autoencoder; commitment definition and conformal `tau`; then run the A4 gates on held-out data and **stop for my review**. This is the science gate — do not proceed past it without me.
- **Phase 3 — runtime trigger (B).** Only after gates pass. Spec + harness (synthetic disagreement/blip/loop cases) + the online loop: clustering (with merge + hysteresis), mutable voting, dispersion + CUSUM, loop channel, question assembly.
- **Phase 4 — evaluation.** Ask-F1 + Pass@k on the HiL-Bench public set; regime-sliced recall (fork / confident-convergent / clear); lead-time analysis; matched-compute baselines (vanilla ask_human, output-divergence at matched N, single-stream should-ask probe).

## Ground rules

- The doc's rule is inviolable: **match runs on the decision (topic), never on step index or action.** Actions and observables are offline *teachers* for labels; they never trigger a read or an alignment at runtime. If anything in the live loop depends on a step index or an action, that's a bug.
- Keep the two "lean" quantities distinct exactly as the doc does: `s` (scalar ambiguity signal, gates + commitment timing) vs `r = L(h)` (vector resolution lean, the fork signal). Never fire disagreement on `s`.
- Don't over-engineer beyond the current phase. Working, inspectable, well-tested prototype quality — this is for experiments and a paper, not production hardening.
- **Migrate before you build.** For any mechanism a cited paper implements, use their repo (see the migrate-vs-build table); hand-write only what's genuinely ours. If you catch yourself reimplementing a paper's method from its description, stop — clone theirs instead, or ask me.
- When you finish a phase, give me: what passed (contract), what the gates *measured* (research, with numbers), what you assumed, which paper repos you migrated (with commit hashes) vs what you wrote yourself, and what you need from me next.

Ask me your Phase-0 questions before writing anything beyond the repo skeleton.
