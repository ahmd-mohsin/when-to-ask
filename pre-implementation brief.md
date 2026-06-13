# **Pre-Implementation Brief — Internal Cross-Trajectory Divergence as a "When-to-Ask" Trigger for Long-Horizon Coding Agents**

**Status:** internal planning doc, for supervisor discussion before method-building. **Framing:** multi-trajectory (best-of-N) is retained as the spine. The contribution is the *internal* divergence signal read **across** the N trajectories — distinct from output-divergence triggers and from single-stream internal probes. **Verification:** every arXiv ID below is tagged `[V]` (verified) or `[?]` (verify before citing). Do not propagate `[?]` into a submission unchecked.

---

## **0\. TL;DR**

In a long-horizon coding-agent run executed as **N parallel trajectories** (best-of-N), the highest-value moments to call `ask_human()` are **branch points where the trajectories' *internal states* diverge** — read from the residual / hidden states, not from their outputs and not from one stream's self-uncertainty. We trigger asking on cross-trajectory internal divergence, generate the question from the behavioral fork (the divergent trajectories supply the candidate options), and evaluate on a HiL-Bench harness with **Ask-F1**.

**The single sentence that defines the paper:** *does internal cross-trajectory divergence beat output cross-trajectory divergence (ClarifyGPT-style) at matched N on Ask-F1, with its win concentrated on genuine-fork blockers?*

---

## **1\. The novelty cell, stated precisely**

Two axes organize the prior art:

|  | Output / verbalized signal | Internal (hidden-state) signal |
| ----- | ----- | ----- |
| **Single trajectory** | verbalized "I'm unsure"; P(True); HiL-Bench `ask_human()` baseline | **dense & crowded:** Confidence Manifold, internal code-correctness, DRIFT, LRP, CCPS, AutoProbe |
| **Multiple trajectories** | **occupied:** ClarifyGPT (code, test-output divergence), LaMSeI (QA, response variation), "Can Multiple Responses…" (QA, output-disagreement diagnosis) | **\<- THIS CELL.** No prior work found that reads internal-state disagreement *across parallel trajectories* to trigger asking in agentic SWE. |

The wedge is the bottom-right cell, in **agentic long-horizon SWE**, scored on **Ask-F1**. It is genuinely narrow: the three neighboring cells are all populated by 2024-2026 work. The contribution is the *signal definition* (cross-trajectory internal geometry) \+ the *application* \+ the *head-to-head against output-divergence at matched N*. It is **not** "internal beats verbalized" (settled) and **not** "multi-sample disagreement \-\> ask" (settled, output-side).

---

## **2\. Grounded positioning — verified related work**

### **2a. Multi-trajectory output-divergence \-\> ask (your direct baselines)**

* **ClarifyGPT** `[V]` (Mu et al., arXiv 2310.10996, ACM FSE 2024). Samples N code solutions, compares **test outputs**: identical \-\> unambiguous, proceed; different \-\> ambiguous \-\> cluster, ask. This is your method with the *output* signal substituted for the *internal* one. **The single most important baseline.** Function-level (MBPP), single-shot.  
* **LaMSeI** `[V]` (TMLR 2025). Samples multiple responses, measures variation, interacts only under high uncertainty, generates question candidates. QA, output-side.  
* **"Can Multiple Responses from an LLM Reveal the Sources of Its Uncertainty?"** `[V]` (arXiv 2509.04464). Collects multiple responses; an auxiliary LLM analyzes disagreement to distinguish **under-specified input vs. missing knowledge** — i.e., the ask-vs-recoverable triage, but on outputs, in QA (AmbigQA). Cite as the closest conceptual neighbor to the triage idea.

### **2b. Single-trajectory internal signal (feasibility ground \+ the "other" baseline)**

These prove the internal signal exists and is strong — and they bound your novelty, since they already do single-stream internal correctness/abstention:

* **The Confidence Manifold** `[V]` (arXiv 2602.08159). Internal probes 0.80-0.97 AUC vs. output methods 0.44-0.64; "the correctness signal exists internally but is not expressed in outputs"; a model "may produce low-entropy outputs while encoding that the response is likely incorrect." **This is the grounding for the confident-convergence wedge — and a single-stream baseline.**  
* **On LLMs' Internal Representation of Code Correctness** `[V]` (arXiv 2512.07404). A correctness direction from contrasting hidden states of correct/incorrect code **outperforms log-likelihood ranking and verbalized confidence**, in code, without test execution. (See §11 — CARD collision.)  
* **DRIFT** `[V]` (arXiv 2601.14210): intermediate-layer representational inconsistencies, single forward pass, routes high-risk queries — explicitly positioned as *not* needing multi-sampling. (A reviewer will use this to ask "why pay for N?" — see §5e/§5f for the answer.)  
* **Latent Representation Probing** `[V]` (2511.19806), **CCPS** `[V]` (2505.21772), **AutoProbe** `[V]` (code quality, S0164121225002390 / 2501.12934), **intra-layer information scores** `[V]` (2603.22299, cross-*layer* agreement \-\> abstention).

### **2c. Multi-agent / cross-trajectory activation directions (mechanism foundation, not asking)**

* **Latent Agents** `[V]` (arXiv 2604.24881, BU, Apr 2026). Distills debate into one model; finds **agent-specific subspaces** via contrastive activation addition (difference-in-means); steers to *suppress* malicious agents. Proves "different-perspective directions exist in activation space" — foundation for a cross-trajectory direction existing. No human, no asking, post-hoc.  
* **STARS** `[V]` (arXiv 2601.22010, ICLR 2026). Steers concurrent runs *toward* divergence (geometric volume on the Stiefel manifold) for diversity. We **read** divergence, not inject it; no human, no escalation.

### **2d. The SWE when-to-ask cluster (home turf \+ baselines)**

* **HiL-Bench** `[V]` (2604.09408, Scale Labs, Apr 2026). Home benchmark. Single agent \+ `ask_human()`; **Ask-F1** \= harmonic mean of question precision and blocker recall; Pass@3. Failure patterns include "high uncertainty yet errors persist" (verbalized is weak) and "overconfident wrong beliefs, no gap detection" (confident-convergence). Necessity guarantee: without asking, pass-rate \<5%.  
* **UA-Multi / "Ask or Assume?"** `[V]` (2603.26233). Uncertainty-aware **multi-agent** (Intent \+ Main) scaffold, language-space uncertainty; 69.4% vs 61.2% on underspecified SWE-bench Verified. (Multi-agent \!= multi-trajectory; it's a baseline, re-run on your harness.)  
* **Ambig-SWE** `[V]` (2502.13069), **ClarEval** `[V]` (2603.00187) — eval frameworks / metric sources.

### **2e. RL / cost-aware clarification family (from Consensus — VERIFY each `[?]`)**

Optimize when-to-ask with a question cost, but in non-SWE settings on **text/confidence/heuristic** triggers — none on internal representations across trajectories:

* **AskBench \+ Rubric-Guided RLVR** `[?]` (Zhao et al., 2602.11199): accuracy, rubric adherence, interaction efficiency (questions/task), verifier-reward triggers.  
* Chi et al. 2019 (VLN) `[?]`; Wang & Ai 2021 (risk-aware DQN, conv. search) `[?]`; Ramrakhya et al. 2025 (embodied ask-for-help, RL) `[?]`; Rao 2017/18 `[?]`; Xu 2019 (KBQA) `[?]`; Lee 2023 (QA) `[?]`; CodeClarQA / Li et al. ACL 2023 `[?]`.

**Consensus caveat:** its summary concluded a divergence \+ internal-signal SWE agent is "a genuinely new formulation." That is only partly right — it missed ClarifyGPT, LaMSeI, and the entire single-stream internal cluster (§2b). Treat its novelty verdict as **over-optimistic**; the RL papers are related work / cost-baselines, not evidence the space is empty.

---

## **3\. The defensible contribution (multi-trajectory-native, narrowed)**

1. **A cross-trajectory internal-divergence signal** for branch-point detection in long-horizon coding agents: read mid-layer hidden states across the N parallel rollouts at decision points and measure their *dispersion*. Neither single-stream internal probes (§2b — no across-trajectory axis) nor output-divergence triggers (§2a — outputs, not internals) compute this.  
2. **The head-to-head result:** internal cross-trajectory divergence beats **output** cross-trajectory divergence (ClarifyGPT-on-your-harness) at **matched N** on Ask-F1, with the gain concentrated on **genuine-fork blockers** — fired earlier and more precisely because internal conflict can precede visible output divergence, and is present even when two trajectories land on the same artifact for different latent reasons.  
3. **A clean agentic Ask-F1 evaluation** on a HiL-Bench harness with the full baseline ladder (§5d), same backbone, compute-matched.

**Honest venue framing (ICML/top-tier):** this is an application \+ diagnostic contribution in a dense neighborhood. To clear the bar it needs (a) a large, qualitative margin over output-divergence on fork blockers, and/or (b) the **intervention** in §6 (gate/steer), not just a probe. A small Ask-F1 bump over output-divergence will read as incremental.

---

## **4\. Core scientific bets \+ which signal catches which blocker**

Three blocker regimes, three different signals — be explicit, because it determines the win region:

| Blocker regime | Output divergence (ClarifyGPT) | Cross-trajectory internal divergence (ours) | Single-stream internal correctness probe |
| ----- | ----- | ----- | ----- |
| **Genuine fork** (trajectories split on interpretation) | fires (late, coarse) | **fires earlier/precisely — our win** | partial (per-stream confidence, no fork structure) |
| **Confident-convergent wrong** (all N identically wrong) | misses (outputs agree) | **likely misses** (internals also agree; sharply-peaked dist.) | **catches** (correctness direction separates correct/wrong even at low entropy — Confidence Manifold) |
| **Clear / unambiguous** | correctly silent | correctly silent | correctly silent |

**Implications:**

* **C1 (the bet that must hold):** cross-trajectory internal divergence separates fork-blockers from proceed-cases **better than output divergence at matched N**, on Ask-F1. *De-risk first (§7).*  
* **Confident-convergence is a stated limitation of the divergence signal** (same-model N shares the blind spot — grounded in the semantic-entropy-fails literature). You may *include* a single-stream correctness probe (§5d, B3) to cover that regime, but credit those gains to prior work (Confidence Manifold / 2512.07404); the **novelty claim rests on divergence beating output-divergence on forks**, not on confident-convergence.  
* This complementarity is itself a clean ablation story: divergence and probe catch disjoint blocker types; the union is the strongest system, the divergence component is the novel one.

---

## **5\. Experimental design**

### **5a. Benchmark & harness**

* **HiL-Bench, 200 public tasks** (300 total; 100 private held-out \-\> leaderboard numbers are **not** comparable; state "200 public" explicitly).  
* **Your own harness on the HiL-Bench dataset is fine and partly necessary** (the stock `harbor` stack runs API models; you need white-box activation access). Two hard constraints:  
  * **Reuse the judge mechanic verbatim:** the frozen **Llama-3.3-70B semantic judge** \+ blocker registry *defines* Ask-F1. Reimplement it faithfully (same model, same matching rubric) or your Ask-F1 is incomparable. Validate your judge reproduces theirs on sample question/blocker pairs.  
  * **Reuse the execution layer** (SWE/SQL test running) rather than re-deriving it.

### **5b. Backbone & trajectories**

* **Open-weight, white-box** (Qwen3-Coder / GLM / DeepSeek / Llama-3.x) — mandatory for residual access. Every baseline runs on the **same** backbone \-\> you **re-run** baselines, you do not reuse reported numbers.  
* **N trajectories** (start N=5) via temperature sampling / diverse prompting from the same agent scaffold (SWE-agent style), executed in the harness.

### **5c. The signal — how cross-trajectory internal divergence is computed**

* **Where:** at **decision points** — operationalized as tool-call boundaries and/or steps where \>=1 trajectory is about to act on an under-determined choice. Monitor continuously; threshold-crossing \= candidate ask.  
* **Alignment (the hard part, multi-trajectory-specific):** the N runs reach states at different steps. Align by (i) step index for early steps, then (ii) a semantic anchor (same sub-goal / same file under edit) once they diverge. Report sensitivity to the alignment scheme.  
* **What:** extract mid-layer hidden states for each trajectory i (mid layers favored by the probing literature). Divergence \= dispersion of the set of N hidden-state vectors at the aligned point — candidate metrics: mean pairwise cosine distance, total variance, or geometric volume (STARS-style). Pick one primary, ablate the rest.  
* **Trigger:** divergence \> tau at a decision point \-\> call `ask_human()` once (controller above the N; see §5f dedup).  
* **Question content:** generated from the **behavioral fork** — the divergent trajectories supply concrete candidate options ("traj A edits config.py / assumes UTC; traj B edits settings.py / assumes local"). Internal divergence sets *when*; behavioral fork sets *what*.  
* **Resolution injection:** the judge's returned resolution is injected into all N trajectories, which continue.

### **5d. Baselines (the full ladder)**

| \# | Baseline | Signal | Built how | Reuse number? |
| ----- | ----- | ----- | ----- | ----- |
| B0 | Vanilla `ask_human()` | verbalized self-report, single stream | stock agent, your backbone | only same-backbone row, else re-run |
| B1 | **Output cross-trajectory divergence (ClarifyGPT-style)** | test-output disagreement across N | re-implement on your harness, **matched N** | no |
| B2 | Verbalized-across-N | each of N self-reports; aggregate/vote to ask | your harness, matched N | no |
| B3 | **Single-stream internal correctness probe** | learned hidden-state direction (Confidence-Manifold / 2512.07404 / AutoProbe-style) per trajectory | your harness | no |
| B4 | Uniform / random routing | ask at fixed interval / random steps | your harness | no |
| B5 | UA-Multi scaffold | language-space uncertainty, multi-agent | re-implement (code public), score Ask-F1 | no |
| B6 | HiL-Bench RLVR / AskBench-RLVR | trained ask policy | checkpoint or re-run, same backbone | only if same backbone/split |

**B1 is the load-bearing comparison** (same setup, output-\>internal swap, matched compute). **B3 is the "is your divergence just a probe run N times?" control** — your divergence must add value *over* B3 on fork blockers, and B3 will (correctly) win on confident-convergence. **B2** controls for "is internal needed, or does verbalized-across-N suffice?".

### **5e. Metrics (non-overlapping suite; all on HiL-Bench 200-public, same backbone, matched N)**

| Metric | Role | Source |
| ----- | ----- | ----- |
| **Ask-F1** | headline judgment quality | HiL-Bench |
| Question Precision / Blocker Recall (split) | the tradeoff; recall on forks is where you win | HiL-Bench |
| **Pass@k** | correctness guard (no trading accuracy for asking) | HiL-Bench |
| One **efficiency** score (EAR / turns-to-clarify style) | questions-vs-correctness cost; your steering-efficiency angle | adapt from ClarEval (cite) |
| **Blocker-recall sliced by regime** (fork / confident-convergent / clear) | isolates the divergence win and the honest limitation | taxonomy via "Asking What Matters" \+ your labels |
| **Lead-time @ decision point** | does internal divergence fire earlier than output divergence? | your measurement |
| **Compute** (forward passes / tokens) | fairness, reported beside Ask-F1 | — |

### **5f. The four multi-trajectory hazards — and how the harness handles each**

1. **Question-spam \-\> precision collapse:** controller asks **once per detected fork**; dedup across the N. (Ask-F1 punishes volume.)  
2. **Compute fairness:** all sampling baselines (B1, B2, B3-if-ensembled) run at **matched N**; report compute. Gains must survive N-matching. (This is also the answer to DRIFT's "why pay for N?": the cross-trajectory fork structure and the ready-made candidate options are what a single forward pass cannot give — and the comparison to B3 quantifies exactly that added value.)  
3. **Trajectory alignment:** §5c scheme; report sensitivity. This is novel methodological work, not a bug.  
4. **Luck-floor validity:** N independent tries raise HiL-Bench's \<5%-without-asking floor and can decouple Pass from Ask-F1. **Protocol fix:** score Ask-F1 on the *asking decisions*, and report Pass under a **single committed trajectory** (or majority) so parallelism can't launder a lucky pass. State this explicitly.

---

## **6\. The optional intervention (lifts venue tier)**

When the signal says "recoverable, not missing-info," the cheapest move is to **gate** the question (don't ask; have the trajectories reconsider/continue) rather than escalate. Try the **gate first**; reach for an **activation steering** intervention only if the gate underperforms (Occam — reviewers will ask why activation machinery if routing suffices). Grounding: contrastive activation steering can shift a model's confidence to match its internal accuracy signal ("Closing the Confidence-Faithfulness Gap," 2603.25052 `[V]`); Latent Agents shows perspective-subspace steering. **Caveat to state:** steering cannot manufacture genuinely missing information, so it only ever substitutes for the human on recoverable wobbles, never on true information gaps.

---

## **7\. De-risking experiment (two weeks, before method-building)**

1. One open-weight coder, **N trajectories**, on a slice of HiL-Bench public tasks, through your harness.  
2. At each decision point log, per trajectory: mid-layer hidden states, test outputs (for B1), verbalized confidence (B2), and a single-stream correctness-probe score (B3).  
3. Compute cross-trajectory **internal** divergence and cross-trajectory **output** divergence at aligned points.  
4. **Success criterion (the real fork):** does internal divergence separate "should-have-asked" from "fine-to-proceed" **better than output divergence (B1) at matched N**, with the margin on **fork blockers**, and does it fire **earlier**?  
   * Yes \-\> spine confirmed; build the controller (+ §6 intervention).  
   * Internal divergence only matches B1, or only matches B3 \-\> fold: B1/B3 already cover those regimes, no multi-trajectory-internal paper.

Note vs. the earlier (discarded) criterion: "beat verbalized/random" is too easy and is the wrong comparison. The discriminating baselines are **B1 (output divergence)** and **B3 (single-stream probe)**, on the **fork-blocker slice**, at **matched N**.

---

## **8\. Make-or-break claims (for supervisor)**

* **C1:** cross-trajectory internal divergence \> output cross-trajectory divergence (B1) on Ask-F1 at matched N, win on fork blockers. *De-risk first.*  
* **C2:** the full method achieves higher Ask-F1 at equal-or-better Pass@k than B0-B6 at matched compute.  
* **C3 (optional):** divergence is not reducible to B3 — it adds blocker recall on forks beyond a per-trajectory probe (ablation).  
* **Limitation, stated up front:** confident-convergent blockers are a blind spot of the divergence signal (mitigated only by adding B3 / cross-model, credited to prior work).

---

## **9\. Go / no-go checklist (this week)**

* \[ \] HiL-Bench public data \+ frozen judge \+ blocker labels downloadable and runnable locally. **(Verified available for 200 public tasks.)**  
* \[ \] Chosen open-weight backbone exposes mid-layer hidden states *mid-trajectory inside the agent loop*. (Prototype: one model, N=2, one public task, residuals logged at a decision point.)  
* \[ \] B1 (ClarifyGPT-style output divergence) implemented on your harness — this is the comparison the whole claim rests on.  
* \[ \] Confirm no prior work reads cross-trajectory **internal** divergence for asking (re-run the §2 searches; the cell looked open as of this draft but must be re-checked at submission).  
* \[ \] Verify all `[?]` citations.

---

## **10\. Citation verification status**

**Verified `[V]`:** HiL-Bench (2604.09408); UA-Multi (2603.26233); STARS (2601.22010); Latent Agents (2604.24881); ClarifyGPT (2310.10996); LaMSeI (TMLR 2025); "Can Multiple Responses..." (2509.04464); The Confidence Manifold (2602.08159); Internal Representation of Code Correctness (2512.07404); DRIFT (2601.14210); Latent Representation Probing (2511.19806); CCPS (2505.21772); AutoProbe (2501.12934 / ScienceDirect); intra-layer information scores (2603.22299); Closing the Confidence-Faithfulness Gap (2603.25052); ClarEval (2603.00187); Ambig-SWE (2502.13069); ensemble semantic entropy (2603.27098); semantic-entropy-fails survey (2601.09929). **Verify `[?]` before citing:** AskBench/RLVR (2602.11199); Chi 2019; Wang & Ai 2021; Ramrakhya 2025; Rao 2017/18; Xu 2019; Lee 2023; CodeClarQA/Li ACL 2023; Devoto et al. (2410.16090); Gnosis (2512.20578); semantic-geometric co-evolution (2603.13325); SE-Agent (2508.02085); TDScaling (2602.03219); ENTROPO (2509.12434); Kong et al. (2605.17193).

---

- [ ] ***FROM FABLE 5\.***

# **Novelty & Publishability Assessment: Cross-Trajectory Internal Divergence for Agentic Ask-Triggering**

## **TL;DR**

* **The 2×2 novelty cell (multi-trajectory × internal-signal, for agentic ask-triggering) is genuinely OPEN but narrow.** No paper reads internal/hidden-state dispersion ACROSS parallel agent trajectories to trigger `ask_human()`/escalation. The nearest neighbor, INSIDE/EigenScore (ICLR 2024), already computes internal-embedding dispersion across multiple sampled generations — but for single-turn QA hallucination detection, not multi-step agentic asking. The proposal's defense ("not agentic, no asynchronous-alignment problem, no fork-question loop, no ask-triggering") holds, but it defends an *application \+ diagnostic* contribution, not a new primitive.  
* **Calibrated estimates: P(publishable at ICLR 2027 | executed as written AND C1 holds) ≈ 0.55–0.65; P(C1 holds empirically) ≈ 0.55; P(≥1 additional novel finding emerges) ≈ 0.85.** The headline benchmark result alone is borderline; the lead-time / asynchronous-alignment / fork-question-loop contributions are what move it from workshop to main conference.  
* **The researcher's self-assessment ("dense neighborhood, application \+ diagnostic") is ACCURATE and slightly over-cautious.** The neighborhood is dense, but the specific cell is open and the temporal early-warning framing is underexplored. Do NOT collapse this into "EigenScore on agents" — but DO pre-empt that reviewer reflex with explicit head-to-head baselines and the alignment/lead-time contributions.

## **Key Findings**

1. **The defining cell is open.** Internal cross-trajectory divergence at aligned tool-call boundaries to trigger asking — beating output cross-trajectory divergence on fork blockers — is not occupied by any prior or concurrent work found. The risk is not occupancy; it is that a reviewer reassembles the cell from three existing pieces (INSIDE \+ ClarifyGPT \+ HiL-Bench) and questions the increment.  
2. **All \[?\] citations verified; one misattribution.** Of the flagged citations, all resolve to real papers and 15/16 are accurately described. The exception: "AutoProbe" (2501.12934) resolves to a framework named **OPENIA**, not AutoProbe — fix before the supervisor meeting.  
3. **Recent direct competitors exist.** "Ask or Assume?" (2603.26233) is essentially the B5 (UA-Multi scaffold) baseline as a standalone paper; AskBench/Rubric-Guided RLVR (2602.11199) is a concurrent "when-and-what-to-ask" RLVR work; both use output/behavioral signals, not internal cross-trajectory dispersion. HiL-Bench (2604.09408, April 2026\) is brand-new and not yet adopted outside Scale Labs.

## **Details**

### **Novelty cell — the four quadrants, verified**

**Cell 1 (single-trajectory, output/verbalized): Occupied.** Verbalized confidence, P(True), and the HiL-Bench RLVR baseline. HiL-Bench (arXiv 2604.09408, Scale Labs, April 2026\) introduces Ask-F1 (harmonic mean of question precision and blocker recall) and shows frontier models collapse from 75–89% pass@3 with full information to 4–24% when they must decide whether to ask `ask_human()`.

**Cell 2 (single-trajectory, internal): Densely occupied.** The Confidence Manifold (2602.08159; Cho et al.) supplies the headline numbers the proposal leans on: internal probes 0.80–0.97 AUC vs output-based methods (P(True), semantic entropy) 0.44–0.64 AUC, with causal activation steering producing 10.9 pp error-rate changes. Companions: DRIFT (2601.14210, intermediate-layer pre-generation hallucination probe), internal code correctness (2512.07404, hidden-state correctness representation for code selection), CCPS (2505.21772, perturbed-representation stability), OPENIA (2501.12934, white-box code-correctness from internal states — the paper mislabeled "AutoProbe"), latent representation probing (2511.19806, VLM abstention), intra-layer information scores (2603.22299), Semantic Entropy Probes (Kossen, Han, Razzak, Schut, Malik & Gal, 2024; 2406.15927), and InternalInspector (2406.12053). Crowded but single-stream.

**Cell 3 (multi-trajectory, output): Occupied.** ClarifyGPT (Mu, Shi, Wang et al., *Proc. ACM Softw. Eng.*, FSE 2024, Article 103; arXiv 2310.10996) samples N code solutions, compares test outputs via a code-consistency check, and asks when outputs disagree — the exact output-divergence baseline B1. Its headline results: it "elevates the performance (Pass@1) of GPT-4 from 70.96% to 80.80% on MBPP-sanitized," and improves GPT-4/ChatGPT averages across four benchmarks from 68.02%→75.75% and 58.55%→67.22% respectively. Also LaMSeI (TMLR 2025), "Can Multiple Responses from an LLM Reveal the Sources of Its Uncertainty?" (2509.04464, EMNLP Findings 2025), and the semantic-entropy family (Kernel Language Entropy, NeurIPS 2024, 2405.20003; Semantic Density, 2405.13845).

**Cell 4 (multi-trajectory, internal) — the claimed-open cell:** No paper triggers asking from internal cross-trajectory dispersion in agentic SWE. The closest, INSIDE/EigenScore (Chen, Liu, Chen, Gu, Wu, Tao, Fu & Ye, ICLR 2024; 2402.03744), proposes — verbatim — "a simple yet effective EigenScore metric … which exploits the eigenvalues of responses' covariance matrix to measure the semantic consistency/diversity in the dense embedding space." This IS multi-sample internal dispersion, but it is (a) single-turn QA hallucination detection, not multi-step agentic trajectories; (b) has no alignment problem across asynchronous tool-call boundaries; (c) has no `ask_human()` triggering or fork-based question generation; (d) has no temporal/lead-time analysis.

### **How close is each nearest neighbor**

* **INSIDE/EigenScore (2402.03744, ICLR 2024\) — CLOSEST on mechanism.** Same core operation (covariance/dispersion of hidden embeddings across multiple generations). A skeptic's strongest move is "EigenScore applied to agents." Defense strength: moderate-to-strong — the agentic multi-step setting, the asynchronous-trajectory-alignment problem, the fork-question generation loop, and the ask-decision are all genuinely absent from INSIDE. But the *signal computation* is not new in kind; novelty lies in where/when/how it is read and acted upon.  
* **STARS (2601.22010, ICLR 2026\) — CLOSEST on the multi-trajectory-internal primitive, opposite direction.** STARS collects hidden activations of concurrent runs and steers them apart (injects divergence for diversity); the proposal reads divergence for escalation. Clean distinction — cite as the explicit contrast.  
* **SE-Agent (2508.02085, 2025\) — CLOSEST cross-trajectory work in SWE.** Uses cross-trajectory "inspiration" (revision/recombination/refinement) on SWE-bench Verified for self-evolution/search, not escalation, and operates on trajectory text/outcomes, not hidden-state dispersion.  
* **"Ask or Assume?" (2603.26233) — CLOSEST task competitor.** Uncertainty-aware multi-agent scaffold on underspecified SWE-bench Verified; decouples underspecification detection from execution; 69.40% resolve vs 61.20% single-agent. This is the B5 baseline made flesh; it does NOT use internal cross-trajectory signals.  
* **Latent Agents (2604.24881) — adjacent mechanism.** Contrastive activation directions for agent perspectives in internalized debate; not asking-related.  
* **Supporting prior on the early-warning premise:** Semantic Entropy Probes (Kossen et al., 2406.15927) "directly approximate SE from the hidden states of a single generation," report AUROC "between 0.7 and 0.95" rising in later layers, and crucially show "semantic entropy can be predicted before generating … with a single forward pass." This is strong directional support for the proposal's core bet that internal signals lead output signals — but again in single-turn QA, not agentic forks.  
* **Field context:** "Uncertainty Quantification in LLM Agents" (2602.05073) names internal-hidden-state methods and trajectory-level UQ as open challenges, and an ICML 2026 workshop on Uncertainty in Agentic Systems exists — confirming the area is recognized as open AND becoming crowded.

### **Citation verification table**

| Citation | Status | Correct attribution / note |
| ----- | ----- | ----- |
| Chi et al. 2019 (VLN ask-for-help) | VERIFIED | "Just Ask: An Interactive Learning Framework for VLN," arXiv 1912.00915 (sometimes cited 2020\) |
| Wang & Ai 2021 (risk-aware DQN clarification) | VERIFIED | "Controlling the Risk of Conversational Search via RL," WWW 2021, arXiv 2101.06327 |
| Xu et al. 2019 (KBQA clarification) | VERIFIED | "Asking Clarification Questions in KBQA," EMNLP-IJCNLP 2019 (CLAQUA) |
| Lee et al. 2023 (QA clarification) | VERIFIED | "Asking Clarification Questions to Handle Ambiguity in Open-Domain QA," Findings EMNLP 2023 (CAmbigNQ) |
| CodeClarQA / Li et al. ACL 2023 | VERIFIED | "Python Code Generation by Asking Clarification Questions," ACL 2023 |
| Rao & Daumé 2017/2018 | VERIFIED | "Learning to Ask Good Questions," ACL 2018 (Best Long Paper) |
| Ramrakhya et al. 2025 (embodied ask-for-help RL) | VERIFIED | "Grounding Multimodal LLMs to Embodied Agents that Ask for Help with RL," arXiv 2504.00907 |
| AskBench / Zhao et al. (2602.11199) | VERIFIED | "When and What to Ask: AskBench and Rubric-Guided RLVR," Zhao, Fang, Cheng, Feb 2026 |
| Devoto et al. (2410.16090) | VERIFIED | "Analysing the Residual Stream of LMs Under Knowledge Conflicts" |
| Gnosis (2512.20578) | VERIFIED | Intrinsic self-verification probe of hidden states/attention |
| semantic-geometric co-evolution (2603.13325) | VERIFIED (title only) | "Auditing Cascading Risks in Multi-Agent Systems via Semantic-Geometric Co-evolution" |
| SE-Agent (2508.02085) | VERIFIED | "SE-Agent: Self-Evolution Trajectory Optimization," 2025 |
| TDScaling (2602.03219) | VERIFIED (title only) | "Beyond Quantity: Trajectory Diversity Scaling for Code Agents" |
| ENTROPO (2509.12434) | VERIFIED (title only) | "Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization" |
| Kong et al. (2605.17193) | VERIFIED | "Multi-LLM Systems Exhibit Robust Semantic Collapse" |
| DRIFT (2601.14210) | VERIFIED | Intermediate-layer pre-generation hallucination probe \+ router |
| internal code correctness (2512.07404) | VERIFIED | Hidden-state correctness representation for code selection |
| CCPS (2505.21772) | VERIFIED | "Calibrating LLM Confidence by Probing Perturbed Representation Stability" |
| AutoProbe (2501.12934) | **MISATTRIBUTED** | ID resolves to **OPENIA**, a white-box code-correctness framework; not named "AutoProbe" |
| latent representation probing (2511.19806) | VERIFIED | Latent Representation Probing for VLM abstention in Scene-Text VQA |
| intra-layer information scores (2603.22299) | VERIFIED | "Between the Layers Lies the Truth," Badash, Belinkov, Freiman, Mar 2026 |
| Confidence Manifold (2602.08159) | VERIFIED | Cho et al.; 0.80–0.97 internal AUC vs 0.44–0.64 output AUC; 10.9 pp steering effect |
| ClarifyGPT (2310.10996) | VERIFIED | Mu, Shi, Wang et al., FSE 2024; Pass@1 70.96%→80.80% on MBPP-sanitized |
| STARS (2601.22010) | VERIFIED | ICLR 2026, Stiefel activation steering for diversity (injects divergence) |
| HiL-Bench (2604.09408) | VERIFIED | Scale Labs, April 2026, Ask-F1 metric |
| INSIDE/EigenScore (2402.03744) | VERIFIED | Chen et al., ICLR 2024; EigenScore from response-covariance eigenvalues |

### **Probability estimates with reasoning**

**P(publishable at ICLR 2027 | proposal executed as written AND C1 holds) ≈ 0.55–0.65.** With C1 confirmed, the paper has a clean, well-motivated positive result, a strong baseline ladder (B0–B6), regime-sliced blocker recall, lead-time analysis, and a matched-compute protocol — a competent ICLR submission. The drag: (a) the "EigenScore-on-agents" framing risk; (b) HiL-Bench is a single harness from one lab, and reviewers increasingly demand ≥2 benchmarks, multiple backbones, and seed-level significance for agentic papers; (c) two concurrent competitors ("Ask or Assume?", AskBench) crowd the "when to ask" story. With the methodological contributions foregrounded, this clears the bar; as a thin benchmark-only paper, it is borderline reject.

**P(C1 holds empirically — internal cross-trajectory divergence beats output cross-trajectory divergence on fork blockers at matched N) ≈ 0.55.** For higher: the single-stream literature is consistent and directional — internal beats output (Confidence Manifold 0.80–0.97 vs 0.44–0.64 AUC; SEPs encode and predict semantic entropy before generation at AUROC 0.7–0.95; INSIDE's internal-embedding dispersion beats lexical/logit dispersion). The mechanistic prior that internal conflict precedes visible output divergence is well-supported. For caution: those gains are single-turn QA, not multi-step agentic forks; the asynchronous trajectory-alignment step injects noise that could erode the internal advantage; output divergence at tool-call boundaries may be a surprisingly strong baseline (ClarifyGPT works precisely because test-output disagreement is a clean ambiguity signal); and "beats at matched N on the fork-blocker slice specifically" is a narrow, demanding target. Net: a coin flip tilted slightly positive.

**P(≥1 additional publishable finding emerges during experimentation) ≈ 0.85.** Likely-novel standalone observations: (a) **lead-time** (internal divergence preceding output divergence by K steps) — underexplored, genuinely contributive; (b) **asynchronous trajectory-alignment-scheme sensitivity** as a methodological result — novel because nobody has aligned hidden states across asynchronous multi-step agent rollouts; (c) **divergence-direction semantics** (what the fork direction encodes) — novel; (d) **complementarity** between divergence and correctness probes — moderately novel. Expected confirmations (low marginal value): layer-wise localization (already established by Confidence Manifold, intra-layer scores, DRIFT, and SEPs that intermediate layers carry the signal).

## **Recommendations**

**Stage 1 — Before the supervisor meeting (framing, no experiments):**

1. Reframe the headline as the **temporal early-warning claim** (internal divergence as a leading indicator of output divergence by K steps), not merely "internal beats output." This is the most defensible open ground and the hardest to collapse into EigenScore. Validation threshold: non-trivial median lead-time K\>0 on the fork-blocker slice.  
2. Name the **asynchronous trajectory-alignment problem** as an explicit methodological contribution with a named algorithm — no prior work aligns hidden states across asynchronous multi-step agent rollouts.  
3. Fix the OPENIA/AutoProbe citation (2501.12934) and add the direct competitors (Ask or Assume? 2603.26233; AskBench 2602.11199; SE-Agent; STARS) to related work so reviewers see you own the neighborhood.

**Stage 2 — Experimental priorities (contribution-per-unit-risk):**

1. Run B1 (ClarifyGPT-style output divergence at matched N) as the make-or-break head-to-head — this IS C1. If internal loses here, pivot at once to the complementarity/lead-time framing rather than "beats."  
2. Close the **fork-structured question-generation loop** (signal → question → resolution injection → continuation). Nobody has closed this loop with internal signals; it converts a diagnostic into a method and is the single biggest lift over INSIDE.  
3. Add the **gate-vs-ask triage** (recoverable wobble vs genuine information gap, routed by whether divergence is reducible by intra-model reconsideration) — open, and a second contribution axis.

**Stage 3 — De-risking for review:**

1. Add a second benchmark beyond HiL-Bench (the underspecified SWE-bench Verified variant from "Ask or Assume?", or AskBench) plus ≥2 backbones with seed-level significance — directly addresses the dominant agentic-paper review complaint.  
2. Include the **causal intervention** (activation steering at fork points, borrowing the Confidence Manifold's 10.9 pp protocol) to show the divergence direction is causally linked to the eventual fork.

**Benchmarks that would change the recommendation:** If C1 fails at matched N, abandon the "beats" framing and publish the complementarity \+ lead-time \+ alignment-method package. If a concurrent paper appears reading internal cross-trajectory signals for asking before submission, pivot hard to the closed-loop question-generation \+ alignment-method contributions, which remain differentiated.

## **Caveats**

* **HiL-Bench adoption:** As of June 2026, no evidence of adoption outside Scale Labs. A single-harness evaluation is a real review risk; treat HiL-Bench as primary but not sole.  
* **Concurrent-work velocity is high.** Agentic UQ is a recognized open area (2602.05073 survey; ICML 2026 workshop) — this validates importance but raises collision probability before a \~September 2026 submission.  
* **Title-only citations (2603.13325, 2602.03219, 2509.12434)** exist with matching titles, but full abstracts were not machine-readable; verify their exact claims before citing specific numbers.  
* **ICLR 2027 timing** (submission \~September 2026, the cycle after ICLR 2026\) is inferred from ICLR's standard annual cadence; confirm exact dates when the official CFP posts.  
* The probability estimates are calibrated judgments from the probing/UQ literature and current agentic-paper review norms, not guarantees.

