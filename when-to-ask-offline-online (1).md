# How the When-to-Ask Detector Actually Works — Offline vs Online (v2)

A plain-English companion to the implementation spec. It answers three questions and nothing else:

1. What do you have to **learn / build** for each signal?
2. **Where exactly** does the autoencoder get used (and where it does not)?
3. What runs **offline** (once, before deployment) vs **online** (live, during every run)?

> **What changed from v1:** "lean" was hiding two different quantities. They are separated below (**ambiguity signal** vs **resolution lean**), and the disagreement trigger now fires on the right one. The resolution lean is a *learned vector and a real runtime component* — v1 wrongly called it training scaffolding. Several sections changed as a result; the reasoning is in §"Three signals, named precisely".

---

## The one rule everything obeys

You compare runs by **the decision they are facing**, never by step number and never by which action they took. Step and action alignment is the thing we threw out. Hold this and the rest follows:

- **Actions and observables are a *teacher*, used offline to make training labels.**
- **They are never a *trigger*** — they don't decide when you read, and they never decide how you line runs up at runtime.

If you find yourself reading the residual stream "right before the tool call," or matching runs "at the same step," you've broken the rule. Read during the reasoning *before* the action, and match on topic.

---

## Three signals, named precisely (this kills a confusion we were carrying)

Up to now "lean" meant two different things, and they are **not the same axis**. Name them apart:

- **Ambiguity signal `s = dot(h, d)`** — a single number. *How under-specified / should-ask does this state look.* Same idea for every decision, so it's one fixed direction `d`. Its jobs: flag that a decision is being faced, and help tell when a run has stopped deliberating. **It is NOT what you fire disagreement on.**
- **Resolution lean `r = L(h)`** — a small vector. *Which interpretation this run is going toward* (retry-all vs retry-transient; tabs vs spaces). This differs from decision to decision, so it **cannot** be a single fixed direction — it has to be learned. **This is the thing the disagreement trigger measures.**
- **Commitment** — *how settled `r` is*: `r` has stopped moving over the last few reads (and `s` has dropped). A run only votes once its `r` is settled.
- **Topic `T(h)`** — a small vector: *which decision is being faced*, built to ignore `r` so that forking runs land in the same place.

**Why the s-vs-r split is load-bearing.** The whole reason this method exists is the **confident fork** — two runs each sure, but sure about different interpretations. In that state both runs look *un-ambiguous*: `s` is LOW for both. So if you fired on spread of `s`, you'd see near-zero spread and miss the fork — the exact blindness the method is designed to beat. The fork lives in `r`: run A's `r` points at retry-all, run B's at retry-transient, even though both are confident. **You fire on spread of `r` within a decision-bucket, never on `s`.**

`s` is a scalar you get for free from a difference-of-means direction (no network). `r` and `T` are vectors that come out of the one autoencoder. That asymmetry is exactly why the autoencoder shows up in one place and not others.

---

## Where you read `h`, and when

Hook **one mid layer** of the frozen agent and grab the residual vector `h`. The important part is *when*:

- Read **throughout the reasoning span** — at a fixed cadence (e.g. every K tokens) or at deliberation cues ("hmm", "wait", "let me", "actually"). Do **not** wait for the tool call / edit.
- A run's **decision point is detected from the signals themselves**: `s` is high (an under-specified decision is in play) and the topic has settled onto one sub-problem. You don't need to know an action is coming to notice this.
- The run casts its vote when its `r` settles (commitment) — which happens *before* it emits the action. That gap (internal commitment -> visible action) is your lead-time, and it only exists if you read early.

Reading at the action boundary makes the internal signal and the behavior coincide, so the lead-time looks like zero even when it isn't. That is the single most common way to accidentally kill the project.

---

# PART A — OFFLINE (build and train once, before any live use)

Everything in Part A is done ahead of time on logged data. Nothing here runs during a live task.

## A0. Collect trajectory data

Run the agent **N times** on under-specified tasks. Force the runs to actually disagree by varying **seed and temperature** (and optionally a light persona nudge). If all N runs pick the same interpretation, there's no fork to study — so check and report how much diversity you got.

For every run, at each read position across the reasoning span, log:

- the **residual vector `h`** (raw material for all signals), and
- the **observables** — which file/function is being edited, which code region/span is in focus, which test is failing, what sub-goal the run stated, and (once it acts) **which action it committed to**.

The observables are not part of the live detector. They exist only to make training labels below.

## A1. Ambiguity signal — build the should-ask direction `d` (NO autoencoder)

Not a learned network. A single direction, built by subtracting two averages.

1. Take labeled decision states: some **under-specified / should-ask**, some **specified / proceed**. (Label source: HiL-Bench's blocker registry, or any ambiguity-labeled corpus.)
2. Average the "should-ask" residuals -> `mu_plus`. Average the "proceed" residuals -> `mu_minus`.
3. `d = (mu_plus - mu_minus)`, normalized. (Difference-in-means / CAA recipe — Rimsky et al., Marks & Tegmark.)

At runtime the **ambiguity signal is `s = dot(h, d)`**. Its three jobs: (1) mark which junctures are ambiguous decisions worth watching, (2) help time commitment (`s` drops as a run settles), (3) anchor the resolution lean in A2 so "different lean" means a real resolution difference, not an incidental action difference. **`d` is NOT the lean.** No network, no autoencoder.

## A2. Topic AND resolution lean — the disentangling autoencoder (the only autoencoder)

This is the one and only place an autoencoder is used, and it's used **offline only**. It produces **both** `T(h)` (topic) and `L(h)` (resolution lean). Both are real runtime components — v1 wrongly called `L` scaffolding.

Why an autoencoder? You can't get a lean-free topic by clustering raw `h`: the fork *is* a difference in `h`, so any raw-similarity method splits exactly the runs you want to match. You have to *learn* a representation with the lean stripped out. And you need a `r` that separates interpretations *per decision*, which a fixed direction can't do. The autoencoder learns both.

**Shape of the network (backbone: ReDAct, arXiv 2602.19396):**

- **Encoder** — a shared 2-layer MLP body feeding two heads: `T(h)` (topic vector) and `L(h)` (resolution-lean vector).
- **Decoder** — a small MLP that takes `(T, L)` and rebuilds `h`.

**The four "pulls" you train it with:**

1. **Topic names the decision.** Train `T` so a simple classifier can recover *which* juncture this is from `T` alone — using the observable decision-identity label from A0. This gives the topic its meaning.
2. **Resolution lean names the interpretation.** Train `L` to predict the **interpretation class** — from HiL-Bench's blocker registry where it exists ("retry-all" vs "retry-transient" as *semantic categories*, NOT raw action strings and NOT hand-assigned +1/-1); fall back to clusters of the observable committed action where no registry class exists.
   - **Where specificity comes from (read this — it is NOT an alignment to `d`).** You want "different lean" to mean *a different resolution of the ambiguous decision*, not a trivial action difference (different variable name, formatting). That specificity comes from the *supervision*, not from aligning `L` to the direction `d`: the registry's class list only contains ambiguity-relevant interpretations, so `L` is trained to encode only those — trivial differences are not in the label set, so `L` never learns to represent them. For the off-registry fallback (where you cluster observable actions), specificity comes from **only including junctures where the ambiguity signal `s` is high** — i.e. `s` *gates which decision points count*, it does **not** supply a signed target for `L`. There is no "align `L` to `d`" step, and therefore no per-fork signed scalar. (This sentence exists because the earlier phrasing "tie `L`'s variation to `d`" was misread as exactly the signed-scalar alignment we are avoiding.)
3. **Topic blind to lean — the load-bearing pull.** A small adversary tries to predict `r` from `T`; train `T` so it **fails** (gradient reversal), and keep `T` and `L` pointing in different directions (orthogonality). This earns the invariance; without it, topic quietly carries the lean and forking runs scatter.
4. **`(T, L)` rebuild `h`.** Reconstruction forces the two summaries to retain the real information in `h`, so neither collapses to something trivial.

**Total objective:** `topic-supervision + lean-supervision(interpretation class) + (adversarial-invariance + orthogonality) + reconstruction`.

**What you keep:** freeze the network; keep **both the topic encoder and the lean encoder**. Discard the decoder for runtime.

> **On the signed-scalar question (resolved).** You do NOT assign +1/-1 per fork. `L` predicts the interpretation *class* (a categorical label), and the geometry — different interpretations landing at different points in `L`-space — emerges. At runtime you measure how spread-out the `L` vectors are within a bucket. The only human-judgment source is the registry's list of interpretation classes: a fixed artifact built once, not per-fork manual work. Off-registry decisions fall back to observable-action clusters (noisier — see A4 checks and the co-divergence fallback).

## A3. Commitment — derived from `r`'s stability; you only calibrate a threshold (NO separate probe)

Commitment is **not** a network you train. It's a function of `r` and `s`:

```
commitment = "r has stopped moving over the last few reads"  (low recent movement of L)
             corroborated by "s has dropped"                 (state no longer looks ambiguous)
```

A run mid-deliberation: `r` still moving, `s` high -> not committed. A run that's decided: `r` steady, `s` low -> committed.

The only thing to **set** is the threshold `tau` for "settled." Don't hand-tune it. Use **split conformal prediction** (LYNX, arXiv 2512.05325):

- **Calibration label** (per run, per decision): the **stabilization point** — the last read before the action where `r` settled and did not flip until the action.
- **Nonconformity score:** instability at a candidate point (recent movement of `L`; equivalently `1 - stability`).
- **Guarantee it buys:** `P(r moves materially after we declared committed) <= delta`.

This is computable because **stability is observable from the trajectory** — unlike answer *correctness*, which needs a right answer that forks don't have. So: borrow LYNX's calibration procedure; do **not** borrow the correctness target. (Papers that predict answer correctness are uncomputable at your forks by construction.)

## A4. Validate before you trust it (the gate) — with the conflation and separation checks

Run these on held-out data. **Do not build the live trigger until they pass.**

1. **Topic blind to lean.** Predict `r` from `T` alone -> should be **no better than chance** (quantify with ReDAct's eta^2 / ANOVA). If it succeeds, invariance broke — fix A2.
2. **Topic names the decision.** The same `T` must predict decision-identity **well**.
3. **Forking runs collocate — and this sets the bucketing threshold.** On held-out forks (same decision, opposite resolution), topic cosine must be **high**. The distribution of same-decision vs different-decision cosines gives you `theta` for runtime bucketing (§B): set `theta` at the crossover.
4. **Conflation check (the new risk).** Take pairs that touch the *same observable* (same file) but are *different decisions*; confirm they do **not** collocate in topic space. If they do, your labels conflated two decisions — enrich the labels (see note below) before trusting the bucketing.
5. **Resolution lean separates interpretations.** Within held-out buckets, confirm different-interpretation runs have **well-separated `L` vectors** — otherwise the trigger has nothing to fire on.
6. **Transfer.** Train on some task families; measure bucket purity on unseen ones. Report as an honest limitation.
7. **Lead-time (make-or-break).** On known forks, confirm bucket disagreement rises **K > 0 steps before** the runs take different actions. Positive K -> the internal signal leads behavior. K ~ 0 -> you're reading too late (revisit "where you read `h`") or the signal doesn't lead (pivot the framing).

> **Labeling note (mitigating conflation).** Do not label a decision by the file alone — that conflates two decisions that touch the same file. Use a **composite label**: file + code region/span + stated sub-goal + error signature. If labels are still noisy, add a **co-divergence** signal as an extra label source: runs whose resolution leans diverge *together* at overlapping reasoning spans are evidence of the same underlying decision. This is used **only offline** to build labels and is validated by checks 3-5, so it is not circular with the runtime detector.

---

# PART B — ONLINE (runs live, per run, per read — nothing is trained here)

At runtime you only run **forward passes of already-frozen things**. No training, no decoder, no reconstruction. The autoencoder from A2 contributed exactly two things to this stage: the frozen **topic encoder `T`** and the frozen **lean encoder `L`**.

Per live run, at each read position across the reasoning span:

```
h        = residual at mid layer (read during reasoning, BEFORE the action)
s_i      = dot(h, d)                  # ambiguity signal (scalar): gates / helps commitment timing
r_i      = L(h)                       # resolution lean (VECTOR): the fork signal
topic_i  = T(h)                       # decision vector: the matching key
commit_i = (recent movement of r is low) and (s_i has dropped)   # settled?
```

Bucket assignment — leader/threshold clustering on the topic stream (async-safe, persistent buckets):

```
b = nearest bucket to topic_i by cosine
if cosine(topic_i, centroid[b]) < theta:   b = new bucket()     # theta from A4 check 3
update centroid[b] with topic_i                                 # running mean

# hysteresis (stop one run's decision fragmenting into two buckets):
#   a run stays in its current bucket unless its topic leaves theta for M consecutive reads.
# merge (fix order-dependence): if two centroids come within theta of each other,
#   merge the buckets (pool their votes). Greedy leader-clustering is arrival-order
#   dependent; the merge step makes the final bucketing order-invariant in practice.
```

Two things the bare "nearest-bucket" rule misses, both handled above: **order-dependence** (greedy assignment can seed the same decision as two buckets depending on which run's read arrives first — the merge step collapses them) and **within-run fragmentation** (a single run's topic wobbles across reads and spawns a spurious second bucket — the hysteresis holds it). `M` and the merge threshold are small sweeps in Phase 2.

Voting — **mutable** (fixes stale votes):

```
if commit_i:
    bucket[b].vote[run] = r_i          # the run's CURRENT committed lean; overwrite if r moves
else:
    bucket[b].vote.pop(run)            # run went back to deliberating -> retract its vote
```

A "vote" is the run's current committed resolution lean, tagged with the decision-bucket its topic falls in. It updates as the run's `r` moves and is retracted if the run de-commits. Because a slow run's topic is resolution-invariant, its vote lands in the right bucket whenever it arrives — that's async dissolving.

Trigger:

```
# for any bucket with >= 2 votes:
spread_t = commitment-weighted dispersion of the r vectors in the bucket
           (e.g. mean pairwise distance, or trace of covariance;
            soft weights w = sigmoid(alpha*(commitment - threshold)), never hard-exclude a run)

S_t = max(0, S_{t-1} + spread_t - reference - slack)      # online CUSUM
fire ask_human() when S_t > h_threshold
```

Fire **only on persistent** dispersion. A spread that collapses as runs reason more was a recoverable wobble (and mutable votes will have updated by then) — don't interrupt. A spread that holds is a genuine fork — ask. A run stuck in a loop (repeated environment states) never commits, so it can't vote; give it a **separate** channel `D_total = D_disagreement + lambda * D_loop` so a thrashing run still counts.

On fire: the divergent `r` vectors (and the runs' committed actions) are two readings of the same ambiguous decision — turn them into the question's options, inject the human's answer into all runs, continue.

**What does NOT run online:** the decoder, the reconstruction, the adversary, any training. Live, you run the frozen `T` and `L` forward passes + one dot product (`s`) + clustering + a CUSUM. That's the entire cost.

---

## Quick reference — what you learn, how, and whether an autoencoder is involved

| Signal | Runtime form | What you do offline to get it | Autoencoder? |
|---|---|---|---|
| **Topic** `T(h)` | small vector | Train the disentangling autoencoder on observable-labeled junctures (A2); freeze the topic encoder. | **Yes — the one autoencoder, offline only.** |
| **Resolution lean** `r = L(h)` | small vector | Same autoencoder's lean head, supervised by interpretation *class* (A2); freeze it. | **Yes — same autoencoder.** |
| **Ambiguity signal** `s = dot(h, d)` | scalar | Build `d = mu_plus - mu_minus` by averaging labeled should-ask vs proceed states (A1). | No — a difference of two averages. |
| **Commitment** | `r` steady + `s` dropped | Nothing to train; calibrate `tau` with split-conformal on the stabilization label (A3). | No — derived from `r` and `s`. |

---

## Open items that stay empirical (not specced — settle with data in Phase 2)

- **Label quality (conflation).** Whether composite labels avoid conflating two same-file decisions is empirical. A4 checks 4-5 are the gate; the co-divergence signal is the fallback if labels are too noisy.
- **`r`-spread scale across buckets.** Different decisions may have different natural `L`-scales, so raw dispersion may not be comparable across buckets before the CUSUM threshold. May need per-bucket normalization. Sweep in Phase 2.
- **Specificity of the interpretation-class set.** How coarse/fine the registry's class list is (and, off-registry, how high to set the `s`-gate) trades sensitivity against false forks — too coarse and you miss real forks, too fine and you fire on hair-splitting. A design choice to sweep, tied to the registry, not a signed-scalar knob.
- **Conformal exchangeability.** Split-conformal's coverage guarantee assumes the calibration and runtime points are exchangeable. Per-step reads *within* one trajectory are autocorrelated, which strains that assumption (LYNX sidesteps it by calibrating at sparse cue tokens). Calibrate on one read per (run, decision) rather than every step, and treat the guarantee as approximate under shift.
- **Oscillating runs (a third state).** A run whose `r` never stably settles is neither a clean vote nor a clear loop. Mutable votes + CUSUM absorb brief oscillation, and the loop channel catches perpetual non-commitment, but a run that oscillates *without* repeating environment states falls between them. Decide whether such a run contributes a low-weight vote or is excluded; sweep in Phase 2.

---

## Why actions are the teacher but never the trigger

Actions and observables are everywhere in the offline work and nowhere in the live trigger, and that's deliberate:

- **Offline (teacher):** the file/region/sub-goal an action touches becomes the *decision-identity label* (topic supervision); the interpretation class it commits to becomes the *resolution label* (lean supervision); the stabilization point becomes the *commitment calibration label*. No human annotation, no answer key — just the run's own observable behavior, plus a one-time registry of interpretation classes.
- **Online (never the trigger):** you don't wait for an action to read `h` (you read earlier, during reasoning), and you never line runs up by their actions or steps (you bucket by topic). The learned encoders are *finer and earlier* than the noisy trace they were trained from — which is the whole point of learning them instead of matching on observables forever.

If anything in the live loop depends on an action or a step index, that's the bug. The actions already did their job; they did it offline.
