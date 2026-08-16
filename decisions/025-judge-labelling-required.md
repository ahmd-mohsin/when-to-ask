# 025: labelled forks are the binding constraint — judge-labelling PRE-REGISTERED

Date: 2026-08-16. Status: PROPOSED (owner to accept before any new gate number
is produced). Follows the R2 gate run of 2026-08-15/16 (decisions/011 STOP
point) and answers the one criticism the A4 gate numbers cannot currently
survive: statistical power.

**Pre-registration timing.** This entry is written BEFORE any judge-labelled
data exists and BEFORE any gate has been recomputed on it. That ordering is the
whole point — swapping the labelling teacher after seeing a null is the move
that would make the negative indefensible.

## 1. What the R2 gate run established

Gate 5 fails at every layer, and the layer confound is closed:

    L  frac | g5_ratio  +-sd   g5_sil | g1_acc chance | g7_medK  fpos
    0  0.20 |   0.832  0.461   0.094  |  0.735  0.482 |   19.9   0.95
    1  0.30 |   0.890  0.366   0.094  |  0.768  0.482 |   20.8   0.95
    2  0.41 |   0.840  0.294   0.089  |  0.782  0.482 |   20.4   0.95
    3  0.50 |   0.894  0.279   0.091  |  0.778  0.482 |   20.0   0.95  <- PRE-REGISTERED
    4  0.59 |   0.869  0.362   0.088  |  0.747  0.482 |   20.2   0.95
    5  0.70 |   0.764  0.313   0.042  |  0.729  0.482 |   20.8   0.95
    6  0.80 |   0.746  0.322   0.063  |  0.756  0.482 |   20.7   0.95
    7  0.84 |   0.625  0.192   0.051  |  0.747  0.482 |   19.6   0.95

The between/within ratio never reaches 1.0 across the full depth span
0.20–0.84, and the pre-registered mid-layer is the BEST of the eight. Per
decisions/015's precedent at 7B, "wrong layer" is ruled out. Corroborated by:
CPU vs GPU agreeing to 0.029 at layer 3 (an order of magnitude below the fold
sd of 0.44), and the completed 1415-run collection tightening layer 3 to
0.805 +- 0.148 on 38 decisions.

## 2. Why that is not yet a publishable negative

**32 pooled forked decisions, 3–9 per fold, 4% class coverage.** A null of that
size cannot separate "the lean is not linearly readable from mid-layer states"
from "we lack the labelled forks to see it". The layer sweep closes the layer
confound; nothing in the current run closes this one. It is the criticism a
referee reaches for first, and the honest answer today is that we cannot
answer it.

Two results say the signal is weak-but-present rather than absent, which is
exactly why power matters here:

- Raw-h cross-run separability (decisions/015's decisive diagnostic, no A2,
  micro-averaged as in 015): **structural +0.097, value +0.126** over chance at
  layer 3, against 7B's +0.015. Something is there.
- The near/far diagnostic refutes the literal-copying explanation for the value
  arm: FAR reads (>=24 tok from any signature mention) carry the separability
  (+0.135) while NEAR reads (<=12 tok) sit below chance. The lean is durably
  represented, not a copied token in context.

## 3. The binding constraint is LABELLING, not collection

All 214 decisions already have >=4 runs committed. The loss is entirely at the
labelling step:

    commitment (run,decision) pairs: 4931
      labeled  : 1574 (31.9%)
      UNLABELED: 3357 (68.1%)
    why unlabeled:
      3272  no signature hits      <- 97.5% of the loss
        85  tie between top classes

The runs DID commit to an interpretation; the class artifact's signature
lexicon cannot read which one. Where labelling succeeds, **43% of decisions
turn out to be forks** (62 of 145 with >=2 labeled runs). Labelling the
remaining two-thirds at that rate plausibly takes 62 forked decisions past 100.

More seeds would NOT help — every decision already clears the run-count bar.
More tasks would burn the sealed pool (decisions/019). This is the cheap lever.

## 4. Decision

Judge-label the 3,272 unlabelled commitments, subject to three conditions.

**(a) Validate the judge first.** `scripts/validate_judge.py` scores any judge
against the frozen `data/judge_validation_pairs.json`. A judge at 80% accuracy
manufactures fake forks and would produce a *worse* artifact than the current
sparse one. Validation accuracy is recorded here before labelling runs.

**(b) The labelling teacher change is pre-registered by this entry.** The
current teacher is signature-lexicon matching over the trace
(`wta.labeling.build_labels`). Adding a judge-sourced label path changes the
teacher, so gate numbers computed on judge-labelled data are NOT comparable to
the 2026-08-15/16 run and must be reported as a separate, labelled arm — never
substituted for the numbers in §1.

**(c) The §1 numbers stand regardless of the outcome.** If gate 5 remains flat
with 100+ forked decisions, that is a strengthened negative and the stronger
paper. If it moves, the §1 run is reported as the underpowered precursor. Both
outcomes are pre-committed here; neither is allowed to retro-select which run
becomes the headline.

## 5. Execution note

Labelling runs off-box (owner's laptop, Claude Fable 5 via the Batch API).
Cost estimated at $15–80 for all 3,272 items depending on trace-window size —
negligible against the 33–45 GPU-hours the collection cost. Implementation is a
new judge: `harness/.../ApiJudge` targets an OpenAI-compatible endpoint
(configs/derisk.yaml: Llama-3.3-70B), and the Anthropic API is not
OpenAI-compatible, so this is not a base-URL swap.

Judge output must carry, per item: chosen class id, confidence, and the
evidence span it read. The evidence span is what makes the new labels
auditable in the same way `labels_debug.jsonl` makes the current ones auditable
— without it there is no way to review a judge label after the fact.

## 6. Status of the sealed pool

Untouched. decisions/019 gates the sealed test on gate 5 passing; it has not
passed, so no sealed-test execution is authorised by this entry. Nothing here
changes that gate.
