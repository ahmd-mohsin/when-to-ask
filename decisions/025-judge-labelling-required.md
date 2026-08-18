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

## Amendment A (2026-08-16, owner in-session — BEFORE any judge label exists)

Written while this entry is still PROPOSED and before any judge-labelled
datum was produced, preserving the pre-registration ordering. Three changes
to the execution plan; the decision in §4 and its three conditions stand.

**(1) §5's transport is replaced.** The owner has no Anthropic API key (a
Claude subscription only), so labelling runs as **Claude Fable 5 subagents
inside the owner's Claude Code session**, not via the Batch API. The judging
protocol is a frozen prompt (specs/judge_labels.md §3, wta/judge_labels.py
SYSTEM_PROMPT); items are batched ~10 per subagent grouped by prompt size;
raw responses are written to disk before scoring. Disclosed limitation: an
agent harness wraps the judge instead of a bare API call, and there is no
API request log — the frozen items file + raw response JSONL + frozen
artifact are the audit trail. In-repo precedent for in-session judge labour
with a frozen audit trail: the 024 fork-type map. The Batch-API path stays
implemented in wta/judge_labels.py should a key ever exist. Cost is paid in
subscription usage, not the §5 dollar estimate; the full 3,272-item pass is
chunked and resumable in case usage limits interrupt it.

**(2) §4a's named harness cannot score this judge — a purpose-built gate
replaces it.** `scripts/validate_judge.py` is a two-path agreement harness
for the ask-injection task (agent question -> blocker id, 34 frozen pairs,
requires the vendored hil-bench judge behind a local vLLM endpoint at
:8808); the labelling task has a different input (commitment trace window),
a different output (class id + confidence + evidence span), and a different
metric. Validation therefore runs on a NEW frozen set,
`data/label_judge_validation.json`: 300 lexicon-labelled commitments (all
trace-sourced + a seed-20260816 sample of actions-sourced), judged blind
with the production prompt builder. PRE-REGISTERED PASS BAR (specs/
judge_labels.md §6): accuracy vs lexicon labels >= 0.90 overall AND >= 0.90
on the actions-sourced stratum, abstention <= 0.20, on non-abstained items.
Any miss -> STOP, owner review, no full labelling run. Known optimism of
this bound (validation traces contain signature phrases by construction) is
recorded in the spec; mitigation is the per-label verbatim evidence span,
which must be locatable in the raw trace for a label to be accepted at all,
plus a sampled human audit of production labels before any gate run.

**(3) Scope runs on the merged 1415-run collection** (30 recovered runs
merged 2026-08-16; HANDOFF_R2_GATES §5), enumerated from a fresh lexicon
pass over it — not the 1385-run snapshot's literal 3,272 list. The 1385
snapshot and its §1 numbers remain frozen and untouched.

Validation accuracy (recorded per §4a before any labelling run):

**RESULT 2026-08-16/17: STOP. The judge misses all three pre-registered
criteria.** 292 of 300 items judged (8 lost to a subscription session limit
mid-run and not retried — see below); Claude Fable 5, in-session subagents,
production prompt builder, blind to the lexicon label.

    accepted   196      abstained 96 (32.9% of judged; bar <= 20%)  STOP
    accuracy   0.765    (bar >= 0.90)                               STOP
      actions  0.832    (bar >= 0.90)  n=107                        STOP
      trace    0.685                   n=89
    other/missing 8 (the un-run work file)

Pre-registered confidence sensitivity (spec §7: primary arm 0.7, report
0.0/0.9) — the judge IS calibrated, accuracy rises monotonically with its
own confidence, but no threshold reaches the bar:

    conf>=0.0  n=196  0.765      conf>=0.9   n=75  0.893
    conf>=0.7  n=149  0.826      conf>=0.95  n=35  0.943

No full labelling run is authorised by this result. The bar is NOT moved and
the 8 missing items are not re-run to chase it: on the 292 judged the miss is
7-13 points, far outside what 8 items could close.

**Diagnostic caveat that the owner must weigh (not an appeal of the STOP).**
The validation "ground truth" is the lexicon teacher's own label, and the
trace stratum is the teacher's KNOWN-noisy path — spec labels.md §v2 documents
verified trace-path mislabels (that is why action-based commitment labelling
was built at all). Judge-vs-lexicon accuracy is therefore a disagreement rate,
not an error rate: 0.685 on trace vs 0.832 on actions is exactly the ordering
"the trace labels are themselves wrong a lot" predicts. Whether the 46
disagreements are judge errors or lexicon errors is a separate, answerable
question (blind adjudication, decisions/024's method); its outcome does not
alter this STOP, but it decides which lever is worth pulling next.

## Amendment A diagnostic (2026-08-17) — the STOP stands, the yardstick is bent

Three post-hoc analyses, run to interpret the STOP (not to appeal it).
Nothing here changes the pre-registered bar or the §1 numbers.

**(i) Where the failure lives.** Both failed criteria are concentrated in one
stratum, and one is a defect in OUR code, not the judge:

    stratum   abstention   accuracy      trace policy   abstention
    actions    7.8%         0.832        full            26.6%
    trace     49.4%         0.685        excerpt         72.5%

The `excerpt` path (spec §5, fires above 30k chars) abstains at 72.5% — it
elides the very evidence the judge needs. 40/292 validation items and
276/3,361 production items are excerpted, so this is a real but bounded
implementation defect. `scripts/validate_label_judge.py` numbers above are
NOT re-run with a fixed policy; doing so would be tuning to the bar.

**(ii) Blind adjudication of all 46 disagreements: 46-0 for the judge**
(`scripts/adjudicate_label_disagreements.py`, blind A/B, provenance hidden,
per-item deterministic order; adjudicator = Claude Opus 5, a DIFFERENT model
from the judge, deliberately). 18-0 on the actions stratum — the project's
gold standard. Verdicts and reasoning: results/label_disagreement_adjudication.json.

Discount this appropriately: the adjudicator applies the same
actions-over-prose rubric the judge was given, so it is not fully independent.
It is not circular either — that rubric is the REPO'S definition of
behavioural commitment (decisions/017, spec labels.md v2), adopted precisely
because whole-trace matching was proven wrong at 14B. The adjudicator is
applying the project's own ground truth, not the judge's private preference.

**(iii) The decisive evidence needs no model at all.** On
swe_19/contradictory_remove_vs_deprecate_unsafeproxy, run s0, the winning
signature `"class unsafeproxy"` (class keep_class_with_identity_preservation)
scores 4 hits — three of them inside

    sed -i '/class UnsafeProxy:/,/^$/d' lib/ansible/utils/unsafe_proxy.py

a command that DELETES the class. The lexicon read a deletion as evidence of
keeping. That signature is built from the blocker's own anchors ("class",
"unsafeproxy"), i.e. it matches the TOPIC, not the resolution — the exact
leak class scripts/audit_class_artifact.py warns about (318 warnings, 0
errors on the current artifact; these warnings are load-bearing, not noise).

**A distinct failure mode from the one §3 identified.** §3 says the loss is
coverage (68% unlabelled). This is different and worse in kind: the run IS
labelled, and labelled WRONGLY, in a direction that ERASES forks. On swe_19
the lexicon labels 15/15 runs "keep" — a genuine 3-way decision collapses to
a unanimous non-fork, and a decision with one committed class leaves the
gate-5 set entirely. Note decisions/018 cites this same blocker as a
strongest-separating structural fork at 14B (LORO 1.00).

Scale (`scripts/audit_signature_discrimination.py`, model-free, over all 214
blockers x 1415 runs; "weak" = won by score 1 with margin 1, "anchor-built" =
winning signature contains or is contained by one of the blocker's anchors):

    blockers with >=1 action-stage label   159
      forked (>=2 committed classes)        59
      unanimous                            100
      unanimous, >=4 runs                   71
        of which mostly-weak evidence       23
        of which ALSO anchor-built           3   <- strict suspects

The strict count is **3** (swe_19 remove-vs-deprecate, swe_0
non_linux_distribution_normalization, swe_10 conflicting_normalized_value...),
up to 23 under the looser "unanimity resting on single-hit evidence" reading.
So this does NOT wholesale explain the gate-5 null — it is a narrow defect.
Its weight is that two methods sharing no mechanism converge on the same
blockers: all 3 strict suspects are also where the judge disagreed (7 of the
46 disagreements land on them).

**(iv) What this says about the validation design (our error, not the
judge's).** The set is 60% trace-sourced (181 of 300) because §6 specified
ALL trace-sourced items plus a sample of actions-sourced ones. That
over-weights the teacher's known-noisy path, and scores judge-vs-lexicon
disagreement as if it were judge error. A yardstick built from lexicon labels
cannot settle a dispute about lexicon labels. Compounding it, no validation
set built from lexicon-LABELABLE items is distributionally representative of
production items, which are by definition the ones the lexicon CANNOT label
(spec §6 recorded this limit; its magnitude was underestimated).

**Open, and owner's to decide** (options in the handoff, none actioned):
a real ground truth requires labels the lexicon did not produce — i.e. owner
hand-labelling of a production-distribution sample. Until then neither "the
judge is good enough" nor "the judge is unusable" is established. The STOP
holds; no production labelling has been run.
