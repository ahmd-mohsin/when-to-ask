# Audit of the R2 negative — 2026-08-18

Multi-agent audit (7 auditors + adversarial verifiers, all findings reproduced by
running code against the real repo/data; scratch scripts in the session
scratchpad). Question asked: is the R2 all-gates-negative real, or is a defect
suppressing signal? Answer: **the negative is NOT established — the labels that
every gate consumed are corrupted by two confirmed coordinate bugs — and the
"encouraging" small-model results that justified scaling were baseline
artifacts.** Both the negative story and the positive story are currently
unsupported. A label rebuild + gate rerun decides it.

## A. Confirmed label-corruption bugs (false-negative direction)

**A1. CRLF newline-translation mismatch — CRITICAL, independently verified.**
`src/wta/labeling.py:237` reads the trace with `Path.read_text` (universal
newlines: CRLF→LF) while offsets are accumulated from the raw `.segments.json`
strings (`:243-250`), which preserve `\r\n`. Every `\r\n` shifts all downstream
offsets by one char. 286/1415 runs are affected — 275 of them the temp-1.3 arm
(high temperature emits literal CRLF), which carries **55% of all reads**
(377,725/683,285; high-temp runs are ~14x longer). Measured impact on
`models/v3_32b/labels.npz`: decision coverage 0.055 on affected reads vs 0.418
on clean ones (**8x collapse**); 78,584 reads index past the translated EOF
(forced unlabeled); held-out **seed 7 is poisoned** (dec coverage 0.078 vs seed
6's 0.369) — half the s6,s7 holdout behind gate2 and A1. Worst runs carry
65-69K CRLFs (~40% of their text).
*Fix:* read with `newline=''`, or better `text = '\n\n'.join(segments)` when the
sidecar exists (verified byte-identical to raw txt, 3/3 runs). Same fix needed
at `scripts/gate2_text_control.py:166`.

**A2. Anchor-window coordinate mismatch — CRITICAL.**
Read char offsets are computed in RAW segment-join coordinates
(`labeling.py:331-334`) but the scoring window is sliced from the
whitespace-collapsed, lowercased `text_norm` (`:239`, `:336`):
`win = text_norm[char-400:char+400]`. Cumulative whitespace collapse displaces
the window — median 93-360 chars on clean runs, same order as the window
itself. A faithful replica reproduces the shipped labels 14,346/14,346 on a
27-run sample; correcting only the coordinates **changes 30.3% of
decision-labeled reads** (9.6% flip to a different decision, 20.7% become
unlabeled) and adds **+28% missed coverage**. The debug trail masked this:
`window_snippet` (`:363`) is taken from RAW text, so human audits saw
correctly-aligned context while scoring used the drifted window.
*Fix:* slice the window from the same coordinate system the offsets live in
(slice raw text, then normalize the window content).

**A3. Mixed units in the phase boundary — MAJOR.**
Trace-source `commit_char = text_norm.find(sig)` (`:292-294`, normalized
coords) is compared to the read's raw-join char at `:354`. Norm coords are
systematically smaller → trace-sourced commit positions land too early → reads
BEFORE the true commitment get phase=1 "settled". Pollutes A1's
should-ask/settled pools (the 0.599 AUROC) and gate5/gate7 class broadcast.
Affects 173/1574 commitments (the trace-source share).

**Blast radius:** decision/class/phase labels feed gate1, gate2, gate3, gate5,
gate7, A1, A3, and the permutation test's 35-decision eligible set. Every
R2 headline number consumed these labels. Additionally, the gate-2 text control
is **asymmetrically biased in favor of text**: the TF-IDF baseline slices the
same displaced window the label was derived from (circularly self-consistent
even when displaced) while activations align to the true read position — this
directly inflates "text 0.704 beats activations 0.504".

## B. The statistics overstate the negative (independent of A)

- **"0/35 decisions p<0.05" is false under an exact test.** Decisions 73 and
  210 (7 runs, 5v2 split, 21 distinct assignments) have their observed labeling
  as the UNIQUE MAXIMUM of the entire permutation distribution: exact
  p = 1/21 = 0.0476. The MC-200 estimator + (1+c)/(1+B) correction pushed them
  to 0.055/0.060. Honest count: **2/35** (vs ~0.65 expected under null; a mild,
  non-significant tilt).
- **~20-25 of the 35 decisions are structurally untestable**: their run-class
  splits admit too few distinct label assignments to ever reach p<0.05 (three
  2-run decisions have exactly 1 partition → p≡1.0 forever). Only ~10-13
  decisions could ever fire. "0/35" should read "0-2 of ~10-13 testable".
- **The per-decision criterion was never powered**: planted-effect simulation on
  the real splits caps per-decision power at ~33% even for effects 2x the
  within-read spread.
- **A proper global test still comes out negative** (this is the honest
  negative): Stouffer sum of per-decision z over 32 nondegenerate decisions vs
  its own permutation distribution gives p=0.268 (B=1000); and the global test
  IS well powered (≥80% for class separation ≥~0.2x within-spread). But it was
  measured on the corrupted labels of §A, so it inherits their attenuation.
- The permutation machinery itself is CORRECT: run-level shuffle, shared JL
  projection, identical code path; all 35 observed ratios reproduced bit-exactly
  by independent reimplementation. Planted signal at 0.25x within-spread IS
  detected (p=0.005-0.045) on well-populated decisions — the pipeline can see
  signal where the label structure permits.

## C. The small-model "encouraging" results were artifacts (false-positive direction)

Independently verified: the 14B structural raw-h LORO 0.71-0.73 (decisions/015,
cited verbatim in decisions/021 as the scaling justification) **never beat a
trivial baseline**. `scripts/v2_value_diag.py:81-101` skips any fold whose
training set loses its minority class — single-run-minority decisions only ever
test majority reads — yet reports chance = 1/n_classes. The correct trivial
baseline (train-majority read predictor) is 0.916 pooled vs the observed 0.729.
Per decision: swe_19 1.000 = baseline 1.000; swe_24 0.923 < 1.000; swe_0 0.724 <
0.922; swe_21 0.350 < 0.750. Exact run-level permutation on the two
non-degenerate cells: p=0.071 and p≥0.33. At 32B the four "winners" show
textbook winner's curse (two unmeasurable, one collapsed to 0.426, one "held" at
0.859 on a degenerate 17:1 split with baseline 1.000).

Same defect kills HANDOFF §3's salvage numbers: structural micro "+0.097 over
chance" and value "+0.126" are actually ~0.19-0.21 BELOW the majority-read
baseline (0.789/0.768); FAR-read "+0.135/+0.101" likewise. Gate5 was never
positive at ANY scale (7B 0.726±0.303, 14B 0.896±0.773, 32B 0.894±0.279).

## D. What is clean (verified)

- **The recovered-runs merge is clean**: 1415 complete 4-file run sets, zero
  duplicates, temp arms 360/360/360/335, all 120 recovered files sha256-identical
  in both locations, all 14 manifests agree on Qwen/Qwen3-32B, spot-checked npz
  (R,8,5120) fp16 finite. The 25 missing runs are exactly the unrecovered
  temp-1.3 set.
- **Label↔activation ROW alignment is clean**: labels.npz h rows bit-exact equal
  to run npz sliced at slot 3 = model layer 32; token_idx matches row-for-row.
- **Layer slotting is consistent everywhere** (the #1 a-priori suspect — ruled out).
- **Tokenizer = Qwen/Qwen3-32B confirmed** in log + all 7 shard manifests.
- The 1415 "tightening" (0.805±0.148/38) was computed on a full GPU-box labels
  build, not the len-1 debug regen — but it inherits the §A bugs identically.
- Headline numbers all used the frozen 1385 snapshot consistently on both arms
  of each comparison (no n mismatch).

## E. Also missing from the 32B verdict

- **Gate3 and gate6 were never measured at 32B**: single-split holdout had
  n_same=0 (s6,s7 = 2/24 seeds), and `run_full_gates.py` kfold path
  (`:239-247`) pools only gates 1/5/7. Two gates that were positive at 7B AND
  14B (theta 0.87 / 0.705) have no 32B number at all. Cheap fix: add g2/g3/g6
  to the kfold path.
- The l_he (A2 L-space) run-level permutation — HANDOFF 1b's own "missing step"
  — still hasn't run.
- gate7 lead-time has no null model at any scale and its medK unit (reads) is
  not comparable across scales (R2 read density ~3x).

## F. Ordered plan

1. **Fix the labeler** (one commit, record as a decisions/ ADR — this is defect
   repair, not gate tuning; the pre-registered gates and bars stay frozen):
   labeling.py newline handling (A1), window coordinates (A2), phase-boundary
   units (A3); same newline fix in gate2_text_control.py. Add regression tests:
   CRLF fixture run; window == raw-slice invariant.
2. **Rebuild labels** on the box or with the slim trick (full 1415 set), rerun
   `generate_labels` + fork census (the 34 gate-measurable forks count will
   change; +28% coverage expected, temp-1.3 arm becomes labelable at all).
3. **Rerun gates with corrected statistics**: exact permutation where the
   assignment space is small, report testable-decision counts, the global
   Stouffer test as the headline, majority baselines for every LORO-style
   diagnostic, gates 2/3/6 added to kfold, and the l_he permutation.
4. **Then read the answer.** If positive → the original paper, now with a stats
   toolkit reviewers can't dent. If negative → it is for the first time a
   genuinely controlled negative, and the negative-result/methods paper
   (pre-registered option C, decisions/015) is real: pre-registration +
   run-level exact permutation + lexical controls + majority baselines + label
   noise machinery is exactly what the prior-art survey showed the field lacks.
5. Independent of 1-4: the labelling lever (68% unlabeled) and the judge arm
   (blocked at its validation gate) — the PPI/CDI route (arXiv 2408.15204) with
   a few-hundred-item owner-labeled production sample remains the principled
   unlock, and the §A fixes will change the judge-validation set too
   (trace-sourced items were labeled by the buggy teacher).

Deferred (moot until after the rebuild): gate2 probe-fairness audit
(standardization/convergence/nonlinear probe on full 5120d), A1 AUROC deep-dive
(0.599 is downstream of A1/A3 label corruption).
