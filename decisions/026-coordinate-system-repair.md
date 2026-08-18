# 026: the labels were measured in three coordinate systems — REPAIR, not tuning

Date: 2026-08-18. Status: ACCEPTED (owner approved the repair plan in-session,
2026-08-18). Follows the R2 negative-result audit (AUDIT_R2_NEGATIVE.md,
7-agent audit + adversarial verification, all key claims reproduced by running
code against the real repo/data). Answers the question the R2 numbers cannot
currently survive: were the gates measured against the labels the spec
describes?

**Pre-registration timing.** This entry is written and committed BEFORE the
labels are rebuilt and BEFORE any gate is recomputed. The fix direction, the
verification protocol, and the comparability rules below are frozen first, so
whatever the rerun says — positive or negative — the ordering is defensible.
This mirrors 025's ordering discipline exactly.

## 1. The defect

`build_labels` (src/wta/labeling.py) mixed three coordinate systems:

    RAW-join   "\n\n".join(segments) from <run>.segments.json (CR preserved).
               The system offsets are BORN in: tokenizer offset_mapping per raw
               segment + offs[] arithmetic. Read chars and actions-source
               commit positions live here.
    TRANS      the .txt read via Path.read_text() = universal newlines,
               \r\n -> \n. Shorter than RAW wherever CRLF existed.
    NORM       _norm(TRANS) = lowercased, whitespace runs collapsed to one
               space. ~9% shorter than TRANS on these traces.

Three mixes, each verified by running code on `data/a0_v3_32b` +
`models/v3_32b/labels.npz`:

- **(a) Window slice.** Read char (RAW-join) sliced the scoring window out of
  NORM: `win = text_norm[char-400:char+400]`. Median displacement 93–360 chars
  on clean runs — the same order as the window itself. A replica that
  reproduces the shipped labels exactly (14,346/14,346 reads on a 27-run
  sample), then corrects only the coordinates, changes **30.3% of
  decision-labeled reads** (9.6% flip decision, 20.7% become unlabeled) and
  adds **+28% missed coverage**.
- **(b) CRLF drift.** 286/1415 runs (275 of them the temp-1.3 arm — high
  temperature emits literal CRLF) have TRANS shorter than RAW by up to ~36K
  chars. Those runs carry **55% of all reads** (high-temp runs are ~14x
  longer). Their decision coverage collapses 8x (0.055 vs 0.418); 78,584 reads
  index past EOF (forced unlabeled); **held-out seed 7 is poisoned** (coverage
  0.078 vs seed 6's 0.369) — half the s6,s7 holdout behind gate 2 and A1.
- **(c) Phase units.** `phase = 1 if char >= commit_char` compares a RAW-join
  `char` against a `commit_char` that is RAW-join for `actions` source, NORM
  for `trace` source (`text_norm.find`), and TRANS for `judge` source. NORM
  positions are systematically early → pre-commitment reads read as `settled`,
  polluting A1's matched contrast (the 0.599 AUROC) and gate5/7 class
  broadcast. Affects the 173 trace-source + any future judge-source
  commitments.

The debug trail masked all three: `snippet`/`window_snippet` slice TRANS at
those mixed offsets, so human audits saw plausibly-aligned context while the
scorer used a displaced window.

Blast radius: decision/cls/phase feed every gate (1,2,3,5,7), A1, A3, and the
permutation test's eligible-decision set. `scripts/gate2_text_control.py`
carries its own copy (TRANS text sliced at RAW offsets) with an asymmetric
effect: the text baseline is circularly self-consistent with the displaced
labels while activations align to the true read — inflating "text beats
activations".

## 2. Why this is defect repair, not tuning

Frozen and unchanged: `window_chars=400`, `min_anchor_hits=1`,
`min_sig_hits=1`, argmax + strict-margin rules, `phase` >= boundary semantics,
anchor/signature vocabularies, `_MUTATING_TOKENS`, judge-merge precedence
(fill-in only, never override), the labels.npz schema, and every gate
statistic definition. The spec (labels.md, "Labels produced" table) already
defines the window as "the ±window_chars text around the read's char position"
— slicing a DIFFERENT string at that position was the bug; the repair makes
the code do what the pinned spec says. `_norm(raw) == _norm(trans)`, so
trace-stage signature SCORES are bit-identical before/after — only positions
and windows move, which keeps the label diff mechanically attributable to the
coordinate repair.

## 3. The canonical basis: RAW-join

One coordinate system everywhere: `text = "\n\n".join(segments)` when the
sidecar exists (verified byte-identical to the on-disk .txt read raw, 25/25
sampled runs), else the .txt read with `newline=""` (no translation). All
offsets — read char, all three commit sources, debug snippets — are RAW-join.
The scoring window is sliced from raw text and its CONTENT normalized before
scoring: `_norm(text[char-400:char+400])`.

Rejected alternative — canonical NORM with raw→norm maps at every read: it
would convert 787K positions per build through a map that must survive
`str.lower()` length expansions, and it changes the effective size of the
pre-registered ±400 window. RAW-join converts only the minority direction
(trace-source commit: NORM find → raw via a new exact `_norm_map`), keeps
gate2's `char` contract, and `specs/judge_labels.md` §4 already mandates raw
offsets for judge evidence (the old implementation violated its own spec).

## 4. Exact semantic deltas (complete list)

1. Window content: `_norm(text_raw[char±400])` instead of
   `text_norm[char±400]` — same parameter, correct string.
2. Trace-source `commit_char`: earliest normalized signature occurrence mapped
   back to its raw position via `_norm_map` (monotone map — earliest-occurrence
   semantics preserved exactly).
3. Judge-source `commit_char`: raw offsets (spec judge_labels.md §4
   conformance); `judge_labels.py` reads traces raw and its `_norm_with_map`
   (which had leading/trailing-whitespace off-by-ones and a lowercase-expansion
   desync) is replaced by the shared `_norm_map`.
4. Fallback path (no sidecar — v1 data): .txt read with `newline=""`. Verified
   no-op for v1 (`data/a0`: 0 sidecars, 0 CR bytes in all 160 traces) — v1
   labels unchanged.
5. Debug `snippet`/`window_snippet`: expressions unchanged, now slices of raw
   text at raw offsets — honest. New observable (spec v3.1): every read row's
   `window_snippet` and `anchor_scores` are REPRODUCIBLE from the raw trace and
   `char` alone.
6. New diagnostics counters (behavior-neutral): `txt_join_mismatch` (sidecar
   join vs on-disk txt divergence), `segment_clamped`/`token_clamped` (the
   previously silent `min()` clamps). Expected 0/~0 on the real collection.
7. Memory engineering (value-identical): two-pass preallocation replaces
   list-of-views + `np.stack` (peak ~32GB → ~16.5GB for the 1415 build);
   `run_full_gates.py` gains `--labels-npz` to load a prebuilt labels.npz
   instead of rebuilding (default behavior unchanged).

Explicit punts (unchanged, recorded so nobody mistakes them for oversights):
the min-over-all-signature-terms commit position (matches spec); clamp
BEHAVIOR (counter only); `v2_value_diag.py`/`value_read_analysis.py` TRANS
reads (v1-era diagnostics, separately discredited by the audit); the judge
validation set rebuild (see §6).

## 5. Statistics corrections (bundled, additive — owner decision 2026-08-18)

The audit showed the R2 readout also overstated the negative independently of
the labels. Corrections, all ADDITIVE reporting (no gate statistic redefined):

- `gate5_permutation_test.py`: exact enumeration of the run-label assignment
  space where small (every R2 decision had ≤6,435 distinct assignments), MC
  2000 otherwise; per-decision `n_assignments` / `min_p` / `testable`; and the
  honest headline — a GLOBAL permutation test (Stouffer sum of per-decision z
  against the permutation distribution of the same sum). Context: the R2
  "0/35 p<0.05" was an exact-test 2/35 with only ~10-13 decisions testable.
- `run_full_gates.py --kfold`: pools gates 2/3/6 alongside 1/5/7 — gate 3 and
  gate 6 (positive at 7B and 14B) had NO 32B measurement because the
  single-split holdout had zero fork pairs and the kfold path never computed
  them.
- `v2_value_diag.py` LORO: eligibility requires every class carried by ≥2 runs
  (no silently skipped minority folds) and reports the train-majority baseline
  + balanced accuracy — the 14B "0.71-0.73 vs 0.50 chance" scaling
  justification was 0.729 vs a 0.916 trivial baseline.
- New `gate5_lhe_permutation.py`: the HANDOFF §1b missing step — the same
  run-level permutation on A2's learned L space, so §1's 0.894 can finally be
  stated (or not) in permutation terms.

## 6. Comparability and the judge arm

Rebuilt labels land in `models/v3_32b_fixed/`. The frozen `models/v3_32b/`
artifacts and every R2 number in HANDOFF_R2_GATES.md stand as the
corrupted-label run — reported, never overwritten, never retro-selected
against (mirror of 025 §4's rule). Old `results/` is archive-renamed before
any rerun (two scripts write hardcoded/default paths into it).

The judge arm stays STOPPED (025 Amendment A). One consequence of this repair:
`data/label_judge_validation.json` was built with trace-sourced items labeled
by the buggy teacher, so before any judge revival the validation set must be
rebuilt from fixed-teacher labels. That is a dependency recorded here, not
work done in this entry.

## 7. Verification protocol (results to be recorded by amendment)

1. Six new contract tests (`harness/contract/test_labeling_coords.py`) run
   against the PRE-fix code first — the CRLF twin, whitespace-window,
   phase-units, norm-map expansion, and debug-honesty tests must FAIL there
   (captured), then pass post-fix. Windows-nonempty/monotone mechanizes spec
   check 4.
2. Full suite green, INCLUDING the real-sample v1 tests with labels unchanged.
3. Stratified ~40-run before/after diff (10 per temp arm, ≥8 CRLF runs): old
   labels from a git worktree at the pre-fix commit, same sample, same
   tokenizer. Report per-temp-arm coverage delta + past-EOF count. Expected
   direction (from the audit's independent replica): ~+28% coverage overall,
   temp-1.3 arm 0.055 → ~0.4, past-EOF → 0.
4. Determinism: double-build array equality (arrays + meta, not zip bytes).
5. Full 1415 rebuild on the box: rows == 787,281, `txt_join_mismatch == 0`,
   clamp counters ~0, fork census recorded vs the frozen run.
6. Gates: kfold layer 3 (pre-registered) first, then single-split layer 3 (a1/
   a2/a3 artifacts) + l_he permutation, then the remaining layers; then the
   corrected permutation test + all three gate2 text-control variants against
   the fixed labels/debug pair. STOP for owner review per decisions/011.

## Amendment A (2026-08-18, same session — verification results, PRE-rebuild)

Written after implementing §4/§5 and running protocol items 1-4, BEFORE the
full 1415 rebuild or any gate recomputation. Nothing here changes §1-§6; one
expectation from the audit is corrected DOWN (item 3 below) — recorded now so
the gate rerun is read against honest priors.

**(1) Tests.** Suite green: 232 (226 prior + 6 new in
harness/contract/test_labeling_coords.py). Against the pre-fix code (commit
ee90ace, clean worktree): **5 of 6 fail** — crlf_run_matches_lf_twin,
whitespace_collapse_window_alignment, trace_commit_phase_units (fails
directly on the wrongly-settled read, not via unlabeling), norm_map_roundtrip
(old judge `_norm_with_map` import/behavior), debug_trail_reproducible.
windows_nonempty_monotone passes both (guard, not proof). Real-sample v1
tests pass with labels unchanged, as predicted (0 sidecars, 0 CR bytes).

**(2) Sample diff** (stratified 40 runs, 10/temp arm, 13 CR-bearing of the
collection's 287; old labels built from the pre-fix worktree, same tokenizer;
11,667 rows, row-aligned): determinism double-build PASS (arrays + meta).
Per-read changes old→new: **13.4% of all reads change** decision/cls/phase
(677 unlabeled→labeled, 602 labeled→unlabeled, 278 labeled→DIFFERENT
decision); class labels 769→899 (+17%); phase changes on 862 reads; **95
reads pointed past the translated EOF** under old coordinates.
txt_join_mismatch = 0/40. token_clamped = 156 (1.3% of reads, diffuse across
27 tasks, overshoot ≤2 tokens median 1 — spec labels.md caveat-5
re-tokenization drift, benign, now merely visible). Eyeball case (swe_2-s8
read 250): at the same raw char the OLD displaced window scored 1 stray hit
for the wrong blocker; the correct window scores 4 hits for
missing_overflow_sentinel_int. Debug-trail honesty on a full CR run
(swe_18-s7): 1,524/1,524 stored (blocker, read) scores recompute exactly
from raw text + char (spec v3.1 observable 5).

**(3) Correction to the audit's projection.** AUDIT_R2_NEGATIVE.md §A
projected "+28% coverage; temp-1.3 arm 0.055 → ~0.4". On this stratified
sample the aggregate coverage did NOT recover: by arm, 0.574→0.596 /
0.472→0.496 / 0.358→0.345 / **0.113→0.111**; CR-bearing runs 0.191→0.190.
The audit's own verifier had flagged the confound: high-temp runs are ~14x
longer, so their low coverage is substantially a property of the traces (long
off-anchor stretches), not only of the displaced windows, and the "8x
collapse" conflated the two. What stands unweakened: the mechanism proofs in
(1), the 13.4% per-read label churn, the past-EOF reads, and the wrong-window
scoring — the labels the gates consumed were substantially wrong AT THE READ
LEVEL even where aggregate coverage was flat. Consequence for reading the
rerun: the repair's effect on gate numbers is of UNKNOWN sign and magnitude;
neither a flip nor a confirmation is the "expected" outcome. That is exactly
the posture §2 pre-registered.

**(4) Statistics additions smoke-verified** (§5): the exact-enumeration
permutation recovers a planted [5,2]-split signal at its floor p=1/21=0.0476,
reports 1v1 splits untestable (p≡1.0), no false positive on a null [6,5]
decision, global Stouffer p=0.043 on the planted set; the kfold path now
pools gates 2/3/6 (v1-sample smoke: g2 0.474 vs 0.025 chance, g3 theta 0.844,
g6 purity 0.299 — consistent with decisions/013-era values) and dumps
per-fold l_he for gate5_lhe_permutation.py, which runs end-to-end.
