# 024: fork-type map frozen — data/fork_type_annotations.json is FINAL

Date: 2026-08-14. Status: ACCEPTED. Closes the §2g prerequisite opened by
decisions/022 (the blocker → {structural|value} map that `score_eval.py`
slices the headline table by, and that 021 §7 pre-registers as two separate
hypotheses). The owner delegated the hand review and sign-off (2026-08-14,
in-session); this entry is the sign-off record.

**Tally: 214 blockers = 161 structural / 53 value.** 38 entries differ from
the mechanical draft (`scripts/derive_fork_types.py`), which is expected — the
draft is digit statistics, and both of its failure directions showed up (see
§3). The map is frozen BEFORE any R2 label/gate number has been looked at:
the R2 tarball is unextracted at signing time, so this stays a clean
pre-registration.

## 1. The classification policy (frozen with the map)

Applied uniformly to what the interpretation classes actually SAY (names,
signatures, anchors), never to digit counts:

- **VALUE**: the competing classes differ ONLY in which literal value is
  chosen — a number (limit 100 vs 50, timeout, offset base, version to pin)
  **or a literal token/string set** (which wire labels, which enum tokens,
  which sentinel, which default path, which dict key) — with the same code
  shape and behavior pattern around it. The string-literal extension goes
  beyond 021 §7's numeric examples; it follows the research meaning
  (decisions/015: "literal-value forks" — the distinguishing content is a
  literal token choice, which is what the value-invisibility finding was
  about), and the digit-only heuristic could never see it.
- **STRUCTURAL**: the classes differ in behaviour or code shape — different
  API/module/ownership, remove-vs-keep, clamp-vs-reject-vs-disable, different
  strategy/algorithm, different processing order, different condition scope,
  different fallback chain.
- **MIXED → STRUCTURAL**: if ANY axis between classes is behavioural (e.g. a
  numeric-limit fork that also contains a "no limit / omit the check" class,
  or a version-literal fork that also contains a different-API class), the
  blocker is structural. This is the conservative direction for the paper's
  claim: mislabeling a partly-value fork as structural dilutes the structural
  slice and weakens the headline, never inflates it.
- Behavioral responses to a special value (zero handled by reject vs clamp vs
  disable) are structural; digits inside signatures (line numbers, quoted
  code) do not make a value fork.

## 2. Method

1. Mechanical draft regenerated on the laptop and verified **byte-identical**
   to the box-generated draft in the R2 tarball (both from the committed
   `data/interpretation_classes.json`): 182 high-confidence + 32 NEEDS-REVIEW.
2. **Blind pass**: 12 independent classifier agents, each over an 18-blocker
   batch containing only task/blocker/anchors/classes (no mechanical verdict),
   under the §1 policy. Full 214-blocker coverage.
3. **Adjudication**: contested = mechanical "review" ∪ blind "hard" ∪
   blind-vs-mechanical disagreement = **59 blockers**. Each batch's contested
   set was adjudicated by 2 further independent agents (argue value, argue
   structural, decide). **All 59 unanimous; zero escalations.**
4. **Hand verification** (reviewer of record): 10 hardest flagged cases read
   in full BEFORE seeing agent verdicts — agent consensus matched all 10 —
   plus 7 of the 38 draft-flips sampled and verified after (both flip
   directions covered).

## 3. Notable calls (the draft's failure modes, both directions)

- `swe_47/ambiguous_checkbox_toggle_behavior`: draft said value (digits in
  sigs); classes are *which condition clears the selection* (any-selected vs
  fully-selected) → **structural**. The known trap from the draft commit.
- `swe_47/missing_selection_state_wire_labels`: draft said structural (no
  digits); classes differ only in which literal label strings go on the wire
  ('clear/mixed/complete' vs 'none/some/all' vs 'empty/partial/full') →
  **value**. The inverse trap, invisible to a digit heuristic.
- `swe_22/ambiguous_retry_backoff_strategy`: fibonacci vs exponential vs
  linear vs fixed are different algorithms → **structural** despite
  numeric-heavy signatures.
- `swe_11/max_fields_count_ambiguous`, `swe_11/max_nesting_depth_policy_
  conflict`, `swe_26/html_expiration_threshold`, `swe_38/missing_max_active_
  entries_limit`, `swe_53/upgrade_*`: numeric-limit forks that each contain a
  no-limit/omit-the-check (or wrong-exception) class → **structural** by the
  mixed rule.
- `swe_14/default_data_path`: **value** (four literal path choices), the
  closest call in the set. Flagged: (a) one realization of the tmp class is
  `os.tempdir()` (an API call, arguably code shape); (b) this fork separated
  0.69 in the 7B raw-h diagnostic (decisions/015) when "structural" was still
  informal. The policy is semantic and outcome-blind — the interpretation
  axis is *which location*, a literal choice; three independent semantic
  reads agreed. If the value slice shows structure at analysis time, this
  entry is a candidate explanation and is called out here in advance.

## 4. What changed downstream

- `data/fork_type_annotations.json` — the frozen §2g artifact (already
  gitignore-whitelisted since 022).
- `data/fork_type_annotations.review.json` — full per-blocker audit trail
  (mechanical verdict, blind verdict, adjudication flag, decisive rationale),
  whitelisted alongside it.
- `harness/contract/test_fork_type_annotations.py` — the map must cover
  exactly the artifact's blockers with only the two kinds; a blocker added or
  renamed without a fork-type entry now fails the suite.

Per 022 §2, this file may no longer change once the sealed pool is unsealed.
Any pre-unsealing amendment must be recorded in a further decisions entry.
