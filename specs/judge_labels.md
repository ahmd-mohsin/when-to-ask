# Judge labels — LLM labelling of lexicon-unlabelable commitments (025 arm)

Pre-registered by decisions/025 (+ Amendment A). This spec is written BEFORE any
judge-labelled data exists. The judge is a SECOND labelling teacher: it fires
only where the frozen signature lexicon abstained, and its output is a frozen,
audited artifact — never a live model call inside `build_labels`.

## 1. Scope

- Items = `kind="commitment"` records in a `labels_debug.jsonl` with
  `chosen=null` and `reason="no signature hits"` (3,272 on the 1385-run
  snapshot; ~3,36x on the merged 1415). `--include-ties` adds the
  "tie between top classes" records (85/1385) — OFF by default, matching
  025 §4's scope ("the 3,272 unlabelled commitments").
- The judge NEVER relabels or overrides a lexicon-labelled pair. Where the
  lexicon labelled, its label stands (comparability with the v2 teacher).
- Sealed pool untouched; judge labels feed the SEPARATE labelled arm only
  (025 §4b) — gate numbers on judge-labelled data never replace the
  2026-08-15/16 run.

## 2. Judge

- Model: Claude Fable 5. Transport (025 Amendment A): **subagents inside the
  owner's Claude Code session** (subscription — the owner has no API key).
  Each subagent receives the frozen protocol prompt + a small batch of items
  (grouped by prompt-size budget) and returns schema-forced JSON; the raw
  responses are written to disk before any scoring. The Anthropic Message
  Batches API path (025 §5's original assumption) remains implemented in
  `wta.judge_labels` as the alternative transport if a key ever exists.
- Thinking is always on for this model and `temperature` is not accepted —
  the ask-judge's frozen `JUDGE_TEMPERATURE=0.05` convention cannot carry
  over. Determinism is NOT assumed; auditability replaces it (§4: every
  accepted label carries a verbatim, verified evidence span).
- Disclosure (for the paper): the judge runs inside an agent harness rather
  than a bare API call; the entire judging protocol lives in the fixed prompt
  (this spec + `wta.judge_labels.SYSTEM_PROMPT`), items are batched ~10 per
  subagent, and there is no API request log — the frozen items file, the raw
  response JSONL, and the frozen artifact are the audit trail. In-repo
  precedent: the decisions/024 fork-type map (blind in-session classifiers +
  adjudicators, frozen with an audit trail).
- Not a base-URL swap of `xtid.harness.ApiJudge` (different task shape and a
  non-OpenAI-compatible API); new module `wta.judge_labels`.

## 3. Prompt (per item)

Context given to the judge:

- blocker id + the hil-bench registry `description` (prose definition of the
  ambiguity) from `third_party/hil-bench/harbor_swe/<task>/shared/
  ask-human-data/blocker_registry.json`;
- the artifact's `anchors` (topic identifiers);
- the candidate classes: `name` + `signatures` (as "indicator phrases", i.e.
  examples of how a commitment to that class can surface — the judge must
  generalize beyond them);
- the run's trace (policy in §5).

Bias controls (both MUST hold):

- the registry `resolution` field and the artifact `canonical` flag are
  EXCLUDED from the prompt (they name the intended answer);
- class presentation order is shuffled per item with a deterministic
  permutation seeded by sha256(run_id + "|" + blocker) — kills position bias
  toward the canonical class (always class 0 in the artifact) while staying
  reproducible.

Required output (JSON only): `{"class": <name or null>, "confidence": 0..1,
"evidence": "<verbatim quote from the trace>", "reasoning": "..."}`.
The judge is instructed that commitment = what the run's actions/final work
product actually implement or assert, not options it merely deliberated;
abstain (`class: null`) when the run never engaged the decision or the
evidence is ambiguous ("unlabeled beats mislabeled", spec labels.md).

## 4. Acceptance — a label exists only if it is auditable

A judge response becomes a label iff ALL hold:

1. parses as JSON (``` fences tolerated) with a `class` field;
2. `class` is one of the item's class names (else `bad_class`);
3. `evidence` is locatable in the run's raw `.txt`: exact substring first,
   else whitespace-normalized match mapped back to raw offsets
   (else `evidence_not_found` — the label is REJECTED, not repaired);
4. the response was not a refusal / API error.

`commit_char` = raw char offset of the located evidence span (the judge's
cited commitment moment — the analog of the trace path's earliest-signature
position). Every item's outcome (accepted / abstained / rejected + reason,
confidence, evidence, spans) is frozen to
`data/judge_labels_<collection>.jsonl` + rendered into an audit .md
(sampled, audit_labels.py-style) for owner review.

## 5. Trace-window policy (pre-registered; the cost lever of 025 §5)

- Full raw `.txt` when ≤ 30,000 chars (≈ p90 of the 1415-run collection).
- Else a targeted excerpt, capped at 30,000 chars: head 2,000 chars
  + all mutating-action turns (the v2 teacher's commitment surface)
  + windows of ±1,000 chars around each anchor hit for THIS blocker (max 8)
  + tail 8,000 chars; elisions marked `[... trace elided ...]`.
- Evidence verification (§4.3) always runs against the FULL raw `.txt`.

## 6. Validation gate (STOP before any full labelling run)

025 §4a as written names `scripts/validate_judge.py` + the frozen 34 pairs —
that harness validates the ask-injection judge (question -> blocker match)
and mechanically cannot score a labelling judge (different task shape,
different metric; Amendment A records this). The labelling-judge gate is:

- Frozen validation set `data/label_judge_validation.json`: ALL trace-sourced
  labelled commitments + a seeded (20260816) sample of actions-sourced ones,
  300 total, built once by `scripts/validate_label_judge.py --build-set` from
  the current `labels_debug.jsonl`; committed; overwrite requires `--force`.
- The judge labels them blind (identical prompt builder as production; the
  lexicon's label is withheld).
- PRE-REGISTERED PASS BAR (all must hold, computed on non-abstained items):
  accuracy vs lexicon labels >= 0.90 overall AND >= 0.90 on the
  actions-sourced stratum; abstention rate <= 0.20.
  Any miss -> STOP, owner review; no full run.
- Known limit (recorded, not hidden): validation items are by construction
  lexicon-labelable, so their traces contain signature phrases the production
  items lack; validation accuracy is an optimistic bound. Mitigation: the
  evidence spans (§4) make a post-hoc human audit of production labels
  possible; the audit .md samples them for owner review before any gate run.

## 7. Integration (the teacher change, 025 §4b)

- `build_labels(..., judge_labels=None)`: an optional pre-computed dict
  `{(run_id, blocker): (class_name, commit_char, confidence)}` consulted only
  when both the actions and trace stages abstained; sets
  `label_source="judge"` (debug record gains `judge_conf`). Determinism:
  `build_labels` stays deterministic GIVEN the frozen judge-labels file; no
  model call happens inside it (spec labels.md "no model forward passes"
  holds for the builder — the LLM ran offline, upstream, frozen).
- `load_judge_labels(path, min_conf)` filters accepted labels at a confidence
  threshold. Primary labelled arm: `min_conf = 0.7` (pre-registered);
  the audit reports sensitivity at 0.0 and 0.9.
- Outputs land in a SEPARATE models dir (e.g. `models/v3_32b_judged/`);
  the un-judged `models/v3_32b*` artifacts are never overwritten.

## 8. Interfaces

- `wta/judge_labels.py`: `build_judge_items`, `build_judge_prompt`,
  `parse_judge_response`, `locate_evidence`, `class_permutation`,
  `write_item_files` (session transport: per-subagent work files),
  `session_responses` (session results -> the freeze input shape),
  `make_batch_requests` / `submit_batches` / `poll_batches` /
  `fetch_batch_results` (API transport, lazy SDK import),
  `freeze_results`, `load_judge_labels`, `estimate_cost`.
- `scripts/judge_label_commitments.py`: `--build` (items + per-subagent work
  files + manifest + size estimate) / `--freeze` (session results JSONL ->
  `data/judge_labels_<name>.jsonl` + audit md).
- `scripts/validate_label_judge.py`: `--build-set` (frozen 300-item set) /
  `--export` (work files via the SAME builder) / `--score` (accuracy vs the
  frozen set; prints PASS or STOP against the §6 bar).
- Contract tests are CPU-only: no network, no SDK, no subagents.
