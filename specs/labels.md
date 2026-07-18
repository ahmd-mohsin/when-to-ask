# Spec — Offline label builder (observables → training labels)

Actions/observables are the **teacher, never the trigger** (the ground rule).
This component runs offline only, turning A0 logs + blocker registries into
the labels A1/A2/A3 train on. Written against the real v1 3-task sample
(2026-07-05), not guessed.

## Reality of v1 logs (what there is to work with)

One generation per run: ~24 cadence reads (cue words never fire on
Qwen2.5-Coder — AWS finding), the full decoded trace text, **no action
events**. So v1 observables are: the trace text, its token→char alignment,
the task's blocker registry, and the frozen interpretation-class artifact
(`data/interpretation_classes.json`, decisions/005 — class 0 is always the
canonical resolution; anchors name the DECISION, signatures name the
RESOLUTION).

## Labels produced, per read

| label | how | value |
|---|---|---|
| `decision_id` | anchor-lexicon scoring of the ±`window_chars` text around the read's char position (token→char via tokenizer `offset_mapping` on the trace — an approximation of generation-time positions, documented) | global int, −1 = background/no decision |
| `class_id` | per (run, blocker): signature scoring over the FULL trace; argmax with ≥`min_hits` and strict margin over runner-up, else unlabeled. Applied only to reads whose `decision_id` matches and that fall at/after the behavioural commitment point | global flattened int (decision-local classes offset into one vocabulary), −1 = unlabeled |
| `phase` | behavioural commitment proxy: the first read whose preceding text contains a signature of the run's committed class for that decision → reads before = `should_ask`, at/after = `settled` | for A1's matched contrast + A3's stabilization label + lead-time reference |

## v2: action-based commitment (2026-07-18, after the 14B mislabel finding)

v2 runs log `ActionEvent`s (decisions/017) — the agent's actual commands.
On multi-turn agent traces, whole-trace signature scoring mislabels VALUE
commitments: agents mention several candidate values while deliberating
before writing one (verified mislabels on wta-a0-v2-14b: swe_2 sentinel
labeled from a discussion mention while the action clamps; a `return 0` in an
unrelated function matched `zero_or_placeholder_sentinel`).

Fix, per (run, blocker), when the run has actions:

1. Score class signatures over the concatenation of **mutating** action texts
   only (commands matching `sed -i`, `>`/`>>` redirection, `tee `, `patch `,
   `git apply`, `perl -i` — writing to files IS the behavioural commitment;
   read-only exploration like `grep 30 file` must not count).
2. Same argmax + `min_sig_hits` + strict-margin rules as trace scoring.
3. Commitment position = the FIRST mutating action containing a winning-class
   signature, mapped through its own segment's token→char table (same
   mapping as reads). Reads before that action are `should_ask` even if the
   winning signature was *mentioned* earlier — that is the point of the fix.
4. Fallback: no mutating actions, no hits, or a tie → whole-trace scoring
   (the v1 path, unchanged). Every commitment's `label_source`
   (`actions` | `trace`) is recorded in the debug trail so coverage by
   source is auditable.

Honesty caveats: the mutating-command lexicon is a heuristic (a `>` inside
quoted code counts as mutating); v1 data has no actions and is untouched;
prose-only commitments (agent states a choice, never edits) fall back to the
noisy trace path and stay measurable in the audit.

## Design decisions (and their honesty caveats)

1. **Class ids are flattened globally** (each (blocker, class) pair is one id)
   so A2's lean CE sees real interpretations, not an arbitrary shared 0..C
   indexing across decisions. Gate 5 / Part B still measure within-decision
   separation only.
2. **Unlabeled beats mislabeled**: any read/run without a clear anchor or
   signature margin gets −1. Coverage is REPORTED (a low number is a finding
   about the labeler or the class set, not something to force).
3. Traces can commit to interpretations outside the derived class set (seen
   in the sample: custom per-OS probing on swe_0). The class sets include the
   observed non-canonical families where the registry implies them; anything
   else stays unlabeled and is counted in the coverage report.
4. Anchor collisions between blockers of the same task (e.g. `platform.system`
   appears in two swe_0 blockers) are expected label noise — exactly what the
   conflation gate (A4 check 4) exists to measure. The co-divergence fallback
   (method doc) is the escalation if gates say labels are too noisy.
5. Token→char alignment re-tokenizes the saved trace; generation-time token
   positions can drift by a few tokens near unusual byte sequences. Window
   size (default ±400 chars) dwarfs the drift.

## Interface

```
build_labels(a0_dir, registry_root, classes_path, tokenizer_name, window_chars=400,
             min_anchor_hits=1, min_sig_hits=1) -> LabeledDataset
LabeledDataset: h (n_reads, H) float32 | decision (n,) | cls (n,) | phase (n,)
                | run_id (n,) | task_id (n,) | read_token_idx (n,)
                + vocab maps + per-task coverage report; .save(npz)/.load
```

Deterministic; no model forward passes; no GPU.

## Observable behaviour that verifies this spec (on the real sample)

1. Runs end-to-end on `data/a0` and labels ≥ 25% of reads with a decision and
   ≥ 4 of the 13 blockers with ≥ 2 distinct committed classes across runs
   (a real fork to study). Below that: the harness prints the coverage table
   and fails — a finding to review, not to tune away silently.
2. Determinism: two builds produce identical arrays.
3. Every labeled `class_id` belongs to its read's `decision_id` (flattening
   consistency), and `phase == settled` only at/after that run's commitment
   read for that decision.
4. Token→char sanity: every read's window is non-empty and windows are
   monotonically ordered in the trace.
