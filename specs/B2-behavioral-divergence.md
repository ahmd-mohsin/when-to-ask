# Spec — B2: registry-free behavioral divergence features for the Part B trigger

decisions/027. The trigger machinery is specs/B-online-trigger.md VERBATIM
(CUSUM on within-bucket spread of r, never s; leader/merge/hysteresis
bucketing; h in observe units; offline-calibrated reference). This spec
defines only what B-online-trigger left abstract: where `topic_vec`, `r_vec`,
and `weight` come from — now from observable behavior, not activations.

## Inputs (all present in v2/v3 run logs; no registry, no hidden states)

Per run, per turn t: `action_text` (the executed command), and its
observables `files`, `region`, `subgoal`, `error_signature`
(logging_schema.ActionEvent; population verified on R2: 99.9% / 65.7% /
5.7% / 53.9% / 96.4%).

## Feature definitions (frozen by 027 §3)

- `is_commitment(t)` = `labeling._is_mutating(action_text)` — the frozen
  mutating-command lexicon. Non-mutating turns NEVER produce votes.
- `r_vec(t)` = l2-normalized signed-hash char-3-gram TF vector, dim 512,
  hash seed 0, over `_norm(action_text) + " " + " ".join(region) + " " +
  error_signature`. (`_norm` = labeling's whitespace/case normalizer.)
- `topic_vec(t)` = l2-normalized signed-hash TF vector, dim 256, hash seed 1,
  over word tokens of `" ".join(files)` + `_norm(subgoal)`, accumulated over
  the run's last `topic_window=3` turns (bag union). If empty at t (no files,
  no subgoal yet), carry the run's previous topic_vec; if never any, skip the
  observe (a run that has touched nothing has no topic).
- `weight(t)`: STICKY commitment — 0.0 until the run's first mutating action,
  1.0 from then on, with `r_vec` re-asserting the LATEST mutating action's
  vector on every subsequent turn (observe pumps the CUSUM per event, and
  passing weight=0 after a commitment would retract the vote — v1 has no
  retraction; a new mutating action simply replaces the vote).

Hashing: python sha1 of the token/ngram string, bucket = digest mod dim,
sign = digest bit — deterministic across platforms/versions (NOT `hash()`,
which is salted).

## Offline replay protocol (stage 1 of 027 §5)

For each task: order each run's turns by (segment_idx); replay in lockstep
rounds — round k feeds every run's turn k (runs shorter than k drop out) —
mirroring eval/orchestrator.py's round-robin. Each turn = ONE
`AskTrigger.observe(run_id, topic_vec, r_vec, s=0.0, weight)` call when
topic_vec exists. Loop channel OFF (no env-state hashes in offline replay).
Record every fired AskDecision: (round, bucket_id, runs, options).
One trigger instance per task; `--n-runs` subsampling (seed 0, sorted seeds
first) supported so live-eval N (e.g. 8) can be simulated from the 24-seed
data; default = all available runs.

## Calibration + splits (mirrors A3's offline-teacher discipline)

5-fold split BY TASK (kfold_group_indices, seed 0). On TRAIN folds only:
`reference` = 0.9 quantile of within-bucket spread among vote pairs whose
runs' committed classes AGREE per ground-truth labels (the registry-labeled
analog of a3_commitment.benign_spread_reference); `theta` and `h` from the
pre-declared grid theta ∈ {0.3, 0.5, 0.7} x h ∈ {3, 6, 12} maximizing
task-level detection F1 on train folds. Eval folds are touched once, with
the frozen selection. The deployed detector consumes NO labels.

## Ground truth + metrics (eval folds only)

- Task NEEDS-ASK iff ≥1 decision has ≥2 runs committed to DIFFERENT classes
  (FIXED-labeler commitment rows, 026; label_source actions|trace).
- Detection: task-level precision / recall / F1 (fired-at-all vs needs-ask).
- Attribution (reported, not gated): among true-positive tasks, fraction
  where the firing bucket contains ≥2 runs that ground-truth committed to
  different classes of the same decision.
- Lead time: fire round vs the gate7-style action-divergence round (min over
  differing-class run pairs of max(commit round_i, commit round_j)); median,
  in turns; positive = fired before the fork completed... fired at or before.
- Baselines, budget-matched: random-ask (Bernoulli per task matched to the
  detector's overall fire rate, 20 seeds, mean F1), fixed-step ask at the
  detector's median fire round, ask-at-first-mutating-action.

## Interface

```
src/wta/divergence.py
  behavior_features(actions: list[ActionEvent]) -> list[TurnFeatures]
      TurnFeatures(round, r_vec, topic_vec, weight)   # per turn, ordered
  replay_task(runs: dict[run_id, list[TurnFeatures]], cfg: TriggerConfig)
      -> list[FireEvent(round, bucket_id, run_ids)]
scripts/offline_ask_headtohead.py
  --a0 --classes --labels-debug --folds 5 --n-runs N --out results/...
  stage-1 driver: features -> calibrate on train folds -> replay -> score
```

Deterministic; CPU-only; no model calls. The introspection arm (027 stage-1b)
is a separate driver consuming the SAME ground-truth/split artifacts.

## Design decisions (and honesty caveats)

1. Hashed n-grams, not embeddings: deterministic, dependency-free, and biased
   AGAINST us (surface-form splits count as divergence — e.g. two sed
   spellings of the same fix look different). An embedding upgrade is a
   later sweep, pre-registered as such; v1 keeps the conservative feature.
2. The lexicon labeler's known weaknesses (signature collapse, swe_19-style
   fork erasure) now sit in the GROUND TRUTH, not the detector — detection
   F1 is measured against imperfect truth and is therefore a lower bound in
   expectation, but errors are shared identically by every arm/baseline.
3. Lockstep-by-segment ignores wall-clock/token skew between runs; the live
   orchestrator has the same rhythm, so stage-1 numbers transfer in spirit,
   not exactly.
4. `region` is sparse (5.7%) — r_vec leans on command text + error
   signature; that is the honest population reality of R2.

## Observable behaviour that verifies this spec

1. Determinism: two runs of the driver produce identical JSON.
2. Leak rule: `divergence.py` imports nothing from the class artifact and
   the driver's detector path never reads `--classes` (contract test greps
   the module + the feature/replay call graph).
3. Planted fork: synthetic two-run task with different mutating edits on a
   shared topic fires within h-consistent rounds; identical-edit twin never
   fires (contract test).
4. Blip tolerance: one-round divergence followed by agreement stays under
   h (contract test, mirrors B-online-trigger contract #4).
