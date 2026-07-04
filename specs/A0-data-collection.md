# Spec A0 — Trajectory data collection

Status: Phase-1 scope covers the read policy + log schema (CPU-testable) and the
single-task hook proof (`scripts/prove_hook.py`, runs on the owner's AWS GPU box).
Full N=8 collection over harbor_swe is a later run, not code.

## Purpose

Run the agent N times per under-specified task; at read positions across the
reasoning span, log the mid-layer residual `h` and, separately, the observables
that become offline training labels. Observables are a *teacher* (labels only);
they are never a trigger and never an alignment key at runtime.

## Inputs

- Task (HiL-Bench harbor task), backbone (config `backbone.kind: hf`,
  default `Qwen/Qwen2.5-Coder-7B-Instruct`), `mid_layer` (fraction or index,
  default 0.5 — decisions/007).
- N = 8 runs: seeds 0–7, temperature cycled {0.7, 0.85, 1.0} (decisions/008).

## Read policy (decisions/006)

A read is taken during generation, per generated token stream:

- **cadence**: at every K-th generated token (K = 32 default; sweep {16,32,64});
- **cue**: when the detokenized generated text — lowercased, whitespace runs
  collapsed to single spaces, and trailing non-alphanumeric characters
  ignored — ends with a deliberation cue from
  {"hmm", "wait", "let me", "actually", "should i", "alternatively"}
  **at a word boundary** (the character before the cue, if any, is
  non-alphanumeric). So: cues span token boundaries ("let" + " me" fires),
  trailing punctuation doesn't block ("Wait," fires — the dominant surface
  form in real output), and words merely containing a cue ("awaits", "hmmm")
  do not fire. Matching is on detokenized text, never per-token string
  equality. (Amended 2026-07-02: the original wording "text ends with a cue"
  under-specified these semantics; review finding.)

One read per token position; if cadence and cue coincide, the read is recorded
with trigger `cue`. **Forbidden:** any path that produces exactly one `h` per
action, or reads only at the tool-call boundary (`HFWhiteBoxModel.generate`'s
per-action averaging must not be reused for wta).

## Per-read record

| field | type | invariant |
|---|---|---|
| `segment_idx` | int | one generation (`generate()` call) = one segment; a real agent run is several |
| `token_idx` | int | `(segment_idx, token_idx)` strictly increasing across the run's reads |
| `trigger` | `"cadence"` \| `"cue"` | enforced by `RunLog.validate()` |
| `cue` | str \| None | set iff trigger == `"cue"` (enforced by `RunLog.validate()`) |
| `h` | float16[H] | H = backbone hidden size, constant per run |

## Per-action record (observables — offline labels only)

| field | type |
|---|---|
| `token_idx` | int (position where the action was emitted) |
| `action_text` | str |
| `observables` | dict: `file`, `region`, `subgoal`, `error_signature` (any may be None) |

## Run log

`RunLog(run_id, task_id, seed, temperature, model_id, mid_layer, reads[], actions[])`,
persisted as `<run_id>.npz` (the read matrix, float16[R, H]) + `<run_id>.json`
(everything else; no `h` in the JSON). Round-trip load must be exact.

## Observable behaviour that verifies this spec

1. On a synthetic token stream of length ≥ 3K with planted cues, the selector
   returns cadence reads at K−1, 2K−1, … and cue reads exactly at cue ends,
   including cues that span token boundaries.
2. `token_idx` strictly increasing; no duplicate positions.
3. Log round-trip: save → load equality (h to float16 precision, metadata exact).
4. `prove_hook.py` (AWS): two seeds on one real task produce two RunLogs with
   ≥ 5 reads each, h shape (R, hidden_size) at the configured mid layer, and
   nonzero variance across reads within each run.
