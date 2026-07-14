# 017 — v2 collection: real agent trajectories (the end-state collector)

---
status: agreed (owner 2026-07-14: "build it... end product")
date: 2026-07-14
---

**What it is.** `scripts/collect_v2.py` runs N seeded multi-step agent
trajectories per task INSIDE the task's own hil-bench docker container:
observe → think → act (one shell command per turn, mini-swe-agent convention,
adapted with provenance), reading mid-layer residuals during every turn's
generation. Everything previously learned is baked in: 4-layer capture
(decisions/014), cadence + cue + **value** reads (decisions/016, ON by
default), image-loading ladder (ADR 012's extractor helpers, reused),
resumability, manifest + events.jsonl diagnostics.

**Pieces.**
- `wta/agent_env.py` — DockerTaskEnv: persistent container per run
  (`--entrypoint sh ... sleep infinity`), `sh -lc` exec with timeout,
  teardown; trivially fakeable.
- `wta/hf_reader.py::generate_segment` — one agent TURN from a chat-history
  message list; fresh read selector per segment (cue buffers must not leak
  across turns); seed mixed with segment_idx for deterministic-but-distinct
  turns; reads carry `segment_idx`.
- `wta/agent_loop.py` — the loop: strict one-bash-block protocol, last-block
  parsing, `echo TASK_DONE` submit marker (logged as an action, NOT executed),
  observation truncation (head 1500 / tail 500), missing-block reprompt,
  max_steps stop. Every action logged as `ActionEvent(segment_idx, token_idx,
  command, observables={files, step})` — the offline teacher for v2 labels;
  reads never depend on actions (ground rule intact).
- **Labeling is segment-aware**: `<run_id>.segments.json` (per-turn texts) maps
  each read through ITS OWN segment's token→char table; `.txt` stays the
  human-readable join, still used for whole-run signature/commitment matching.
  v1 single-segment data takes the legacy path untouched.

**Data layout per run:** `<run_id>.npz` (R, L, H) + `.json` (reads/actions) +
`.segments.json` + `.txt`.

**Contracts tested (docker/model-free):** loop end-to-end with scripted turns
(segment ordering, marker semantics, file observables, validate()), missing
block + max_steps robustness, parser/observable/truncation units, and a real
on-disk two-segment run labeled through the correct segment text. 108 tests.

**What v2 buys the paper:** real (not proxy) lead-time — internal disagreement
read-index vs the actual tool-call ActionEvent; Ask-F1/Pass@k become
measurable; commitment happens near real edits. This is the collection the
headline numbers come from (decisions/016 reporting plan).

**Knowingly out of scope here** (Phase-4 eval, unchanged plan): the frozen
judge / ask_human injection during runs, Pass@k scoring, baseline arms.
Collection first; the eval harness consumes this data next.
