# 028: the ICLR experimental program — one matrix, five supports

Date: 2026-08-19. Status: PROPOSED (owner directive in-session: "a worthy
paper for ICLR in about a month; design experiments + ablations, implement,
then writing takes ~10 days"). Follows 027 Amendments A/B (mechanism dropped;
the measurement paper IS the paper). This entry pre-registers every remaining
experiment, its budget, and its stop rule, so the next working session
executes without redesigning. Paper skeleton: paper/OUTLINE.md.

**Boundary (from 027 Amendment B, restated so nobody drifts):** everything
below is MEASUREMENT of where the fork signal lives. No mechanism claims this
cycle, no trigger tuning, sealed pool stays sealed. If a cell turns out
strong enough to make a mechanism viable, that is FUTURE WORK in this paper
and possibly the next one.

## T1 — THE table: representation x fork-signal matrix (paper centerpiece)

One question per row: can THIS representation of a run's state tell that two
runs resolved the same blocker differently? Columns: (a) same-vs-diff class
separation AUROC on the frozen commitment-pair pool (7,321 same / 1,246 diff
pairs — the pool that scored v1=0.555, v2=0.580), (b) stage-1 detection F1
where applicable, (c) access requirements (single-run? ensemble? registry?).

| row | representation | status |
|---|---|---|
| R1 | raw activations, 8 layers (32B) | DONE — permutation null (026 B) |
| R2 | learned L-space (A2) | DONE — null, n_testable=0 (026 B) |
| R3 | surface behavior hashes | DONE — 0.555 (027 A) |
| R4 | MiniLM embedding of mutating turn | DONE — 0.580 (027 B) |
| R5 | strong embedder (BAAI/bge-large-en-v1.5 via transformers, CPU) | NEW, ~1h |
| R6a | single-run LLM introspection: Fable subagent reads ONE run's prefix (registry-blind), binary ask/no-ask + free-text "what unresolved decision?" — scored as detection F1 vs census | NEW, ~240 judgments |
| R6b | ensemble LLM comparison: Fable subagent sees TWO runs' commitment excerpts (registry-blind), judges same/different resolution — scored as separation AUROC on a 200-pair stratified sample of the pool | NEW, ~200 judgments |

R1-R5 are representations; R6a/b measure whether the signal is recoverable
by full-context LLM JUDGMENT — single-run (R6a, the Ask-or-Assume-style
detector the field builds) vs ensemble-with-comparison (R6b). Expected shape
of the story if the negatives hold and R6b succeeds: the signal exists, is
invisible to every generic representation and to single-run judgment, and
becomes visible only under decision-aware comparison ACROSS runs. Every cell
reported as-run regardless.

R6 protocol (frozen): prompts registry-blind (contract-test greppable);
excerpts = the mutating turn's subgoal + command + error signature + ±400
raw chars of trace context (the labeler's own window rule); pair sample
stratified 100 same / 100 diff, seed 0; R6a prefix points = the round before
each run's first commitment on a forked blocker + matched rounds on
non-forked tasks; transport = Fable subagents in-session (025 Amendment A
precedent), chunked 20/agent, resumable via per-chunk result files; ~460
judgments ≈ well under one session budget. STOP rule: if session limits bite,
report the completed fraction — no silent truncation.

## F2 — the rollout-budget ablation (the "how many runs buy the signal" figure)

Census vs N: for N in {2, 4, 8, 12, 16, 24} seeds (20 resamples each, seed 0),
recompute: fraction of tasks with >=1 fork, forked-decision count, and the
per-decision run-level TESTABILITY floor distribution (026's exact-test
machinery). Data exists; pure CPU analysis (~1h implement + minutes to run).
This figure turns the "2/3 of tasks fork at N=24" census fact and the
"n_testable=0 at ~6 runs" floor finding into one design-guidance plot —
reviewers' "so what should I do" answer.

## T3 — probe-robustness appendix (closes the last gate-2 escape)

Full-dim (5120) linear probe + a 2-layer MLP probe (hidden 512, the one
nonlinear family, pre-declared here) on the FIXED labels, gate-2 task, same
s6,s7 split as the text control, vs the causal-masked text baseline 0.730.
Box, ~2-4h. Whatever the number, it goes in: if the MLP closes the gap the
internals story gets a caveat; if not, the negative is escape-proof.

## T5 — cross-scale replication (generality row)

R3/R4 AUROC + fork census on the 14B collection (data/a0_v2, local) and the
v1 7B sample. CPU, ~1 day implement+run. Activations rows at those scales
already exist historically (015/018) — cite with 026's corrected-baseline
caveats, do not rerun.

## F4 — lead-time and fork anatomy (descriptive, all data exists)

Commitment-dating distributions; fork onset vs commitment (gate7 machinery);
splits by fork type (structural/value, data/fork_type_annotations.json) and
by temperature arm. Feeds the census section; ~half a day.

## T6 — ground-truth error bounds (framing, no new labels)

Bound census error with what exists: audit_signature_discrimination (3
strict suspects / 214), the 46-disagreement adjudication record, judge
validation numbers (025 A). One paragraph + one appendix table. Optional
(owner time permitting, NOT blocking): ~50-item owner hand-label of fork/
no-fork tasks for a human-agreement point.

## Execution order (dependencies, ~2 weeks of experiments)

1. F2 + F4 (local CPU, no new deps) — start immediately, they de-risk nothing
   and feed the census section.
2. R5 (local, ~1h) then T5 (local, ~1 day).
3. R6a/R6b (Fable subagents; the only session-budget item) — after R5 so the
   embedder rows are frozen first.
4. T3 on the box (parallel to any of the above; owner runs or pastes).
5. Analysis freeze -> writing (paper/OUTLINE.md section order), target ~10
   writing days, ICLR deadline buffer >= 5 days.

## Amendment A (2026-08-20 — as-implemented instantiation, frozen BEFORE first runs)

Recorded before any 028 number exists. Nothing above is edited; these pin
the implementation choices the entry left open, each settled pre-run:

1. **F2 floor universe (pre-run adversarial review finding, confirmed on
   real data).** "026's exact-test machinery" computes floors over the
   GATE5 universe — per decision, runs carrying >=1 class-labeled READ
   (debug-trail phase==1), eligibility >=4 class-labeled reads and >=2
   classes — which on the full fixed labels has 37 eligible decisions at
   median 6 runs vs 63/10 in the commitment trail. F2's quantity (c) is
   therefore computed in the gate5 universe (execution-checked to reproduce
   exactly 37 eligible / median 6 / 13 testable before the first run);
   quantities (a)/(b) stay census (commitment-trail). Commitment-universe
   floors are emitted as clearly-labeled supplementary data.
2. **F4 frozen definitions**: commitment round per offline_ask_headtohead.
   commit_rounds; fork onset = min datable commitment round in the fork;
   fork completion = gate7 formula per blocker; ask window = completion -
   onset; modal class = largest committed class (ties by name sort).
3. **R6 instantiation** (scripts/r6_build_items.py IS the frozen
   instantiation, committed before any judgment): R6a = 4 judged runs per
   task x 60 tasks (equal per-task fire chances; fewer where fewer eligible
   runs), forked-task cut = the run's first commitment round on a forked
   blocker, non-forked cuts cycled from the sorted forked-k pool; prefix =
   turn-by-turn ActionEvent summaries (subgoal/command<=400ch/error) + the
   verbatim baseline/instruction.md; R6b stratification = round-robin over
   (task, blocker) groups, rng seed 0; judge-visible payload keys pinned by
   contract test (registry-blind, 4 new tests; suite 240). R6b AUROC score
   = confidence signed by the different/same answer, AUROC computed as the
   exact Mann-Whitney U with ties counted 0.5 (scores take 10 discrete
   values; a non-tie-aware rank formula was caught pre-run and replaced).
   Transport: items shuffled deterministically (seed 0) before 20-item
   chunking so no judge sees a truth-homogeneous chunk; payload item_ids
   re-keyed to opaque salted hashes (build-order ids monotonically encoded
   ground truth — an id-threshold rule scored 100% with zero excerpt use;
   caught pre-run, private map results/r6_items/blind_id_map.json
   translates back at scoring); judges instructed to read ONLY their
   payload file (transcripts audited for stray reads after the run);
   chunks stored via scripts/r6_store_chunks.py only when judgment ids
   exactly match the payload's id set (resumable: unstored chunks re-run).
4. **T5 data fact**: the v1 7B collection recorded ZERO ActionEvents
   (predates composite logging), so its R3/R4 pair pool is structurally
   empty; the 7B row reports census only, with this reason. 14B labels
   regenerated with the fixed labeler (models/t5_v2_14b_fixed: 244
   commitments, 182 action-sourced, 16 forked blockers — matches the
   historical 14B numbers); 7B likewise (models/t5_a0_7b_fixed: 354
   commitments, all trace-sourced, 32 forked blockers).
5. **R5 embedder**: BgeEmbedder = CLS pooling + l2 norm (model-card
   recipe), truncation 256 to match the MiniLM row, batch 32; driver
   recomputes v1/v2 alongside as consistency checks (no gate — 027 B's
   fallback stands; the row is reported as-run).
6. **T3**: split code replicated verbatim from gate2_text_control.py;
   MLP = sklearn MLPClassifier hidden 512, relu, adam, alpha 1e-4, batch
   256, lr 1e-3, max_iter 100, early stopping, random_state 0; every
   unpinned hyperparameter stays at the sklearn default (a stricter
   patience that crept into the draft script was caught pre-run and
   removed; the as-run config including patience and validation fraction
   is recorded in the output JSON); a 256-d JL linear probe runs first as
   a consistency check vs the recorded 0.2745; input-identity guards abort
   on any npz that is not the fixed-label set (787,281 reads; s6,s7 test
   split 9,180 / 110 classes); results persist after each probe.

7. **AUROC uncertainty (added 2026-08-22, BEFORE the intervals were
   computed).** Every T1/T5 separation AUROC gains a TASK-CLUSTERED
   bootstrap 95% CI (2000 draws, seed 0, scripts/t1_auroc_ci.py). This is
   additive: it does not rerun or vary any pre-registered cell, it puts a
   sampling interval around the same statistic. Clustering is required for
   the same reason gate5's permutation had to be run-level rather than
   read-level (026): commitment pairs are not independent — they share runs
   and tasks — so Hanley-McNeil or a naive pair bootstrap understates the
   SE. Both are reported, the naive one only to show the inflation. The
   point estimates already recorded are unchanged and remain the as-run
   numbers.

8. **Trace-blind vs trace-informed registry split (added 2026-08-22, BEFORE
   the numbers were computed).** The class artifact was authored in two
   passes with different exposure to the data, which gives a free
   construct-validity contrast that partially substitutes for the never-run
   collection-R3 sealed/OOD leg:
   - **trace-informed (20 tasks)**: swe_0, swe_1, swe_2, swe_10..swe_26 —
     drafted "grounded in registry+traces" and subsequently anchor-leak
     repaired (77 uniquely-predictive anchors dropped).
   - **trace-blind (40 tasks)**: swe_3..swe_9, swe_27..swe_59 — derived
     2026-07-18 REGISTRY-ONLY, before any traces for those tasks existed.
   Reported per group, as-run: fork census (fraction of tasks forked,
   forked blockers, commitments) and the commitment-pair separation AUROC
   with its task-clustered CI. Pre-declared reading: if the census and the
   AUROCs agree across groups, the fork construct and the negatives are not
   artifacts of registry authors having seen traces; a large gap is a
   construct-validity finding that MUST be reported and discussed either
   way. This is a subgroup analysis of existing cells, not a rerun; no
   pre-registered point estimate changes.

9. **T7 — cross-family replication (added 2026-08-22, BEFORE any second-family
   data exists).** The GPU box came free while the R6 LLM cells wait on
   subscription budget. Reviewer gap #4 is "one model family"; T5 already
   covers SCALE (7B/14B/32B) but every collection is Qwen. T7 re-runs the
   collection protocol VERBATIM on the same 60 tasks with the same frozen
   class artifact and the same labeler, changing ONLY the model family.
   Because the artifact is per-TASK, this needs no new registry work and no
   LLM budget.

   - **Protocol frozen = R2's**: `--classes data/interpretation_classes.json`
     (which also keeps the sealed pool untouched by construction),
     `--n-tasks 60`, `--max-steps 50`, defaults elsewhere (cadence 8,
     max_new_tokens 2048, temps 0.7/0.9/1.1/1.3, layers 0.2..0.85 fractional
     so they map across any depth), thinking OFF, nudge ON.
   - **Model**: primary `mistralai/Mistral-Small-3.2-24B-Instruct-2506`;
     fallback `google/gemma-3-27b-it` if the smoke gate fails.
   - **SMOKE GATE, frozen now** (3 tasks x 2 seeds -> a `_smoke` dir, before
     any full run): (a) no chat-template or protocol breakage and no
     reasoning-trace leakage into the transcript; (b) >=1 mutating action in
     >=50% of smoke runs; (c) median reads/run >= 10. PASS -> full run.
     FAIL on the primary -> try the fallback ONCE. FAIL on both -> record the
     NO-GO here and drop T7; no third candidate, no protocol tuning to make a
     model comply (that would make the families non-comparable).
   - **Seeds staged**: `--n-runs 12` first (complete and balanced at every
     point; the collector skips existing run JSONs so it resumes), then
     re-launch with `--n-runs 24` to extend to R2 parity if the box stays
     free.
   - **Deliverable, as-run either way**: fork census + the T1 behavioral rows
     (R3 hashed / R4 MiniLM / R5 bge) on the second family, each with the
     task-clustered CI of item 7. Activations ARE collected (default layers)
     so the internals rows remain possible, but replicating R1/R2 needs the
     full A1/A2/A3 + gates pipeline and is a STRETCH goal, not a commitment.
   - **Pre-declared reading**: if the second family lands in the same
     .54-.60 band, the negative is not a Qwen artifact and gap #4 shrinks to
     benchmark+scaffold. If it lands materially higher, that is a
     family-dependence finding and MUST be reported as such — it would bound
     the paper's claim rather than break it.
   - **NOT in scope**: the sealed pool stays sealed; harbor_sql OOD still
     needs its own class artifacts and is deferred; the 25 missing temp-1.3
     R2 runs are NOT backfilled (adding runs to `data/a0_v3_32b` would change
     the frozen universe every current number is computed on).

## Stop rules and reporting

Every cell lands in the paper as-run. No cell is rerun with variations
unless this entry is amended BEFORE the rerun. The T1 pool, splits, seeds,
prompts, and sample sizes above are frozen by this entry once accepted.
Suite must stay green; every experiment ships with its driver script in
scripts/ and its output in results/ (fresh files, never overwriting R2-era
or 026/027 artifacts).

## Amendment B (2026-08-23 — T7 loader compatibility, frozen BEFORE the run)

Amendment A item 9 named `mistralai/Mistral-Small-3.2-24B-Instruct-2506` as
T7's primary model and forbade "protocol tuning to make a model comply." On
the box, that model turns out not to be loadable by the collector at all —
not a tuning question, a hard incompatibility found before any GPU time was
spent. This amendment records the diagnosis and the owner's chosen fix.

**Diagnosis (verified on the box, no weights downloaded — meta-device
instantiation from the published config).**

1. `Mistral-Small-3.2` is `Mistral3ForConditionalGeneration`, a multimodal
   wrapper. `mistral3` is **not present in transformers'
   `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`**, so `hf_reader`'s
   `AutoModelForCausalLM.from_pretrained` raises rather than loading it.
2. Its depth and width live in `config.text_config` (40 layers, hidden
   5120); top-level `config.num_hidden_layers` / `config.hidden_size` are
   absent, so `hf_reader`'s layer resolution would break even if the load
   succeeded.
3. Its decoder blocks are at `model.model.language_model.layers`, not
   `model.model.layers`, so `layer_capture._decoder_layers` raises too.
4. The pre-registered fallback `google/gemma-3-27b-it` is `gated=manual` on
   the Hub and no HF token is configured on the box, so the fallback path
   as written is unreachable there.

**Options considered.** (a) Swap to `Mistral-Small-24B-Instruct-2501` — the
text-only predecessor, ungated, `MistralForCausalLM`, and *identical* depth
and width (40 x 5120) to the 3.2 text tower; zero code change, model id
only. (b) Teach the reader to load multimodal wrappers and read the nested
config. (c) Token + gemma fallback. (d) Drop T7.

**Decision (owner, 2026-08-23): option (b).** Keep the pre-registered model
literal and extend the loader. The change is a *compatibility* extension,
not a protocol tune: it does not alter what is captured, when, or how — the
layer fractions (0.2..0.85), cadence, cue selection, sampling params and
max_steps are untouched, and the Llama/Qwen path is byte-for-byte unchanged
(the nested lookups are fallbacks that only fire when the flat layout is
absent).

**Frozen specifics.**

- Load: try `AutoModelForCausalLM`; on a wrapper architecture fall back to
  `AutoModelForImageTextToText`. Text-only models are unaffected.
- Config: read `num_hidden_layers` / `hidden_size` from `config.text_config`
  when present, else top-level.
- Capture: `_decoder_layers` accepts `model.model.language_model.layers` as
  a fallback when `model.model.layers` is absent; the final-norm lookup
  follows the same object, preserving the last-layer post-norm semantics
  that make hook capture bit-compatible with the R2 path.
- **Contract test** (rule 4) pins all three on the real published config via
  meta-device instantiation, so it needs no weights and no GPU.

**Stated risk, recorded rather than hidden.** T7 captures from a different
family's decoder stack; the layer *fractions* are identical but the blocks
they land on are a different model's. That is inherent to any cross-family
comparison and is exactly what T7 measures. What this amendment must NOT be
read as licensing: no per-model tuning of fractions, cadence, or sampling to
improve a result. If the smoke gate fails on Mistral-3.2 under this loader,
the NO-GO stands as written in Amendment A item 9.

**Blocker still open at the time of writing.** The hil-bench task docker
images were destroyed by an ephemeral-NVMe wipe on the box, and rebuilding
them pulls from `ScaleAI/hil-bench-swe-images`, which returns 401
unauthenticated. Until an HF token is present on the box, T7 cannot run
regardless of this amendment. The loader fix is landed and tested; the
collection is not started.

### Amendment B — as-run outcome (2026-08-23, same day)

The loader fix was validated on REAL weights, not just the meta device
(`scripts/xfam_loader_smoke.py`, `results/xfam_loader_smoke.json`). It works:

    loaded: Mistral3ForConditionalGeneration
    n_layers=40 hidden=5120 layers=[8, 12, 16, 20, 24, 28, 32, 34]

That is the full Amendment B path exercised end to end — wrapper load,
nested-config depth/width, and the frozen 0.2..0.85 fractions resolving onto
the second family's stack. **Blocker 1 is genuinely closed.**

**But the smoke check then FAILED on a blocker Amendment B did not
anticipate:** `Mistral-Small-3.2-24B-Instruct-2506` **ships no HF
`chat_template`** (`tokenizer.chat_template is None`; its template lives only
in `mistral-common`). `generate_segment` builds every agent turn with
`apply_chat_template`, so the collector cannot construct a single prompt for
this model. Loading it is necessary but not sufficient.

Supplying a chat template is NOT the same kind of change as the loader fix.
The loader did not affect what the model sees; a chat template defines
exactly that — the prompt format the whole trajectory is conditioned on.
Hand-writing or importing one is squarely the "protocol tuning to make a
model comply" that Amendment A item 9 forbids, and it would make the two
families non-comparable in precisely the way that rule exists to prevent.

Two further facts recorded for whoever picks this up:

- `mistralai/Mistral-Small-24B-Instruct-2501` — the text-only predecessor,
  same family, same size class, **same 40 x 5120 depth and width** — **does**
  ship a chat template, and is a plain `MistralForCausalLM`. It needs neither
  the loader fix nor a template decision.
- transformers warns on BOTH Mistral tokenizers: *"incorrect regex pattern …
  This will lead to incorrect tokenization. You should set
  `fix_mistral_regex=True`"*. Since read positions are token indices, this
  bears on capture and must be settled before any Mistral collection —
  independently of which of the two models is chosen.

**Status: T7 remains NOT launchable.** No protocol tuning was performed. The
model choice is back with the owner.

## Amendment C (2026-08-23 — T7 model swap + tokenizer flag, frozen BEFORE the run)

Supersedes the model choice in Amendment A item 9 and closes out Amendment B.
Two owner decisions and one correction of the record.

### C.1 Model: 3.2-24B -> Mistral-Small-24B-Instruct-2501

Amendment B verified that the Amendment B loader genuinely loads
`Mistral-Small-3.2-24B-Instruct-2506` on real weights, and then found that the
model **ships no HF `chat_template`**, so `generate_segment` cannot build a
single agent turn for it. Supplying a template would define what the model
sees — the one thing the loader fix did not touch — and is the protocol
tuning Amendment A item 9 forbids.

**Decision (owner, 2026-08-23): use
`mistralai/Mistral-Small-24B-Instruct-2501`.** Verified on the box before
adopting:

| | 3.2-24B-2506 (was) | 24B-2501 (now) |
|---|---|---|
| architecture | `Mistral3ForConditionalGeneration` | `MistralForCausalLM` |
| depth x width | 40 x 5120 (in `text_config`) | **40 x 5120** |
| `chat_template` | **absent** | **present** |
| gated | no | no |

Same family, same size class, same depth and width as the 3.2 text tower, and
it brings its OWN chat template — so the prompt format is the model vendor's,
exactly as R2 used Qwen's shipped template. **Nothing about the protocol is
chosen by us; only the model id moves.** This keeps T7 answering the question
it was pre-registered to answer (does the .54-.60 band survive a change of
model FAMILY) with strictly fewer researcher degrees of freedom than the
original pick would have required.

The Amendment B loader support stays in the tree: it is tested, it does not
fire for text-only models, and it is what any future wrapper model would need.

### C.2 Tokenizer: pass `fix_mistral_regex=True`

transformers warns on **both** Mistral tokenizers that the shipped regex
"will lead to incorrect tokenization" and asks for this flag. Reads are
recorded at token indices, so tokenization is not cosmetic here.

**Decision: pass it unconditionally**, in one code path for every model
rather than a per-family branch. Verified safe: Qwen3 (the R2 family) yields
**byte-identical token ids** with and without the flag, so R2-era behaviour is
untouched. Tokenizers that reject the kwarg fall back to the plain load.

**Reported as-run, not overstated:** on 13 varied probe strings (digits,
whitespace, tabs, camelCase, code) the flag produced **identical ids** on
2501 too. So on the evidence it is currently a no-op that silences a warned-
about defect, rather than a change that measurably moves read positions. It
is set because the warning concerns exactly the quantity our reads are
indexed by, not because a difference was observed.

### C.3 Correction: the HF-token "blocker" in Amendment B was WRONG

Amendment B and STATUS.md recorded that restoring the destroyed hil-bench
images needs an HF token, because `ScaleAI/hil-bench-swe-images` returned
401. **That was my error, and the owner was right to push back on it.** The
401 came from querying the *model/dataset* API for something that is an HF
**bucket** — a different namespace. The bucket is **publicly readable**:
`hf buckets ls` and `hf buckets cp` both work anonymously, which is exactly
the path `warmup_images.sh` uses.

The real blocker was the wiped docker/containerd roots: both daemons come up
"active" pointed at an empty volume, so `docker load` fails with
`metadata.db: no such file or directory`. Fix is to stop docker+containerd,
recreate both roots, start containerd then docker. Restoration is otherwise
credential-free and is now scripted in
`scripts/restore_hilbench_images.py` (60 tasks, 174 GB, sealed pool excluded
by construction because the task list is derived exactly as collect_v2
derives it).

**No HF token is required for T7.**

## Amendment D (2026-08-23 — error_signature input bound, frozen DURING the T7 run)

Found while diagnosing an idle GPU 5h into the T7 collection. Recorded before
the fix was applied, per rule 2.

**The defect.** `agent_loop.run_agent` computed the action's failure
fingerprint from the FULL, untruncated command output, and only truncated
afterwards for the transcript:

```python
event.observables["error_signature"] = error_signature(code, out)   # full output
obs = truncate_obs(out, cfg.obs_head, cfg.obs_tail)                 # then truncate
```

On `swe_1-s10` the model issued `grep -r -` in `/app` — a malformed grep that
matches the literal `-` in every file, recursively. Measured in a scratch
container from the same image: **723 MB of output** inside the 120 s exec
window. `error_signature` then scanned all 723 MB, capturing whole lines
(`^.*?(...)$`) and allocating a normalized copy of each match. Shard 1 spent
**over an hour at 97% CPU on a single metadata field** with a loaded 24B model
idle beside it, while shards 0/2/3 ran normally at 93-96% GPU.

It is deterministic — seeds are fixed and generation is seeded per segment —
so restarting the shard replays the same command and wedges again. It is also
present in the R2 code path unchanged, and can recur on any runaway command
across the remaining ~1,385 runs.

**Decision (owner, 2026-08-23): fingerprint the observation the model actually
received.** Truncate first, then compute the signature from the truncated
observation:

```python
obs = truncate_obs(out, cfg.obs_head, cfg.obs_tail)
event.observables["error_signature"] = error_signature(code, obs)
```

The justification is not only speed. The docstring defines the field as the
"normalized failure fingerprint of the observation an action produced", and
the observation the agent conditioned on WAS the truncated one — the dropped
middle never entered the trajectory. Fingerprinting text the model never saw
was arguably the bug, independent of cost.

**Honest statement of the blast radius.** This is NOT limited to pathological
outputs. `truncate_obs` fires on any output over `obs_head + obs_tail`
(1500 + 500 = 2000 chars), so the signature can change for ordinary actions
too — specifically when an error-matching line falls in the dropped middle and
shifts which hit is selected. In practice the function prefers the LAST typed
`Name: message` line, and typed exceptions land at the end of output, which
`obs_tail` retains — so most signatures should be unchanged. **That
expectation is not measured and must not be reported as if it were:** the
collector never persists observations (only the model's generated text), so
the impact on already-collected runs cannot be recomputed after the fact.

**What this change does NOT touch — checked, because the first framing of it
overstated the risk.** `truncate_obs(out, cfg.obs_head, cfg.obs_tail)` is
byte-identical before and after: same call, same arguments, moved two lines
earlier. `obs_head=1500 / obs_tail=500` are unchanged and last moved in
`31be8b4` (v2 collection, decisions/017) — *before* R2, unchanged since. The
line that feeds the model,
`messages.append({... f"[exit {code}]\n{obs}\n\nNext step?"})`, does not
appear in the diff at all.

So the agent was ALREADY conditioned on the truncated observation, in R2 too.
The 723 MB the old code scanned was text the model never saw — discarded by
`truncate_obs` on the very next line. **Trajectories, reads, activations and
fork behaviour are bit-identical before and after this amendment.** Nothing
here shortens a trajectory or suppresses a disagreement; the shortened-window
failure mode the owner has previously measured concerns `obs_head`/`obs_tail`,
which this amendment does not modify.

The only quantity that can differ is the `error_signature` string, on actions
whose raw output exceeded 2000 chars and whose error line fell in the dropped
middle.

**Disposition of the 55 pre-amendment runs (owner, 2026-08-23): KEEP them,
restart all four shards on the fixed code.** Rationale: since the model's
input is unchanged and seeds are deterministic, re-collecting those runs would
reproduce the same trajectories and differ only in that metadata string —
5.5 h of GPU time to alter one field, recovering no lost behaviour. They stay
identifiable (collected before 2026-08-23 17:17Z) and can be dropped at
analysis time if `error_signature` ever proves load-bearing for a T7 number.
All four shards were restarted at 17:17Z so every subsequent run uses one code
path.
## Amendment E (2026-08-25 — R6a/R6b as-run outcome, including the NO-GO)

Both LLM cells are complete at 100% coverage. Recorded here as-run, whichever
way they landed. **Nothing in the protocol was tuned, and no cell was rerun
with variations** — the only re-runs were whole chunks that the transport
rejected before any judgment of theirs was stored (D.3).

### E.1 The numbers

`results/r6_llm_cells.json`, `results/r6b_auroc_ci.json`.

| cell | coverage | headline | context |
|---|---|---|---|
| R6a single-run introspection | 238/238 items, 60/60 tasks | task-detection **F1 0.709** (P 0.718 / R 0.700; 39 fired / 40 true) | divergence detector 0.507 · random budget-matched 0.582 · **always-ask 0.800** |
| R6b ensemble comparison | 200/200 pairs, 46 tasks | **AUROC 0.5786**, task-clustered 95% CI **[0.481, 0.670]** | R3 hashed .555 · R4 MiniLM .580 · R5 bge .573 |

R6a run-level: accuracy 0.580, ask rate 0.504 — **below the 66.4% (158/238)
always-ask base rate**, i.e. worse than trivial at the run level even though
it beats the fitted detectors at the task level.

R6b decision accuracy 0.585. The judge answered "different" on **129 of 200**
pairs against a pool that is 100/100 by construction, and its per-arm
accuracy is lopsided: **73/100 correct on diff, 44/100 on same**. The AUROC
is computed on the confidence-signed score, so this yes-bias moves accuracy
without by itself moving AUROC; it is reported because it characterises HOW
the judge fails, not to discount the number.

### E.2 The pre-declared reading, and the NO-GO

The 028 entry pre-declared: *"Expected shape of the story if the negatives
hold and R6b succeeds: the signal exists, is invisible to every generic
representation and to single-run judgment, and becomes visible only under
decision-aware comparison ACROSS runs."*

**R6b did not succeed.** 0.5786 sits inside the same .54–.60 band as every
generic representation — statistically indistinguishable from MiniLM's .580
— and its clustered interval **contains chance**. The pre-declared story
shape is therefore NOT available and must not be written. This is recorded
as a NO-GO on that framing.

What it does to the reviewer gaps in STATUS.md:
- **Gap #3 ("you didn't try hard enough") is NOT closed by a ceiling.** R6b
  was the designated way to show one, and there is no ceiling to show. The
  honest statement is stronger in one direction and weaker in another: the
  negative now extends from generic representations to full-context LLM
  judgment with cross-run comparison, but we still cannot distinguish "the
  signal is not there" from "nothing we tried recovers it." Say both.
- **Gap #1 (label validity) gains weight, not loses it.** An LLM judge
  reading the same excerpts the labeler used agrees with the labels at
  0.585. That is consistent with a weak signal AND with noisy ground truth,
  and this cell cannot separate the two. **T6 plus the ~50-item owner
  hand-label is now the highest-leverage remaining item, more so than before
  this run.**
- **Gap #7 (do not over-claim "zero signal") applies to R6b too.** Its CI
  contains chance; R4's does not. Do not report R6b as "chance" or as
  "signal" — report .579 [.481, .670] and let the band speak.

Scope boundary holds: this is measurement. No mechanism claim is made or
implied by either cell.

### E.3 Transport, as-run (the 028 STOP rule in action)

The STOP rule ("if session limits bite, report the completed fraction — no
silent truncation") was exercised for real:

- **First launch**: all 12 R6a chunks ran; **all 10 R6b chunks died on a
  session limit** before judging anything. Zero R6b judgments existed, so
  nothing was contaminated.
- **Two chunks were silently truncated by their judge** — `r6a_chunk_01`
  returned 16 of 20 items, `r6b_chunk_02` returned 19 of 20.
  `r6_store_chunks.py`'s all-or-nothing id check rejected both and left them
  pending. Each was re-judged WHOLE. **No partial chunk was ever stored, and
  no item was kept from a rejected chunk** — that check is what stops
  "re-run the missing items" from becoming selection on the outcome.
- **One chunk failed on a network error** (ENOTFOUND) with no judgment
  produced; re-run.
- Final coverage is 100% on both cells, so the completed-fraction clause did
  not have to be invoked in the paper.

**Registry-blindness audit (Am.A item 3 obligation, discharged).** Across all
14 workflow runs the judge transcripts contain **70 `Read` calls and 24
`StructuredOutput` calls and no other tool use whatsoever** — no Bash, Grep,
or Glob. Every one of the 23 distinct Read targets is a payload chunk file
under `results/r6_chunks/`. No judge opened the registry, the item files with
their truth fields, or `blind_id_map.json`.

### E.4 Item 7 extended to R6b (additive, no cell changed)

Amendment A item 7 requires a task-clustered bootstrap CI on every T1/T5
separation AUROC; R6b is one, so it gets the same treatment on the same
terms. `scripts/r6b_auroc_ci.py` reuses `t1_auroc_ci.py`'s estimator and
constants verbatim (exact Mann-Whitney U with ties 0.5, 2000 draws, seed 0,
clustering on task), substituting the confidence-signed judgment score for
the representation distance. It carries a hard guard that aborts unless the
recomputed point estimate reproduces the recorded cell exactly; it does.
Naive pair CI [0.500, 0.656] is reported alongside only to show the
inflation, per item 7.

### E.5 One deviation from the committed runner, recorded

`scripts/r6_runner_workflow.js` as committed did not pin a judge model, so
its subagents would have inherited whatever model the driving session ran.
The frozen protocol says the transport is **Fable subagents**, so
`model: 'fable'` was set explicitly on both agent calls before the first
judgment was collected. This makes the runner match the pre-registration
rather than depart from it; it is recorded because it is a change to a
committed file made during an experiment.
