# 023: R1 diversity pilot — RESULT: PASS, R2 is a go

Date: 2026-08-08. Status: ACCEPTED. Closes the R0/R1 gate opened by
decisions/021 §1 and its R1 pre-registration.

Run on the local 2x RTX PRO 6000 Blackwell box (96GB cards, 2 data-parallel
shards), Qwen3-32B, 6 fork-bearing tasks x 12 seeds = **72/72 runs, 6.4
GPU-hours**. Both pre-registered criteria pass. The criteria were fixed in
decisions/021 before the run and are evaluated here unchanged.

## R0 — the clamp was real, confirmed on the box

`effective_generation_config()` against the actual weights:

```
shipped:   do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
           min_p=None, repetition_penalty=1.0
overrides: top_p=1.0, top_k=0, min_p=0.0
```

So the entire prior 32B collection ran with the candidate set truncated to 20
tokens and nucleus 0.95 — the mechanical explanation decisions/021 §1
suspected for the single-run-minority forks. The fixed path is the default.

Also verified: 32.76B params bf16 loads **single-device** (65.5GB allocated on
one 102GB card, `hf_device_map` one entry), which is the premise of the
data-parallel shard design.

## R1 — criterion A: forks. 4/6 tasks, bar >= 3

">= 2 interpretation classes each committed by >= 2 DISTINCT runs", per
blocker (two classes under *different* blockers are not a fork):

| task | blocker | split |
|---|---|---|
| swe_30 | `ambiguous_non_collection_spec_ownership` | `delegate_wrapped_find_spec` 2 / `synthesize_module_spec` 2 |
| swe_36 | `controller_replacement_import_path` | `keep_shim_controller` 8 / `stdlib_collections_abc` 4 |
| swe_47 | `contradictory_enum_definition_location` | `hooks_useselectioncontrols` 9 / `state_useselection` 3 |
| swe_50 | `unspecified_mime_detection_precedence_and_fallback_chain` | `type_then_extension_then_default` 4 / `content_sniffing` 4 |

swe_12 and swe_4 converged on every blocker across all 12 runs each. These are
genuine non-forks, not partial-data artifacts — both tasks were complete when
scored. **Unclamping did not make everything diverge**, which is the right
shape for a real effect rather than an artifact.

## R1 — criterion B: reads. median 216/run, bar >= 25

Across all 72 runs. Never in doubt; the `--cadence 4` fallback in
decisions/021 is not needed.

## Reading the number correctly (trap)

`scripts/generate_labels.py`'s summary line reports **"forked blockers: 5"**,
which is NOT the pre-registered criterion — it counts blockers whose second
class was committed by only ONE run. The registered metric is 4/6 and must be
computed from the `labels_debug.jsonl` commitment rows (`kind == "commitment"`,
group `chosen` by `blocker`, require >= 2 classes with >= 2 distinct runs).
swe_4 and swe_50 are where the two metrics diverge.

## Measured cost — budget R2 off the MEAN

| | value |
|---|---|
| per-run median | 136s |
| per-run **mean** | **319s** |
| per-run max | 4922s (82 min, one run) |
| reached TASK_DONE | 52/72 (72%) |
| output size | **~31MB/run** (2.2GB for 72) |

The distribution has a fat right tail: the slowest runs cluster on seeds where
`seed % 4 == 3`, i.e. **temperature 1.3** (`--temps 0.7,0.9,1.1,1.3`), which
produce max-token trajectories every turn. `swe_30-s3` alone ran 4922s with
7856 reads and never finished. Any R2 estimate taken from the median will be
~2.3x optimistic.

R2 projection (1440 runs) at the mean: **~128 GPU-hours** — ~64h wall on 2
cards, ~16h on 8. Disk: **~44GB**, so `--out` must be on the NVMe.

## Infrastructure defects found and fixed en route

1. `resolve_tokenizer` read only `collection_manifest.json`, but sharded
   collections write `collection_manifest.s0.json` — so `auto` silently fell
   back to the Qwen2.5-7B default and labeled a Qwen3-32B collection with the
   wrong tokenizer. Every data-parallel collection was affected, R2 included.
   Fixed + tested (globs, and raises if manifests disagree on `model_id`).
2. `scripts/clone_third_party.sh` aborted on the first repo on any fresh box:
   each `third_party/<name>/` already holds a tracked `PROVENANCE.md`, and
   `git clone` refuses a non-empty target under `set -e`.
3. docker 29 uses the **containerd image store**, so `data-root` alone leaves
   image blobs on the root disk; root hit 99% mid-run. containerd's own `root`
   must move too. swe_12 unpacks to 26.2GB from a 7.56GB archive.
4. `hf buckets cp` — how every task image is fetched — does not exist in
   huggingface-hub 0.36.2 (what transformers 4.57 pins); needs hub 1.x.

## Decision

**R1 PASSES. R2 is a go.** The clamp explanation in decisions/021 §1 holds:
removing the truncation produced multi-run forks on 4 of 6 tasks where the
clamped collection could not diverge.

Evidence retained in-repo: `data/a0_pilot_32b/events.s*.jsonl` and
`collection_manifest.s*.json` (per-run steps/reads/timings/finish-reason).
The 2.2GB activation collection and 336MB `labels.npz` were NOT retained —
they exceed git and the box was terminated after this run.
