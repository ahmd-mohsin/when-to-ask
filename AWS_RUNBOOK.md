# AWS runbook — what to run when the GPU instance is ready

Order matters; each step gates the next (decisions/011). Paste each step's
output back to Claude before moving on — several steps are stop-and-review
points by design.

## 0. Setup (once)

```bash
# instance: 1 GPU with >= 24 GB (g5.xlarge / g6e.xlarge class), Ubuntu DLAMI
git clone <this repo> && cd <repo>
pip install -e .[gpu,dev]
python -m pytest -q          # 90 tests, all CPU — should pass as on the laptop
```

## 1. Hook proof (Phase-0 go/no-go, ~5 min)

```bash
python scripts/prove_hook.py
```

Downloads Qwen2.5-Coder-7B-Instruct (~15 GB), runs one under-specified task
N=2, prints PASS/FAIL per spec-A0 check. **If FAIL, stop and report.**

## 2. A0 collection v1.6 (grounded ADR 012 + multi-layer ADR 014) —
##    the full-batch procedure; this is the LAST GPU step

```bash
# (a) extract pre-patch source context from the task docker images (leak
#     analysis in the script header; falls back gracefully, recorded in the
#     manifest). If data/task_context/ is empty (ephemeral NVMe wiped on a
#     stop/start), re-run this; --scratch-dir keeps multi-GB archives off root.
python scripts/extract_task_context.py --n-tasks 20 --scratch-dir /opt/dlami/nvme/wta-scratch

# (b) collect capturing 4 LAYERS in one forward pass (--layers, ADR 014) so
#     the layer sweep is laptop-only forever after. Same seeds/prompts as
#     before -> directly comparable; only the saved activations change.
python scripts/collect_a0.py --n-tasks 20 --layers 0.4,0.5,0.6,0.7
```

Writes `data/a0/<task>/<run>.{npz,json,txt}` (npz `h` now **(R, 4, 3584)**) +
per-task `prompt.txt` + `collection_manifest.json` (versions, GPU, grounding
mode, `reader.layer_indices`, per-run timing/finish-reason) + `events.jsonl`.
Verify: npz shape (R, 4, 3584) float16, finite, nonzero variance; 20/20
`mode=docker`; distinct-signature count (< 30% fires decisions/008 — tell
Claude). Then tarball `data/a0/` + `data/task_context/`, print sha256, report.
The box can be stopped after — all sweeps/gates run on the laptop.

## 2b. v2 collection — REAL agent trajectories (decisions/017) — CURRENT step

```bash
git pull && python -m pytest -q          # expect 108 green
python scripts/collect_v2.py --n-tasks 20 --scratch-dir /opt/dlami/nvme/wta-scratch
```

Runs 8 seeded agent trajectories per task inside each task's docker container
(4-layer capture + cadence/cue/value reads baked in). Budget ~4-8 GPU-hours
(15 turns × ~1k tokens × 160 runs). Watch the manifest per run: `steps`,
`finished` (reached TASK_DONE), `reads_by_trigger` (expect nonzero `value`),
`actions`. Broken images are skipped and logged, not fatal. Tarball
`data/a0_v2/` back to the laptop when done. For the 32B pass afterwards:
same command + `--model-id Qwen/Qwen2.5-Coder-32B-Instruct` on a g5.12xlarge.

## 2c. SCALE collection — 60 train tasks at Qwen3-32B (decisions/018+019) —
##     CURRENT step. Instance: **g7e.2xlarge** (1x RTX PRO 6000 Blackwell,
##     96 GB VRAM, ~$3.36/hr on-demand us-east-1; verified 2026-07-19).
##     Single-GPU fit for 32B bf16 (~65 GB + KV), ~3x faster than the sharded
##     4xA10G alternative -> est. $60-90 total. Fallback if no capacity:
##     g5.12xlarge (4x A10G, spot). NOT g6e.xlarge (48 GB < 65 GB).
##     At boot on g7e: (1) use a CURRENT DLAMI (Blackwell needs cu128+;
##     torch cu130 as on the 14B box is fine); (2) `df -h` to find the local
##     NVMe mount — it may not be /opt/dlami/nvme on this family; point
##     --scratch-dir at it.

```bash
git pull && python -m pytest -q          # expect 118 green

# (a) SMOKE FIRST (~1-2 h, <$10) — Qwen3-32B has never run in this harness.
#     3 tasks x 2 runs; verifies: no <think> blocks in segments (thinking
#     pinned OFF by default, decisions/019), one-bash-block protocol
#     compliance, memory fits, and gives a measured per-run time for the
#     full-run cost estimate. PASTE THE MANIFEST + one .txt back to Claude
#     BEFORE launching the full run.
python scripts/collect_v2.py \
    --model-id Qwen/Qwen3-32B \
    --n-tasks 3 --n-runs 2 \
    --classes data/interpretation_classes.json \
    --out data/a0_v2_32b_smoke \
    --scratch-dir /opt/dlami/nvme/wta-scratch

# (b) FULL RUN (only after smoke review) — 60 train tasks x 8 seeds.
python scripts/collect_v2.py \
    --model-id Qwen/Qwen3-32B \
    --n-tasks 60 \
    --classes data/interpretation_classes.json \
    --out data/a0_v2_32b \
    --scratch-dir /opt/dlami/nvme/wta-scratch
```

Notes that matter:
- `--classes` is REQUIRED: restricts collection to the 60 artifact (train)
  tasks. Without it, sorted-dir order reaches swe_60+ — the SEALED TEST
  POOL — before swe_7/8/9. Never collect swe_60+.
- Thinking mode is OFF by default and recorded in the manifest (do NOT pass
  --enable-thinking). If smoke segments still contain <think> text, STOP and
  report — that's a template regression, not something to work around.
- Rough full-run budget: 480 runs; at 14B a run averaged ~2.5 min, sharded
  32B is ~2-4x slower → expect ~40-80 box-hours (~$100-200 spot). The smoke
  run replaces this guess with a measured number — trust the measurement.
- Broken images are skipped and logged (they still count toward --n-tasks;
  report "SKIPPED" manifest entries).
- Watch per-run: steps, finished, reads_by_trigger (nonzero value), actions.
- When done: tarball data/a0_v2_32b/ back to the laptop, print sha256.
  Laptop then reruns audit_labels + sweep + run_full_gates --kfold 5
  (step 3) against the 32B data.

## 3. Offline training + sweeps + gates (laptop, CPU — no AWS)

```bash
python scripts/audit_labels.py                       # human-readable label audit
python scripts/sweep.py --layers 0,1,2,3             # rank layers (gate1+gate5) + eps/window
python scripts/run_full_gates.py --layer <best> --eps-settle <best> --window <best> --kfold 5
```

`sweep.py` prints the layer table (best by gate-5 lean-separation) and the
eps×window table (best A3 settle rate), then the exact `run_full_gates.py`
command for the trustworthy k-fold gate numbers. That gate run is the owner
STOP point (decisions/011/013/014).

train_offline prints: label coverage + class balance, A1 held-out AUROC, the
GRL-treadmill check, A3 settle rate + benign-spread reference — each number's
audit trail lands in models/ (labels_debug.jsonl, a2_history.jsonl). Read the
audit file; if labels look wrong, the fix is the class artifact / lexicons,
never the downstream.

## 2d. THE REAL RUN — local RTX PRO 6000 Blackwell box (decisions/021)

**Box reality (verified on the box, 2026-08-08): 2 cards, 102GB each; NVMe
at `/opt/dlami/nvme` (NOT /mnt/nvme); root disk has ~13GB free, so the venv
(`/opt/dlami/nvme/wta-venv`, torch cu130) and `HF_HOME` live on the NVMe.
transformers v4 or v5 both work as of commit after b94357d (the
torch_dtype->dtype rename is handled in hf_reader).** All shard loops below
are written for 2 GPUs — scale the loop and `--num-shards` to your card
count if more arrive.

Qwen3-32B bf16 (~65GB) fits on **one** card (verified: 65.5GB allocated,
single-device map), so each GPU is an independent worker over a disjoint
task slice (`--shard i --num-shards N`). No tensor sharding, near-linear
scaling; all shards write the same `--out` (per-task subdirs never collide;
events/manifest are suffixed per shard).

**R1 preflight — the clone (the tasks are NOT in git, by licence design):**
`third_party/hil-bench` is reference-use-only (decisions/003) and is never
committed, so a fresh box has only PROVENANCE.md. Restore it with:

```bash
bash scripts/clone_third_party.sh   # pins hil-bench @352d14c; verify with: git -C third_party/hil-bench rev-parse --short HEAD
```

`data/task_context/` is NOT needed for collect_v2 (that was the v1.5
collect_a0 grounding path): v2 reads each task's `baseline/instruction.md`
directly and pulls docker images itself via the HF bucket
(`extract_task_context.try_load_archive`).

**Three more preflight facts, each found the hard way on 2026-08-08:**

1. **`hf buckets cp` needs huggingface-hub 1.x.** That is the call that
   fetches every task image. **This failure is silent and expensive:** if
   `hf` is missing or too old, `try_load_archive` returns None, the task is
   logged `task_skipped`, and the run continues on whatever images happen to
   be cached in the local docker store. Observed 2026-08-08: a 60-task R2
   launched with no `hf` on PATH would have collected **4 tasks** and skipped
   56, with the only evidence four `task_skipped` lines across four JSONL
   files. The bucket serves anonymously — **no auth, no token needed.**

   Always check before launching, on any box:
   ```bash
   hf buckets --help >/dev/null 2>&1 && echo "hf OK" || echo "hf MISSING"
   ```
   On the current verified stack (transformers 5.14.1) huggingface-hub 1.27.0
   comes with it, so `/opt/dlami/nvme/wta-venv/bin/hf` already works — just
   put the venv first on PATH. **No separate CLI venv is needed.** Only on the
   OLD transformers 4.57 stack (which pins hub 0.36.2, where the subcommand
   fails with `invalid choice: 'buckets'`) must you isolate the CLI so it
   cannot perturb torch:
   ```bash
   uv venv /opt/dlami/nvme/hfcli-venv && uv pip install \
       --python /opt/dlami/nvme/hfcli-venv/bin/python huggingface-hub
   ```
   (`uv` is NOT preinstalled on the DLAMI — `pip install uv`, or fall back to
   `python -m venv` + `pip install huggingface-hub`.)
2. **Move image storage onto the NVMe BEFORE pulling anything — BOTH
   halves.** The ~29GB root disk cannot hold the 6 pilot images (~27GB of
   archives). Setting docker's `"data-root": "/opt/dlami/nvme/docker"` in
   `/etc/docker/daemon.json` (keep the existing `nvidia` runtime block) is
   **not sufficient on docker 29**: it uses the containerd image store —
   `docker info` reports `Storage Driver: overlayfs`, not `overlay2` — so
   image blobs go to `/var/lib/containerd`, which `data-root` does not
   control. Root still hit 99% mid-run. Also set containerd's root:
   ```bash
   systemctl stop docker docker.socket containerd
   rsync -a --remove-source-files /var/lib/containerd/ /opt/dlami/nvme/containerd/
   printf '\nroot = "/opt/dlami/nvme/containerd"\n' >> /etc/containerd/config.toml
   systemctl start containerd && systemctl start docker
   ```
   Verify with `du -x -sh /var/lib/containerd` (should vanish) and
   `docker images` (pulled images survive the move). **Check the append
   landed at top level** — `root` must not fall inside a TOML table. On the
   stock DLAMI file every `[section]` is commented out so appending is safe;
   on a customised one, insert `root` above the first live `[table]` instead:
   ```bash
   grep -vE '^\s*#|^\s*$' /etc/containerd/config.toml   # expect: disabled_plugins, root
   ```
   Confirm new blobs actually land on the NVMe (this is what `data-root`
   alone silently fails to do):
   ```bash
   docker pull alpine:latest   # then: root should grow ~0, nvme/containerd ~13MB
   ```
3. **Export `TMPDIR` onto the NVMe too.** Image archives are up to 9.2GB
   each and staging them on root was observed to spike it to 95% full.

4. **Never hardcode the card count — detect it.** This runbook has been run
   on 2-card and 4-card boxes; the commands below derive `--num-shards` from
   what is actually present, and every box should do the same rather than
   copy-pasting a number. Sharding is over TASKS
   (`eligible[shard::num_shards]`) and seeds within a task are serial, so the
   useful shard count is capped by the task count of the stage:
   ```bash
   NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
   echo "this box has $NGPU cards"
   ```
   Launch one process per card with `--num-shards $SHARDS --shard $i` and
   `CUDA_VISIBLE_DEVICES=$i`, where `SHARDS = min(NGPU, n_tasks)`. Extra cards
   beyond `n_tasks` sit idle. A stage's wall-clock floor is one task's full
   seed sweep, no matter how many cards you add.

Run collections under **tmux**, not a foreground shell — these are multi-hour
jobs and a dropped connection kills an unprotected run. `scripts/launch_r2.sh`
does all of the above (env, detection, per-shard logs) for R2.

### R0 — config verification (5 minutes, do this FIRST)

The single most important check in this whole plan (decisions/021 §1):
generation used to inherit the model's shipped `generation_config`, and Qwen3
ships `top_k=20`, which caps interpretation diversity no matter the
temperature. The collector now prints and records the effective config:

```bash
python -c "import sys; sys.path.insert(0,'src'); from wta.hf_reader import HFStreamReader; r=HFStreamReader('Qwen/Qwen3-32B'); print(r.effective_generation_config())"
```

Record what `shipped` says. If `top_k` is 20 (or `top_p` < 1), that confirms
the clamp was real and the 32B fork scarcity has a mechanical explanation.

### R1 — diversity pilot — GO/NO-GO GATE (~1-1.5h wall on 2 cards; floor is one task's 12 runs)

6 known fork-bearing tasks x 12 seeds. **Pre-registered success criterion,
fixed before the run:** >= 3 of the 6 tasks show >= 2 interpretation classes
each committed by >= 2 distinct runs, AND median reads/run >= 25.

Note `python` below must be the NVMe venv's interpreter (the system python
has no torch), and whichever venv provides `hf` must precede it on PATH — see
preflight 1. Shard count is detected, not hardcoded — see preflight 4.

```bash
export PATH=/opt/dlami/nvme/wta-venv/bin:$PATH      # must provide `hf`
export HF_HOME=/opt/dlami/nvme/hf TMPDIR=/opt/dlami/nvme/tmp
PY=/opt/dlami/nvme/wta-venv/bin/python
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
SHARDS=$(( NGPU < 6 ? NGPU : 6 ))    # R1 has only 6 tasks; extra cards idle
echo "R1 on $SHARDS of $NGPU cards"
for i in $(seq 0 $((SHARDS-1))); do CUDA_VISIBLE_DEVICES=$i "$PY" scripts/collect_v2.py --model-id Qwen/Qwen3-32B --classes data/interpretation_classes_pilot.json --n-tasks 6 --n-runs 12 --shard $i --num-shards $SHARDS --out data/a0_pilot_32b --scratch-dir /opt/dlami/nvme/wta-scratch & done; wait
```

Sharding is over TASKS (`eligible[shard::num_shards]`), and seeds within a
task are serial — so R1 cannot use more than 6 cards, and its wall-clock
floor is one task's 12 runs no matter how many cards you add. The split is
`eligible[i::SHARDS]` over the sorted eligible list; on a 2-card box that is
shard 0 = swe_12/swe_36/swe_47, shard 1 = swe_30/swe_4/swe_50.

**Measured on this box (first 6 runs, 2026-08-08): ~55-141s per run** —
i.e. ~1.7 min, not the 6-10 min the R2 estimate below assumes. 15-40 steps
and 51-236 reads per run.

(`interpretation_classes_pilot.json` = the 6-task subset swe_36, swe_47,
swe_50, swe_4, swe_12, swe_30.) Then on the laptop:

```bash
python scripts/generate_labels.py --a0 data/a0_pilot_32b --out models/pilot_32b
```

**Status 2026-08-08: R1 was run on a DIFFERENT box and CLEARED the gate.**
Its `data/a0_pilot_32b/` artifacts therefore do not exist on this machine —
absence here is not evidence the gate was skipped. R2 on this box is
correctly gated. Re-run R1 locally only if you need the pilot artifacts
themselves.

- **Criterion met** -> the clamp explanation holds; run R2.
- **Diversity missed** -> forks are genuinely rare for this backbone. Do NOT
  scale a null: pivot to a larger/different backbone, the structural-fork
  slice, or the negative result.
- **Reads missed only** -> drop `--cadence` to 4 and re-pilot (cheap).

### R2 — main collection (60 tasks x 24 seeds, ~1440 runs)

Use the launcher, which does env + card detection + per-shard logging:

```bash
tmux new-session -d -s r2 'bash scripts/launch_r2.sh'
tmux attach -t r2      # logs also at /opt/dlami/nvme/logs/r2/shard*.log
```

Equivalent by hand:

```bash
export PATH=/opt/dlami/nvme/wta-venv/bin:$PATH      # must provide `hf`
export HF_HOME=/opt/dlami/nvme/hf TMPDIR=/opt/dlami/nvme/tmp
PY=/opt/dlami/nvme/wta-venv/bin/python
SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)   # 60 tasks: any card count shards cleanly
echo "R2 on $SHARDS cards"
for i in $(seq 0 $((SHARDS-1))); do CUDA_VISIBLE_DEVICES=$i "$PY" scripts/collect_v2.py --model-id Qwen/Qwen3-32B --classes data/interpretation_classes.json --n-tasks 60 --n-runs 24 --shard $i --num-shards $SHARDS --out data/a0_v3_32b --scratch-dir /opt/dlami/nvme/wta-scratch & done; wait
```

R2 has 60 tasks, so unlike R1 it shards cleanly across any card count you
have — detect it, do not copy a number. The old ~6-10 min/run figure was a
guess carried over from the sharded-4xA10G plan; R1 measured **~1.7 min/run**
on a single Blackwell card, so re-derive the budget from R1's own manifest
timings before quoting a number. Measured on the 4-card Blackwell box
(2026-08-08), first runs of R2: 17-190s per run. Budget ~80 min on top for
the one-time image fetches (~84s per uncached image, ~7GB archive each).
Interrupt-safe — re-running the same command resumes (existing `<run>.json`
files are skipped), including across a change in `--num-shards`.

Disk: each run writes ~12MB (npz dominates, 8 layers x reads), so R2's ~1440
runs need **~18GB** — keep `--out` on the NVMe, not the root disk.

### R3 — OOD + sealed test (after R2, or on spare cards)

Needs class artifacts derived first (`scripts/audit_class_artifact.py` must
exit 0 errors): ~20 `harbor_sql` tasks x 12 seeds gives gate 6 its first real
held-out family, and the sealed `swe_60+` pool x 12 seeds becomes the eval
split. Point `--tasks-dir third_party/hil-bench/harbor_sql` for the OOD half.

### Merging shards before analysis

```bash
cat data/a0_v3_32b/events.s*.jsonl > data/a0_v3_32b/events.jsonl
python -c "import json,glob;m=[json.load(open(f)) for f in sorted(glob.glob('data/a0_v3_32b/collection_manifest.s*.json'))];out=m[0];[out['tasks'].update(x['tasks']) for x in m[1:]];json.dump(out,open('data/a0_v3_32b/collection_manifest.json','w'),indent=1)"
```

## 4. A4 gates — STOP POINT

```bash
python scripts/run_gates.py --data data/a4_heldout.npz --a2 models/a2.pt
```

Prints the seven gate numbers unfiltered. **This is the science gate: the
owner reviews before ANY Part B result is trusted (decisions/011). Red gates
are findings, not bugs — nothing gets tuned to pass.**

## 5. Phase-4 eval — R5 (ONLY after step-4 sign-off; decisions/022)

Everything below is built and CPU-validated (contract tests +
`scripts/run_eval_smoke.py`). Ordered GPU procedure on the local Blackwell
box; specs: `specs/eval.md` + `specs/eval-bridge.md`. The sealed pool runs
ONCE (decisions/018 §4) — smoke every stage on a TRAIN task (swe_0) first.

```bash
# 5.0 preflight (repo + suite green on the box)
git pull && python -m pytest -q && python scripts/run_eval_smoke.py
pip install vllm   # first time only; uncomment vllm in requirements-gpu.txt

# 5.1 judge (frozen config, one GPU) — leave running for the whole eval
CUDA_VISIBLE_DEVICES=0 vllm serve casperhansen/llama-3.3-70b-instruct-awq \
    --port 8808 &
curl -s http://127.0.0.1:8808/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"casperhansen/llama-3.3-70b-instruct-awq","messages":[{"role":"user","content":"say ok"}]}'

# 5.2 JUDGE VALIDATION (brief §5a) — STOP on failure, paste report to owner
JUDGE_BASE_URL=http://127.0.0.1:8808 python scripts/validate_judge.py

# 5.3 backbone for bridge rows (second GPU; the DETECTOR arm never uses this)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-32B --port 8809 &
export AGENT_SWE_BASE_URL=http://127.0.0.1:8809/v1
# verify the litellm route: the -m value below, the key in
# configs/hilbench/config_mappings.yaml, and vLLM's served model name must agree.

# 5.4 materialize flat tasks (train smoke first, then sealed)
python scripts/materialize_hilbench_tasks.py --tasks swe_0 \
    --out data/hilbench_flat_smoke --extract-scripts --scratch-dir /opt/dlami/nvme/wta-scratch
python scripts/materialize_hilbench_tasks.py --tasks swe_60..swe_99 \
    --out data/hilbench_flat_sealed --extract-scripts --scratch-dir /opt/dlami/nvme/wta-scratch

# 5.5 BRIDGE ROWS in hil-bench's own harness (smoke on swe_0, then sealed).
# --max-steps 100 pins the SAME step budget as our-loop arms
# (configs/eval.yaml max_steps: 100) so the scaffold delta is not a
# step-budget delta; their SWE-Agent default is 200, disclosed.
cd third_party/hil-bench
python -m hil_bench.cli swe ../../data/hilbench_flat_smoke --all-modes \
    -m openai/Qwen/Qwen3-32B --passes 1 --num-workers 2 --max-steps 100 \
    --config-mapping ../../configs/hilbench/config_mappings.yaml \
    --judge-config ../../configs/hilbench/judge_config.yaml \
    --output-dir ../../results/bridge_smoke
# paste the smoke output back for review, THEN the sealed run:
python -m hil_bench.cli swe ../../data/hilbench_flat_sealed --all-modes \
    -m openai/Qwen/Qwen3-32B --passes 3 --num-workers 4 --max-steps 100 \
    --config-mapping ../../configs/hilbench/config_mappings.yaml \
    --judge-config ../../configs/hilbench/judge_config.yaml \
    --output-dir ../../results/bridge
cd ../..

# 5.6 OUR-LOOP ARMS (in-process HFStreamReader; detector thresholds ONLY
# from --artifacts). On the 2-card box this phase runs AFTER the bridge rows
# finish: stop the :8809 backbone vLLM first, keep the :8808 judge up, and
# run our loop on the freed card (single shard). With more cards, shard as
# in 2d. PRE-REQ: configs/eval.yaml n_runs set from the post-R1 decisions/ entry.
kill %2   # the :8809 vllm from 5.3; judge on :8808 stays
CUDA_VISIBLE_DEVICES=1 HF_HOME=/opt/dlami/nvme/hf JUDGE_BASE_URL=http://127.0.0.1:8808 \
  python scripts/run_eval.py --artifacts models/v3_32b_gates \
    --arms no_ask,full_info,model_initiated,detector,output_divergence,verbalized \
    --tasks swe_60..swe_99 --out data/eval --scratch-dir /opt/dlami/nvme/wta-scratch
# B4 random runs LAST, budget = detector's measured asks/task (decisions/022 §2i):
#   read it from data/eval/*/detector/*/ask_log.json, then
#   python scripts/run_eval.py --arms random --b4-budget <asks_per_task> ...

# 5.7 scoring + headline table (invokes their evaluator under docker)
python scripts/score_eval.py --our data/eval --bridge results/bridge \
    --flat-tasks data/hilbench_flat_sealed --resolve \
    --resolved-cache results/resolved_cache.json --out results/eval
# pre-req: data/fork_type_annotations.json committed + owner-signed BEFORE
# unsealing (decisions/022 §2g), else the fork slice reports unsplit.

# 5.8 scaffold-robustness (decisions/021 §8 item 4)
python scripts/scaffold_robustness.py --traj results/bridge --our data/eval \
    --out results/scaffold_robustness.json

# 5.9 tarball results + manifests back to the laptop
tar czf wta-eval-results.tar.gz results/ data/eval/*/eval_manifest*.json
```

## Cost notes (modest budget, decisions/004)

- Steps 1-2 are the only GPU-bound steps until eval; a g5.xlarge spot
  instance covers them. A0 at 20 tasks × N=8 × ~768 tokens ≈ a few GPU-hours.
- Everything else (training A2 on logged activations, calibration, gates) is
  CPU and can run on the laptop overnight at real-data scale.
