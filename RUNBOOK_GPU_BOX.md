# Runbook: GPU box work queue (2026-08-22)

Two independent jobs. **T3 needs no GPU** — run it first, it is the last
blocking item for the paper's section 4. **T7 is the GPU job** and can run
underneath it on all 4 cards.

Both are pre-registered: T3 in decisions/028 T3 + Amendment A item 6,
T7 in Amendment A item 9. Report numbers as-run.

```bash
git pull            # box tracks main; both branches have the commits
python -m pytest -q # expect 252 collected: 240 passed, 12 skipped
```

---

## Step 0 — box prep after ANY VM stop (the instance store is wiped)

`/opt/dlami/nvme` is ephemeral and is rebuilt empty on every stop/start. Three
things die with it and every one of them is load-bearing:

1. **The venv** `/opt/dlami/nvme/wta-venv`, which this runbook and
   `launch_xfam.sh` hard-code. Rebuild: `/usr/bin/python3.12 -m venv`, then
   `pip install numpy scipy scikit-learn pyyaml pytest`, the cached torch
   wheel in `/opt/dlami/nvme/wheels`, then `transformers accelerate
   huggingface_hub`. `hf buckets` must resolve or `launch_xfam.sh` aborts.
2. **The docker/containerd roots.** Both daemons come up "active" pointed at
   an empty volume, `docker images` lists the old tags at **0B**, and
   `docker load` fails with `metadata.db: no such file or directory`. Fix:
   `sudo systemctl stop docker docker.socket containerd`, `sudo mkdir -p
   /opt/dlami/nvme/docker /opt/dlami/nvme/containerd`, start **containerd
   first**, then docker.
3. **The task images.** Restore with
   `python scripts/restore_hilbench_images.py` (60 tasks, ~174 GB download /
   ~500 GB unpacked, ~40 min). The bucket is **public — no HF token needed**.

### Storage placement — do this BEFORE the full run

`launch_xfam.sh` writes to `data/xfam_<slug>`, and `data/` is on the **29 GB
root disk (~2 GB free)**. R2's 1,415 runs are 41 GB, so T7's 720 runs project
to **~21 GB** — the full collection WILL fill root and die partway through a
24h job. `data/*` is gitignored, so this cannot be fixed in the repo; it must
be done on the box, exactly as R2 did it:

```bash
mkdir -p /ssd2/wta_data/xfam_<slug> /ssd2/wta_data/xfam_<slug>_smoke
ln -sfn /ssd2/wta_data/xfam_<slug>       data/xfam_<slug>
ln -sfn /ssd2/wta_data/xfam_<slug>_smoke data/xfam_<slug>_smoke
```

Heavy irreproducible data (traces, activations) goes on the EBS volumes
(`/ssd`, `/ssd2`); only reproducible things (weights, venv, docker images)
belong on the ephemeral NVMe.

---

## Job 1 — T3 probe robustness (CPU, ~2-4h, do this first)

Closes the last gate-2 escape: can a full-dim or NONLINEAR probe beat the
causal+anchors-masked TEXT baseline of **0.730**?

```bash
python scripts/gate2_probe_robustness.py --labels models/v3_32b_fixed/labels.npz --out results/gate2_probe_robustness.json
```

Guards are built in: it aborts unless the npz is the fixed-label set
(787,281 reads) and the s6,s7 split comes out 9,180 test / 110 classes, and
it persists after each probe so a late crash never loses finished numbers.
The 256-d consistency check should print ≈ **0.2745**; if it is far off,
STOP and paste the log rather than interpreting the run.

Full detail: [RUNBOOK_T3_PROBE.md](RUNBOOK_T3_PROBE.md).

---

## Job 2 — T7 cross-family replication (GPU, 4 cards)

**Why this one.** Reviewer gap #4 is "one model family". T5 already covers
scale (7B/14B/32B) but every collection is Qwen. T7 changes ONLY the family:
same 60 tasks, same frozen class artifact, same labeler, same protocol. The
artifact is per-task, so this needs **no new registry work and no Fable
budget** — which is exactly why it fits the window while R6 waits.

### 2a. Smoke gate first (~20 min) — mandatory

```bash
scripts/launch_xfam.sh smoke
```

```bash
python scripts/xfam_smoke_gate.py --a0 data/xfam_mistralai-mistral-small-3.2-24b-instruct-2506_smoke --out results/xfam_smoke_primary.json
```

Pre-registered criteria: leakage-free, ≥1 mutating action in ≥50% of runs,
median reads/run ≥ 10. **PASS** → 2b. **FAIL** → run the fallback once:

```bash
MODEL=google/gemma-3-27b-it scripts/launch_xfam.sh smoke
```

FAIL on both → record the NO-GO in 028 item 9 and stop. Do **not** tune the
protocol to make a model comply; that destroys comparability with R2.

### 2b. Full collection, staged

```bash
scripts/launch_xfam.sh full 12
```

60 tasks × 12 seeds ≈ 720 runs, ~24h on 4 cards. Complete and balanced when
it finishes. Then, if the box stays free, extend to R2 parity — the
collector skips existing run JSONs, so this ADDS seeds 12–23:

```bash
scripts/launch_xfam.sh full 24
```

Then tar the output dir and send it back; labeling and the T1 rows are CPU
work on the laptop.

---

## Do NOT do these

- **Do not backfill the 25 missing temp-1.3 runs into `data/a0_v3_32b`.**
  Every current number is computed on that exact 1,415-run universe; adding
  runs silently invalidates all of them.
- **Do not collect the sealed pool (swe_60+).** `--classes` already prevents
  it. Un-sealing is an owner decision and a pre-registration change.
- **Do not collect harbor_sql yet.** OOD needs its own class artifacts
  derived first (LLM work); collecting traces without them produces data
  nothing can label.
