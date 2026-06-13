# xtid — Cross-Trajectory Internal Divergence

A de-risk experiment for the research question in `pre-implementation brief.md`:

> In a long-horizon coding agent run as **N parallel trajectories**, are the best moments to
> call `ask_human()` the **branch points where the trajectories' *internal states* diverge**
> (read from mid-layer hidden states), rather than where their *outputs* diverge?

**This repo builds exactly the de-risk path (brief §7)** — enough to answer the make-or-break
claim **C1** before committing to the full method:

> **C1.** At matched N, on the **fork-blocker slice**, does *internal* cross-trajectory divergence
> separate "should-have-asked" from "fine-to-proceed" **better than output divergence (B1,
> ClarifyGPT-style)**, and does it fire **earlier** (positive lead-time)?

Deferred to a later pass (interfaces are stubbed so they slot in): the full trigger/dedup
controller, fork-based question generation, resolution injection, gate-vs-ask triage, activation
steering, baselines B4–B6, and the full metrics suite.

## What is migrated vs. built

**Migrated** (cloned into `third_party/`, core algorithm ported into our adapters — see
`third_party/VERSIONS.md`):

| From | Algorithm | Into |
| --- | --- | --- |
| HiL-Bench | Llama-3.3-70B semantic judge, Ask-F1, task/blocker data, executor | `xtid/harness/` |
| ClarifyGPT | test-output consistency check (B1) | `xtid/signals/output_divergence.py` |
| INSIDE/EigenScore | covariance-eigenvalue dispersion | `xtid/signals/internal_divergence.py` |
| OPENIA | linear correctness probe on internal states (B3) | `xtid/signals/probe.py` |
| STARS | Stiefel-manifold geometric volume (reimplemented; repo empty) | `xtid/signals/internal_divergence.py` |
| mini-swe-agent | minimal observe→query→act loop | `xtid/agent/loop.py` |

**Ours, from scratch** (the contribution):

- `backbone/model.py` — white-box wrapper exposing **mid-layer hidden states mid-trajectory**.
- `agent/multi.py`, `agent/decision_points.py` — **N-trajectory runner** + decision-point detection.
- `signals/alignment.py` — **asynchronous cross-trajectory hidden-state alignment** (novel).
- `signals/internal_divergence.py` — the **cross-trajectory** reading of dispersion (the candidate signal).
- `recording/recorder.py` — per-decision-point, per-trajectory logging.
- `analysis/` — regime labels, AUROC separation, lead-time → **the C1 table**.

## Layout

```
configs/      smoke.yaml (CPU, no deps) · derisk.yaml (GPU)
scripts/      clone_third_party · provision_gpu.sh · run_derisk.py
src/xtid/     harness · backbone · agent · signals · recording · analysis
third_party/  vendored upstream repos (git-ignored; VERSIONS.md tracked)
tests/        CPU smoke + unit tests
```

## Quickstart (CPU, this machine — no torch, no downloads)

The smoke path uses a numpy `FakeWhiteBoxModel` + rule-based `MockJudge` + synthetic tasks, so the
**full pipeline runs anywhere** with only the base deps.

```bash
pip install -e ".[dev]"          # numpy, scikit-learn, pyyaml, pytest
pytest                            # unit + smoke tests
python scripts/run_derisk.py --config configs/smoke.yaml   # end-to-end, prints a (toy) C1 table
```

## Real run (rented GPU — RunPod / Lambda)

The white-box backbone (hidden states) and the Llama-3.3-70B judge cannot run on the dev machine.
On a GPU box:

```bash
bash scripts/provision_gpu.sh                 # installs requirements-gpu.txt, downloads the backbone
export JUDGE_BASE_URL=...  JUDGE_API_KEY=...   # hosted Llama-3.3-70B judge (Together/Fireworks/DeepInfra)
python scripts/run_derisk.py --config configs/derisk.yaml
```

- **Backbone:** default `Qwen/Qwen2.5-Coder-7B-Instruct` (dense → hidden states trivial). Scale up later.
- **Judge:** recommended via an OpenAI-compatible API (avoids self-hosting ~140 GB). Self-host option
  documented in `scripts/provision_gpu.sh`.
- **Recommended instance:** 1× H100 80GB or 1× A100 80GB; judge via API.

The run emits the **C1 table**: per-regime (fork / confident-convergent / clear) AUROC for *internal*
divergence vs. *output* divergence (B1) vs. the single-stream probe (B3), plus internal-vs-output
lead-time on the fork slice. That table is the go / no-go decision.
