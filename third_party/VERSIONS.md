# Vendored upstream repos — pinned versions

Cloned into `third_party/` for reference and algorithm migration. **We do not edit
these**; we port their core algorithm into `src/xtid/...` with an in-file citation.
This file is the only tracked thing under `third_party/` (the rest is git-ignored).

Cloned on 2026-06-12.

| Repo | Remote | Commit | What we migrate | Into |
| --- | --- | --- | --- | --- |
| hil-bench | https://github.com/hilbenchauthors/hil-bench | `352d14c` | Llama-3.3-70B semantic judge (`hil_bench/ask_human_server.py`), Ask-F1 (`hil_bench/utils/compute_hil_metrics.py`), task/blocker data (`harbor_swe/`, `harbor_sql/`), executor | `xtid/harness/*` |
| ClarifyGPT | https://github.com/ClarifyGPT/ClarifyGPT | `543b34b` | test-output consistency check (`src/clarify/run_clarify_chatgpt_mbpp.py::runTests_getTaskID`) → B1 | `xtid/signals/output_divergence.py` |
| eigenscore | https://github.com/D2I-ai/eigenscore | `ea8062a` | EigenScore (`func/metric.py::getEigenScore`) = `mean(log10(svd(cov(Z)+αI)))` | `xtid/signals/internal_divergence.py` |
| OPENIA | https://github.com/iSE-UET-VNU/OPENIA | `aa96070` | linear correctness probe on internal states → B3 | `xtid/signals/probe.py` |
| mini-swe-agent | https://github.com/SWE-agent/mini-swe-agent | `531dbaf` | minimal observe→query→act agent loop (`agents/default.py`) | `xtid/agent/loop.py` |
| STARS | https://github.com/lythk88/STARS | — (empty/unavailable) | Stiefel-manifold geometric volume | reimplemented from arXiv 2601.22010 in `xtid/signals/internal_divergence.py` |

## Notes
- **STARS** cloned with no files on the default branch (empty/unavailable as of 2026-06-12).
  The geometric-volume divergence metric is reimplemented from the paper (closed-form
  log-volume of the centered activation matrix). The leftover `third_party/STARS/` dir
  (`.git` only) is harmless and git-ignored.
- **Confidence Manifold** (arXiv 2602.08159) has no public repo; its probe geometry
  (3–8 discriminative dims, centroid-distance ≈ trained-probe AUC) informs B3 but is not
  cloned.
- HiL-Bench judge model default: `casperhansen/llama-3.3-70b-instruct-awq`, temperature 0.05.
