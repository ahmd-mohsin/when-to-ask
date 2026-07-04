# Provenance — HiL-Bench

- **Repo:** https://github.com/hilbenchauthors/hil-bench
- **Commit:** 352d14c (cloned 2026-06-12, xtid pass)
- **Licence:** none in repo (all rights reserved by default). Owner reports
  personal permission from authors for research use (2026-07-02, decisions/003).
  Reference-use only; not redistributed.
- **What we use (verbatim per the build brief — reimplementing the benchmark
  would break comparability):** harness (`run_hil_bench.py`), frozen judge
  (`hil_bench/ask_human_server.py`, Llama-3.3-70B), Ask-F1
  (`hil_bench/utils/compute_hil_metrics.py`), task + blocker data
  (`harbor_swe/`, `harbor_sql/` — 100 + 100 `blocker_registry.json`).
- **Adaptations (xtid pass):** judge/Ask-F1/executor wrapped into
  `src/xtid/harness/*`; reused by wta per decisions/001. The blocker
  registries additionally become the source for the derived
  interpretation-class artifact (decisions/005) — a new use, no upstream edit.
