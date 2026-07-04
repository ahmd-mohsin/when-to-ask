# Provenance — ClarifyGPT

- **Repo:** https://github.com/ClarifyGPT/ClarifyGPT
- **Commit:** 543b34b (cloned 2026-06-12, xtid pass)
- **Licence:** none in repo (all rights reserved by default). Owner reports
  personal permission from authors for research use (2026-07-02, decisions/003).
  Reference-use only; not redistributed.
- **What we migrate:** the test-output consistency check
  (`src/clarify/run_clarify_chatgpt_mbpp.py::runTests_getTaskID`) — the
  output-divergence baseline. Migrated in the xtid pass into
  `src/xtid/signals/output_divergence.py`; the wta eval (Phase 4) reuses that
  port as the matched-N output-divergence baseline.
- **Adaptations:** ported to our task/execution interfaces; consistency logic
  unchanged.
