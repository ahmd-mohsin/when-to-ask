# Provenance — Representation Engineering (RepE)

- **Paper:** Representation Engineering: A Top-Down Approach to AI Transparency
  (Zou et al.), arXiv 2310.01405 (code link in arXiv Comments)
- **Repo:** https://github.com/andyzoujm/representation-engineering
- **Commit:** 5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e (cloned 2026-07-02, depth 1)
- **Licence:** MIT
- **What we use:** activation read/hook conventions (`repe/rep_reading_pipeline.py`,
  `rep_readers.py`) as the reference for reading hidden states at chosen token
  positions → informs `src/wta/hf_reader.py` (per-read mid-layer capture during
  generation). The xtid pass already established per-token capture via
  `output_hidden_states`; RepE is the cited pattern for position-selected reads.
- **Adaptations:** we read during incremental generation (their pipelines read
  on full forward passes over given text); position selection (cadence + cue)
  is ours (decisions/006). Flagged as plumbing, not a mechanism change.
