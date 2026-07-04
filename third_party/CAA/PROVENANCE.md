# Provenance — CAA

- **Paper:** Steering Llama 2 via Contrastive Activation Addition (Panickssery/Rimsky et al.), arXiv 2312.06681 (code link in paper Appendix A)
- **Repo:** https://github.com/nrimsky/CAA
- **Commit:** 5dabbbd9a0bca5f25e174501e959de378806aa48 (cloned 2026-07-02, depth 1)
- **Licence:** MIT (LICENSE file, © 2024 Nina Rimsky)
- **What we migrate:** the difference-in-means steering-vector construction
  (`generate_vectors.py`: `vec = (all_pos_layer - all_neg_layer).mean(dim=0)`)
  → A1 ambiguity direction `d` in `src/wta/a1_direction.py`.
- **Adaptations:** numpy port behind our spec interface (their code is
  torch + Llama-specific prompt plumbing we don't need); mean-of-class
  formulation `mean(pos) - mean(neg)` (equivalent for balanced pairs; we use
  unpaired class means per Geometry-of-Truth's MMProbe). No mechanism change.
