# Provenance — LYNX

- **Paper:** LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning
  (Akgül et al.), arXiv 2512.05325 (code link in paper abstract, HTML version)
- **Repo:** https://github.com/farukakgul/LYNX
- **Commit:** e62c6c2f1b9b07bceb86d44cc6c58a416ffb116f (cloned 2026-07-02, depth 1)
- **Licence:** none in repo (all rights reserved by default). Owner reports
  personal permission from authors for research use (2026-07-02, decisions/003).
  Reference-use only; not redistributed (third_party/ is git-ignored).
- **What we migrate (Phase 2):** the split-conformal threshold-calibration
  procedure (calibration-set nonconformity scores → quantile threshold with
  coverage guarantee) → A3 commitment threshold `tau` in `src/wta/a3_commitment.py`.
  We borrow the calibration procedure, NOT the correctness target: our
  calibration label is the stabilization point (observable from the trajectory),
  since answer-correctness is uncomputable at forks by construction.
  Their cue-token reading ("hmm", "wait") is also the precedent for our cue set
  (decisions/006).
- **Adaptations:** calibration label and nonconformity score are ours (spec A3);
  the quantile/coverage machinery is theirs. Flagged: this is the boundary the
  method doc itself draws ("borrow LYNX's calibration procedure; do not borrow
  the correctness target").
