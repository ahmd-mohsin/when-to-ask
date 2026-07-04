# Provenance — Geometry of Truth

- **Paper:** The Geometry of Truth: Emergent Linear Structure in LLM Representations
  of True/False Datasets (Marks & Tegmark), arXiv 2310.06824, COLM 2024
  (repo link in paper Introduction)
- **Repo:** https://github.com/saprmarks/geometry-of-truth
- **Commit:** 5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb (cloned 2026-07-02, depth 1)
- **Licence:** none in repo (all rights reserved by default). Owner reports
  personal permission from authors for research use (2026-07-02, decisions/003).
  Reference-use only; not redistributed (third_party/ is git-ignored).
- **What we migrate:** the mass-mean probe direction
  (`probes.py::MMProbe`: `direction = pos_acts.mean(0) - neg_acts.mean(0)`)
  → A1 ambiguity direction `d` in `src/wta/a1_direction.py` (co-source with CAA).
- **Adaptations:** numpy port; we keep only the direction construction, not
  the IID/covariance variant (their `MMProbe` covariance option is noted as a
  Phase-2 alternative). No mechanism change.
