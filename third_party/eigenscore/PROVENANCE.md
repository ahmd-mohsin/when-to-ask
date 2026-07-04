# Provenance — EigenScore (INSIDE)

- **Repo:** https://github.com/D2I-ai/eigenscore
- **Commit:** ea8062a (cloned 2026-06-12, xtid pass)
- **Licence:** MIT (© 2024 Alibaba)
- **What we migrated (xtid pass):** EigenScore
  (`func/metric.py::getEigenScore` = `mean(log10(svd(cov(Z)+αI)))`) →
  `src/xtid/signals/internal_divergence.py`. Available to wta as a
  cross-trajectory internal-divergence comparison signal if needed in eval;
  not part of the core When-to-Ask method.
