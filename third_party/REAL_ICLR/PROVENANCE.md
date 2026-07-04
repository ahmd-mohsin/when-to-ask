# Provenance — REAL (formerly DEAL)

- **Paper:** REAL: Reading Out Transformer Activations for Precise Localization
  in Language Model Steering (Zhan et al.), ICLR 2026 poster, arXiv 2506.08359.
  NOTE: v1 (June 2025) was titled "DEAL: Disentangling Transformer Head
  Activations for LLM Steering" — cite arXiv 2506.08359v1 when the DEAL name is
  used. The method doc's "DEAL (secondary)" refers to this paper.
- **Repo:** https://github.com/liam0949/REAL_ICLR (located via GitHub search;
  README self-identifies as the official ICLR 2026 code; no code link on arXiv)
- **Commit:** c93c04f9aa651b988040c586c6a9541178c5dbc4 (cloned 2026-07-02, depth 1)
- **Licence:** MIT (dual copyright: REAL authors 2026 + Kenneth Li 2023 — the
  repo builds on likenneth/honest_llama, MIT)
- **Role (Phase 2, per decisions/002):** ReDAct (arXiv 2602.19396) has no public
  code, so REAL is the vendored disentangling-machinery reference for A2:
  activation dataset/loader patterns, autoencoder training-loop scaffolding,
  probe evaluation utilities — whatever transfers. The A2 architecture itself
  (shared body → T/L heads, gradient reversal, orthogonality, reconstruction)
  is implemented from the ReDAct paper and cited as "following ReDAct".
- **Adaptations:** to be recorded here when A2 is built (Phase 2). Their
  mechanism is a per-attention-head VQ-AE; we do NOT claim to use it — only the
  surrounding machinery is migrated.
