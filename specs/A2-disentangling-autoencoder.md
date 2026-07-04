# Spec A2 — Disentangling autoencoder: topic `T(h)` and resolution lean `L(h)`

The one and only autoencoder; trained offline; at runtime only the frozen
`T` and `L` forward passes survive. Architecture follows **ReDAct**
(arXiv 2602.19396 — no public code, implemented from the paper per
decisions/002); training machinery pieces migrated from **REAL**
(`third_party/REAL_ICLR`, MIT): `build_mlp`, `supervised_contrastive_loss`
(the InfoNCE-style term), `set_global_determinism`.

## Architecture

- **Encoder**: shared 2-layer MLP body (H → hidden → hidden, LeakyReLU) with
  two linear heads: `T(h) ∈ R^{d_T}` (topic) and `L(h) ∈ R^{d_L}` (lean).
  Defaults d_T = 16, d_L = 8, hidden = 128 (sweepable).
- **Decoder**: MLP `(T ⊕ L) → ĥ` (d_T+d_L → hidden → H).
- **Auxiliary (training only, discarded at runtime):** linear topic classifier
  on `T`; linear lean classifier on `L`; adversary MLP on `T` predicting the
  lean class through a **gradient-reversal layer**.

## The four pulls (total objective)

| # | loss | on | meaning |
|---|---|---|---|
| 1 | CE(topic-classifier(T), decision-identity) + λ_supcon · SupCon(T, decision-identity) | all labeled reads | topic names the decision |
| 2 | CE(lean-classifier(L), interpretation-class) | class-labeled reads (committed reads of ambiguous decisions — a pre-settle read's lean is a deliberation mixture, so labeling it with the eventual class is wrong supervision) | lean names the interpretation |
| 3 | CE(adversary(GRL(T)), interpretation-class) **+** λ_ortho · ‖corr(T, L)‖²_F | class-labeled reads / batch | topic blind to lean (load-bearing) |
| 4 | MSE(decoder(T, L), h) | all reads | (T, L) retain the real information |
| 5 | λ_condmean · mean over (decision, class) of ‖mean(T \| decision, class) − mean(T \| decision)‖² | class-labeled reads | **OURS — conditional-mean alignment** (amended 2026-07-03): measured on fixtures, GRL alone loses the arms race — the encoder perturbs T against the adversary's *current* weights while the class stays linearly readable (fresh-probe acc 1.00 even at w_adv=8, with adversary catch-up steps). Collapsing within-decision class-conditional means removes exactly what a linear probe reads and *is* fork-collocation expressed as a loss (probe: 1.00 → 0.37 ≈ chance). GRL is retained for nonlinear residue. Flagged as an addition to ReDAct's recipe. |

Notes:
- The adversary is trained to *succeed* while GRL makes the encoder make it
  *fail* — standard gradient reversal (λ_grl ramps 0→1 over training).
- Orthogonality between vectors of different dims is implemented as batch
  **decorrelation**: after centering, `C = T_cᵀ L_c / B`, penalty `mean(C²)`.
- Lean supervision is the interpretation **class** (categorical), never a
  signed scalar, never raw action strings (method doc, "resolved" note).
- No eval-split data ever enters training (brief rule 1).

## Interface

```
cfg = A2Config(in_dim, n_topics, n_classes, d_topic=16, d_lean=8, hidden=128,
               epochs, lr, seed, loss weights…)
model = train_a2(h_train, topic_labels, class_labels, cfg)  # class label -1 = unlabeled
model.encode_topic(h: float[..., H]) -> float32[..., d_T]   # numpy in/out
model.encode_lean(h: float[..., H])  -> float32[..., d_L]
model.reconstruct(h) -> float32[..., H]                      # offline diagnostics only
save/load round-trip (state dict + config)
```

Deterministic given seed (REAL's determinism helper, CPU-adapted).

## Observable behaviour that verifies this spec (contract, on fixtures)

1. **Wiring:** at init, all four loss terms are finite and non-zero, and a
   training step produces non-zero gradients in body, both heads, decoder,
   and adversary.
2. **Learning:** after training on fixture reads (held-out split): linear
   probe on `T` recovers decision-identity ≥ 0.9 acc; linear probe on `L`
   recovers interpretation class ≥ 0.9 acc (ambiguous reads).
3. **Invariance (fixture-level):** probe on `T` predicting interpretation
   class ≤ chance + 0.15 (the planted structure is orthogonal, so this is an
   engineering check here; the real-data version is A4 gate 1 and is NOT
   tuned).
4. **Reconstruction floor:** held-out R² ≥ 0.5.
5. Determinism: same seed → identical encodings; save/load round-trip exact.
