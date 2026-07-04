# Spec A4 — The seven research validation gates

**These are hypotheses, not tests to make pass.** On real data they run on
held-out splits, produce numbers, and STOP for owner review (build brief;
decisions/011 made this a usage gate for Part B results, unchanged in
substance). The fixture harness only verifies the gate *machinery* — on
planted-structure data the pipeline should pass, which checks the gate code,
not the science.

Each gate is a pure function `(frozen encoders, labeled held-out data[, pairing]) →
GateResult(name, numbers, direction, note)`; `scripts/run_gates.py` assembles
the report. No gate function may mutate or retrain anything.

| # | gate | measures | reported numbers |
|---|---|---|---|
| 1 | topic-leakage | predict interpretation class from `T` alone | held-out probe accuracy vs chance; ReDAct-style eta² (ANOVA of T by class) |
| 2 | decision-recovery | `T` predicts decision-identity | held-out probe accuracy vs chance |
| 3 | fork-collocation | same-decision/opposite-resolution topic cosine high | same-decision vs different-decision cosine distributions (means, overlap), crossover → **theta** for Part B |
| 4 | conflation | same-observable(file)/different-decision pairs must NOT collocate | their cosine distribution vs the same-decision distribution (pairing supplied from observables) |
| 5 | lean-separation | within a decision, different classes separate in `L` | between-class / within-class distance ratio; silhouette |
| 6 | OOD transfer | bucket purity on unseen task families | train probe on subset of decisions; purity of leader-clusters on held-out decisions |
| 7 | lead-time | bucket disagreement rises K > 0 reads before actions diverge | per-fork K distribution (first read where within-decision L-dispersion of committed-weighted runs exceeds its pre-fork baseline vs the action-divergence read); median K |

Notes:
- Gate 7's "action-divergence read" comes from logged actions on real data;
  the fixture harness uses the planted settle points as the action proxy
  (fixtures emit no actions) — flagged in the report as `proxy=True`.
- eta² (gate 1): fraction of per-dimension T variance explained by class,
  averaged over T dims — matches ReDAct's ANOVA usage.
- All probes are linear (logistic regression on frozen embeddings), fit on a
  train split of the held-out data, scored on the rest — encoders NEVER see
  gate data during A2 training.

## Contract (machinery) checks on fixtures

With planted orthogonal structure, after A2 training: gate 1 accuracy ≈
chance and eta² small; gate 2 ≥ 0.9; gate 3 same-decision cosine ≫
different-decision, a finite theta exists; gate 5 ratio > 2; gate 7 median
K > 0 (internal signal leads the planted settle point by construction of the
pre-settle lean drift). Gate 4 machinery: given an adversarial pairing that
labels different decisions "same file", the gate reports non-collocation on
fixtures. Gate 6 (amended 2026-07-03): purity ≥ 0.8 on held-out reads of
**seen** decisions verifies the machinery; on **unseen** fixture decisions
the transfer failure is genuine by construction (an MLP has no inductive
bridge between orthogonal planted directions), so the unseen number is
reported, not asserted — exactly the "honest limitation" role gate 6 has on
real data.
