# harness/

Tests and checks written **before** each component is implemented, split into
two clearly separated kinds (do not mix them):

## `contract/` — engineering tests (make these pass)
Ordinary software correctness on synthetic fixtures with known structure:
shapes/dtypes, all four autoencoder losses wired and non-zero, reconstruction
floor, leader-clustering assign/merge/hysteresis behaviour, vote
write/retract semantics, CUSUM fires on persistent synthetic disagreement and
not on transient blips, loop-channel contribution.

## `gates/` — research validation gates, A4 (do NOT "make these pass")
Hypotheses about the learned representation, run on **held-out** data only.
Each gate reports its number and stops for owner review:

1. Topic-leakage (predict `r` from `T` → ~chance; ReDAct-style eta^2)
2. Decision-recovery (`T` predicts decision-identity well)
3. Fork-collocation (+ same-vs-different cosine distributions → sets `theta`)
4. Conflation (same-file/different-decision pairs must NOT collocate)
5. Lean-separation (within-bucket `L` vectors well separated)
6. OOD transfer (bucket purity on unseen task families — reported as limitation)
7. Lead-time (bucket disagreement rises K > 0 steps before actions diverge)

Never close a gate gap by training on the eval split or hand-selecting
examples. A red gate is a finding, not a bug.
