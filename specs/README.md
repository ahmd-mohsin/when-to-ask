# specs/

One file per component of the When-to-Ask detector (canonical method doc:
`when-to-ask-offline-online (1).md`; process: `fable-build-prompt.md`).

Each spec is short, precise, and testable: inputs, outputs, shapes/dtypes,
invariants, and the observable behaviour that would tell us it's correct.

Planned files (created per phase, not all up front):

| File | Component |
|---|---|
| `A0-data-collection.md` | N-run trajectory logging: residuals `h` across the reasoning span + observables |
| `A1-ambiguity-direction.md` | Difference-in-means direction `d`; scalar signal `s = dot(h, d)` |
| `A2-disentangling-autoencoder.md` | Topic `T(h)` + resolution lean `L(h)`; four-loss training |
| `A3-commitment.md` | Commitment definition (`r` steady + `s` dropped); split-conformal `tau` |
| `A4-gates.md` | The seven held-out research validation gates (hypotheses, not tests to "pass") |
| `B-online-trigger.md` | Bucketing (leader + merge + hysteresis), mutable voting, dispersion + CUSUM, loop channel, question assembly — **not written until A4 gates pass** |
| `eval.md` | Ask-F1 + Pass@k on HiL-Bench, regime slices, lead-time, matched-compute baselines |

Spec changes forced by reality are edited here explicitly and flagged to the
project owner — never silently absorbed into code.
