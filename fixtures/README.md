# fixtures/

Synthetic-fixture generators with **known** topic/lean/commitment structure,
built before any real model activations are involved. They exist so the whole
pipeline (autoencoder, clustering, voting, CUSUM) can be validated end-to-end
on data where the right answer is known — de-risking the plumbing separately
from the science.

Required structure knobs per generator: number of decisions (topics), number
of interpretation classes per decision, per-run commitment trajectories
(deliberate → settle), transient-blip vs persistent-fork spread patterns, and
loop-state runs that never commit.
