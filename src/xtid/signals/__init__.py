"""Signals -- the divergence signals compared in the de-risk experiment.

  * alignment.py            -- OURS (novel): asynchronous cross-trajectory hidden-state alignment.
  * internal_divergence.py  -- OURS: cross-trajectory dispersion of aligned hidden states
                               (mean-pairwise-cosine [primary], total-variance, EigenScore[migrated],
                               Stiefel-volume[migrated]).  <-- the candidate winning signal.
  * output_divergence.py    -- B1, MIGRATED from ClarifyGPT: test-output consistency across N.
  * verbalized.py           -- B2: aggregate of N verbalized self-reports.
  * probe.py                -- B3, MIGRATED from OPENIA: single-stream correctness probe.
"""
