"""Analysis -- OURS. The C1 test.

  * regimes.py     -- label decision points: fork / confident-convergent / clear.
  * separation.py  -- per-signal AUROC for should-ask-vs-proceed, sliced by regime.
  * lead_time.py   -- internal-vs-output threshold-crossing lead-time on the fork slice.

Together these produce the go/no-go table described in brief S7.
"""
