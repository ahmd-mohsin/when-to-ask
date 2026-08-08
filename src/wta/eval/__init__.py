"""Phase 4 evaluation package (spec eval + eval-bridge, decisions/022).

Everything in here is measurement plumbing: loading frozen offline artifacts,
driving live N-run trajectories with ask/answer injection, assembling
questions, and scoring. Nothing trains; nothing tunes thresholds on eval
tasks (frozen-thresholds rule, specs/eval.md).
"""
