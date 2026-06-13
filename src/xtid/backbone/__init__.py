"""Backbone -- OURS.

White-box model wrapper that exposes mid-layer residual hidden states
*mid-trajectory inside the agent loop* (the go/no-go item in brief S9).

  * HFWhiteBoxModel  -- real Hugging Face causal LM (torch); GPU runs.
  * FakeWhiteBoxModel -- numpy stand-in for CPU smoke tests (no torch, no downloads).

Both implement the same WhiteBoxModel interface so the rest of the pipeline is
device- and dependency-agnostic.
"""
