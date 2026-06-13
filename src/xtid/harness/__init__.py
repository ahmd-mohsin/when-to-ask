"""Harness -- MIGRATED from HiL-Bench (arXiv 2604.09408, Scale Labs).

Reuses, per the brief's hard constraint (S5a):
  * the frozen Llama-3.3-70B semantic judge (verbatim matching rubric),
  * the Ask-F1 metric,
  * the SWE/SQL execution layer,
  * the 200 public tasks + blocker registry.

Upstream: third_party/hil-bench  (see third_party/VERSIONS.md for the pinned commit).
"""
