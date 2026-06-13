"""Agent -- OURS (single-trajectory step loop adapted from mini-swe-agent).

  * loop.py            -- one trajectory: observe -> think -> act, step by step.
  * decision_points.py -- flags decision points (tool-call boundaries / under-determined choices).
  * multi.py           -- N-trajectory runner (temperature / diverse prompting) with a
                          controller seam above the N (used later for ask-triggering).
"""
