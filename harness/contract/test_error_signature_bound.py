"""Contract: error_signature is computed from the TRUNCATED observation
(decisions/028 Amendment D).

The collector used to fingerprint the full, untruncated command output. A
malformed `grep -r -` produced 723 MB inside one exec window and wedged a
collection shard for over an hour at 97% CPU with its GPU idle. Worse than
slow, it described text the agent never saw: the model conditions on the
truncated observation.

These tests pin both halves -- the ordering in run_agent, and that the cost is
now bounded by obs_head + obs_tail rather than by the command's output size.
"""

from __future__ import annotations

import inspect
import time

from wta.agent_loop import error_signature, run_agent, truncate_obs


def test_signature_is_computed_after_truncation():
    """Source-level pin: truncate_obs must precede the error_signature call.
    If someone reorders these, the 723 MB hang comes back silently."""
    src = inspect.getsource(run_agent)
    i_trunc = src.index("obs = truncate_obs(")
    i_sig = src.index('event.observables["error_signature"]')
    assert i_trunc < i_sig, (
        "error_signature must be computed from the truncated observation "
        "(028 Amendment D), not the raw output")
    # and it must be fed obs, not out
    line = [ln for ln in src.splitlines()
            if 'event.observables["error_signature"]' in ln][0]
    assert "error_signature(code, obs)" in line, line


def test_cost_is_bounded_by_the_truncation_window():
    """The real regression test: a huge output must not cost more than a small
    one, because only the truncated window is ever scanned."""
    head, tail = 1500, 500
    # content shaped like the real failure: recursive grep hits across files
    unit = "src/mod/file.py:12:  value = compute(x) - offset  # Error: nope\n"
    small = unit * 20
    huge = unit * 400_000            # ~26 MB, same shape as the 723 MB case

    t0 = time.perf_counter()
    sig_small = error_signature(0, truncate_obs(small, head, tail))
    t_small = time.perf_counter() - t0

    t0 = time.perf_counter()
    sig_huge = error_signature(0, truncate_obs(huge, head, tail))
    t_huge = time.perf_counter() - t0

    assert sig_small.startswith("exit 0")
    assert sig_huge.startswith("exit 0")
    # generous bound: truncation makes input size irrelevant, so the two calls
    # should be within an order of magnitude even on a loaded box
    assert t_huge < max(t_small * 50, 0.5), (
        f"huge={t_huge:.4f}s vs small={t_small:.4f}s -- cost still scales "
        "with output size, so the Amendment D bound is not in effect")


def test_truncation_keeps_head_and_tail_so_typed_errors_survive():
    """Amendment D's stated expectation: typed exceptions land at the END of
    output, and obs_tail retains them, so the usual signature is preserved."""
    noise = "chatter line that matches nothing\n" * 5000
    out = noise + "ValueError: bad config key 'retries'\n"
    obs = truncate_obs(out, 1500, 500)
    assert "ValueError: bad config key" in obs
    sig = error_signature(1, obs)
    assert "ValueError: bad config key 'retries'" in sig


def test_signature_unchanged_when_output_fits_in_the_window():
    """Below the truncation threshold the fix is a no-op, so every ordinary
    short action keeps byte-identical signatures."""
    out = "Traceback (most recent call last):\n  ...\nKeyError: 'name'\n"
    assert len(out) < 2000
    assert error_signature(2, out) == error_signature(
        2, truncate_obs(out, 1500, 500))
