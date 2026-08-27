"""Contract: the labeler and the collector must tokenize identically
(decisions/028 Amendment F).

`token_idx` in a run log is recorded in the COLLECTION model's tokenizer
units. `wta.labeling` rebuilds a token->char map by re-tokenizing the trace
and indexes it with that `token_idx`. If the two paths load the tokenizer
differently, every action's character offset drifts and the commitment
anchoring is computed against the wrong text.

That is not hypothetical: the labeler used a bare
`AutoTokenizer.from_pretrained` while the collector passed
`fix_mistral_regex=True`, and on real Mistral transcripts the two disagreed
on 23 of 40 traces (0 of 40 on Qwen). These tests pin the paths together so
they cannot drift apart again silently.
"""

from __future__ import annotations

import inspect

import pytest


def test_labeler_uses_the_collector_tokenizer_loader():
    """Source-level pin: build_labels must not construct its own tokenizer."""
    from wta import labeling
    src = inspect.getsource(labeling.build_labels)
    assert "_load_tokenizer(" in src, (
        "labeling must load the tokenizer via hf_reader._load_tokenizer "
        "(028 Amendment F), not construct its own")
    assert "AutoTokenizer.from_pretrained(" not in src, (
        "a bare AutoTokenizer.from_pretrained in build_labels re-introduces "
        "the token->char drift Amendment F fixed")


def test_loader_passes_the_flag_and_tolerates_rejection(monkeypatch):
    """The shared loader must pass fix_mistral_regex and fall back cleanly."""
    import transformers

    from wta import hf_reader

    seen = []

    class _Fake:
        @staticmethod
        def from_pretrained(model_id, **kw):
            seen.append(kw)
            if kw.get("fix_mistral_regex") and model_id == "picky":
                raise TypeError("unexpected kwarg")
            return f"tok::{model_id}"

    monkeypatch.setattr(transformers, "AutoTokenizer", _Fake)
    assert hf_reader._load_tokenizer("normal") == "tok::normal"
    assert seen[0] == {"fix_mistral_regex": True}
    assert hf_reader._load_tokenizer("picky") == "tok::picky"
    assert seen[-1] == {}


@pytest.mark.parametrize("model_id,must_match", [
    ("Qwen/Qwen3-8B", True),        # R2 family: flag is a verified no-op
])
def test_flag_is_a_noop_for_the_r2_family(model_id, must_match):
    """Amendment F's blast-radius claim: fixing the labeler cannot move any
    existing Qwen-collected number, because the flag changes nothing there."""
    transformers = pytest.importorskip("transformers")
    try:
        a = transformers.AutoTokenizer.from_pretrained(model_id)
        b = transformers.AutoTokenizer.from_pretrained(
            model_id, fix_mistral_regex=True)
    except Exception as exc:
        pytest.skip(f"cannot reach {model_id}: {type(exc).__name__}")
    sample = (
        "THOUGHT:\nI will inspect the config loader.\n\n```bash\n"
        "grep -rn 'retries' src/ | head -20\n```\n"
        "[exit 0]\nsrc/cfg.py:12:    retries = 3\n\nNext step?"
    )
    assert a(sample)["input_ids"] == b(sample)["input_ids"]
