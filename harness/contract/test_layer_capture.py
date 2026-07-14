"""Contract: hook capture == the old output_hidden_states path, exactly.

Runs a TINY Qwen2 built from config (no download) twice with greedy decoding:
once with output_hidden_states=True (ground truth), once under LayerCapture.
Every captured vector must match hidden_states[idx+1][0, -1] at every step --
the equivalence the 32B memory fix rests on (decisions/017 addendum)."""

import numpy as np
import pytest
import torch

from wta.layer_capture import LayerCapture, _decoder_layers


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=64, vocab_size=99,
                      max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg).eval()
    input_ids = torch.randint(0, 99, (1, 7))
    return model, input_ids


def _generate(model, input_ids, **kw):
    with torch.no_grad():
        return model.generate(input_ids, max_new_tokens=5, do_sample=False,
                              pad_token_id=0, return_dict_in_generate=True, **kw)


def test_capture_matches_output_hidden_states(tiny):
    model, input_ids = tiny
    truth = _generate(model, input_ids, output_hidden_states=True)

    layers = [1, 3]
    with LayerCapture(model, layers) as cap:
        run2 = _generate(model, input_ids)

    # greedy decoding -> identical token sequence, identical forwards
    assert torch.equal(truth.sequences, run2.sequences)
    assert cap.n_steps == len(truth.hidden_states)
    for step in range(cap.n_steps):
        got = cap.get(step)  # (L, H) float32
        assert got.shape == (2, 32)
        for j, idx in enumerate(layers):
            want = truth.hidden_states[step][idx + 1][0, -1, :].float().numpy()
            assert np.allclose(got[j], want, atol=1e-6), f"step {step} layer {idx}"


def test_hooks_removed_after_context(tiny):
    """LayerCapture must remove exactly the hooks it added. (Baseline-relative:
    transformers' own generate(output_hidden_states=True) — run by the other
    test on this shared model — leaves ITS hooks on every layer permanently,
    which is amusingly one more reason to prefer this capture path.)"""
    model, input_ids = tiny
    before = [len(m._forward_hooks) for m in _decoder_layers(model)]
    with LayerCapture(model, [0]) as cap:
        during = [len(m._forward_hooks) for m in _decoder_layers(model)]
        _generate(model, input_ids)
    n = cap.n_steps
    _generate(model, input_ids)  # outside the context: must not capture
    assert cap.n_steps == n
    assert during[0] == before[0] + 1  # our hook was installed
    assert [len(m._forward_hooks) for m in _decoder_layers(model)] == before


def test_unsupported_architecture_rejected():
    class NotALM:
        pass

    with pytest.raises(ValueError):
        with LayerCapture(NotALM(), [0]):
            pass
