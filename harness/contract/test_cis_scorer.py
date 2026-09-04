"""Contract: the branched-KV-cache scorer equals a from-scratch forward
(decisions/029 G0.4), and position capture equals output_hidden_states.

Tiny Qwen2 from config (the test_layer_capture fixture), fp32, no download.
Sequences are random ids: the scorer is tokenizer-agnostic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wta.cis_scorer import PositionCapture, TeacherForcedScorer


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=64, vocab_size=99,
                      max_position_embeddings=128)
    return Qwen2ForCausalLM(cfg).eval()


def _rand(n, rng):
    return [int(x) for x in rng.integers(0, 99, size=n)]


def test_branch_equals_from_scratch_and_crop_restores(tiny):
    rng = np.random.default_rng(1)
    sc = TeacherForcedScorer(tiny, layer_indices=[1, 3])
    G = _rand(20, rng)
    sc.prefill(G)
    assert sc.prefix_len == 20 and sc.cache.get_seq_length() == 20

    hdr, seg = _rand(3, rng), _rand(10, rng)
    for _ in range(3):                                    # three variants
        user = _rand(6, rng)
        branch = user + hdr + seg
        off = len(user) + len(hdr)
        pos = [off - 1, off + 4]                         # P_k-1 and a mid-seg position
        lp_b, f_b = sc.branch(branch, off, len(seg), positions=pos)
        lp_s, f_s = sc.from_scratch(G + branch, len(G) + off, len(seg),
                                    positions=[len(G) + p for p in pos])
        assert lp_b.shape == (10,) and np.allclose(lp_b, lp_s, atol=1e-5)
        assert f_b.shape == (2, 2, 32) and np.allclose(f_b, f_s, atol=1e-5)
        assert sc.cache.get_seq_length() == 20            # cropped back

    # extend to G_{k+1} and branch again: still equals from-scratch
    ext = _rand(9, rng)
    sc.extend(ext)
    assert sc.prefix_len == 29
    user2, seg2 = _rand(5, rng), _rand(7, rng)
    b2 = user2 + hdr + seg2
    off2 = len(user2) + len(hdr)
    lp_b, _ = sc.branch(b2, off2, len(seg2))
    lp_s, _ = sc.from_scratch(G + ext + b2, len(G) + len(ext) + off2, len(seg2))
    assert np.allclose(lp_b, lp_s, atol=1e-5)


def test_position_capture_equals_output_hidden_states(tiny):
    rng = np.random.default_rng(2)
    ids = torch.tensor([_rand(15, rng)])
    with torch.no_grad():
        truth = tiny(input_ids=ids, output_hidden_states=True, use_cache=False)
    layers, pos = [0, 3], [2, 9, 14]
    with PositionCapture(tiny, layers, pos) as cap:
        with torch.no_grad():
            tiny(input_ids=ids, use_cache=False)
    got = cap.get(0)                                       # (L, P, H)
    assert got.shape == (2, 3, 32)
    for j, idx in enumerate(layers):
        want = truth.hidden_states[idx + 1][0, pos, :].float().numpy()
        assert np.allclose(got[j], want, atol=1e-6), f"layer {idx}"
    # last layer takes the final norm, exactly as LayerCapture does
    with PositionCapture(tiny, [3], [14]) as cap:
        with torch.no_grad():
            tiny(input_ids=ids, use_cache=False)
    assert np.allclose(cap.get(0)[0, 0], truth.hidden_states[4][0, 14].numpy(), atol=1e-6)


def test_logprob_gather_is_teacher_forced(tiny):
    """log p(seg[j]) must use the logit at position (seg_offset + j - 1)."""
    rng = np.random.default_rng(3)
    sc = TeacherForcedScorer(tiny, layer_indices=[1], logit_chunk=4)
    ids = _rand(24, rng)
    off, n = 10, 9
    lp, _ = sc.from_scratch(ids, off, n)
    with torch.no_grad():
        logits = tiny(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
    ref = torch.log_softmax(logits.float(), -1)
    want = np.array([ref[off + j - 1, ids[off + j]].item() for j in range(n)])
    assert np.allclose(lp, want, atol=1e-6)
