"""Contract: multimodal-wrapper loading for T7 (decisions/028 Amendment B).

Amendment B extends the collector to load models whose text tower is nested
inside a multimodal wrapper -- specifically T7's pre-registered primary,
`mistralai/Mistral-Small-3.2-24B-Instruct-2506`
(`Mistral3ForConditionalGeneration`). Three things must hold, and the flat
Llama/Qwen path must be provably UNCHANGED, because every number in the paper
so far was captured through it.

No weights and no GPU: the real-architecture checks instantiate from the
published config on the meta device.
"""

from __future__ import annotations

import pytest

from wta.hf_reader import _text_config, resolve_layers
from wta.layer_capture import _decoder_layers, _text_stack

MISTRAL3 = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"


# --- flat layout: must behave exactly as before ---------------------------

class _Layer:
    pass


class _FlatInner:
    def __init__(self, n):
        self.layers = [_Layer() for _ in range(n)]
        self.norm = "flat-norm"


class _FlatModel:
    def __init__(self, n=8):
        self.model = _FlatInner(n)


class _FlatConfig:
    num_hidden_layers = 8
    hidden_size = 512


def test_flat_layout_resolves_to_model_dot_model():
    m = _FlatModel(8)
    assert _text_stack(m) is m.model
    assert _decoder_layers(m) is m.model.layers
    assert _text_stack(m).norm == "flat-norm"


def test_flat_config_reads_top_level():
    cfg = _FlatConfig()
    assert _text_config(cfg) is cfg  # unchanged path
    assert _text_config(cfg).num_hidden_layers == 8


def test_unsupported_layout_still_raises():
    class _Bare:
        model = None

    with pytest.raises(ValueError, match="unsupported architecture"):
        _decoder_layers(_Bare())


# --- nested layout: the Amendment B addition ------------------------------

class _NestedInner:
    def __init__(self, n):
        self.language_model = _FlatInner(n)
        self.language_model.norm = "nested-norm"


class _NestedModel:
    def __init__(self, n=40):
        self.model = _NestedInner(n)


def test_nested_layout_resolves_to_language_model():
    m = _NestedModel(40)
    assert _text_stack(m) is m.model.language_model
    assert len(_decoder_layers(m)) == 40
    # the final norm MUST come from the same object that owns the blocks
    assert _text_stack(m).norm == "nested-norm"


def test_nested_config_prefers_text_config():
    class _Nested:
        num_hidden_layers = None
        hidden_size = None

        class text_config:  # noqa: N801
            num_hidden_layers = 40
            hidden_size = 5120

    cfg = _text_config(_Nested())
    assert cfg.num_hidden_layers == 40
    assert cfg.hidden_size == 5120


# --- the real published architecture --------------------------------------

def _meta_mistral3():
    """Instantiate T7's primary from its published config on the meta device
    (no weights fetched beyond config.json, no GPU)."""
    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    pytest.importorskip("torch")
    try:
        cfg = transformers.AutoConfig.from_pretrained(MISTRAL3)
        with accelerate.init_empty_weights():
            return transformers.AutoModelForImageTextToText.from_config(cfg), cfg
    except Exception as exc:  # offline box / hub unreachable
        pytest.skip(f"cannot reach {MISTRAL3} config: {type(exc).__name__}")


def test_real_mistral3_is_absent_from_causal_lm_mapping():
    """The reason the fallback load path exists at all. If this ever flips,
    Amendment B's diagnosis is stale and should be revisited."""
    m = pytest.importorskip(
        "transformers.models.auto.modeling_auto")
    assert "mistral3" not in m.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert "mistral" in m.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def test_real_mistral3_layout_and_depth():
    model, cfg = _meta_mistral3()
    # config: depth/width only reachable through text_config
    tc = _text_config(cfg)
    assert tc.num_hidden_layers == 40
    assert tc.hidden_size == 5120
    assert getattr(cfg, "num_hidden_layers", None) is None

    # modules: blocks nested under language_model, flat lookup absent
    assert getattr(model.model, "layers", None) is None
    layers = _decoder_layers(model)
    assert len(layers) == 40
    assert _text_stack(model) is model.model.language_model
    assert getattr(_text_stack(model), "norm", None) is not None


def test_real_mistral3_layer_fractions_resolve():
    """The frozen 0.2..0.85 fractions must map onto the second family's depth
    without any per-model tuning (Amendment B's stated boundary)."""
    _, cfg = _meta_mistral3()
    n = _text_config(cfg).num_hidden_layers
    idxs = resolve_layers(n, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85])
    assert len(idxs) == 8
    assert idxs == sorted(set(idxs))
    assert all(0 <= i < n for i in idxs)
