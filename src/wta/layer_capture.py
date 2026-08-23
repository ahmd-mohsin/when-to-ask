"""Forward-hook activation capture -- OURS (decisions/017 addendum).

Replaces ``generate(output_hidden_states=True)``, which materializes ALL
layers over the WHOLE prompt on every turn (a ~15-20 GB transient at 32B with
long agent histories -- an OOM in waiting on 4x24 GB). Hooks on just the
requested decoder layers grab only the LAST position of each forward pass:
kilobytes per step instead of gigabytes per turn, and it also speeds up 7B.

Captured values are identical to ``hidden_states[idx + 1][0, -1]`` from the
old path -- proven by the contract test on a tiny local Qwen2 (the +1 is the
embedding entry in transformers' hidden_states tuple; a forward hook on
``model.model.layers[idx]`` sees that block's output directly).
"""

from __future__ import annotations

import numpy as np


def _text_stack(model):
    """The module that OWNS the decoder blocks and the final norm.

    Flat Llama/Qwen-family causal LMs put both on ``model.model``. Multimodal
    wrappers (Mistral3ForConditionalGeneration, decisions/028 Amendment B)
    nest the text tower one level deeper, at ``model.model.language_model``,
    and expose no ``layers`` on ``model.model`` at all. The flat layout is
    tried FIRST, so every previously collected model resolves exactly as
    before and the R2 capture path is unchanged.
    """
    inner = getattr(model, "model", None)
    if getattr(inner, "layers", None) is not None:
        return inner
    nested = getattr(inner, "language_model", None)
    if getattr(nested, "layers", None) is not None:
        return nested
    raise ValueError(
        f"unsupported architecture for hook capture: {type(model).__name__} "
        "(expected model.model.layers, the Llama/Qwen layout, or "
        "model.model.language_model.layers, the multimodal-wrapper layout)")


def _decoder_layers(model):
    """The decoder-block list for Llama/Qwen-family causal LMs."""
    return _text_stack(model).layers


class LayerCapture:
    """Context manager: capture last-position hidden states at chosen layers
    for every forward pass inside the block (one pass per generated token,
    prefill included -- same per-step semantics as the old path)."""

    def __init__(self, model, layer_indices: list[int]):
        self.layer_indices = list(layer_indices)
        self._buf: dict[int, list[np.ndarray]] = {i: [] for i in self.layer_indices}
        self._model = model
        self._handles = []

    def _make_hook(self, idx: int, final_norm):
        # transformers' hidden_states[i+1] is the input to layer i+1 -- i.e.
        # layer i's raw output -- EXCEPT for the last layer, where the tuple
        # stores the output AFTER the model-level final norm. Reproduce that
        # exactly so hook capture is bit-compatible with the old path (and
        # with all previously collected data).
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            vec = hidden[0, -1, :]
            if final_norm is not None:
                vec = final_norm(vec)
            self._buf[idx].append(vec.detach().to("cpu").float().numpy())
        return hook

    def __enter__(self) -> "LayerCapture":
        stack = _text_stack(self._model)
        layers = stack.layers
        # final norm must come from the SAME module that owns the blocks, or
        # the last-layer post-norm semantics above silently break on nested
        # (multimodal) layouts.
        norm = getattr(stack, "norm", None)
        last = len(layers) - 1
        for idx in self.layer_indices:
            self._handles.append(layers[idx].register_forward_hook(
                self._make_hook(idx, norm if idx == last else None)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def n_steps(self) -> int:
        return len(self._buf[self.layer_indices[0]]) if self.layer_indices else 0

    def get(self, step: int) -> np.ndarray:
        """(L, H) float32 for one generation step, layers in requested order."""
        return np.stack([self._buf[i][step] for i in self.layer_indices])
