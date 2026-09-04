"""CIS teacher-forced scorer with a branched KV cache -- OURS (decisions/029).

The first teacher-forced / log-prob path in this repo (everything before it
called `model.generate`). One `DynamicCache` holds G_k (history through
assistant_{k-1}); each variant of the last user turn -- the original, and one
per injected resolution -- is forwarded as a BRANCH (user suffix + generation
header + recorded segment), scored, and cropped away. Then the cache is
EXTENDED with the history form of the turn to reach G_{k+1}.

Why not one causal forward over the whole transcript: Qwen3's template puts
the think header only on the generation prompt, so no single sequence
contains every per-turn context (see cis_context). Why the branch point is
G_k and not H_k: the injection edits the last user turn, so that turn must be
inside the branch.

Position capture forks `layer_capture.LayerCapture._make_hook`, which keeps
only `hidden[0, -1, :]`; here the requested positions (relative to the
branch input) are kept. Layer semantics, the final-norm rule and `_text_stack`
are reused unchanged, so a captured vector equals
`hidden_states[idx + 1][0, pos]` of the old path -- pinned by
harness/contract/test_cis_scorer.py on a tiny local Qwen2.

torch imports lazily (decisions/004: importable on the CPU laptop).
"""

from __future__ import annotations

import numpy as np

from wta.layer_capture import _text_stack


class PositionCapture:
    """Capture `hidden[0, positions, :]` at chosen layers for every forward
    inside the block. `positions` index the CURRENT forward's input (for a
    cached branch that is the branch input, not the absolute sequence)."""

    def __init__(self, model, layer_indices: list[int], positions: list[int]):
        self.layer_indices = list(layer_indices)
        self.positions = list(positions)
        self._buf: dict[int, list[np.ndarray]] = {i: [] for i in self.layer_indices}
        self._model = model
        self._handles = []

    def _make_hook(self, idx: int, final_norm):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            vec = hidden[0, self.positions, :]
            if final_norm is not None:
                vec = final_norm(vec)
            self._buf[idx].append(vec.detach().to("cpu").float().numpy())
        return hook

    def __enter__(self) -> "PositionCapture":
        stack = _text_stack(self._model)
        layers = stack.layers
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

    def get(self, step: int = 0) -> np.ndarray:
        """(L, P, H) float32 for one forward, layers in requested order."""
        return np.stack([self._buf[i][step] for i in self.layer_indices])


def load_model(model_id: str, dtype: str = "bfloat16", device: str = "cuda"):
    """Exactly HFStreamReader.__init__'s load: shared tokenizer loader
    (Amendment F), dtype kwarg per transformers major, device_map auto."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    from wta.hf_reader import _dtype_kwarg_name, _load_tokenizer
    tok = _load_tokenizer(model_id)
    kwargs: dict = {_dtype_kwarg_name(transformers.__version__): getattr(torch, dtype)}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return model, tok


class TeacherForcedScorer:
    """`prefill(G_k)`, then per variant `branch(ids, seg_offset, seg_len)`,
    then `extend(ids)` to G_{k+1}. All log-probs are fp32 numpy."""

    def __init__(self, model, layer_indices: list[int], logit_chunk: int = 512):
        import torch
        from transformers import DynamicCache
        self._torch = torch
        self._Cache = DynamicCache
        self.model = model
        self.layer_indices = list(layer_indices)
        self.logit_chunk = logit_chunk
        self.cache = None
        self.prefix_len = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _ids(self, ids: list[int]):
        return self._torch.tensor([ids], dtype=self._torch.long, device=self.device)

    def prefill(self, ids: list[int]) -> None:
        self.cache = self._Cache()
        self.prefix_len = 0
        if ids:
            self.extend(ids)

    def extend(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._torch.no_grad():
            out = self.model(input_ids=self._ids(ids), past_key_values=self.cache,
                             use_cache=True)
        self.cache = out.past_key_values
        self.prefix_len += len(ids)

    def _seg_logprobs(self, logits, ids: list[int], seg_offset: int,
                      seg_len: int) -> np.ndarray:
        """log p(ids[seg_offset + j] | ...) for j in [0, seg_len): the logit
        that predicts token t sits at position t - 1. Chunked so a 150k-vocab
        log_softmax over a 2k-token block never materializes at once."""
        torch = self._torch
        out = np.empty(seg_len, dtype=np.float32)
        tgt = torch.tensor(ids[seg_offset:seg_offset + seg_len],
                           device=logits.device)
        start = seg_offset - 1
        for c in range(0, seg_len, self.logit_chunk):
            sl = logits[0, start + c:start + c + min(self.logit_chunk, seg_len - c), :]
            lp = torch.log_softmax(sl.float(), dim=-1)
            out[c:c + lp.shape[0]] = lp.gather(
                1, tgt[c:c + lp.shape[0]][:, None])[:, 0].cpu().numpy()
        return out

    def branch(self, ids: list[int], seg_offset: int, seg_len: int,
               positions: list[int] | None = None):
        """Forward `ids` on top of the cached prefix, score the segment, crop
        the cache back. Returns (logprobs[seg_len], feats (L,P,H) or None)."""
        if self.cache is None:
            raise RuntimeError("prefill first")
        torch = self._torch
        feats = None
        with torch.no_grad():
            if positions:
                with PositionCapture(self.model, self.layer_indices, positions) as cap:
                    out = self.model(input_ids=self._ids(ids),
                                     past_key_values=self.cache, use_cache=True)
                feats = cap.get(0)
            else:
                out = self.model(input_ids=self._ids(ids),
                                 past_key_values=self.cache, use_cache=True)
            lp = self._seg_logprobs(out.logits, ids, seg_offset, seg_len)
        self.cache = out.past_key_values
        self.cache.crop(self.prefix_len)
        return lp, feats

    def from_scratch(self, ids: list[int], seg_offset: int, seg_len: int,
                     positions: list[int] | None = None):
        """Reference path with no cache: one forward over the full sequence.
        Used by the G0.4 contract test and the box-side equality check."""
        torch = self._torch
        feats = None
        with torch.no_grad():
            if positions:
                with PositionCapture(self.model, self.layer_indices, positions) as cap:
                    out = self.model(input_ids=self._ids(ids), use_cache=False)
                feats = cap.get(0)
            else:
                out = self.model(input_ids=self._ids(ids), use_cache=False)
            lp = self._seg_logprobs(out.logits, ids, seg_offset, seg_len)
        return lp, feats
