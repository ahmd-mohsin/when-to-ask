"""Per-read mid-layer capture during generation (GPU path) -- OURS, with cited
patterns.

Capture mechanics follow xtid's `HFWhiteBoxModel` (per-step hidden states via
`output_hidden_states`; src/xtid/backbone/model.py) -- but WITHOUT its
per-action averaging, which is exactly the boundary-read pattern decisions/006
forbids. Position-selected activation reading is the RepE pattern
(third_party/representation-engineering/PROVENANCE.md); positions come from
`wta.reads.StreamReadSelector` (cadence + cues).

torch/transformers import lazily: this module must be importable on the CPU
laptop (decisions/004), but only `prove_hook.py` / real A0 collection call it,
on the owner's AWS GPU instance.
"""

from __future__ import annotations

import numpy as np

from wta.layer_capture import LayerCapture
from wta.logging_schema import ReadRecord, RunLog
from wta.reads import DEFAULT_CUES, StreamReadSelector

# mid-layer resolution reused from the frozen xtid artifact (decisions/001).
from xtid.backbone.model import resolve_mid_layer


def resolve_layers(n_layers: int, specs: list[float | int]) -> list[int]:
    """Resolve a list of layer specs (fractions in (0,1) or explicit indices) to
    distinct, sorted layer indices via `resolve_mid_layer`. Pure -- unit-tested
    without torch (decisions/014)."""
    idxs = sorted({resolve_mid_layer(n_layers, s) for s in specs})
    if not idxs:
        raise ValueError("no layers resolved")
    return idxs


class HFStreamReader:
    """Generate with a frozen HF causal LM, reading mid-layer residuals at
    cadence/cue positions during the reasoning span."""

    def __init__(self, model_id: str, *, mid_layer: float | int = 0.5,
                 layers: list[float | int] | None = None,
                 dtype: str = "bfloat16", device: str = "cuda",
                 cadence: int = 32, cues: tuple[str, ...] = DEFAULT_CUES,
                 value_pattern: str | None = None, value_cooldown: int = 8,
                 load_in_4bit: bool = False,
                 enable_thinking: bool | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.cadence = cadence
        self.cues = cues
        # decisions/019: Qwen3 chat templates default to thinking mode ON;
        # False pins non-thinking (a no-op variable for templates that don't
        # read it, e.g. Qwen2.5); None omits the kwarg entirely (legacy).
        self.enable_thinking = enable_thinking
        self.value_pattern = value_pattern
        self.value_cooldown = value_cooldown
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"torch_dtype": getattr(torch, dtype)}
        if device == "cuda":
            kwargs["device_map"] = "auto"
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self.mid_layer = resolve_mid_layer(self.n_layers, mid_layer)
        # Multi-layer capture (decisions/014): resolve each spec to an index and
        # store the ORDER we stack them in. None -> single mid layer (legacy).
        self.layer_indices = (
            resolve_layers(self.n_layers, layers) if layers else None
        )
        # what the hook capture actually grabs (single-layer mode still hooks
        # exactly one layer; the saved h stays 1-D for schema compatibility)
        self._capture_layers = self.layer_indices or [self.mid_layer]

    def _template_kwargs(self) -> dict:
        return ({} if self.enable_thinking is None
                else {"enable_thinking": self.enable_thinking})

    def _format(self, prompt: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, **self._template_kwargs(),
            )
        return prompt

    def generate_segment(self, messages: list[dict], *, seed: int,
                         temperature: float, max_new_tokens: int,
                         segment_idx: int) -> tuple[list, str]:
        """One agent TURN (v2, decisions/017): generate from a chat-history
        message list, reading residuals at cadence/cue/value positions. Returns
        (reads for this segment with segment_idx set, generated text). The
        caller accumulates segments into one RunLog. A fresh selector per
        segment (cue buffers must not leak across turns); seed is mixed with
        segment_idx so turns differ deterministically."""
        torch = self._torch
        torch.manual_seed(seed * 100003 + segment_idx)
        text_in = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **self._template_kwargs())
        inputs = self.tokenizer(text_in, return_tensors="pt").to(self.model.device)
        with torch.no_grad(), LayerCapture(self.model, self._capture_layers) as cap:
            out = self.model.generate(
                **inputs, do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        selector = StreamReadSelector(cadence=self.cadence, cues=self.cues,
                                      value_pattern=self.value_pattern,
                                      value_cooldown=self.value_cooldown)
        reads, prev_text = [], ""
        n_steps = min(len(gen_ids), cap.n_steps)
        for step_idx in range(n_steps):
            text_so_far = self.tokenizer.decode(gen_ids[: step_idx + 1],
                                                skip_special_tokens=True)
            delta, prev_text = text_so_far[len(prev_text):], text_so_far
            hit = selector.step(delta)
            if hit is None:
                continue
            h = cap.get(step_idx)
            if self.layer_indices is None:
                h = h[0]
            reads.append(ReadRecord(token_idx=step_idx, trigger=hit.trigger,
                                    cue=hit.cue, h=h.astype(np.float16),
                                    segment_idx=segment_idx))
        return reads, text

    def run(self, prompt: str, *, run_id: str, task_id: str, seed: int,
            temperature: float = 0.8, max_new_tokens: int = 512) -> tuple[RunLog, str]:
        """One generation pass -> (RunLog with per-read h, generated text)."""
        torch = self._torch
        torch.manual_seed(seed)
        inputs = self.tokenizer(self._format(prompt), return_tensors="pt").to(self.model.device)
        with torch.no_grad(), LayerCapture(self.model, self._capture_layers) as cap:
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        selector = StreamReadSelector(cadence=self.cadence, cues=self.cues,
                                      value_pattern=self.value_pattern,
                                      value_cooldown=self.value_cooldown)
        log = RunLog(run_id=run_id, task_id=task_id, seed=seed,
                     temperature=temperature, model_id=self.model_id,
                     mid_layer=self.mid_layer, layers=self.layer_indices)
        # Hook capture (wta/layer_capture.py): one (L, H) per generation step,
        # last position only, chosen layers only -- equivalent to the old
        # output_hidden_states path (contract-tested) without its multi-GB
        # prompt-wide transient. Read ONLY where the selector fires (no
        # averaging; decisions/006).
        #
        # Token text comes from DELTAS of the progressively decoded prefix, not
        # from decode([tok_id]): single-token decode drops BPE space markers
        # and mangles multi-byte characters split across tokens, which would
        # feed the cue matcher "letme" instead of "let me".
        prev_text = ""
        n_steps = min(len(gen_ids), cap.n_steps)
        for step_idx in range(n_steps):
            text_so_far = self.tokenizer.decode(
                gen_ids[: step_idx + 1], skip_special_tokens=True
            )
            delta, prev_text = text_so_far[len(prev_text):], text_so_far
            hit = selector.step(delta)
            if hit is None:
                continue
            h = cap.get(step_idx)  # (L, H); single-layer mode -> squeeze to (H,)
            if self.layer_indices is None:
                h = h[0]
            log.reads.append(ReadRecord(token_idx=step_idx, trigger=hit.trigger,
                                        cue=hit.cue, h=h.astype(np.float16)))
        log.validate()
        return log, text
