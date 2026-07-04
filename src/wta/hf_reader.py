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

from wta.logging_schema import ReadRecord, RunLog
from wta.reads import DEFAULT_CUES, StreamReadSelector

# mid-layer resolution reused from the frozen xtid artifact (decisions/001).
from xtid.backbone.model import resolve_mid_layer


class HFStreamReader:
    """Generate with a frozen HF causal LM, reading mid-layer residuals at
    cadence/cue positions during the reasoning span."""

    def __init__(self, model_id: str, *, mid_layer: float | int = 0.5,
                 dtype: str = "bfloat16", device: str = "cuda",
                 cadence: int = 32, cues: tuple[str, ...] = DEFAULT_CUES,
                 load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.cadence = cadence
        self.cues = cues
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"torch_dtype": getattr(torch, dtype), "output_hidden_states": True}
        if device == "cuda":
            kwargs["device_map"] = "auto"
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self.mid_layer = resolve_mid_layer(self.n_layers, mid_layer)

    def _format(self, prompt: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def run(self, prompt: str, *, run_id: str, task_id: str, seed: int,
            temperature: float = 0.8, max_new_tokens: int = 512) -> tuple[RunLog, str]:
        """One generation pass -> (RunLog with per-read h, generated text)."""
        torch = self._torch
        torch.manual_seed(seed)
        inputs = self.tokenizer(self._format(prompt), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        selector = StreamReadSelector(cadence=self.cadence, cues=self.cues)
        log = RunLog(run_id=run_id, task_id=task_id, seed=seed,
                     temperature=temperature, model_id=self.model_id,
                     mid_layer=self.mid_layer)
        # out.hidden_states: one tuple per generated step; each = (embeddings +
        # per-layer tensors of shape (batch, seq, H)). +1 skips the embedding
        # entry. Read the last position's vector at the step where the selector
        # fires -- and ONLY there (no averaging; decisions/006).
        #
        # Token text comes from DELTAS of the progressively decoded prefix, not
        # from decode([tok_id]): single-token decode drops BPE space markers
        # and mangles multi-byte characters split across tokens, which would
        # feed the cue matcher "letme" instead of "let me".
        prev_text = ""
        n_steps = min(len(gen_ids), len(out.hidden_states))
        for step_idx in range(n_steps):
            text_so_far = self.tokenizer.decode(
                gen_ids[: step_idx + 1], skip_special_tokens=True
            )
            delta, prev_text = text_so_far[len(prev_text):], text_so_far
            hit = selector.step(delta)
            if hit is None:
                continue
            h = out.hidden_states[step_idx][self.mid_layer + 1][0, -1, :].float().cpu().numpy()
            log.reads.append(ReadRecord(token_idx=step_idx, trigger=hit.trigger,
                                        cue=hit.cue, h=h.astype(np.float16)))
        log.validate()
        return log, text
