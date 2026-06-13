"""White-box model wrappers -- OURS.

The de-risk experiment needs the model's *mid-layer residual hidden state at a
decision point*, read mid-trajectory inside the agent loop (brief S9 go/no-go item).
That is not something the stock agent stacks expose, so we wrap the model ourselves.

Two implementations behind one interface (`WhiteBoxModel`):

  * `HFWhiteBoxModel`   -- a real Hugging Face causal LM (torch); used on a GPU box.
  * `FakeWhiteBoxModel` -- a numpy stand-in for CPU smoke tests. No torch, no
                           downloads. It is *structured*: it encodes the three blocker
                           regimes from brief S4 (fork / confident-convergent / clear)
                           so the end-to-end pipeline produces a meaningful C1 table on
                           synthetic data and the analysis code is actually exercised.

Everything downstream consumes plain numpy arrays, so the rest of the pipeline is
device- and dependency-agnostic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


@dataclass
class Generation:
    """One model step: the produced action text plus the internal state we read."""

    text: str
    hidden: np.ndarray  # (hidden_dim,) mid-layer residual at the decision point
    mean_logprob: float | None = None  # mean per-token logprob (a cheap confidence proxy)
    extra: dict = field(default_factory=dict)


@runtime_checkable
class WhiteBoxModel(Protocol):
    hidden_dim: int
    n_layers: int
    mid_layer: int

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.8,
        seed: int | None = None,
        max_new_tokens: int = 256,
    ) -> Generation:
        """Generate an action and return it alongside the mid-layer hidden state."""
        ...


def resolve_mid_layer(n_layers: int, spec: float | int) -> int:
    """A `mid_layer` config may be a fraction of depth (0.5) or an explicit index (16)."""
    if isinstance(spec, float) and 0.0 < spec < 1.0:
        return max(1, min(n_layers - 1, round(spec * n_layers)))
    return max(0, min(n_layers - 1, int(spec)))


# ---------------------------------------------------------------------------
# Real backbone (GPU)
# ---------------------------------------------------------------------------


class HFWhiteBoxModel:
    """Hugging Face causal LM exposing mid-layer hidden states.

    torch / transformers are imported lazily so this module stays importable (and the
    whole CPU smoke path stays runnable) on a machine without them.
    """

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        mid_layer: float | int = 0.5,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"torch_dtype": getattr(torch, dtype), "output_hidden_states": True}
        if device == "cuda":
            kwargs["device_map"] = "auto"
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        # hidden_states has n_layers+1 entries (embeddings + each block); count blocks.
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self.mid_layer = resolve_mid_layer(self.n_layers, mid_layer)

    def _format(self, prompt: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.8,
        seed: int | None = None,
        max_new_tokens: int = 256,
    ) -> Generation:
        torch = self._torch
        if seed is not None:
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
                output_scores=True,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # out.hidden_states: tuple over generated steps; each is a tuple over
        # (embeddings + layers); each layer tensor is (batch, seq, hidden_dim).
        # For each generated token take the mid layer's last-position vector, then average.
        # +1 because index 0 of the per-step tuple is the embedding layer.
        vecs = [
            step[self.mid_layer + 1][0, -1, :].float().cpu().numpy()
            for step in out.hidden_states
        ]
        hidden = np.mean(vecs, axis=0) if vecs else np.zeros(self.hidden_dim, dtype=np.float32)

        mean_logprob = None
        if out.scores:
            lps = []
            for logits, tok in zip(out.scores, gen_ids):
                lps.append(torch.log_softmax(logits[0].float(), dim=-1)[tok].item())
            mean_logprob = float(np.mean(lps)) if lps else None

        return Generation(text=text, hidden=hidden.astype(np.float32), mean_logprob=mean_logprob)


# ---------------------------------------------------------------------------
# Fake backbone (CPU smoke) -- structured to encode brief S4 regimes
# ---------------------------------------------------------------------------

# Control marker the synthetic harness embeds in observations so the fake model knows
# the latent regime of the current decision point. A real model would represent this
# internally; here the synthetic task makes it explicit. Example:
#   [[CTRL regime=fork k=3 dp=2 internal_from=1 output_from=2 gold=0]]
_CTRL_RE = re.compile(r"\[\[CTRL\s+([^\]]+)\]\]")


def _parse_ctrl(prompt: str) -> dict:
    m = _CTRL_RE.search(prompt)
    if not m:
        return {"regime": "clear", "dp": 0}
    out: dict = {}
    for kv in m.group(1).split():
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def _unit_vector(key: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-random unit vector keyed by a string (stable across runs)."""
    h = hashlib.sha256(key.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-9)


class FakeWhiteBoxModel:
    """Deterministic numpy model encoding the three regimes from brief S4.

    For a decision point, given the per-trajectory `seed` and the control marker:

      * clear              -- all trajectories share one hidden vector (low internal
                              dispersion), same action -> output agrees. Correct.
      * fork (k interp.)   -- trajectory picks interpretation `seed % k`; its hidden
                              vector separates by interpretation from `internal_from`
                              onward, its action separates from `output_from` onward
                              (>= internal_from, giving internal a lead-time). Correct
                              iff its interpretation == gold.
      * confident_wrong    -- all trajectories share ONE hidden vector in a distinct
                              "wrong" region (low internal dispersion, low output
                              divergence, high verbalized confidence) -- the divergence
                              blind spot; only a correctness probe separates it.

    This makes the synthetic C1 table reproduce the brief's predicted win regions, so
    the alignment / divergence / probe / separation code is genuinely exercised.
    """

    INTERP_SCALE = 6.0  # how far interpretations separate in hidden space
    NOISE = 0.15

    def __init__(self, hidden_dim: int = 64, n_layers: int = 8, mid_layer: float | int = 0.5):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.mid_layer = resolve_mid_layer(n_layers, mid_layer)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.8,
        seed: int | None = None,
        max_new_tokens: int = 256,
    ) -> Generation:
        ctrl = _parse_ctrl(prompt)
        regime = ctrl.get("regime", "clear")
        dp = int(ctrl.get("dp", 0))
        seed = 0 if seed is None else int(seed)

        # A base "topic" vector keyed by the task/decision-point context (seed-independent).
        # Each interpretation is a same-magnitude offset INTERP_SCALE * unit_vector(j), so
        # regimes differ only in WHICH interpretations the N trajectories occupy -- not in
        # magnitude. This keeps the magnitude-sensitive metrics (variance/eigenscore) honest:
        # clear and confident_wrong both converge on a single interpretation (low dispersion),
        # only fork spreads across interpretations (high dispersion).
        topic_key = _CTRL_RE.sub("", prompt).strip()
        base = _unit_vector(f"topic::{topic_key}::dp{dp}", self.hidden_dim)
        rng = np.random.default_rng(seed * 100003 + dp)
        noise = self.NOISE * rng.standard_normal(self.hidden_dim) * (0.5 + temperature)
        k = max(2, int(ctrl.get("k", 3)))
        gold = int(ctrl.get("gold", 0))

        def interp_hidden(j: int) -> np.ndarray:
            return base + self.INTERP_SCALE * _unit_vector(f"interp::{topic_key}::{j}", self.hidden_dim)

        if regime == "fork":
            internal_from = int(ctrl.get("internal_from", 0))
            output_from = int(ctrl.get("output_from", internal_from))
            interp = seed % k
            # Internal interpretation diverges from internal_from; the emitted action only
            # diverges from output_from -> a built-in lead window for internal divergence.
            j_internal = interp if dp >= internal_from else gold
            j_output = interp if dp >= output_from else gold
            hidden = interp_hidden(j_internal) + noise
            text = f"ANSWER interp={j_output}"
            correct = interp == gold
            base_conf = 0.6  # genuinely unsure on a fork
        elif regime in ("confident_wrong", "convergent_wrong"):
            wrong_j = (gold + 1) % k  # all trajectories pick the SAME wrong interpretation
            hidden = interp_hidden(wrong_j) + noise
            text = f"ANSWER interp={wrong_j}"  # outputs agree -> B1 misses it
            correct = False
            # Overconfident wrong belief (brief S4): reports the SAME high confidence as a
            # clear case, so verbalized confidence (B2) is blind to this regime.
            base_conf = 0.9
        else:  # clear -- all trajectories converge on the gold interpretation
            hidden = interp_hidden(gold) + noise
            text = f"ANSWER interp={gold}"
            correct = True
            base_conf = 0.9

        # Light noise so B2/AUROC are not trivially perfect; clear vs confident_wrong stay
        # indistinguishable in confidence (the honest B2 limitation).
        confidence = float(np.clip(base_conf + 0.04 * rng.standard_normal(), 0.02, 0.98))

        return Generation(
            text=text,
            hidden=hidden.astype(np.float32),
            mean_logprob=float(np.log(confidence)),
            extra={"regime": regime, "dp": dp, "correct": correct, "confidence": confidence},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(cfg: dict) -> WhiteBoxModel:
    """Build a backbone from a config dict (the `backbone:` block of a run config)."""
    kind = cfg.get("kind", "fake")
    if kind == "fake":
        return FakeWhiteBoxModel(
            hidden_dim=cfg.get("hidden_dim", 64),
            n_layers=cfg.get("n_layers", 8),
            mid_layer=cfg.get("mid_layer", 0.5),
        )
    if kind == "hf":
        return HFWhiteBoxModel(
            cfg["model_id"],
            dtype=cfg.get("dtype", "bfloat16"),
            device=cfg.get("device", "cuda"),
            mid_layer=cfg.get("mid_layer", 0.5),
            load_in_4bit=cfg.get("load_in_4bit", False),
        )
    raise ValueError(f"unknown backbone kind: {kind!r}")
