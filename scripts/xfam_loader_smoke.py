"""Engineering validation of the 028 Amendment B loader on REAL weights.

NOT an experiment and NOT a pre-registered gate -- it produces no paper
number. It answers one question before GPU time is committed to T7: does
`HFStreamReader` actually load a multimodal-wrapper model and capture
activations through the frozen collection settings, or does it only work on
the meta device where `test_multimodal_loader.py` checks it?

Needs no docker and no hil-bench images -- it drives the reader directly, so
it can run while the image-bucket credential is still missing.

    CUDA_VISIBLE_DEVICES=0 python scripts/xfam_loader_smoke.py \
        --model-id mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
        --out results/xfam_loader_smoke.json

Checks, all of which must hold for T7 to be launchable:
  1. the model loads at all (the CausalLM mapping does not contain mistral3)
  2. depth/width resolve through config.text_config -> 40 x 5120
  3. the frozen layer fractions 0.2..0.85 resolve to 8 distinct blocks
  4. hook capture returns one (8, hidden) vector per read, finite, non-zero
  5. the chat template applies and generation produces text
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

T0 = time.time()

LAYERS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]  # frozen, decisions/021
PROMPT = (
    "You are working in a Python repo. The test suite fails with a "
    "KeyError in config loading. Briefly describe the first two steps you "
    "would take to diagnose it, then state which file you would open first."
)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id",
                    default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--out", default="results/xfam_loader_smoke.json")
    args = ap.parse_args()

    from wta.hf_reader import HFStreamReader

    log(f"loading {args.model_id} (first run downloads weights) ...")
    # collection defaults, verbatim: cadence 8, thinking OFF, no top-k/top-p
    # truncation (decisions/021 R0)
    reader = HFStreamReader(
        args.model_id, layers=LAYERS, dtype="bfloat16", device="cuda",
        cadence=8, enable_thinking=False, top_p=1.0, top_k=0, min_p=0.0)
    log(f"loaded: {type(reader.model).__name__}")

    res: dict = {
        "model_id": args.model_id,
        "wrapper_class": type(reader.model).__name__,
        "n_layers": int(reader.n_layers),
        "hidden_dim": int(reader.hidden_dim),
        "layer_fractions": LAYERS,
        "layer_indices": [int(i) for i in reader.layer_indices],
        "generation_config": reader.effective_generation_config(),
    }
    log(f"n_layers={reader.n_layers} hidden={reader.hidden_dim} "
        f"layers={reader.layer_indices}")

    # A model can load fine and still be uncollectable: the agent loop builds
    # every turn with apply_chat_template, so a tokenizer that ships no HF
    # chat template is a hard stop, not a warning. Record it as-run instead of
    # crashing -- a failed check here IS the result.
    tmpl = getattr(reader.tokenizer, "chat_template", None)
    res["chat_template_set"] = bool(tmpl)
    res["tokenizer_class"] = type(reader.tokenizer).__name__
    if not tmpl:
        res["checks"] = {"loaded": True, "chat_template_set": False}
        res["PASS"] = False
        res["blocker"] = (
            "tokenizer ships no HF chat_template; apply_chat_template raises, "
            "so the collector cannot build agent turns for this model")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"FAIL chat_template_set -> {out}")
        log(f"VERDICT: FAIL ({res['blocker']})")
        return 1

    log("generating one segment with hook capture ...")
    reads, text = reader.generate_segment(
        [{"role": "user", "content": PROMPT}],
        seed=0, temperature=0.7, max_new_tokens=args.max_new_tokens,
        segment_idx=0)

    res["n_reads"] = len(reads)
    res["gen_chars"] = len(text)
    res["gen_head"] = text[:280]
    if reads:
        h = np.stack([r.h for r in reads]).astype(np.float32)
        res["h_shape_per_read"] = list(reads[0].h.shape)
        res["h_finite"] = bool(np.isfinite(h).all())
        res["h_nonzero_frac"] = float((h != 0).mean())
        res["h_abs_mean"] = float(np.abs(h).mean())
        res["h_norm_by_layer"] = [
            float(np.linalg.norm(h[:, i, :], axis=-1).mean())
            for i in range(h.shape[1])]
        res["triggers"] = sorted({r.trigger for r in reads})

    checks = {
        "loaded": True,
        "depth_width_resolved": reader.n_layers > 0 and reader.hidden_dim > 0,
        "eight_distinct_layers": len(set(reader.layer_indices)) == len(LAYERS),
        "reads_captured": len(reads) > 0,
        "h_shape_ok": bool(reads) and list(reads[0].h.shape) == [
            len(LAYERS), reader.hidden_dim],
        "h_finite": bool(res.get("h_finite", False)),
        "generated_text": len(text.strip()) > 0,
    }
    res["checks"] = checks
    res["PASS"] = all(checks.values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    log("")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log(f"n_reads={len(reads)} h/read={res.get('h_shape_per_read')} "
        f"gen_chars={len(text)}")
    log(f"VERDICT: {'PASS' if res['PASS'] else 'FAIL'} -> {out}")
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
