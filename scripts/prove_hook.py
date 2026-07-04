"""Phase-0 hook proof (spec A0, check 4): run on the AWS GPU instance.

Proves we can read mid-layer residuals at cadence/cue positions INSIDE the
generation loop on a real model, N=2 -- the build brief's Phase-0 go/no-go.

    python scripts/prove_hook.py                       # defaults
    python scripts/prove_hook.py --model-id <hf-id> --max-new-tokens 384

Prints a PASS/FAIL line per spec condition and writes RunLogs to
logs/prove_hook/. Honest reporting: failures print, nothing retries silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.hf_reader import HFStreamReader  # noqa: E402
from wta.logging_schema import save_run_log  # noqa: E402

# An under-specified task in the HiL-Bench spirit: several defensible
# interpretations (retry scope, backoff policy, which errors count), so two
# seeds have something to genuinely deliberate about.
DEFAULT_PROMPT = """You are a coding agent working on a Python service.

Task: "Make the HTTP client retry failed requests."

Think through how you would implement this, step by step, reasoning about any
choices the task leaves open, then state the exact change you would make."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--mid-layer", type=float, default=0.5)
    ap.add_argument("--cadence", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="logs/prove_hook")
    args = ap.parse_args()

    reader = HFStreamReader(args.model_id, mid_layer=args.mid_layer, cadence=args.cadence)
    print(f"model={args.model_id} layers={reader.n_layers} H={reader.hidden_dim} "
          f"mid_layer={reader.mid_layer}")

    logs = []
    for seed in (0, 1):
        log, text = reader.run(DEFAULT_PROMPT, run_id=f"provehook-seed{seed}",
                               task_id="prove_hook", seed=seed, temperature=0.8,
                               max_new_tokens=args.max_new_tokens)
        save_run_log(log, args.out)
        logs.append(log)
        cues = sum(1 for r in log.reads if r.trigger == "cue")
        print(f"seed {seed}: {len(log.reads)} reads ({cues} cue-triggered), "
              f"text {len(text)} chars, saved to {args.out}/")

    ok = True
    for log in logs:
        m = log.read_matrix().astype(np.float32)
        checks = {
            f"{log.run_id}: >=5 reads": m.shape[0] >= 5,
            f"{log.run_id}: h dim == hidden_size": m.shape[1] == reader.hidden_dim,
            f"{log.run_id}: nonzero variance across reads": m.shape[0] >= 2
            and float(m.var(axis=0).mean()) > 0,
        }
        for name, passed in checks.items():
            print(("PASS " if passed else "FAIL ") + name)
            ok &= passed
    print("HOOK PROOF:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
